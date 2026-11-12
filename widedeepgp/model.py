# -*- coding: utf-8 -*-
"""WideDeepGP: the DeepGP network with one output per disease, and its training logic.

The network is assembled from the published building blocks and keeps the published
parameter names (``dataEmbed``, ``encoders``, ``norm_f``, ``mhsa``, ``covariate_encoder``,
``cov_norm``, ``combine``), so weights trained with ``models.DeepGP.SNPCOV_Model_mamba`` load
into it. What differs:

* the last linear layer has one output per label and the network returns logits (the
  sigmoid lives in the loss);
* the number of input sequences and of covariates comes from the data;
* the sequence block comes from ``blocks.make_block``;
* under bf16 the encoders run in bf16 while pooling, the head and the loss stay in float32;
* each sequence's encoder can run under activation checkpointing;
* fixed tables (positions, annotations) are not written into saved models; a digest of each
  is, and a model is not rebuilt with tables that differ from the ones it was trained with;
* optional per-SNP annotation vectors scale and shift the token embedding.
"""
import hashlib
import math
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from layers.Embed import DataEmbedding, DataEmbeddingVocab
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import EncoderLayer
from models.DeepGP import FCN, RMSNorm, layer_norm_fn

from . import metrics as M
from .blocks import make_block
from .losses import WideLoss

POOLS = ("atten", "mean", "last")


def positions_digest(pos):
    h = hashlib.sha256()
    for p in pos:
        h.update(np.ascontiguousarray(np.asarray(p, dtype=np.int64)).tobytes())
    return h.hexdigest()[:16]


def annotations_digest(annot):
    """Digest of the annotation matrices a model is given (standardised, float32): a rewritten
    annotation file changes it."""
    h = hashlib.sha256()
    for a in annot:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _drop_from_state(module, name):
    """Keep a buffer on the module but out of state_dict (it is rebuilt from the data)."""
    value = module._buffers.pop(name)
    module.register_buffer(name, value, persistent=False)


class WideDeepGP(pl.LightningModule):
    """hp: a plain dict (it is saved with every checkpoint) -- see ``train.model_hparams``.
    pos: one array of positions per input sequence. annot: optional list of float arrays
    [L_i, A], one per sequence, already standardised.
    """

    def __init__(self, hp, pos, annot=None):
        super().__init__()
        hp = dict(hp)
        self.save_hyperparameters({"hp": hp})
        self.hp = hp
        d = int(hp["d_model"])
        self.n_seq = len(pos)
        self.label_names = list(hp["label_names"])
        self.n_labels = len(self.label_names)
        self.use_cov = bool(hp["use_cov"])
        self.final_pool = hp["final_pool"]
        if self.final_pool not in POOLS:
            raise ValueError(f"final_pool must be one of {POOLS} (the parser's default 'concat' is not implemented "
                             f"in the published model either)")
        self.bf16 = hp.get("precision", "32") == "bf16"
        self.grad_ckpt = bool(hp.get("grad_ckpt", False))
        # Mamba-3's backward reads its saved tensors more than once, which the non-reentrant
        # checkpoint forbids; its encoders are checkpointed the re-entrant way instead.
        self.ckpt_reentrant = hp.get("mixer", "bimamba1") == "mamba3"
        if hp["pos_digest"] != positions_digest(pos) or [len(p) for p in pos] != list(hp["seq_len"]):
            raise ValueError("the SNP positions given do not match the ones this model was configured for")
        embed = {"cov": DataEmbedding, "vocab": DataEmbeddingVocab}[hp["snp_embed"]]

        # built in the order the published class builds them, so one seed gives the same initial weights
        encoders, data_embed, norms = [], [], []
        for p in pos:
            encoders.append(nn.ModuleList([make_block(d, hp.get("mixer", "bimamba1"), bool(hp.get("bi_direct", False)),
                                                      hp.get("m3_d_state")) for _ in range(int(hp["e_layers"]))]))
            data_embed.append(embed(int(hp["enc_in"]), d, np.asarray(p), float(hp["dropout"])))
            norms.append(RMSNorm(d, eps=1e-5))
        self.encoders = nn.ModuleList(encoders)
        self.dataEmbed = nn.ModuleList(data_embed)
        self.norm_f = nn.ModuleList(norms)
        self.mhsa = EncoderLayer(AttentionLayer(FullAttention(mask_flag=False), d_model=d, n_heads=int(hp["n_heads"])),
                                 d_model=d)
        n_vec = self.n_seq
        if self.use_cov:
            self.covariate_encoder = FCN(int(hp["n_cov"]), d)
            self.cov_norm = RMSNorm(d, eps=1e-5)
            n_vec += 1
        self.combine = FCN(n_vec * d, self.n_labels, hidden_dim=int(hp.get("head_hidden", 256)),
                           hidden_layers=int(hp["n_fcn"]))
        for emb in self.dataEmbed:                       # position tables are a function of the SNP list
            _drop_from_state(emb.position_embedding, "pe")

        # per-SNP annotation vectors: one gain map and one offset map shared by all sequences
        self.annot_dim = int(hp.get("annot_dim", 0))
        if self.annot_dim:
            if annot is None or len(annot) != self.n_seq:
                raise ValueError("this model uses annotation vectors: pass one [L_i, A] array per sequence")
            self.annot_gain = nn.Linear(self.annot_dim, d)
            self.annot_offset = nn.Linear(self.annot_dim, d)
            for lin in (self.annot_gain, self.annot_offset):   # start as the un-annotated model
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            for i, a in enumerate(annot):
                a = torch.as_tensor(np.asarray(a), dtype=torch.float32)
                if a.shape != (len(pos[i]), self.annot_dim):
                    raise ValueError(f"annotations of sequence {i} have shape {tuple(a.shape)}, "
                                     f"expected {(len(pos[i]), self.annot_dim)}")
                self.register_buffer(f"annot_{i}", a, persistent=False)
            if hp.get("annot_digest") is None:
                warnings.warn("this model was saved without a digest of its annotation vectors, so the ones given "
                              "cannot be checked against those it was trained with")
            elif hp["annot_digest"] != annotations_digest(annot):
                raise ValueError("the annotation vectors given differ from the ones this model was trained with "
                                 "(has the annotation file been rewritten?)")

        # a model saved before case_share was recorded was trained on even batches: exactly half cases
        self.loss_fn = WideLoss(counts=hp["label_counts"], gamma=float(hp["gamma"]), w_max=float(hp["w_max"]),
                                sampler=hp["sampler"], correction=bool(hp["sampling_correction"]),
                                outputs=hp["loss_outputs"], case_share=float(hp.get("case_share", 0.5)))
        self.lr, self.l2 = float(hp["lr"]), float(hp["l2"])
        self._val, self._train_sum, self._train_cnt = [], None, None
        self.last_val_metrics = None
        self.samples_seen = 0

    # ---- network -------------------------------------------------------------------------------
    def _encode(self, i, x):
        """One input sequence [B, L, 1] -> its pooled vector [B, d_model], in float32."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and x.is_cuda):
            emb = self.dataEmbed[i]
            v = emb.value_embedding(x)
            if self.annot_dim:
                a = getattr(self, f"annot_{i}")
                v = v * (1.0 + self.annot_gain(a)) + self.annot_offset(a)
            h = emb.dropout(v + emb.position_embedding(x))
            residual = None
            for layer in self.encoders[i]:
                h, residual = layer(h, residual)
            norm = self.norm_f[i]
            h = layer_norm_fn(h, norm.weight, norm.bias, eps=norm.eps, residual=residual, prenorm=False,
                              residual_in_fp32=True, is_rms_norm=True)
        if self.final_pool == "last":
            return h[:, -1, :].float()
        # The mean over a chromosome is taken in float32: people differ by a few percent of the
        # pooled vector, which is finer than bf16 resolves.
        return h.float().mean(dim=1)

    def _encode_anchored(self, i, x, anchor):
        return self._encode(i, x) + 0.0 * anchor

    def forward(self, x_all):
        """x_all: the input sequences, then the covariates. Returns logits [B, n_labels]."""
        x_seq = x_all[:self.n_seq]
        pooled = []
        ckpt = self.grad_ckpt and self.training
        if ckpt and self.ckpt_reentrant:
            # the re-entrant checkpoint only carries gradients if one of its inputs requires them
            anchor = torch.zeros((), device=x_seq[0].device, requires_grad=True)
        for i in range(self.n_seq):
            if ckpt and self.ckpt_reentrant:
                pooled.append(checkpoint(self._encode_anchored, i, x_seq[i], anchor, use_reentrant=True))
            elif ckpt:
                pooled.append(checkpoint(self._encode, i, x_seq[i], use_reentrant=False))
            else:
                pooled.append(self._encode(i, x_seq[i]))
        with torch.autocast("cuda", enabled=False):
            if self.use_cov:
                pooled.append(self.cov_norm(self.covariate_encoder(x_all[-1].float())))
            if self.final_pool == "atten":
                out, _ = self.mhsa(torch.stack(pooled, dim=1))
                x = out.reshape(out.size(0), -1)
            else:
                x = torch.cat(pooled, dim=1)
            return self.combine(x)

    def population_logits(self, logits):
        """Logits on the population scale (training weights cases; see losses.WideLoss)."""
        return logits.float() - self.loss_fn.logit_shift.to(logits.device)

    # ---- training ------------------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x_all, y, target = batch
        logits = self(x_all)
        loss, per_label = self.loss_fn(logits, y, int(target))
        self.samples_seen += int(y.shape[0])
        if self._train_sum is None:
            self._train_sum = torch.zeros_like(per_label, dtype=torch.float64)
            self._train_cnt = torch.zeros_like(per_label, dtype=torch.float64)
        t = int(target)
        if self.loss_fn.outputs == "target":
            self._train_sum[t] += per_label[t].double()
            self._train_cnt[t] += 1
        else:
            self._train_sum += per_label.double()
            self._train_cnt += 1
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=y.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x_all, y = batch[0], batch[1]
        self._val.append((self(x_all).float().cpu(), y.float().cpu()))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        logits = torch.cat([a for a, _ in self._val]).numpy()
        y = torch.cat([b for _, b in self._val]).numpy()
        self._val = []
        m = M.per_label_metrics(logits, y, self.loss_fn.logit_shift.cpu().numpy(), self.label_names)
        self.last_val_metrics = m
        self.log("val/auroc_mean", m["auroc_mean"], prog_bar=True)
        self.log("val/ap_mean", m["ap_mean"])
        self.log("val/bce_mean", m["bce_mean"])
        self.log("samples_seen", float(self.samples_seen))
        for j, name in enumerate(self.label_names):
            for key in ("auroc", "ap", "bce"):
                if not math.isnan(float(m[key][j])):
                    self.log(f"val_{key}/{name}", float(m[key][j]))
        if self._train_sum is not None:
            mean = (self._train_sum / self._train_cnt.clamp_min(1)).cpu().numpy()
            cnt = self._train_cnt.cpu().numpy()
            for j, name in enumerate(self.label_names):
                if cnt[j] > 0:
                    self.log(f"train_loss/{name}", float(mean[j]))
            self._train_sum.zero_()
            self._train_cnt.zero_()

    def predict_step(self, batch, batch_idx):
        return self(batch[0]).float().cpu(), batch[1].float().cpu()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        warm = int(self.hp.get("warmup_steps", 0) or 0)
        sched = self.hp.get("lr_schedule", "constant")
        total = int(self.hp.get("total_optimizer_steps", 0) or 0)
        if warm == 0 and sched == "constant":
            return opt                                   # as published: constant learning rate

        def factor(step):
            f = min(1.0, (step + 1) / warm) if warm else 1.0
            if sched == "cosine" and total > warm:
                f *= 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, (step - warm) / (total - warm)))))
            return f

        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, factor),
                                                   "interval": "step"}}

    # ---- saving and loading --------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint):
        checkpoint["samples_seen"] = int(self.samples_seen)

    def on_load_checkpoint(self, checkpoint):
        self.samples_seen = int(checkpoint.get("samples_seen", 0))
        sd = checkpoint.get("state_dict", {})
        for k in [k for k in sd if k.endswith("position_embedding.pe")]:     # written by the published class
            del sd[k]

    def load_weights(self, state_dict, allow_new_head=False, source_labels=None):
        """Load weights saved by this class or by the published one.

        Fixed tables in the source are ignored and the loss's constants are kept as configured.
        With allow_new_head, the output layer is left at its initial values when the source was
        trained for other diseases: a layer of a different size, or, when ``source_labels`` (the
        source's label names) is given, other names or another order, since each row of the
        layer belongs to one disease. Everything else must match.
        Returns the names that were not loaded.
        """
        own = self.state_dict()
        src = {k: v for k, v in state_dict.items() if not k.endswith("position_embedding.pe")
               and not k.startswith("loss_fn.")}
        skipped = []
        last = f"combine.layers.{len(self.combine.layers) - 1}."
        other_labels = source_labels is not None and list(source_labels) != self.label_names
        for k in list(src):
            if allow_new_head and other_labels and k.startswith(last):
                skipped.append(k)
                del src[k]
            elif k in own and own[k].shape != src[k].shape:
                if allow_new_head and k.startswith(last):
                    skipped.append(k)
                    del src[k]
                else:
                    raise ValueError(f"{k}: saved shape {tuple(src[k].shape)} but the model has {tuple(own[k].shape)}")
        unknown = [k for k in src if k not in own]
        if unknown:
            raise KeyError(f"weights not in this model: {unknown[:5]}")
        missing = [k for k in own if k not in src and not k.startswith("loss_fn.") and k not in skipped]
        annot_only = all(k.startswith(("annot_gain.", "annot_offset.")) for k in missing)
        if missing and not annot_only:
            raise KeyError(f"weights missing from the source: {missing[:5]}")
        self.load_state_dict(src, strict=False)
        return skipped + missing
