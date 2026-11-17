"""WideDeepGP: the DeepGP network with one logit per disease, and its training logic (methods.md
sections 4 and 10). Two input modes: 'blocks' (the default of main_multilabel.py), a per-SNP table
pooled by LD block (blockencoder.py) and one mixer over the block tokens plus the covariate token;
'snps', the published per-chromosome SNP encoders, which keep the published parameter names
(dataEmbed, encoders, norm_f, mhsa, covariate_encoder, cov_norm, combine) so weights of
models.DeepGP.SNPCOV_Model_mamba load into them, beside an optional linear PRS path (--wide)."""
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
from .blockencoder import BlockEncoder
from .blocks import make_block
from .losses import WideLoss

POOLS = ("atten", "mean", "last")
WIDE_MODES = ("off", "on", "only")
WIDE_KEYS = ("prs.", "cov_linear", "intercept")        # state_dict keys of the linear PRS path


def vocab_codes(x):
    """Raw dosages [B, L, 1] (the store's codes times 0.01) -> those codes [B, L], 0 to 200, which
    index --snp_embed vocab's table. DataEmbeddingVocab.forward truncates instead of rounding and
    puts 6 of the 201 levels one code low (methods.md 11)."""
    codes = torch.round(x[:, :, 0].float() * 100).long()            # float32 first: exact for every level
    if bool(((codes < 0) | (codes > 200)).any()):                    # a clear error, not a CUDA device assert
        raise ValueError("--snp_embed vocab reads raw dosages in [0, 2]; these inputs give codes outside 0 to 200")
    return codes


def positions_digest(pos):
    h = hashlib.sha256()
    for p in pos:
        h.update(np.asarray(p, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def annotations_digest(annot):
    """Digest of the standardised float32 annotation matrices: a rewritten annotation file changes it."""
    h = hashlib.sha256()
    for a in annot:
        a = np.ascontiguousarray(a, dtype=np.float32)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _case_shares(loss_fn):
    """(pi, pi') per label: the prevalence, and pi' = pi c / Z, the share of cases in the weighted loss."""
    pi = loss_fn.pi.double().clamp(1e-6, 1 - 1e-6)
    c1 = loss_fn.c1.double()
    return pi, pi * c1 / (pi * c1 + 1.0 - pi)


def curvature_ratios(loss_fn):
    """r_j = pi'(1 - pi') / (pi(1 - pi)) = c_j / Z_j^2: how much the class weights inflate the
    curvature of each label's loss (methods.md 5.2)."""
    pi, pw = _case_shares(loss_fn)
    return (pw * (1.0 - pw)) / (pi * (1.0 - pi))


def wide_learning_rate(kappa, prs_var, batch_eff, ratios, n_snps):
    """Adam learning rate of the PRS weights, kappa tau sqrt(B) / (rbar sqrt(M)), rbar the geometric
    mean of the r_j (methods.md 6). Simulated at M up to 60,000, N up to 9,000, B 16-256 and
    prevalence 0.05-0.2, the noise the weights keep then depends on kappa alone (about 0.03 was
    best for a long run), and they settle in about 1/kappa epoch-equivalents."""
    rbar = float(torch.exp(torch.log(ratios.double()).mean()))
    return float(kappa) * float(prs_var) * math.sqrt(float(batch_eff)) / (rbar * math.sqrt(float(n_snps)))


def prs_penalty(loss_fn, n_train, prs_var):
    """Ridge coefficient of each label's PRS weights, r_j / (2 N tau), on the scale of the loss (methods.md 5.2)."""
    return (curvature_ratios(loss_fn) / (2.0 * float(n_train) * float(prs_var))).float()


class WideDeepGP(pl.LightningModule):
    """hp: a plain dict, saved with every checkpoint (train.model_hparams). pos: one array of
    positions per input sequence. annot: optional standardised float arrays [L_i, A], one per sequence.
    blocks: the partition (genostore.datamodule.load_partition), required in blocks mode."""

    def __init__(self, hp, pos, annot=None, blocks=None):
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
        self.wide = hp.get("wide", "off") or "off"
        if self.wide not in WIDE_MODES:
            raise ValueError(f"wide must be one of {WIDE_MODES}")
        self.blocks = hp.get("input_mode", "snps") == "blocks"
        if self.blocks:
            if self.wide != "off" or int(hp.get("annot_dim", 0)):
                raise ValueError("blocks mode has no linear PRS path and no annotation vectors (--wide off, no --annot)")
            if self.final_pool != "mean":
                raise ValueError("blocks mode pools the mixer's tokens by their mean (--final_pool mean)")
            if blocks is None or hp["partition_digest"] != blocks["digest"]:
                raise ValueError("the block partition given does not match the one this model was configured for")
        self.input_mask = float(hp.get("input_mask", 0.0) or 0.0)
        if not 0.0 <= self.input_mask < 1.0 or (self.input_mask > 0 and not self.blocks):
            raise ValueError("input_mask must be in [0, 1) and is a blocks-mode setting (the SNP-token model has none)")
        if self.wide == "only" and int(hp.get("annot_dim", 0)):
            raise ValueError("annotation vectors act on the deep network's input; --wide only has no deep network")
        if self.wide != "off" and hp.get("loss_outputs", "all") == "target":
            raise ValueError("the linear PRS path's penalty assumes every output learns from every batch "
                             "(--loss_outputs all)")
        # --prs_grid: one linear model per prior variance tau_k, side by side; output column
        # k * n_labels + j is disease j under tau_k (methods.md 4.10)
        self.grid = [float(t) for t in (hp.get("prs_grid") or [])]
        if self.grid and self.wide != "only":
            raise ValueError("--prs_grid trains one linear model per prior variance side by side; it needs --wide only")
        if any(not t > 0 for t in self.grid):
            raise ValueError("--prs_grid: prior variances must be positive")
        self.n_grid = max(1, len(self.grid))
        self.n_out = self.n_labels * self.n_grid
        self.out_names = (list(self.label_names) if not self.grid else
                          [f"{name}@{t:g}" for t in self.grid for name in self.label_names])
        self.bf16 = hp.get("precision", "32") == "bf16"
        self.grad_ckpt = bool(hp.get("grad_ckpt", False))
        mixer = hp.get("mixer", "bimamba1")
        self.ckpt_reentrant = mixer == "mamba3"         # its backward reads saved tensors twice (methods.md 4.3)
        if hp["pos_digest"] != positions_digest(pos) or [len(p) for p in pos] != list(hp["seq_len"]):
            raise ValueError("the SNP positions given do not match the ones this model was configured for")
        embed = {"cov": DataEmbedding, "vocab": DataEmbeddingVocab}[hp["snp_embed"]]
        block_kw = dict(bi_direct=bool(hp.get("bi_direct", False)), m3_d_state=hp.get("m3_d_state"),
                        n_heads=int(hp["n_heads"]), dropout=float(hp["dropout"]))

        if self.blocks:
            # one sequence: the block tokens and the covariate token, then the mixer, a plain norm and the mean
            self.encoder = BlockEncoder(blocks, hp["embed_dim"], d, hp["snp_code"], float(hp["dropout"]))
            self.layers = nn.ModuleList([make_block(d, mixer, **block_kw) for _ in range(int(hp["e_layers"]))])
            self.norm_f = nn.LayerNorm(d)
            if self.use_cov:
                self.covariate_encoder = FCN(int(hp["n_cov"]), d)
                self.cov_norm = nn.LayerNorm(d)
            self.combine = FCN((1 + self.use_cov) * d, self.n_labels, hidden_dim=int(hp.get("head_hidden", 256)),
                               hidden_layers=int(hp["n_fcn"]))
            self.lr_table, self.table_decay, self.table_eps = (float(hp[k]) for k in ("lr_table", "table_decay", "table_eps"))

        # built in the published class's order, so one seed gives the same initial weights
        encoders, data_embed, norms = [], [], []
        for p in (pos if self.wide != "only" and not self.blocks else []):
            encoders.append(nn.ModuleList([make_block(d, mixer, **block_kw) for _ in range(int(hp["e_layers"]))]))
            data_embed.append(embed(int(hp["enc_in"]), d, np.asarray(p), float(hp["dropout"])))
            norms.append(RMSNorm(d, eps=1e-5))
        if not self.blocks:
            self.encoders = nn.ModuleList(encoders)
            self.dataEmbed = nn.ModuleList(data_embed)
            self.norm_f = nn.ModuleList(norms)
        if self.wide != "only" and not self.blocks:
            self.mhsa = EncoderLayer(AttentionLayer(FullAttention(mask_flag=False), d_model=d,
                                                    n_heads=int(hp["n_heads"])), d_model=d)
            if self.use_cov:
                self.covariate_encoder = FCN(int(hp["n_cov"]), d)
                self.cov_norm = RMSNorm(d, eps=1e-5)
            n_vec = self.n_seq + self.use_cov            # a vector per sequence, and the covariates'
            self.combine = FCN(n_vec * d, self.n_labels, hidden_dim=int(hp.get("head_hidden", 256)),
                               hidden_layers=int(hp["n_fcn"]))
        for emb in (self.dataEmbed if not self.blocks else []):   # position tables follow from the SNP list: not saved
            emb.position_embedding.register_buffer("pe", emb.position_embedding.pe, persistent=False)

        # the linear PRS path (methods.md 4.10): logit_j += scale * sum_s x_s theta_sj + sum_k cov_k v_kj
        if self.wide != "off":
            self.n_snps = int(sum(len(p) for p in pos))
            self.prs_scale = 1.0 / math.sqrt(self.n_snps)
            self.prs = nn.ParameterList([nn.Parameter(torch.zeros(len(p), self.n_out)) for p in pos])
            if self.use_cov:
                self.cov_linear = nn.Parameter(torch.zeros(int(hp["n_cov"]), self.n_out))
            self.intercept = nn.Parameter(torch.zeros(self.n_out))       # set to the weighted base rate below
            # with a grid, column c's PRS update is scaled by s_c = tau_k / prs_var (optimizer_step)
            self.register_buffer("prs_col_scale", torch.tensor(
                [t / float(hp.get("prs_var", 1.0)) for t in self.grid for _ in self.label_names],
                dtype=torch.float32) if self.grid else None, persistent=False)
            if self.grid:                                # per disease: the best grid point and validation so far
                self.register_buffer("best_auroc", torch.full((self.n_labels,), float("-inf"), dtype=torch.float64))
                self.register_buffer("best_k", torch.full((self.n_labels,), -1, dtype=torch.long))
                self.register_buffer("best_step", torch.full((self.n_labels,), -1, dtype=torch.long))
                self.register_buffer("best_samples", torch.zeros(self.n_labels, dtype=torch.long))
                for i, p in enumerate(pos):
                    self.register_buffer(f"best_prs_{i}", torch.zeros(len(p), self.n_labels))
                if self.use_cov:
                    self.register_buffer("best_cov", torch.zeros(int(hp["n_cov"]), self.n_labels))
                self.register_buffer("best_intercept", torch.zeros(self.n_labels))
            if self.wide == "on":                        # the network's output starts at zero (no random draws)
                nn.init.zeros_(self.combine.layers[-1].weight)
                nn.init.zeros_(self.combine.layers[-1].bias)
            self.wide_freeze = bool(hp.get("wide_freeze", False))
            if self.wide_freeze:                         # stage 2 of a run's search: the PRS of stage 1, fixed
                for w in self.prs:
                    w.requires_grad_(False)

        # per-SNP annotation vectors: one gain map and one offset map shared by all sequences (methods.md 4.2)
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

        counts = hp["label_counts"]
        if self.grid:                                    # every grid point has the diseases' own constants
            counts = {"cases": list(counts["cases"]) * self.n_grid, "non_cases": list(counts["non_cases"]) * self.n_grid,
                      "n": counts["n"]}
        # a model saved before case_share was recorded was trained on even batches: exactly half cases
        self.loss_fn = WideLoss(counts=counts, gamma=float(hp["gamma"]), w_max=float(hp["w_max"]),
                                sampler=hp["sampler"], correction=bool(hp["sampling_correction"]),
                                outputs=hp["loss_outputs"], case_share=float(hp.get("case_share", 0.5)))
        self.lr, self.l2 = float(hp["lr"]), float(hp["l2"])
        self.lr_wide = float(hp.get("lr_wide") or 0.0)
        if self.wide != "off":
            lam = prs_penalty(self.loss_fn, int(hp["label_counts"]["n"]), float(hp.get("prs_var", 1.0)))
            if self.grid:                                # r / (2 n tau_k) in column c, with tau_k = s_c * prs_var
                lam = lam / self.prs_col_scale
            self.register_buffer("prs_lambda", lam, persistent=False)
            with torch.no_grad():                        # intercepts at the weighted base rate, logit(pi')
                pw = _case_shares(self.loss_fn)[1]
                self.intercept.copy_(torch.log(pw / (1.0 - pw)).float())
        self._val, self._train_sum, self._train_cnt = [], None, None
        self.last_val_metrics = None
        self.samples_seen = 0

    def _embed(self, i, x):
        """The embedding of input sequence i [B, L, 1] -> [B, L, d_model]: the published module's,
        with the annotations' gain and offset on the value part."""
        emb = self.dataEmbed[i]
        v = emb.value_embedding(vocab_codes(x) if isinstance(emb, DataEmbeddingVocab) else x)
        if self.annot_dim:
            a = getattr(self, f"annot_{i}")
            v = v * (1.0 + self.annot_gain(a)) + self.annot_offset(a)
        return emb.dropout(v + emb.position_embedding(x))

    def _encode(self, i, x):
        """One input sequence [B, L, 1] -> its pooled vector [B, d_model], in float32."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and x.is_cuda):
            h = self._embed(i, x)
            residual = None
            for layer in self.encoders[i]:
                h, residual = layer(h, residual)
            norm = self.norm_f[i]
            h = layer_norm_fn(h, norm.weight, norm.bias, eps=norm.eps, residual=residual, prenorm=False,
                              residual_in_fp32=True, is_rms_norm=True)
        if self.final_pool == "last":
            return h[:, -1, :].float()
        return h.float().mean(dim=1)                     # in float32, finer than bf16 resolves (methods.md 6)

    def _encode_anchored(self, i, x, anchor):
        return self._encode(i, x) + 0.0 * anchor

    def _block_logits(self, x_all):
        """Blocks mode: the block tokens and the covariate token through the mixer, their mean beside the
        covariate vector into the head."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16 and x_all[0].is_cuda):
            snps = x_all[:-1] if self.use_cov else x_all
            if self.training and self.input_mask > 0:    # input masking: a share of SNPs missing, the rest rescaled
                masks = [torch.rand(x.shape[:2], device=x.device) < self.input_mask for x in snps]
                pooled = self.encoder.pool(snps, masks) / (1.0 - self.input_mask)
            else:
                pooled = self.encoder.pool(snps)
            h = self.encoder.tokens(pooled)
            if self.use_cov:
                cov = self.cov_norm(self.covariate_encoder(x_all[-1]))
                h = torch.cat([h, cov[:, None]], dim=1)
            residual = None
            for layer in self.layers:
                if self.grad_ckpt and self.training:
                    h, residual = checkpoint(layer, h, residual, use_reentrant=False)
                else:
                    h, residual = layer(h, residual)
            h = self.norm_f(h if residual is None else h + residual)
        mean = h.float().mean(dim=1)
        return self.combine(torch.cat([mean, cov.float()], dim=1) if self.use_cov else mean)

    def forward(self, x_all):
        """x_all: the input sequences, then the covariates. Returns logits [B, n_out]."""
        if self.blocks:
            return self._block_logits(x_all)
        if self.wide == "only":
            return self.wide_logits(x_all)
        logits = self._deep_logits(x_all)
        return logits if self.wide == "off" else logits.float() + self.wide_logits(x_all)

    def big_params(self):
        """The per-SNP weights (the PRS, or the table): their own learning rate, logged apart."""
        if self.blocks:
            return list(self.encoder.tables)
        return list(self.prs) if self.wide != "off" else []

    def parameter_counts(self):
        """Blocks mode: parameters by module, for the size axes (the table and the identities are the
        embedding parameters; the rest is Kaplan's N)."""
        e = self.encoder
        count = lambda *mods: int(sum(p.numel() for m in mods for p in m.parameters()))
        return {"table": count(e.tables), "identity": count(e.block_emb, e.chrom_emb), "block_mlp": count(e.mlp),
                "mixer": count(self.layers, self.norm_f),
                "covariates": count(self.covariate_encoder, self.cov_norm) if self.use_cov else 0,
                "head": count(self.combine)}

    def snapshot(self, step):
        """A weights-only snapshot that load_model reads; the table in bfloat16 (it is most of the weights)."""
        table = {f"encoder.tables.{i}" for i in range(self.n_seq)} if self.blocks else set()
        return {"state_dict": {k: v.detach().to("cpu", torch.bfloat16 if k in table else v.dtype)
                               for k, v in self.state_dict().items()},
                "hyper_parameters": {"hp": self.hp}, "global_step": int(step), "samples_seen": int(self.samples_seen)}

    def prs_scores(self, x_all):
        """Each person's PRS per output [B, n_out]: the SNP part of the linear path, in float32."""
        with torch.autocast("cuda", enabled=False):
            return sum(x_all[i].reshape(x_all[i].shape[0], -1).float() @ self.prs[i]
                       for i in range(self.n_seq)) * self.prs_scale

    def wide_logits(self, x_all):
        """The linear path's logits [B, n_out], in float32."""
        with torch.autocast("cuda", enabled=False):
            out = self.prs_scores(x_all) + self.intercept
            if self.use_cov:
                out = out + x_all[-1].float() @ self.cov_linear
            return out

    def penalty(self):
        """The ridge penalty of the PRS weights on the scale of the loss (None without one; the blocks-mode
        table is held by its optimiser's weight decay instead)."""
        if self.blocks or self.wide == "off" or self.wide_freeze:
            return None
        sq = sum((w.float() ** 2).sum(dim=0) for w in self.prs)
        return (self.prs_lambda * sq).mean()

    def _deep_logits(self, x_all):
        pooled = []
        ckpt = self.grad_ckpt and self.training
        if ckpt and self.ckpt_reentrant:
            # the re-entrant checkpoint only carries gradients if one of its inputs requires them
            anchor = torch.zeros((), device=x_all[0].device, requires_grad=True)
        for i in range(self.n_seq):
            if not ckpt:
                pooled.append(self._encode(i, x_all[i]))
            elif self.ckpt_reentrant:
                pooled.append(checkpoint(self._encode_anchored, i, x_all[i], anchor, use_reentrant=True))
            else:
                pooled.append(checkpoint(self._encode, i, x_all[i], use_reentrant=False))
        with torch.autocast("cuda", enabled=False):
            if self.use_cov:
                pooled.append(self.cov_norm(self.covariate_encoder(x_all[-1].float())))
            if self.final_pool == "atten":
                out, _ = self.mhsa(torch.stack(pooled, dim=1))
                x = out.reshape(out.size(0), -1)
            else:
                x = torch.cat(pooled, dim=1)
            return self.combine(x)

    def training_step(self, batch, batch_idx):
        x_all, y, target = batch
        logits = self(x_all)
        if self.grid:                                    # every grid point learns every disease
            y = y.repeat(1, self.n_grid)
        t = int(target)
        loss, per_label = self.loss_fn(logits, y, t)
        pen = self.penalty()
        if pen is not None:
            loss = loss + pen
            self.log("train/prs_penalty", pen, on_step=True, on_epoch=False, batch_size=y.shape[0])
        self.samples_seen += int(y.shape[0])
        if self._train_sum is None:
            self._train_sum = torch.zeros_like(per_label, dtype=torch.float64)
            self._train_cnt = torch.zeros_like(per_label, dtype=torch.float64)
        if self.loss_fn.outputs == "target":
            self._train_sum[t] += per_label[t].double()
            self._train_cnt[t] += 1
        else:
            self._train_sum += per_label.double()
            self._train_cnt += 1
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=y.shape[0])
        # with a grid, the sum over grid points (the loss is their mean): each grid point's weights get
        # the gradient a model of its own would, far from Adam's epsilon (methods.md 6)
        return loss * self.n_grid if self.grid else loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        """With a grid, column c's PRS update is scaled by s_c = tau_k / prs_var: Adam's step is its
        learning rate times a ratio independent of it, so each grid point moves at its own tau_k's rate."""
        if not self.grid or self.wide_freeze:
            return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        before = [w.detach().clone() for w in self.prs]
        out = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        with torch.no_grad():
            for w, b in zip(self.prs, before):
                w.sub_(b).mul_(self.prs_col_scale).add_(b)
        return out

    def validation_step(self, batch, batch_idx):
        x_all, y = batch[0], batch[1]
        self._val.append((self(x_all).float().cpu(), y.float().cpu()))

    def on_validation_epoch_end(self):
        if not self._val:
            return
        logits = torch.cat([a for a, _ in self._val]).numpy()
        y = torch.cat([b for _, b in self._val]).numpy()
        self._val = []
        dm = getattr(self._trainer, "datamodule", None)
        splits = getattr(dm, "splits", None) or {}
        if "val" in splits and len(y) < len(splits["val"]["y"]):
            # a resumed run repeats its checkpoint's pass, restored part-way: score all of it (methods.md 7)
            from .train import predict_split
            logits, y = predict_split(self, dm, "val", self.device)
        m = self.val_metrics(logits, y)
        if self.grid:
            self._keep_grid_best(m)
        self.last_val_metrics = m
        self.log("val/auroc_mean", m["auroc_mean"], prog_bar=True)
        self.log("val/ap_mean", m["ap_mean"])
        self.log("val/bce_mean", m["bce_mean"])
        self.log("samples_seen", float(self.samples_seen))
        for j, name in enumerate(m["labels"]):
            for key in ("auroc", "ap", "bce"):
                if not math.isnan(float(m[key][j])):
                    self.log(f"val_{key}/{name}", float(m[key][j]))
        if self._train_sum is not None:
            mean = (self._train_sum / self._train_cnt.clamp_min(1)).cpu().numpy()
            cnt = self._train_cnt.cpu().numpy()
            for j, name in enumerate(self.out_names):
                if cnt[j] > 0:
                    self.log(f"train_loss/{name}", float(mean[j]))
            self._train_sum.zero_()
            self._train_cnt.zero_()

    def val_metrics(self, logits, y):
        """A validation pass's metrics: every grid point on every disease (n_grid is 1 without a grid)
        and, with a grid, the means over diseases each at its best grid point."""
        m = M.per_label_metrics(logits, np.tile(y, (1, self.n_grid)), self.loss_fn.logit_shift.cpu().numpy(),
                                self.out_names)
        if self.grid:
            self._grid_means(m)
        return m

    def _grid_means(self, m):
        """With a grid: the means are over diseases, each at its best grid point in this pass (by
        AUROC); m["grid_pick"] holds those points (-1 for a disease without a finite AUROC)."""
        n, k = self.n_labels, self.n_grid
        auc = np.asarray(m["auroc"], dtype=np.float64).reshape(k, n)
        pick = np.full(n, -1)
        for j in range(n):
            if np.isfinite(auc[:, j]).any():
                pick[j] = int(np.nanargmax(auc[:, j]))
        ok = pick >= 0
        cols = pick[ok] * n + np.flatnonzero(ok)
        for key in ("auroc", "ap", "bce"):
            v = np.asarray(m[key], dtype=np.float64)[cols]
            v = v[np.isfinite(v)]
            m[key + "_mean"] = float(v.mean()) if len(v) else (0.5 if key == "auroc" else float("nan"))
        m["grid_pick"] = pick.tolist()

    def _keep_grid_best(self, m):
        """Each disease's best grid point and pass so far (best_*), from a pass's metrics m."""
        trainer = self._trainer
        if trainer is not None and trainer.sanity_checking:
            return
        n = self.n_labels
        auc = np.asarray(m["auroc"], dtype=np.float64).reshape(self.n_grid, n)
        pick = np.asarray(m["grid_pick"])
        step = int(trainer.global_step) if trainer is not None else -1
        with torch.no_grad():
            for j in np.flatnonzero(pick >= 0):
                value = float(auc[pick[j], j])
                if not value > float(self.best_auroc[j]):    # strictly better: a repeated pass changes nothing
                    continue
                c = int(pick[j]) * n + int(j)
                for i in range(self.n_seq):
                    getattr(self, f"best_prs_{i}")[:, j] = self.prs[i][:, c]
                if self.use_cov:
                    self.best_cov[:, j] = self.cov_linear[:, c]
                self.best_intercept[j] = self.intercept[c]
                self.best_auroc[j] = value
                self.best_k[j] = int(pick[j])
                self.best_step[j] = step
                self.best_samples[j] = int(self.samples_seen)

    def per_disease_state(self):
        """With a grid: the weights, as a model without a grid, that take every disease at its best
        grid point and validation pass, and a table of the choices. A disease never scored (no cases
        in the tuning set) takes the grid point nearest prs_var at the end of training."""
        n = self.n_labels
        ref = int(np.argmin([abs(math.log(t / float(self.hp.get("prs_var", 1.0)))) for t in self.grid]))
        state = {f"prs.{i}": getattr(self, f"best_prs_{i}").detach().clone().cpu() for i in range(self.n_seq)}
        if self.use_cov:
            state["cov_linear"] = self.best_cov.detach().clone().cpu()
        state["intercept"] = self.best_intercept.detach().clone().cpu()
        rows = []
        for j, name in enumerate(self.label_names):
            k = int(self.best_k[j])
            g = k if k >= 0 else ref
            if k < 0:
                c = ref * n + j
                for i in range(self.n_seq):
                    state[f"prs.{i}"][:, j] = self.prs[i][:, c].detach().cpu()
                if self.use_cov:
                    state["cov_linear"][:, j] = self.cov_linear[:, c].detach().cpu()
                state["intercept"][j] = self.intercept[c].detach().cpu()
            rows.append({"label": name, "prs_var": self.grid[g], "grid_index": g, "chosen": k >= 0,
                         "step": int(self.best_step[j]), "samples_seen": int(self.best_samples[j]),
                         "val_auroc": float(self.best_auroc[j]) if k >= 0 else float("nan")})
        return state, rows

    def configure_optimizers(self):
        if self.blocks:
            # the table: its own rate, a large epsilon (an entry with a tiny, inconsistent gradient barely moves; one
            # with a consistent gradient moves normally) and a per-step decay that stops noise from accumulating
            tables = [t for t in self.encoder.tables if t.requires_grad]
            rest = [p for p in self.parameters() if p.requires_grad and not any(p is t for t in self.encoder.tables)]
            groups = [{"params": rest, "lr": self.lr, "weight_decay": self.l2}]
            if tables:
                groups.append({"params": tables, "lr": self.lr_table, "weight_decay": self.table_decay, "eps": self.table_eps})
            opt = torch.optim.AdamW(groups, lr=self.lr)
        elif self.wide == "off":
            opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.l2)
        else:
            # the PRS weights carry a ridge penalty in the loss instead of AdamW's decay, so that their
            # optimum is the penalised one; the covariate weights and intercepts are few and unpenalised
            prs = [] if self.wide_freeze else list(self.prs)
            small = ([self.cov_linear] if self.use_cov else []) + [self.intercept]
            taken = {id(p) for p in list(self.prs) + small}
            deep = [p for p in self.parameters() if id(p) not in taken]
            groups = ([{"params": deep, "lr": self.lr, "weight_decay": self.l2}] if deep else []) + \
                [{"params": small, "lr": self.lr, "weight_decay": 0.0}] + \
                ([{"params": prs, "lr": self.lr_wide, "weight_decay": 0.0}] if prs else [])
            opt = torch.optim.AdamW(groups, lr=self.lr)
        warm = int(self.hp.get("warmup_steps", 0) or 0)
        sched = self.hp.get("lr_schedule", "constant")
        total = int(self.hp.get("total_optimizer_steps", 0) or 0)
        if sched == "plateau":                           # Sigurdsson et al. 2023: x0.2 after 10 passes without improvement
            plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.2, patience=10)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": plateau, "monitor": "val/auroc_mean",
                                                       "interval": "epoch", "strict": True}}
        if warm == 0 and sched == "constant":
            return opt                                   # as published: constant learning rate

        def factor(step):
            f = min(1.0, (step + 1) / warm) if warm else 1.0
            if sched == "cosine" and total > warm:
                f *= 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, (step - warm) / (total - warm)))))
            return f

        return {"optimizer": opt, "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, factor),
                                                   "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["samples_seen"] = int(self.samples_seen)

    def on_load_checkpoint(self, checkpoint):
        self.samples_seen = int(checkpoint.get("samples_seen", 0))
        sd = checkpoint.get("state_dict", {})
        for k in [k for k in sd if k.endswith("position_embedding.pe")]:     # written by the published class
            del sd[k]

    def load_weights(self, state_dict, allow_new_head=False, source_labels=None):
        """Load weights saved by this class or the published one, leaving out fixed tables and the
        loss's constants. With allow_new_head, the weights that belong to labels (the output layer and
        the linear path) stay at their initial values when the source was trained for other diseases:
        weights of another size or, given ``source_labels``, other names or another order. Annotation
        maps and the linear path may be absent from the source (older models): they then stay at
        zero. Everything else must match. Returns the names that were not loaded."""
        own = self.state_dict()
        src = {k: v for k, v in state_dict.items() if not k.endswith("position_embedding.pe")
               and not k.startswith("loss_fn.") and not (k.startswith("best_") and k not in own)}
        skipped = []
        per_label = WIDE_KEYS
        if self.wide != "only":
            per_label += (f"combine.layers.{len(self.combine.layers) - 1}.",)
        other_labels = source_labels is not None and list(source_labels) != self.label_names
        for k in list(src):
            if allow_new_head and other_labels and k.startswith(per_label):
                skipped.append(k)
                del src[k]
            elif k in own and own[k].shape != src[k].shape:
                if allow_new_head and k.startswith(per_label):
                    skipped.append(k)
                    del src[k]
                else:
                    raise ValueError(f"{k}: saved shape {tuple(src[k].shape)} but the model has {tuple(own[k].shape)}")
        unknown = [k for k in src if k not in own]
        if unknown:
            raise KeyError(f"weights not in this model: {unknown[:5]}")
        missing = [k for k in own if k not in src and not k.startswith("loss_fn.") and k not in skipped]
        optional = ("annot_gain.", "annot_offset.") + WIDE_KEYS
        if missing and not all(k.startswith(optional) for k in missing):
            raise KeyError(f"weights missing from the source: {missing[:5]}")
        self.load_state_dict(src, strict=False)
        if "intercept" in missing:                       # a source without the PRS path keeps its intercepts in its head
            with torch.no_grad():
                self.intercept.zero_()
        return skipped + missing
