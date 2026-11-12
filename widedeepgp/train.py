# -*- coding: utf-8 -*-
"""One multi-disease training run. Used by main_multilabel.py and by the Optuna tuner, so a
tuned configuration is trained by exactly the code that tuned it.

Vocabulary
    epoch-equivalent   as many training draws as there are training participants
    validation pass    every 1 / --val_per_epoch epoch-equivalents; Lightning's "epoch" is one
                       such interval, so every checkpoint written after validation sits on an
                       epoch boundary and a resumed run continues the same batch sequence

A run directory, <run_dir>/<experiment>/<exp_name>/, holds
    config.json      every setting, the derived schedule, the label counts, the covariate scaling
    tb/              TensorBoard event files
    metrics.csv      one row per validation pass, label and metric
    snapshots/       weights along training (log-spaced steps and every epoch-equivalent)
    snapshots.csv    what each snapshot cost: steps, participants seen, training seconds
    checkpoints/     best.ckpt (by mean validation AUROC) and last.ckpt (to resume)
    results.json     validation metrics of the best checkpoint
"""
import json
import math
import os
import time

import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.apply_func import move_data_to_device
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from genostore.datamodule import SNPStoreDataModule, is_store_dataset

from . import metrics as M
from .callbacks import EarlyStoppingOnce, OptunaPruning, RunLog, Snapshots, snapshot_steps
from .losses import label_counts
from .model import WideDeepGP, positions_digest
from .samplers import make_sampler

MONITOR = "val/auroc_mean"


def build_data(cfg):
    if not is_store_dataset(cfg):
        raise FileNotFoundError(f"{cfg.data_dir}/{cfg.label}/{cfg.rd_mode}/store.json not found: the multi-disease "
                                f"model reads disk-backed datasets only (genostore/README.md)")
    return SNPStoreDataModule(cfg)


def train_labels(dm):
    y = np.asarray(dm.splits["train"]["y"], dtype=np.float32)
    return y[:, None] if y.ndim < 2 else y


def load_annotations(dm):
    """Annotation vectors standardised per channel over the model's SNPs (constant channels are
    only centred)."""
    annot, names = dm.annotations()
    if annot is None:
        raise ValueError("--annot was given but the dataset's store.json names no annotations")
    allv = np.concatenate(annot, axis=0).astype(np.float64)
    mean, sd = allv.mean(0), allv.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [((a - mean) / sd).astype(np.float32) for a in annot], names


def schedule(cfg, n_train):
    """Batches per epoch-equivalent and per validation interval, and the number of intervals."""
    acc = max(1, int(cfg.accumulate))
    per_ee = int(cfg.steps_per_epoch_equiv or math.ceil(n_train / cfg.batch_size))
    per_val = max(1, int(math.ceil(per_ee / float(cfg.val_per_epoch))))    # rounded up, so that
    per_val = int(math.ceil(per_val / acc) * acc)        # val_per_epoch intervals cover an epoch-equivalent;
    #                                                      a multiple of acc: the optimiser also steps on an epoch's last batch
    n_val = max(1, int(math.ceil(float(cfg.max_epoch_equiv) * per_ee / per_val - 1e-9)))
    return {"batches_per_epoch_equiv": per_ee, "batches_per_validation": per_val, "validations": n_val,
            "optimizer_steps": n_val * per_val // acc, "accumulate": acc}


def model_hparams(cfg, dm, sched=None, annot_names=None):
    """Everything the model needs to be rebuilt, as a plain dict (saved in every checkpoint)."""
    names = dm.label_names if dm.multilabel else ["label"]
    return {
        "d_model": int(cfg.d_model), "e_layers": int(cfg.e_layers), "dropout": float(cfg.dropout),
        "enc_in": int(cfg.enc_in), "n_heads": int(cfg.n_heads), "n_fcn": int(cfg.n_fcn),
        "snp_embed": cfg.snp_embed, "final_pool": cfg.final_pool, "bi_direct": bool(cfg.bi_direct),
        "mixer": cfg.mixer, "m3_d_state": cfg.m3_d_state, "head_hidden": int(cfg.head_hidden),
        "use_cov": bool(cfg.use_cov) and dm.data_mode == "snps_covs", "n_cov": int(dm.n_cov),
        "label_names": list(names), "label_counts": label_counts(train_labels(dm)),
        "seq_len": [int(n) for n in dm.seq_len], "pos_digest": positions_digest(dm.pos_chr),
        "annot_dim": len(annot_names) if annot_names else 0, "annot_names": list(annot_names or []),
        "gamma": float(cfg.gamma), "w_max": float(cfg.w_max), "sampler": cfg.sampler,
        "sampling_correction": not cfg.no_sampling_correction, "loss_outputs": cfg.loss_outputs,
        "precision": str(cfg.precision), "grad_ckpt": bool(cfg.grad_ckpt),
        "lr": float(cfg.lr), "l2": float(cfg.l2), "warmup_steps": int(cfg.warmup_steps),
        "lr_schedule": cfg.lr_schedule, "total_optimizer_steps": int(sched["optimizer_steps"]) if sched else 0,
    }


def build_model(cfg, dm, sched=None):
    annot, annot_names = load_annotations(dm) if cfg.annot else (None, None)
    return WideDeepGP(model_hparams(cfg, dm, sched, annot_names), dm.pos_chr, annot)


def load_model(path, dm, device="cpu"):
    """Rebuild a model from a snapshot or checkpoint written by a run, for the dataset ``dm``."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]["hp"]
    annot = load_annotations(dm)[0] if hp.get("annot_dim") else None
    model = WideDeepGP(hp, dm.pos_chr, annot)
    model.on_load_checkpoint(ckpt)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_split(model, dm, split, device=None):
    """Logits and labels [N, n] for a split ('train', 'val', 'test', or an external table name)."""
    device = device or next(model.parameters()).device
    if split in dm.splits:
        loader = dm._loader(dm._dataset(dm.splits[split]), False)
    else:
        loader = dm.eval_dataloder(split)
    was_training = model.training
    model.eval()
    logits, ys = [], []
    for batch in loader:
        out = dm.on_after_batch_transfer(move_data_to_device(batch, device), 0)
        logits.append(model(out[0]).float().cpu())
        ys.append(out[1].float().cpu())
    model.train(was_training)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def run_training(cfg, trial=None, extra_callbacks=()):
    """Train one model. Returns a dict with the run directory, the best mean validation AUROC,
    per-label validation metrics of the best checkpoint and the validation history."""
    t0 = time.time()
    pl.seed_everything(int(cfg.seed), workers=True)
    dm = build_data(cfg)
    y_train = train_labels(dm)
    n_train = len(y_train)
    sched = schedule(cfg, n_train)
    dm.train_sampler = make_sampler(cfg.sampler, y_train, cfg.batch_size, sched["batches_per_validation"],
                                    seed=int(cfg.seed))
    model = build_model(cfg, dm, sched)
    if cfg.init_from:
        src = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        left_out = model.load_weights(src["state_dict"], allow_new_head=True)
        print(f"initialised from {cfg.init_from}; not loaded: {left_out or 'nothing'}", flush=True)

    run_dir = os.path.join(cfg.run_dir, cfg.experiment, cfg.exp_name or "run")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    last = os.path.join(ckpt_dir, "last.ckpt")
    resume = last if (cfg.resume and os.path.exists(last)) else None
    n_snps = int(sum(dm.seq_len))
    n_params = sum(p.numel() for p in model.parameters())
    config = {"args": {k: v for k, v in vars(cfg).items() if isinstance(v, (str, int, float, bool, type(None)))},
              "model": model.hp, "schedule": sched, "n_train": n_train,
              "n_val": int(len(dm.splits["val"]["y"])), "n_test": int(len(dm.splits["test"]["y"])),
              "n_snps": n_snps, "n_parameters": int(n_params),
              "n_encoder_parameters": int(sum(p.numel() for p in model.encoders.parameters())),
              "dataset_dir": os.path.abspath(dm.dir), "store_dir": os.path.abspath(dm.store_dir),
              "covariates": dm.cov_names, "resumed": bool(resume),
              "covariate_scaling": ({"mean": [float(v) for v in dm.cov_cm[0]], "sd": [float(v) for v in dm.cov_cs[0]]}
                                    if getattr(dm, "cov_cm", None) is not None else None),
              "snp_scaling": dm.snp_stats, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    acc = sched["accumulate"]
    snaps = Snapshots(run_dir, snapshot_steps(sched["optimizer_steps"], int(cfg.snapshots_per_decade),
                                              every=max(1, sched["batches_per_epoch_equiv"] // acc)), n_train)
    runlog = RunLog(run_dir, n_train, n_snps)
    best_ckpt = ModelCheckpoint(dirpath=ckpt_dir, filename="best", monitor=MONITOR, mode="max", save_top_k=1,
                                save_last=True, save_on_train_epoch_end=False, auto_insert_metric_name=False)
    callbacks = [best_ckpt, snaps, runlog] + list(extra_callbacks)
    if int(cfg.stop_patience) > 0:
        callbacks.append(EarlyStoppingOnce(monitor=MONITOR, mode="max", patience=int(cfg.stop_patience)))
    if trial is not None:
        callbacks.append(OptunaPruning(trial, MONITOR))
    logger = TensorBoardLogger(save_dir=run_dir, name="", version="tb", default_hp_metric=False)
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                         max_epochs=sched["validations"], accumulate_grad_batches=acc, callbacks=callbacks,
                         logger=logger, num_sanity_val_steps=0, enable_progress_bar=False,
                         log_every_n_steps=min(50, max(1, sched["batches_per_validation"] // acc)),
                         enable_model_summary=False)
    trainer.fit(model, dm, ckpt_path=resume)

    # the best checkpoint and its validation metrics (kept from the pass made at that step; scored
    # again only when this process did not make that pass, as after a resume)
    best_path = best_ckpt.best_model_path or last
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    m = runlog.metrics_at.get(int(state.get("global_step", -1)))
    if m is None:
        logits, y = predict_split(model, dm, "val", device)
        m = M.per_label_metrics(logits, y, model.loss_fn.logit_shift.cpu().numpy(), model.label_names)
    out = {"run_dir": run_dir, "best_checkpoint": best_path, "best_val_auroc_mean": float(m["auroc_mean"]),
           "best_step": int(state.get("global_step", 0)), "best_samples_seen": int(state.get("samples_seen", 0)),
           "val": {name: {k: float(m[k][j]) for k in ("auroc", "ap", "bce")} for j, name in enumerate(m["labels"])},
           "val_mean": {k: float(m[k + "_mean"]) for k in ("auroc", "ap", "bce")},
           "history": [{"validation": h[0], "step": h[1], "samples_seen": h[2], "val_auroc_mean": h[3]}
                       for h in runlog.history],
           "samples_seen": int(model.samples_seen), "epoch_equivalents": model.samples_seen / n_train,
           "stopped_early": bool(trainer.should_stop and trainer.current_epoch < sched["validations"]),
           "train_seconds": round(snaps.train_seconds, 1), "wall_seconds": round(time.time() - t0, 1),
           "seconds_per_batch": round(snaps.train_seconds / max(1, trainer.fit_loop.total_batch_idx + 1), 4),
           "samples_per_second": round(model.samples_seen / max(snaps.train_seconds, 1e-9), 2),
           "gpu_peak_train_gib": round(runlog.peak_train_bytes / 2 ** 30, 3),
           "gpu_peak_val_gib": round(runlog.peak_val_bytes / 2 ** 30, 3),
           "n_parameters": int(n_params)}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    flat = {k: v for k, v in model.hp.items() if isinstance(v, (str, int, float, bool))}
    flat.update(batch_size=int(cfg.batch_size), accumulate=acc, seed=int(cfg.seed), n_labels=len(model.label_names))
    logger.log_hyperparams(flat, {"hp/best_val_auroc_mean": out["best_val_auroc_mean"],
                                  "hp/best_val_bce_mean": out["val_mean"]["bce"]})
    logger.finalize("success")
    out["model"], out["datamodule"] = model, dm
    return out
