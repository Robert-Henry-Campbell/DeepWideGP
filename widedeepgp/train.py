"""One multi-disease training run, shared by main_multilabel.py and the search (tuning.py), so a
tuned configuration is trained by exactly the code that tuned it. It writes the run folder
<run_dir>/<experiment>/<exp_name>/ that README.md describes.

An epoch-equivalent is as many training draws as there are training participants. Lightning's
"epoch" is one validation interval, so every checkpoint written after validation sits on an
epoch boundary and a resumed run continues the same batch sequence.
"""
import json
import math
import os
import re
import subprocess
import sys
import time
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.apply_func import move_data_to_device
from pytorch_lightning.loggers import TensorBoardLogger

from genostore.datamodule import SNPStoreDataModule, dataset_dir, is_store_dataset

from . import metrics as M
from .callbacks import EarlyStoppingOnce, OptunaPruning, RunCheckpoint, RunLog, Snapshots, snapshot_steps
from .losses import WideLoss, label_counts
from .model import WIDE_KEYS, WideDeepGP, annotations_digest, curvature_ratios, positions_digest, wide_learning_rate
from .samplers import cases_per_batch, make_sampler

MONITOR = "val/auroc_mean"


def build_data(cfg):
    if not is_store_dataset(cfg):
        raise FileNotFoundError(f"{dataset_dir(cfg)}/store.json not found: the multi-disease model reads "
                                "disk-backed datasets only (genostore/README.md)")
    return SNPStoreDataModule(cfg)


def train_labels(dm):
    y = np.asarray(dm.splits["train"]["y"], dtype=np.float32)
    return y[:, None] if y.ndim < 2 else y


def load_annotations(dm):
    """Annotation vectors standardised per channel over the model's SNPs (constant ones only centred)."""
    annot, names = dm.annotations()
    if annot is None:
        raise ValueError("--annot was given but the dataset's store.json names no annotations")
    allv = np.concatenate(annot, axis=0).astype(np.float64)
    mean, sd = allv.mean(0), allv.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    return [((a - mean) / sd).astype(np.float32) for a in annot], names


def schedule(cfg, n_train):
    """Batches per epoch-equivalent and per validation interval, and the number of intervals.
    The interval is at least --val_min_participants (methods.md section 7) and the budget at
    least --min_participants, so that early stopping has room to act."""
    acc = max(1, int(cfg.accumulate))
    per_ee = int(cfg.steps_per_epoch_equiv or math.ceil(n_train / cfg.batch_size))
    # rounded up, so that val_per_epoch intervals cover an epoch-equivalent
    per_val = max(1, int(math.ceil(per_ee / float(cfg.val_per_epoch))))
    per_val = max(per_val, int(math.ceil(int(getattr(cfg, "val_min_participants", 0) or 0) / cfg.batch_size)))
    cap = float(getattr(cfg, "val_max_epoch_equiv", 0) or 0)
    if cap > 0:                                          # but at least once per `cap` epoch-equivalents
        per_val = min(per_val, max(1, int(math.ceil(cap * per_ee))))
    per_val = int(math.ceil(per_val / acc) * acc)        # a multiple of acc: the optimiser also steps on an epoch's last batch
    n_val = max(1, int(math.ceil(float(cfg.max_epoch_equiv) * per_ee / per_val - 1e-9)))
    n_val = max(n_val, int(math.ceil(int(getattr(cfg, "min_participants", 0) or 0) / cfg.batch_size / per_val - 1e-9)))
    return {"batches_per_epoch_equiv": per_ee, "batches_per_validation": per_val, "validations": n_val,
            "optimizer_steps": n_val * per_val // acc, "accumulate": acc}


def model_hparams(cfg, dm, sched=None, annot_names=None, annot=None):
    """Everything needed to rebuild the model, as a plain dict saved in every checkpoint; ``annot``,
    the standardised annotation matrices, is kept as a digest."""
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
        "annot_digest": annotations_digest(annot) if annot is not None else None,
        "gamma": float(cfg.gamma), "w_max": float(cfg.w_max), "sampler": cfg.sampler,
        "case_share": cases_per_batch(cfg.batch_size) / int(cfg.batch_size) if cfg.sampler == "rotation" else 0.5,
        "sampling_correction": not cfg.no_sampling_correction, "loss_outputs": cfg.loss_outputs,
        "precision": str(cfg.precision), "grad_ckpt": bool(cfg.grad_ckpt),
        "lr": float(cfg.lr), "l2": float(cfg.l2), "warmup_steps": int(cfg.warmup_steps),
        "lr_schedule": cfg.lr_schedule, "total_optimizer_steps": int(sched["optimizer_steps"]) if sched else 0,
        "wide": getattr(cfg, "wide", "off") or "off", **wide_hparams(cfg, dm),
    }


def parse_grid(spec):
    """--prs_grid as a sorted list of prior variances, or None ('none' or not given)."""
    if spec is None or isinstance(spec, (list, tuple)):
        return sorted({float(t) for t in spec}) if spec else None
    spec = str(spec).strip()
    if not spec or spec.lower() == "none":
        return None
    return sorted({float(t) for t in spec.split(",") if t.strip()})


def wide_hparams(cfg, dm):
    """Settings of the linear PRS path; its learning rate follows from --wide_kappa unless given
    (methods.md section 6; with a grid, each grid point's by the same rule: section 4.10)."""
    wide = getattr(cfg, "wide", "off") or "off"
    grid = parse_grid(getattr(cfg, "prs_grid", None))
    if grid and wide != "only":
        raise ValueError("--prs_grid trains one linear model per prior variance side by side; it needs --wide only")
    if wide != "off" and getattr(cfg, "snp_embed", "cov") == "vocab":
        raise ValueError("--snp_embed vocab reads raw dosages, and the linear PRS path assumes standardised "
                         "SNPs (methods.md 4.10): use --wide off")
    if wide == "off":
        return {}
    tau, kappa = float(cfg.prs_var), float(cfg.wide_kappa)
    lr_wide = getattr(cfg, "lr_wide", None)
    if not lr_wide:
        loss = WideLoss(counts=label_counts(train_labels(dm)), gamma=float(cfg.gamma), w_max=float(cfg.w_max))
        lr_wide = wide_learning_rate(kappa, tau, int(cfg.batch_size) * max(1, int(cfg.accumulate)),
                                     curvature_ratios(loss), int(sum(dm.seq_len)))
    return {"prs_var": tau, "wide_kappa": kappa, "lr_wide": float(lr_wide),
            "wide_freeze": bool(getattr(cfg, "wide_freeze", False)), "snp_stats": getattr(cfg, "snp_stats", None),
            "prs_grid": grid}


def build_model(cfg, dm, sched=None):
    annot, annot_names = load_annotations(dm) if cfg.annot else (None, None)
    return WideDeepGP(model_hparams(cfg, dm, sched, annot_names, annot), dm.pos_chr, annot)


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
    loader = dm._loader(dm._dataset(dm.splits[split]), False) if split in dm.splits else dm.eval_dataloder(split)
    was_training = model.training
    model.eval()
    logits, ys = [], []
    for batch in loader:
        out = dm.on_after_batch_transfer(move_data_to_device(batch, device), 0)
        logits.append(model(out[0]).float().cpu())
        ys.append(out[1].float().cpu())
    model.train(was_training)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def code_version(code_dir=None):
    """The directory the code was loaded from, its git commit (only at the top of a git checkout;
    a frozen copy has none), whether a tracked .py file there differs from it (git_dirty; None
    outside git) and the Python interpreter."""
    code_dir = os.path.realpath(code_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    commit = dirty = None
    try:
        r = subprocess.run(["git", "-C", code_dir, "rev-parse", "--show-toplevel", "HEAD"], capture_output=True,
                           text=True, timeout=60)
        top_and_commit = r.stdout.splitlines()
        if r.returncode == 0 and len(top_and_commit) == 2 and os.path.realpath(top_and_commit[0]) == code_dir:
            commit = top_and_commit[1]
            # --no-optional-locks: never write the index of a checkout another process may be using
            r = subprocess.run(["git", "--no-optional-locks", "-C", code_dir, "status", "--porcelain",
                                "--untracked-files=no", "--", "*.py"], capture_output=True, text=True, timeout=60)
            dirty = bool(r.stdout.strip()) if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"code_dir": code_dir, "git_commit": commit, "git_dirty": dirty, "python": sys.executable}


# What a resumed run may change in a model's hp: its length, and activation checkpointing (which
# leaves outputs and gradients unchanged).
RESUMABLE = ("total_optimizer_steps", "grad_ckpt")
# Arguments that define a run's inputs and batches but are not in its hp. (The dataset is checked
# through the hp's digests, labels and counts, so the path it is named by may differ.)
RUN_ARGS = ("snp_stats", "scaling", "batch_size", "accumulate", "seed", "steps_per_epoch_equiv", "val_per_epoch",
            "val_min_participants", "val_max_epoch_equiv")


def last_checkpoint(ckpt_dir):
    """The checkpoint a resumed run continues from, or None: last.ckpt, or the highest
    last-vN.ckpt, which Lightning writes beside a last.ckpt it did not write (a copied run, or one
    started again without --resume). It must also be the newest: a copy that did not keep the
    files' times can make a stale one newer, and then nothing is picked."""
    if not os.path.isdir(ckpt_dir):
        return None
    found = (re.fullmatch(r"last(?:-v(\d+))?\.ckpt", name) for name in os.listdir(ckpt_dir))
    version = {os.path.join(ckpt_dir, m.string): int(m.group(1) or 0) for m in found if m}
    if not version:
        return None
    latest = max(version, key=version.get)
    newer = sorted(os.path.basename(p) for p in version if os.path.getmtime(p) > os.path.getmtime(latest))
    if newer:
        raise ValueError(f"cannot tell where the run in {ckpt_dir} stands: {os.path.basename(latest)} is the latest by "
                         f"name but {', '.join(newer)} was written after it (copied without the files' times?); move "
                         "the stale checkpoints out of the folder")
    return latest


def load_wide_init(model, path):
    """Start the linear PRS path (PRS and covariate weights, intercepts) from the search's first
    stage; it must have the same SNPs, labels, training split and SNP scaling, so that no fit on
    other participants comes in."""
    src = torch.load(path, map_location="cpu", weights_only=False)
    old = src["hyper_parameters"]["hp"]
    for key in ("pos_digest", "seq_len", "label_names", "label_counts", "snp_stats"):
        if old.get(key) != model.hp.get(key) and not (isinstance(old.get(key), list) and list(old[key]) == list(model.hp[key])):
            raise ValueError(f"--init_wide_from {path}: it was trained with other {key}")
    state = {k: v for k, v in src["state_dict"].items() if k.startswith(WIDE_KEYS)}
    need = [k for k in model.state_dict() if k.startswith(WIDE_KEYS)]
    if sorted(state) != sorted(need):
        raise ValueError(f"--init_wide_from {path}: it holds no linear PRS path of this shape")
    model.load_state_dict(state, strict=False)


def check_resume_args(config_path, cfg):
    """Refuse to resume a run with other input scaling, batches or schedule than the config.json
    it wrote when it started records (the resumed run is about to overwrite that file)."""
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        old = json.load(f).get("args", {})
    new = vars(cfg)
    changed = [k for k in RUN_ARGS if k in old and k in new and old[k] != new[k]]
    if changed:
        raise ValueError(f"cannot resume the run in {os.path.dirname(config_path)}: it was started with other "
                         "arguments (" + ", ".join(f"--{k} {old[k]!r}, now {new[k]!r}" for k in changed)
                         + "); --init_from starts a new run from its weights instead")


def check_resume_tables(ckpt_path, hp, loss=None):
    """Refuse to resume from a checkpoint trained on other SNP positions, annotation vectors,
    diseases or training labels, or with other settings than RESUMABLE allows (methods.md
    section 7). ``loss``, this run's loss module, is compared buffer by buffer too, which covers
    checkpoints saved before a setting (such as case_share) was recorded. An annotated
    checkpoint without a digest resumes with a warning."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    old = ckpt["hyper_parameters"]["hp"]
    if old.get("wide", "off") != hp.get("wide", "off"):
        raise ValueError(f"cannot resume from {ckpt_path}: it was trained with --wide {old.get('wide', 'off')}, this run "
                         f"has --wide {hp.get('wide', 'off')} (runs made before the linear PRS path need --wide off)")
    if old.get("pos_digest") != hp.get("pos_digest") or list(old.get("seq_len", [])) != list(hp["seq_len"]):
        raise ValueError(f"cannot resume from {ckpt_path}: it was trained on other SNP positions")
    if int(old.get("annot_dim", 0)) != int(hp.get("annot_dim", 0)):
        raise ValueError(f"cannot resume from {ckpt_path}: it was trained with annot_dim {old.get('annot_dim', 0)}, "
                         f"this run has {hp.get('annot_dim', 0)}")
    if int(hp.get("annot_dim", 0)):
        if old.get("annot_digest") is None:
            warnings.warn(f"{ckpt_path} holds no digest of its annotation vectors, so the ones of this run cannot be "
                          "checked against those it was trained with")
        elif old["annot_digest"] != hp.get("annot_digest"):
            raise ValueError(f"cannot resume from {ckpt_path}: the annotation vectors differ from the ones it was "
                             "trained with (has the annotation file been rewritten?)")
    if "label_names" in old and list(old["label_names"]) != list(hp["label_names"]):
        raise ValueError(f"cannot resume from {ckpt_path}: it was trained on the labels {list(old['label_names'])}, "
                         f"this run has {list(hp['label_names'])}; --init_from starts a new run from its weights, "
                         "with a new output layer")
    if old.get("label_counts") is not None and old["label_counts"] != hp.get("label_counts"):
        raise ValueError(f"cannot resume from {ckpt_path}: its training split had other label counts (has the dataset "
                         "been rebuilt?); --init_from starts a new run from its weights instead")
    checked = {"pos_digest", "seq_len", "annot_dim", "annot_digest", "label_names", "label_counts"}
    changed = [k for k in sorted(old) if k in hp and k not in checked and k not in RESUMABLE and old[k] != hp[k]]
    if changed:
        raise ValueError(f"cannot resume from {ckpt_path}: it was trained with other settings ("
                         + ", ".join(f"{k} {old[k]!r}, now {hp[k]!r}" for k in changed)
                         + "); --init_from starts a new run from its weights instead")
    if loss is not None:
        saved = ckpt.get("state_dict", {})
        for name, value in loss.named_buffers():
            was = saved.get(f"loss_fn.{name}")
            if was is not None and (was.shape != value.shape or not torch.allclose(
                    was.float().cpu(), value.float().cpu(), rtol=1e-6, atol=0.0)):
                raise ValueError(f"cannot resume from {ckpt_path}: the loss's {name} differs from the one it was "
                                 "trained with (other label counts, gamma, w_max or an odd batch size?); "
                                 "--init_from starts a new run from its weights instead")


def write_per_disease(model, dm, run_dir, config, step, device):
    """The model of a grid run (--prs_grid), every disease at its own best prior variance and
    pass (methods.md section 4.10), written as a run of its own, <run_dir>/per_disease/, which
    score_snapshots.py and --init_wide_from read as any run. It is scored on the tuning set
    again, which reproduces the AUROC each disease was chosen by."""
    state, rows = model.per_disease_state()
    hp = dict(model.hp, prs_grid=None, prs_var_per_label=[r["prs_var"] for r in rows], selected_from_grid=model.grid)
    pd_model = WideDeepGP(hp, dm.pos_chr)
    full = pd_model.state_dict()
    full.update(state)
    pd_model.load_state_dict(full)
    pd_model.samples_seen = int(model.samples_seen)
    pd_model.to(device)
    logits, y = predict_split(pd_model, dm, "val", device)
    m = M.per_label_metrics(logits, y, pd_model.loss_fn.logit_shift.cpu().numpy(), pd_model.label_names)
    n_train = int(config["n_train"])
    for j, r in enumerate(rows):
        r.update(epoch_equiv=round(r["samples_seen"] / n_train, 6), val_auroc_rescored=float(m["auroc"][j]))
    pd_dir = os.path.join(run_dir, "per_disease")
    os.makedirs(os.path.join(pd_dir, "checkpoints"), exist_ok=True)
    best = os.path.join(pd_dir, "checkpoints", "best.ckpt")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in pd_model.state_dict().items()},
                "hyper_parameters": {"hp": hp}, "global_step": int(step), "samples_seen": int(model.samples_seen),
                "per_disease": rows, "pytorch-lightning_version": pl.__version__}, best)
    last = os.path.join(pd_dir, "checkpoints", "last.ckpt")
    if os.path.exists(last):
        os.remove(last)
    os.link(best, last)                                  # one file: there is nothing to resume
    with open(os.path.join(pd_dir, "config.json"), "w") as f:
        json.dump(dict(config, model=hp, per_disease_of=os.path.abspath(run_dir)), f, indent=1)
    with open(os.path.join(pd_dir, "per_disease.csv"), "w") as f:
        keys = list(rows[0])
        f.write(",".join(keys) + "\n")
        f.writelines(",".join(str(r[k]) for k in keys) + "\n" for r in rows)
    res = {"run_dir": pd_dir, "best_checkpoint": best, "best_val_auroc_mean": float(m["auroc_mean"]),
           "val": {name: {k: float(m[k][j]) for k in ("auroc", "ap", "bce")} for j, name in enumerate(m["labels"])},
           "val_mean": {k: float(m[k + "_mean"]) for k in ("auroc", "ap", "bce")}, "per_disease": rows,
           "prs_grid": model.grid, "samples_seen": int(model.samples_seen)}
    with open(os.path.join(pd_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    return {"per_disease_dir": pd_dir, "per_disease_checkpoint": best,
            "per_disease_val_auroc_mean": float(m["auroc_mean"]), "per_disease": rows}


def run_training(cfg, trial=None, extra_callbacks=()):
    """Train one model. Returns a dict with the run directory, the best mean validation AUROC,
    per-label validation metrics of the best checkpoint and the validation history."""
    if cfg.dm != "snps_covs" or not cfg.use_cov:
        raise ValueError("WideDeepGP always uses the covariates: a genotype-only model (--dm snps, or --use_cov, "
                         "which switches the covariates off) is not offered")
    t0 = time.time()
    pl.seed_everything(int(cfg.seed), workers=True)
    dm = build_data(cfg)
    if dm.n_cov == 0:
        raise ValueError(f"{dm.dir}/store.json names no covariates; WideDeepGP always uses them")
    y_train = train_labels(dm)
    n_train = len(y_train)
    sched = schedule(cfg, n_train)
    dm.train_sampler = make_sampler(cfg.sampler, y_train, cfg.batch_size, sched["batches_per_validation"],
                                    seed=int(cfg.seed))
    model = build_model(cfg, dm, sched)
    if cfg.init_from:
        src = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
        src_labels = (src.get("hyper_parameters") or {}).get("hp", {}).get("label_names")
        left_out = model.load_weights(src["state_dict"], allow_new_head=True, source_labels=src_labels)
        print(f"initialised from {cfg.init_from}; not loaded: {left_out or 'nothing'}", flush=True)
    if getattr(cfg, "init_wide_from", None):
        load_wide_init(model, cfg.init_wide_from)
        print(f"linear PRS path from {cfg.init_wide_from}{' (fixed)' if model.hp.get('wide_freeze') else ''}", flush=True)

    run_dir = os.path.join(cfg.run_dir, cfg.experiment, cfg.exp_name or "run")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    resume = last_checkpoint(ckpt_dir) if cfg.resume else None
    if resume:
        print(f"resuming from {resume}", flush=True)
        check_resume_tables(resume, model.hp, model.loss_fn)
        check_resume_args(os.path.join(run_dir, "config.json"), cfg)
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
              "snp_scaling": dm.snp_stats if dm.snp_mean is not None else "none",
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), **code_version()}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    acc = sched["accumulate"]
    snaps = Snapshots(run_dir, snapshot_steps(sched["optimizer_steps"], int(cfg.snapshots_per_decade),
                                              every=max(1, sched["batches_per_epoch_equiv"] // acc),
                                              max_count=int(getattr(cfg, "snapshots_max", 0) or 0)), n_train)
    runlog = RunLog(run_dir, n_train, n_snps)
    best_ckpt = RunCheckpoint(dirpath=ckpt_dir, filename="best", monitor=MONITOR, mode="max", save_top_k=1,
                                save_last=True, save_on_train_epoch_end=False, auto_insert_metric_name=False)
    callbacks = [best_ckpt, snaps, runlog] + list(extra_callbacks)
    if int(cfg.stop_patience) > 0:
        callbacks.append(EarlyStoppingOnce(monitor=MONITOR, mode="max", patience=int(cfg.stop_patience)))
    if trial is not None:
        callbacks.append(OptunaPruning(trial, MONITOR, steps_per_pass=sched["batches_per_validation"] // acc))
    logger = TensorBoardLogger(save_dir=run_dir, name="", version="tb", default_hp_metric=False)
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                         max_epochs=sched["validations"], accumulate_grad_batches=acc, callbacks=callbacks,
                         logger=logger, num_sanity_val_steps=0, enable_progress_bar=False,
                         log_every_n_steps=min(50, max(1, sched["batches_per_validation"] // acc)),
                         enable_model_summary=False)
    trainer.fit(model, dm, ckpt_path=resume)

    # the best checkpoint, with the metrics of the pass made at its step (scored again only when
    # another process made that pass, as before a resume)
    best_path = best_ckpt.best_model_path or last_checkpoint(ckpt_dir)
    state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    m = runlog.metrics_at.get(int(state.get("global_step", -1)))
    if m is None:
        logits, y = predict_split(model, dm, "val", device)
        m = model.val_metrics(logits, y)            # as the pass computed them (with a grid: per grid point)
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
    if model.grid:
        # the grid's best checkpoint holds every grid point; the run's model takes each disease's
        # best point and pass, from the best_* buffers at the end of training
        last = torch.load(last_checkpoint(ckpt_dir), map_location="cpu", weights_only=False)
        model.load_state_dict(last["state_dict"])
        out.update(write_per_disease(model, dm, run_dir, config, int(last.get("global_step", 0)), device))
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    flat = {k: v for k, v in model.hp.items() if isinstance(v, (str, int, float, bool))}
    flat.update(batch_size=int(cfg.batch_size), accumulate=acc, seed=int(cfg.seed), n_labels=len(model.label_names))
    logger.log_hyperparams(flat, {"hp/best_val_auroc_mean": out["best_val_auroc_mean"],
                                  "hp/best_val_bce_mean": out["val_mean"]["bce"]})
    logger.finalize("success")
    out["model"], out["datamodule"] = model, dm
    return out
