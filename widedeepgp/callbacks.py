# -*- coding: utf-8 -*-
"""Callbacks for a multi-disease run: weight snapshots, the run's own log files, early stopping
that counts a repeated validation pass once (EarlyStoppingOnce), Optuna pruning."""
import csv
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from . import metrics as M


def snapshot_steps(max_steps, per_decade=8, every=None, max_count=0):
    """Optimiser steps at which weights are saved: evenly spaced on a log axis (``per_decade``
    per tenfold increase), plus every ``every`` steps (one epoch-equivalent), plus the last.
    With ``max_count``, the regular ones are thinned (every second, third, ... epoch-equivalent)
    until the total fits; the log-spaced ones and the last are kept."""
    logged = set()
    k = 0
    while per_decade > 0:                                # per_decade <= 0: no log-spaced snapshots
        s = int(round(10 ** (k / float(per_decade))))
        if s > max_steps:
            break
        logged.add(s)
        k += 1
    logged.add(int(max_steps))
    regular = list(range(int(every), int(max_steps) + 1, int(every))) if every else []
    if max_count and regular:
        stride = 1
        while stride < len(regular) and len(logged | set(regular[stride - 1::stride])) > max_count:
            stride += 1
        regular = regular[stride - 1::stride]
    return sorted(s for s in logged | set(regular) if s >= 1)


class Snapshots(pl.Callback):
    """Weights-only snapshots along training, with a table of what each one cost.

    The table (snapshots.csv) records optimiser step, batches, participants seen,
    epoch-equivalents and training time, which is what a compute axis needs. Snapshots hold
    weights and the model's configuration, not optimiser state: resuming uses the run's last
    checkpoint (train.last_checkpoint).
    """

    FIELDS = ["step", "batches", "samples_seen", "epoch_equiv", "train_seconds", "file"]

    def __init__(self, run_dir, steps, n_train):
        self.dir = os.path.join(run_dir, "snapshots")
        self.table = os.path.join(run_dir, "snapshots.csv")
        self.steps = set(int(s) for s in steps)
        self.n_train = int(n_train)
        self.done = set()
        self.train_seconds = 0.0
        self._t = None

    def on_fit_start(self, trainer, pl_module):
        os.makedirs(self.dir, exist_ok=True)
        if os.path.exists(self.table):
            with open(self.table) as f:
                self.done = {int(r["step"]) for r in csv.DictReader(f)}
        else:
            with open(self.table, "w", newline="") as f:
                csv.DictWriter(f, self.FIELDS).writeheader()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._t = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._t is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.train_seconds += time.perf_counter() - self._t
        step = int(trainer.global_step)
        if step in self.steps and step not in self.done:
            self.done.add(step)
            name = f"step_{step:09d}.ckpt"
            trainer.save_checkpoint(os.path.join(self.dir, name), weights_only=True)
            with open(self.table, "a", newline="") as f:
                csv.DictWriter(f, self.FIELDS).writerow({
                    "step": step, "batches": int(trainer.fit_loop.total_batch_idx) + 1,
                    "samples_seen": int(pl_module.samples_seen),
                    "epoch_equiv": round(pl_module.samples_seen / self.n_train, 6),
                    "train_seconds": round(self.train_seconds, 2), "file": os.path.join("snapshots", name)})

    def state_dict(self):
        return {"train_seconds": self.train_seconds}

    def load_state_dict(self, state):
        self.train_seconds = float(state.get("train_seconds", 0.0))


class RunLog(pl.Callback):
    """Writes metrics.csv (one row per validation pass, label and metric) and logs throughput,
    memory, learning rate, gradient norm and the cases seen per disease to the trainer's logger.

    ``metrics_at`` keeps the metrics of each pass this process made, by optimiser step, so that
    results.json takes the best step's from there instead of scoring the tuning set again; a
    pass made before a resume is not in it, and that step is scored again from its checkpoint.
    """

    FIELDS = ["validation", "step", "samples_seen", "epoch_equiv", "label", "metric", "value", "n", "n_cases"]

    def __init__(self, run_dir, n_train, n_snps):
        self.path = os.path.join(run_dir, "metrics.csv")
        self.n_train, self.n_snps = int(n_train), int(n_snps)
        self.cases_seen = None
        self._t, self._n = None, 0
        self.n_validations = 0
        self._last_step = -1
        self.history = []                       # (validation, step, samples_seen, mean AUROC)
        self.peak_train_bytes = self.peak_val_bytes = 0     # peak GPU memory, training and validation apart
        self.metrics_at = {}                    # optimiser step -> the metrics of the validation pass made there

    def on_fit_start(self, trainer, pl_module):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.FIELDS).writeheader()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _writes_now(trainer):
        """True during the batches whose step metrics Lightning 2.0 writes to the logger: those
        that complete every ``log_every_n_steps``-th optimiser step. Values logged on other
        batches are dropped, so step metrics are logged on these batches only."""
        loop = trainer.fit_loop.epoch_loop
        return (not loop._should_accumulate()) and (loop._batches_that_stepped + 1) % trainer.log_every_n_steps == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        y = batch[1]
        if torch.cuda.is_available():
            self.peak_train_bytes = max(self.peak_train_bytes, torch.cuda.max_memory_allocated())
        seen = torch.nan_to_num(y, nan=0.0).sum(dim=0).double()
        self.cases_seen = seen if self.cases_seen is None else self.cases_seen + seen
        self._n += int(y.shape[0])
        now = time.perf_counter()
        if self._t is None:
            self._t, self._n = now, 0
        elif self._writes_now(trainer) and now > self._t:
            rate = self._n / (now - self._t)
            groups = trainer.optimizers[0].param_groups
            log = {"perf/samples_per_s": rate, "perf/million_sample_snps_per_s": rate * self.n_snps / 1e6,
                   "train/lr": groups[0]["lr"]}             # the deep network's (or, with --wide only, the covariates')
            prs = {id(p) for p in getattr(pl_module, "prs", [])}
            if prs and any(id(p) in prs for g in groups for p in g["params"]):
                log["train/lr_wide"] = [g["lr"] for g in groups if any(id(p) in prs for p in g["params"])][0]
            if torch.cuda.is_available():
                log["perf/gpu_peak_gib"] = self.peak_train_bytes / 2 ** 30
            pl_module.log_dict(log)
            self._t, self._n = now, 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if self._writes_now(trainer):
            prs = {id(p) for p in getattr(pl_module, "prs", [])}
            sq = sum(float(p.grad.detach().float().pow(2).sum()) for p in pl_module.parameters()
                     if p.grad is not None and id(p) not in prs)
            pl_module.log("train/grad_norm", sq ** 0.5)    # everything but the PRS weights, which would dominate it
            if prs:
                sq = sum(float(p.grad.detach().float().pow(2).sum()) for p in pl_module.prs if p.grad is not None)
                pl_module.log("train/grad_norm_wide", sq ** 0.5)

    def on_validation_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_validation_end(self, trainer, pl_module):
        if torch.cuda.is_available():            # evaluation batches may be larger than training batches
            self.peak_val_bytes = max(self.peak_val_bytes, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        m = pl_module.last_val_metrics
        if m is None or trainer.sanity_checking:
            return
        if int(trainer.global_step) == self._last_step:      # the validation a resumed run repeats
            return
        self._last_step = int(trainer.global_step)
        self.n_validations += 1
        self.metrics_at[int(trainer.global_step)] = m
        ee = pl_module.samples_seen / self.n_train
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.FIELDS).writerows(M.to_rows(
                m, validation=self.n_validations, step=int(trainer.global_step),
                samples_seen=int(pl_module.samples_seen), epoch_equiv=round(ee, 6)))
        self.history.append((self.n_validations, int(trainer.global_step), int(pl_module.samples_seen),
                             float(m["auroc_mean"])))
        if trainer.logger is not None and self.cases_seen is not None:
            seen = self.cases_seen.cpu().numpy()
            trainer.logger.log_metrics({f"cases_seen/{name}": float(seen[j])
                                        for j, name in enumerate(pl_module.label_names)}, step=int(trainer.global_step))
            trainer.logger.log_metrics({"epoch_equiv": float(ee)}, step=int(trainer.global_step))

    def state_dict(self):
        return {"n_validations": self.n_validations, "history": self.history, "last_step": self._last_step,
                "cases_seen": None if self.cases_seen is None else self.cases_seen.cpu().tolist()}

    def load_state_dict(self, state):
        self.n_validations = int(state.get("n_validations", 0))
        self._last_step = int(state.get("last_step", -1))
        self.history = [tuple(h) for h in state.get("history", [])]
        if state.get("cases_seen") is not None:
            self.cases_seen = torch.tensor(state["cases_seen"], dtype=torch.float64,
                                           device="cuda" if torch.cuda.is_available() else "cpu")


class EarlyStoppingOnce(EarlyStopping):
    """Early stopping that counts every validation pass once.

    A checkpoint written after a validation pass makes a resumed run repeat that pass before it
    trains further; the repeat must not use up patience.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen_step = -1

    def _run_early_stopping_check(self, trainer):
        if int(trainer.global_step) == self._seen_step:
            return
        self._seen_step = int(trainer.global_step)
        super()._run_early_stopping_check(trainer)

    def state_dict(self):
        return dict(super().state_dict(), seen_step=self._seen_step)

    def load_state_dict(self, state_dict):
        """Restore the count of passes without improvement but keep this run's patience
        (--stop_patience): a resumed run may change how long it trains, and Lightning would
        otherwise put back the checkpoint's patience."""
        self._seen_step = int(state_dict.get("seen_step", -1))
        patience = self.patience
        super().load_state_dict(state_dict)
        self.patience = patience



class RunCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that carries a run's best and last checkpoints over to a copy of its
    directory, or to the same directory named another way.

    Lightning restores the best score and the name of the last checkpoint only when the
    checkpoint folder is named exactly as when the state was saved (with this fsspec it keeps
    the path as given: a symlink or a relative --run_dir counts as another folder). A copied
    run directory, once resumed, would start a new best (best-v1.ckpt) compared with nothing
    from before the resume, and a new last-v1.ckpt beside the copied last.ckpt. When the files
    are there, the saved paths are moved into this folder first, so the run carries on in
    best.ckpt and last.ckpt with its best score, as a run resumed in place does. Copy a run
    directory only while nothing is writing to it."""

    @property
    def state_key(self):
        # ModelCheckpoint's own key, so that checkpoints written before this class existed (every
        # run of the pilot) restore it, and the ones it writes restore a plain ModelCheckpoint
        return "ModelCheckpoint" + super().state_key[len(type(self).__qualname__):]

    def load_state_dict(self, state_dict):
        saved_dir = state_dict.get("dirpath")
        if saved_dir and self.dirpath and saved_dir != self.dirpath:
            def here(path):
                return os.path.join(self.dirpath, os.path.basename(path)) if path else path
            moved = {k: here(state_dict.get(k, "")) for k in ("best_model_path", "kth_best_model_path",
                                                             "last_model_path")}
            best_k = {here(path): score for path, score in (state_dict.get("best_k_models") or {}).items()}
            if all(os.path.exists(path) for path in [*moved.values(), *best_k] if path):
                state_dict = {**state_dict, **moved, "best_k_models": best_k, "dirpath": self.dirpath}
        super().load_state_dict(state_dict)

class OptunaPruning(pl.Callback):
    """Report the watched validation metric to an Optuna trial after every validation pass and
    stop the run when the trial's pruner says so."""

    def __init__(self, trial, monitor="val/auroc_mean"):
        self.trial, self.monitor = trial, monitor
        self.n = 0
        self._last_step = -1

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None or int(trainer.global_step) == self._last_step:
            return
        self._last_step = int(trainer.global_step)
        self.n += 1
        self.trial.report(float(value), step=self.n)
        if self.trial.should_prune():
            import optuna
            raise optuna.TrialPruned(f"pruned at validation {self.n} ({self.monitor} = {float(value):.4f})")
