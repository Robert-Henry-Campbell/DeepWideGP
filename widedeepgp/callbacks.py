"""Callbacks of a multi-disease run: weight snapshots, the run's log files, early stopping and
Optuna pruning that count a pass a resumed run repeats once, and checkpoints that follow a copied
run folder."""
import csv
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from . import metrics as M


def snapshot_steps(max_steps, per_decade=8, every=None, max_count=0):
    """Optimiser steps at which weights are saved: ``per_decade`` per tenfold increase on a log
    axis, every ``every`` steps (one epoch-equivalent), and the last. With ``max_count`` the
    regular ones are thinned (every second, third, ...) until the total fits."""
    logged = set()
    k = 0
    while per_decade > 0:
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
    """Weights-only snapshots (a resume reads the last checkpoint instead), and snapshots.csv,
    what each one cost: the compute axis of methods.md section 7."""

    FIELDS = ["step", "batches", "samples_seen", "epoch_equiv", "train_seconds", "file"]

    def __init__(self, run_dir, steps, n_train):
        self.dir = os.path.join(run_dir, "snapshots")
        self.table = os.path.join(run_dir, "snapshots.csv")
        self.steps = {int(s) for s in steps}
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
            torch.save(pl_module.snapshot(step), os.path.join(self.dir, name))
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


def _grad_sq(params):
    return sum(float(p.grad.detach().float().pow(2).sum()) for p in params if p.grad is not None)


class RunLog(pl.Callback):
    """Writes metrics.csv (one row per validation pass, label and metric) and logs throughput,
    memory, learning rates, gradient norms and cases seen per disease to the trainer's logger.
    ``metrics_at`` keeps each pass this process made, by optimiser step, for results.json; that
    includes the pass a resumed run repeats (the model scores it whole), while a pass made before
    a resume is scored again from its checkpoint."""

    FIELDS = ["validation", "step", "samples_seen", "epoch_equiv", "label", "metric", "value", "n", "n_cases"]

    def __init__(self, run_dir, n_train, n_snps):
        self.path = os.path.join(run_dir, "metrics.csv")
        self.n_train, self.n_snps = int(n_train), int(n_snps)
        self.cases_seen = None
        self._t, self._n = None, 0
        self.n_validations = 0
        self._last_step = -1
        self.history = []                       # (validation, step, samples_seen, mean AUROC)
        self.peak_train_bytes = self.peak_val_bytes = 0
        self.metrics_at = {}

    def on_fit_start(self, trainer, pl_module):
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, self.FIELDS).writeheader()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @staticmethod
    def _writes_now(trainer):
        """Lightning 2.0 writes step metrics only on the batches that complete every
        log_every_n_steps-th optimiser step, and drops values logged on the others."""
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
                   "train/lr": groups[0]["lr"]}             # the network's (with --wide only, the covariates')
            big = {id(p) for p in pl_module.big_params()}
            own = [g["lr"] for g in groups if any(id(p) in big for p in g["params"])]
            if own:
                log["train/lr_table" if pl_module.blocks else "train/lr_wide"] = own[0]
            if torch.cuda.is_available():
                log["perf/gpu_peak_gib"] = self.peak_train_bytes / 2 ** 30
            pl_module.log_dict(log)
            self._t, self._n = now, 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self._writes_now(trainer):
            return
        big = pl_module.big_params()
        ids = {id(p) for p in big}                       # the per-SNP weights apart: they would dominate the norm
        pl_module.log("train/grad_norm", _grad_sq(p for p in pl_module.parameters() if id(p) not in ids) ** 0.5)
        if big:
            pl_module.log("train/grad_norm_table" if pl_module.blocks else "train/grad_norm_wide", _grad_sq(big) ** 0.5)

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
        step = int(trainer.global_step)
        if step == self._last_step:                      # the pass a resumed run repeats, scored whole by the
            self.metrics_at[step] = m                    # model: kept for results.json, not logged or counted
            return
        self._last_step = step
        self.n_validations += 1
        self.metrics_at[step] = m
        ee = pl_module.samples_seen / self.n_train
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, self.FIELDS).writerows(M.to_rows(
                m, validation=self.n_validations, step=step, samples_seen=int(pl_module.samples_seen),
                epoch_equiv=round(ee, 6)))
        self.history.append((self.n_validations, step, int(pl_module.samples_seen), float(m["auroc_mean"])))
        if trainer.logger is not None and self.cases_seen is not None:
            seen = self.cases_seen.cpu().numpy()
            trainer.logger.log_metrics({f"cases_seen/{name}": float(seen[j])
                                        for j, name in enumerate(pl_module.label_names)}, step=step)
            trainer.logger.log_metrics({"epoch_equiv": float(ee)}, step=step)

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
    """Early stopping that counts every validation pass once: a resumed run repeats the pass its
    checkpoint was written after, and the repeat must not use up patience."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._seen_step = -1

    def _run_early_stopping_check(self, trainer):
        step = int(trainer.global_step)
        if step == self._seen_step:
            return
        self._seen_step = step
        super()._run_early_stopping_check(trainer)

    def state_dict(self):
        return dict(super().state_dict(), seen_step=self._seen_step)

    def load_state_dict(self, state_dict):
        # keeps this run's --stop_patience (a resume may train longer); Lightning would put back the checkpoint's
        self._seen_step = int(state_dict.get("seen_step", -1))
        patience = self.patience
        super().load_state_dict(state_dict)
        self.patience = patience


class RunCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that carries a run's best and last checkpoints over to a copy of its
    folder, or to the same folder named another way (methods.md section 7). Lightning restores
    the best score and the last checkpoint's name only when the folder is named exactly as when
    saved (a symlink or a relative --run_dir counts as another), so a resumed copy would start a
    new best-v1.ckpt, compared with nothing from before the resume, and a new last-v1.ckpt. When
    the files are there, the saved paths are moved into this folder first."""

    @property
    def state_key(self):
        # ModelCheckpoint's: the pilot's checkpoints restore into this class, and its own into a plain ModelCheckpoint
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
    """Reports the watched metric to an Optuna trial after every validation pass and stops the
    run when the pruner says so. The step reported is the pass's number in the run (optimiser
    steps over steps per pass), so a resumed trial meets the pruner at the right rung."""

    def __init__(self, trial, monitor="val/auroc_mean", steps_per_pass=None):
        self.trial, self.monitor = trial, monitor
        self.steps_per_pass = steps_per_pass
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
        if self.steps_per_pass:
            self.n = max(1, int(round(int(trainer.global_step) / float(self.steps_per_pass))))
        self.trial.report(float(value), step=self.n)
        if self.trial.should_prune():
            import optuna
            raise optuna.TrialPruned(f"pruned at validation {self.n} ({self.monitor} = {float(value):.4f})")
