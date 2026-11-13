# -*- coding: utf-8 -*-
"""Hyper-parameter search inside every run.

Robert's decision (12 Nov 2026): every run of WideDeepGP redoes the tuning, because the best
settings change with the training size of the scaling study. main_multilabel.py therefore
calls ``tune_run`` whenever --tune_trials > 0 (its default). With the linear PRS path on
(--wide on, the default) a run searches in two stages:

  1. linear   the linear model alone (PRS and covariates, --wide only), large batches, one
              validation per epoch-equivalent, twice the run's budget of epoch-equivalents,
              no pruning: its weights settle slowly (about 1/wide_kappa epoch-equivalents), so early
              validations would judge them unfairly. Searched: the ridge penalty (prs_var), the
              pace of the PRS weights (wide_kappa) and the covariate weights' learning rate.
  2. joint    the deep network on top, started from stage 1's best linear path with the PRS
              weights fixed (--init_wide_from, --wide_freeze); the covariate weights and
              intercepts keep learning. Searched: learning rate, weight decay, dropout, with
              successive-halving pruning.

The run's model is the joint one only if its mean tuning AUROC is strictly higher than the
linear one's; otherwise the linear model is. --wide only searches stage 1 alone, --wide off
the deep network alone (stage "deep"). The run's folder, <run_dir>/<experiment>/<exp_name>/,
holds one subfolder per stage, each with its Optuna study (optuna_journal.log) and one ordinary
training run per trial (trial_NNNN/), plus

    results.json   the chosen stage and trial, and every trial's settings, value and cost
    search.json    the arguments the search was started with (a rerun with others is refused)

Trials are spread over GPUs, one worker process per GPU (main_multilabel.py with
--tune_worker); --tune_gpus auto takes the GPUs with nothing running on them minus one, the
lab's rule, and waits until at least two are free (picked again at the start of each stage).
With --tune_gpus none, or without a GPU, the trials run here, one after another. Every trial trains with the run's seed; each worker's sampler is seeded with the run's
seed plus its index. When a search is over, the weights of every trial but the chosen one and
stage 1's best are deleted (records stay). Rerunning the same command continues the search:
trials that a crash left running are marked failed and replaced. One search per run at a time
(a lock file refuses a second).
"""
import copy
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time

import optuna
from optuna.trial import TrialState

FINISHED = (TrialState.COMPLETE, TrialState.PRUNED)
# arguments that may change when a search is continued
FREE_ARGS = {"tune_gpus", "tune_worker", "num_workers", "eval_batch_size", "resume", "tune_trials",
             "tune_trials_linear"}


def run_root(cfg):
    return os.path.join(cfg.run_dir, cfg.experiment, cfg.exp_name or "run")


def stages(base):
    """[(stage, trials)] of a run's search."""
    wide = getattr(base, "wide", "off") or "off"
    if wide == "on":
        return [("linear", int(base.tune_trials_linear)), ("joint", int(base.tune_trials))]
    if wide == "only":
        return [("linear", int(base.tune_trials_linear))]
    return [("deep", int(base.tune_trials))]


def stage_cfg(base, stage, init=None):
    """The settings every trial of a stage starts from (before the search's suggestions)."""
    cfg = copy.copy(base)
    cfg.tune_trials, cfg.tune_worker, cfg.resume = 0, False, False
    cfg.init_wide_from, cfg.wide_freeze = None, False
    if stage == "linear":
        cfg.wide, cfg.annot = "only", False
        cfg.batch_size, cfg.accumulate = max(int(base.batch_size), 256), 1
        cfg.max_epoch_equiv = 2.0 * float(base.max_epoch_equiv)     # 60 at the default: it settles in about 1/kappa
        cfg.stop_patience = 10 if int(base.stop_patience) > 0 else 0
        cfg.val_per_epoch = 1.0
        cfg.val_min_participants, cfg.val_max_epoch_equiv, cfg.min_participants = 0, 1.0, 0
        cfg.snapshots_max = min(int(getattr(base, "snapshots_max", 0) or 10), 10)
    elif stage == "joint":
        cfg.wide, cfg.init_wide_from, cfg.wide_freeze = "on", init, True
    else:
        cfg.wide = "off"
    return cfg


def open_stage_study(base, stage, worker_index=0):
    """A stage's study. Each worker's sampler is seeded from the run's seed plus its index
    (with one seed for all, every worker would start with the same random settings)."""
    import tune_optuna
    return tune_optuna.open_study(stage, run_root(base), seed=int(base.seed) + int(worker_index), n_startup_trials=5,
                                  prune=stage != "linear")


def free_gpus():
    """Indices of the GPUs with no compute process on them (empty without nvidia-smi)."""
    try:
        busy = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.split()
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=60).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for row in rows:
        idx, uuid = [x.strip() for x in row.split(",")]
        if uuid not in busy:
            out.append(idx)
    return out


def pick_gpus(spec, poll_seconds=300):
    """The GPUs a search uses: 'auto' (the free ones minus the lowest-numbered, once two are
    free), a comma-separated list, or None (one worker in this process)."""
    if not spec or str(spec).lower() in ("none", "cpu"):
        return []
    if spec != "auto":
        return [g.strip() for g in str(spec).split(",") if g.strip()]
    while True:
        free = free_gpus()
        if len(free) >= 2:
            return free[1:]
        print(f"{time.strftime('%H:%M:%S')} waiting for GPUs: {len(free)} free, the search needs two "
              f"(it uses the free ones minus one)", flush=True)
        time.sleep(poll_seconds)


def drop_weights(trial_dir):
    """Delete a trial's weights (checkpoints and snapshots); keep its records."""
    for sub in ("checkpoints", "snapshots"):
        p = os.path.join(trial_dir, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)


def objective(trial, base, stage, init):
    import torch
    import tune_optuna
    from widedeepgp.train import run_training
    cfg = stage_cfg(base, stage, init)
    tune_optuna.suggest(trial, cfg, "linear" if stage == "linear" else "deep")
    cfg.exp_name = os.path.join(base.exp_name or "run", stage, f"trial_{trial.number:04d}")
    trial_dir = os.path.join(cfg.run_dir, cfg.experiment, cfg.exp_name)
    trial.set_user_attr("run_dir", os.path.abspath(trial_dir))
    try:
        out = run_training(cfg, trial=None if stage == "linear" else trial)
    except optuna.TrialPruned:
        drop_weights(trial_dir)
        raise
    except torch.cuda.OutOfMemoryError:
        drop_weights(trial_dir)
        torch.cuda.empty_cache()
        raise optuna.TrialPruned("out of GPU memory")
    trial.set_user_attr("epoch_equivalents", out["epoch_equivalents"])
    trial.set_user_attr("train_seconds", out["train_seconds"])
    trial.set_user_attr("best_checkpoint", os.path.abspath(out["best_checkpoint"]))
    return out["best_val_auroc_mean"]


def worker(base, stage, n_trials, worker_index=0, init=None):
    """Run ``n_trials`` trials of a stage."""
    if int(n_trials) <= 0:
        return
    study = open_stage_study(base, stage, worker_index)
    study.optimize(lambda t: objective(t, base, stage, init), n_trials=int(n_trials), gc_after_trial=True)


def worker_from_env(base):
    """A worker process started by tune_run (main_multilabel.py --tune_worker)."""
    env = os.environ
    worker(base, env["WIDEDEEPGP_TUNE_STAGE"], int(env["WIDEDEEPGP_TUNE_N"]), int(env.get("WIDEDEEPGP_TUNE_WORKER", 0)),
           env.get("WIDEDEEPGP_WIDE_INIT") or None)


def fail_stale(study):
    """Trials still marked running when a search starts were left by a crash: mark them failed."""
    stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for t in stale:
        study.tell(t.number, state=TrialState.FAIL)
    return len(stale)


def run_workers(base, argv, script, stage, n, gpus, init):
    """The trials of a stage: in this process without GPUs, else one worker per GPU."""
    if not gpus:
        worker(base, stage, n, 0, init)
        return
    k = min(len(gpus), n)
    share = [n // k + (1 if i < n % k else 0) for i in range(k)]
    log_dir = os.path.join(run_root(base), stage, "workers")
    os.makedirs(log_dir, exist_ok=True)
    procs = []
    for i in range(k):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[i]), WIDEDEEPGP_TUNE_STAGE=stage,
                   WIDEDEEPGP_TUNE_N=str(share[i]), WIDEDEEPGP_TUNE_WORKER=str(i), WIDEDEEPGP_WIDE_INIT=init or "")
        log = open(os.path.join(log_dir, f"gpu{gpus[i]}.log"), "a")
        procs.append((subprocess.Popen([sys.executable, script, *argv, "--tune_worker"], env=env, stdout=log,
                                       stderr=subprocess.STDOUT), log))
    codes = [p.wait() for p, _ in procs]
    for _, log in procs:
        log.close()
    if any(codes):
        print(f"{stage}: worker exit codes {codes}; logs in {log_dir}", flush=True)


def stage_summary(base, stage):
    import tune_optuna
    study = tune_optuna.load_study(stage, run_root(base))
    trials = study.get_trials(deepcopy=False)
    done = [t for t in trials if t.state == TrialState.COMPLETE]
    rows = [{"trial": t.number, "state": t.state.name.lower(), "value": t.value, "params": t.params,
             "run_dir": t.user_attrs.get("run_dir"), "epoch_equivalents": t.user_attrs.get("epoch_equivalents"),
             "train_seconds": t.user_attrs.get("train_seconds")} for t in trials]
    out = {"stage": stage, "n_trials": len(trials), "n_complete": len(done),
           "n_pruned": sum(t.state == TrialState.PRUNED for t in trials),
           "train_seconds": round(sum(r["train_seconds"] or 0.0 for r in rows), 1), "trials": rows}
    if done:
        best = max(done, key=lambda t: (t.value, -t.number))
        out.update(best_trial=best.number, best_value=best.value, best_params=best.params,
                   best_trial_dir=best.user_attrs["run_dir"], best_checkpoint=best.user_attrs.get("best_checkpoint"))
    return out


class _RunLock:
    """One search per run at a time."""

    def __init__(self, root):
        self.path = os.path.join(root, ".search.lock")

    def __enter__(self):
        self.f = open(self.path, "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close()
            raise RuntimeError(f"another search of {os.path.dirname(self.path)} is running")
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def check_identity(base, root):
    """A search may be continued only with the arguments it was started with (but more trials)."""
    now = {k: v for k, v in vars(base).items() if isinstance(v, (str, int, float, bool, type(None))) and k not in FREE_ARGS}
    path = os.path.join(root, "search.json")
    if os.path.exists(path):
        with open(path) as f:
            then = json.load(f)
        changed = sorted(k for k in set(then) | set(now) if then.get(k) != now.get(k))
        if changed:
            raise ValueError(f"{root} holds a search started with other arguments: "
                             + ", ".join(f"--{k} {then.get(k)!r} then, {now.get(k)!r} now" for k in changed))
    else:
        with open(path, "w") as f:
            json.dump(now, f, indent=1)


def tune_run(base, argv, script):
    """The search of one run. ``argv`` and ``script`` start the workers (main_multilabel.py)."""
    root = run_root(base)
    os.makedirs(root, exist_ok=True)
    import torch
    with _RunLock(root):
        check_identity(base, root)
        spec = base.tune_gpus if torch.cuda.is_available() else None
        summaries, init = {}, None
        for stage, n in stages(base):
            study = open_stage_study(base, stage)
            stale = fail_stale(study)
            if stale:
                print(f"{stage}: {stale} trial(s) left running by an earlier attempt marked failed", flush=True)
            remaining = n - len(study.get_trials(deepcopy=False, states=FINISHED))
            if remaining > 0:                            # GPUs picked per stage: the rule applies when trials start
                run_workers(base, argv, script, stage, remaining, pick_gpus(spec), init)
            summaries[stage] = stage_summary(base, stage)
            if "best_trial" not in summaries[stage]:
                raise RuntimeError(f"no trial of stage {stage} in {root} completed; see the trials' folders")
            if stage == "linear":
                init = summaries[stage]["best_checkpoint"]
        summary = choose(base, summaries)
    print(json.dumps({k: summary[k] for k in ("chosen_stage", "best_trial_dir", "best_val_auroc_mean", "best_params")},
                     indent=1), flush=True)
    return summary


def choose(base, summaries):
    """The run's model, its results.json, and the deletion of the other trials' weights."""
    root = run_root(base)
    order = [s for s in ("deep", "linear", "joint") if s in summaries]
    chosen = order[0]
    if "joint" in summaries and summaries["joint"]["best_value"] > summaries["linear"]["best_value"]:
        chosen = "joint"
    elif "linear" in summaries:
        chosen = "linear"
    best = summaries[chosen]
    with open(os.path.join(best["best_trial_dir"], "results.json")) as f:
        trial = json.load(f)
    out = {"tuning": True, "chosen_stage": chosen,
           "best_trial_dir": os.path.relpath(best["best_trial_dir"], os.path.abspath(root)),
           "best_checkpoint": best["best_checkpoint"], "best_val_auroc_mean": best["best_value"],
           "best_params": best["best_params"], "val": trial.get("val"), "val_mean": trial.get("val_mean"),
           "train_seconds_all_trials": round(sum(s["train_seconds"] for s in summaries.values()), 1),
           "stages": summaries, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(root, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    keep = {best["best_trial_dir"]}
    if "linear" in summaries:                            # the joint stage was started from it
        keep.add(summaries["linear"]["best_trial_dir"])
    for s in summaries.values():
        for row in s["trials"]:
            d = row["run_dir"]
            if d and d not in keep and os.path.isdir(d):
                drop_weights(d)
    return out
