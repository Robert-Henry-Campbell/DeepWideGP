# -*- coding: utf-8 -*-
"""Hyper-parameter search inside every run.

Robert's decision (12 Nov 2026): every run of WideDeepGP redoes the tuning, because the best
settings change with the training size of the scaling study. main_multilabel.py therefore
calls ``tune_run`` whenever --tune_trials > 0 (its default). With the linear PRS path on
(--wide on, the default) a run searches in two stages:

  1. linear   the linear model alone (PRS and covariates, --wide only) for every prior variance
              of --prs_grid at once (one training run; the diseases do not interact in the
              linear model, so each grid point is the model that variance alone would give).
              Each disease keeps the variance and validation pass with its best tuning AUROC,
              as the polygenic-score comparators are tuned per disease; that model is written
              to linear/per_disease/. Large batches, one validation per epoch-equivalent (and
              at most one per --val_min_participants), the learning rates on a cosine to zero
              over the run's budget.
  2. joint    the deep network on top, started from stage 1's per-disease model with the PRS
              weights fixed (--init_wide_from, --wide_freeze); the covariate weights and
              intercepts keep learning. Searched (--tune_trials, Optuna): learning rate, weight
              decay, dropout, with successive-halving pruning.

The run's model is always the joint one (the network on top of the linear model), whatever its
tuning AUROC relative to the linear model alone, which results.json records as an ablation
("linear_only"; Robert's decision of 15 Nov 2026). --wide only runs stage 1 alone, --wide off
searches the deep network alone (stage "deep"). The run's folder,
<run_dir>/<experiment>/<exp_name>/, holds linear/ (the grid run, with per_disease/ inside) and,
for an Optuna stage, a subfolder with its study (optuna_journal.log) and one ordinary training
run per trial (trial_NNNN/), plus

    results.json   the chosen stage and model, and every trial's settings, value and cost
    search.json    the arguments the search was started with (a rerun with others is refused)

Trials are spread over GPUs, one worker process per GPU (main_multilabel.py with
--tune_worker); --tune_gpus auto takes the GPUs with nothing running on them minus one, the
lab's rule, and waits until at least two are free (picked again at the start of each stage).
With --tune_gpus none, or without a GPU, the trials run here, one after another. Every trial trains with the run's seed; each worker's sampler is seeded with the run's
seed plus its index. When a search is over, the weights of every trial but the chosen one and
stage 1's per-disease model are deleted (records stay). Rerunning the same command continues
the search: an unfinished grid run resumes from its last checkpoint, and trials that a crash
left running are marked failed and replaced. One search per run at a time (a lock file refuses
a second).
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
FREE_ARGS = {"tune_gpus", "tune_worker", "num_workers", "eval_batch_size", "resume", "tune_trials"}
# prior variances of the PRS (on the logit scale) for the linear stage: half-decade steps over
# the range a disease's PRS can have (a liability-scale heritability of a few percent to well over
# one half), down to values at which the PRS is negligible, so that a disease whose SNPs give the
# model nothing usable falls back to its covariates (on the pilot's proxy region, 6 of 30 diseases
# chose 0.03, then the lowest value)
DEFAULT_PRS_GRID = "0.003,0.01,0.03,0.1,0.3,1,3,10,30"
# learning rate of the linear stage's covariate weights and intercepts (the network's default,
# 1e-4, would leave them short of their optimum on the smallest training sets: about 500 steps)
LINEAR_LR = 1e-2


def run_root(cfg):
    return os.path.join(cfg.run_dir, cfg.experiment, cfg.exp_name or "run")


def stages(base):
    """[(stage, trials)] of a run's search; the linear stage is one grid run."""
    wide = getattr(base, "wide", "off") or "off"
    if wide == "on":
        return [("linear", 1), ("joint", int(base.tune_trials))]
    if wide == "only":
        return [("linear", 1)]
    return [("deep", int(base.tune_trials))]


def stage_cfg(base, stage, init=None):
    """The settings every trial of a stage starts from (before the search's suggestions)."""
    cfg = copy.copy(base)
    cfg.tune_trials, cfg.tune_worker, cfg.resume = 0, False, False
    cfg.init_wide_from, cfg.wide_freeze = None, False
    if stage == "linear":
        cfg.wide, cfg.annot = "only", False
        cfg.prs_grid = base.prs_grid if getattr(base, "prs_grid", None) else DEFAULT_PRS_GRID
        cfg.batch_size, cfg.accumulate = max(int(base.batch_size), 256), 1
        cfg.lr, cfg.lr_schedule, cfg.warmup_steps = LINEAR_LR, "cosine", 0   # the PRS settles in about 1/kappa
        #                                                  epoch-equivalents; the cosine then removes the noise
        cfg.stop_patience = 10 if int(base.stop_patience) > 0 else 0
        cfg.val_per_epoch, cfg.val_max_epoch_equiv = 1.0, 0.0  # once per epoch-equivalent, or per val_min_participants
        cfg.snapshots_per_decade, cfg.snapshots_max = 0, 1   # the grid's weights are not kept
        cfg.exp_name = os.path.join(base.exp_name or "run", "linear")
    elif stage == "joint":
        cfg.wide, cfg.init_wide_from, cfg.wide_freeze = "on", init, True
        cfg.prs_grid = None
    else:
        cfg.wide, cfg.prs_grid = "off", None
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
    tune_optuna.suggest(trial, cfg, "deep")             # an enqueued trial gets its queued settings back
    resume_from = trial.user_attrs.get("resume_from")
    if resume_from:                                      # a trial a crash interrupted: carry on in its folder
        cfg.exp_name = os.path.relpath(resume_from, os.path.abspath(os.path.join(cfg.run_dir, cfg.experiment)))
        cfg.resume = True
    else:
        cfg.exp_name = os.path.join(base.exp_name or "run", stage, f"trial_{trial.number:04d}")
    trial_dir = os.path.join(cfg.run_dir, cfg.experiment, cfg.exp_name)
    trial.set_user_attr("run_dir", os.path.abspath(trial_dir))
    try:
        out = run_training(cfg, trial=trial)
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
    if env["WIDEDEEPGP_TUNE_STAGE"] == "linear":
        linear_run(base)
        return
    worker(base, env["WIDEDEEPGP_TUNE_STAGE"], int(env["WIDEDEEPGP_TUNE_N"]), int(env.get("WIDEDEEPGP_TUNE_WORKER", 0)),
           env.get("WIDEDEEPGP_WIDE_INIT") or None)


def linear_dir(base):
    return os.path.join(run_root(base), "linear")


def linear_run(base):
    """Stage 1: the grid run, resumed from its last checkpoint if an earlier attempt left one."""
    from widedeepgp.train import last_checkpoint, run_training
    cfg = stage_cfg(base, "linear")
    cfg.resume = last_checkpoint(os.path.join(linear_dir(base), "checkpoints")) is not None
    run_training(cfg)


def linear_stage(base, argv, script, gpus):
    """Stage 1 on the first of ``gpus`` (in a worker process) or here; nothing to do if done."""
    if not os.path.exists(os.path.join(linear_dir(base), "results.json")):
        if gpus:
            log_dir = os.path.join(run_root(base), "workers")
            os.makedirs(log_dir, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpus[0]), WIDEDEEPGP_TUNE_STAGE="linear",
                       WIDEDEEPGP_TUNE_N="1", WIDEDEEPGP_TUNE_WORKER="0", WIDEDEEPGP_WIDE_INIT="")
            with open(os.path.join(log_dir, f"linear_gpu{gpus[0]}.log"), "a") as log:
                code = subprocess.Popen([sys.executable, script, *argv, "--tune_worker"], env=env, stdout=log,
                                        stderr=subprocess.STDOUT).wait()
            if code:
                print(f"linear: worker exit code {code}; log in {log_dir}", flush=True)
        else:
            linear_run(base)
    return linear_summary(base)


def linear_summary(base):
    """Stage 1 as a one-trial stage: with a grid, the per-disease model."""
    d = linear_dir(base)
    path = os.path.join(d, "results.json")
    if not os.path.exists(path):
        raise RuntimeError(f"the linear stage in {d} did not finish; see its folder and workers/linear_gpu*.log")
    with open(path) as f:
        r = json.load(f)
    # paths inside the stage's folder, not those recorded when it ran: the folder may have moved
    if r.get("per_disease_checkpoint"):
        best_dir, value = os.path.join(d, "per_disease"), r["per_disease_val_auroc_mean"]
        ckpt = os.path.join(best_dir, "checkpoints", os.path.basename(r["per_disease_checkpoint"]))
        params = {"prs_var_per_label": {row["label"]: row["prs_var"] for row in r["per_disease"]}}
    else:                                                # --prs_grid none: one prior variance for every disease
        best_dir, value = d, r["best_val_auroc_mean"]
        ckpt = os.path.join(d, "checkpoints", os.path.basename(r["best_checkpoint"]))
        with open(os.path.join(d, "config.json")) as f:
            params = {"prs_var": json.load(f)["model"]["prs_var"]}
    row = {"trial": 0, "state": "complete", "value": value, "params": params, "run_dir": os.path.abspath(d),
           "epoch_equivalents": r.get("epoch_equivalents"), "train_seconds": r.get("train_seconds")}
    return {"stage": "linear", "n_trials": 1, "n_complete": 1, "n_pruned": 0,
            "train_seconds": round(r.get("train_seconds") or 0.0, 1), "trials": [row], "best_trial": 0,
            "best_value": value, "best_params": params, "best_trial_dir": os.path.abspath(best_dir),
            "best_checkpoint": os.path.abspath(ckpt)}


def fail_stale(study):
    """Trials still marked running when a search starts were left by a crash: mark them failed,
    and queue each again with the same settings. A trial that had written a checkpoint carries
    on from its last checkpoint in its own folder (resume_from), so a crash at full size, where a
    trial can run for days, loses only the time since that checkpoint."""
    from widedeepgp.train import last_checkpoint
    stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for t in stale:
        study.tell(t.number, state=TrialState.FAIL)
        if not t.params:
            continue
        attrs = {"resumes_trial": t.number}
        d = t.user_attrs.get("run_dir")
        try:
            if d and last_checkpoint(os.path.join(d, "checkpoints")):
                attrs["resume_from"] = d
        except ValueError:                               # an ambiguous checkpoint folder: start afresh
            pass
        study.enqueue_trial(dict(t.params), user_attrs=attrs)
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
            if stage == "linear":
                done = os.path.exists(os.path.join(linear_dir(base), "results.json"))
                summaries[stage] = linear_stage(base, argv, script, [] if done else pick_gpus(spec))
                init = summaries[stage]["best_checkpoint"]
                continue
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
        summary = choose(base, summaries)
    print(json.dumps({k: summary[k] for k in ("chosen_stage", "best_trial_dir", "best_val_auroc_mean", "best_params")},
                     indent=1), flush=True)
    return summary


def choose(base, summaries):
    """The run's model, its results.json, and the deletion of the other trials' weights.

    The run's model is the last stage's best trial: with the linear path on, the network on top
    of the linear model, whatever its tuning AUROC relative to the linear model alone (Robert's
    decision of 15 Nov 2026: WideDeepGP is reported as the whole model, never as the ridge PGI
    inside it). The linear model alone is recorded as an ablation ("linear_only")."""
    root = run_root(base)
    chosen = [s for s in ("deep", "linear", "joint") if s in summaries][-1]
    best = summaries[chosen]
    with open(os.path.join(best["best_trial_dir"], "results.json")) as f:
        trial = json.load(f)
    out = {"tuning": True, "chosen_stage": chosen,
           "best_trial_dir": os.path.relpath(best["best_trial_dir"], os.path.abspath(root)),
           "best_checkpoint": best["best_checkpoint"], "best_val_auroc_mean": best["best_value"],
           "best_params": best["best_params"], "val": trial.get("val"), "val_mean": trial.get("val_mean"),
           "train_seconds_all_trials": round(sum(s["train_seconds"] for s in summaries.values()), 1),
           "stages": summaries, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    if chosen == "joint":                                # the ablation: the same run without its network
        lin = summaries["linear"]
        out["linear_only"] = {"best_trial_dir": os.path.relpath(lin["best_trial_dir"], os.path.abspath(root)),
                              "best_val_auroc_mean": lin["best_value"],
                              "network_minus_linear_only": best["best_value"] - lin["best_value"]}
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
