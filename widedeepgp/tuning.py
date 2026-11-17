"""The hyper-parameter search inside every run (methods.md section 8), which main_multilabel.py
starts with ``tune_run`` whenever --tune_trials > 0. With --wide off, the default, a run searches
the network alone (stage "deep": Optuna trials over learning rate, weight decay, dropout and, in
blocks mode, the table's prior variance). With --wide on it searches in two stages:

  1. linear   one grid run of the linear model alone (--wide only) over the prior variances of
              --prs_grid; each disease keeps its best variance and validation pass, and that
              model is written to linear/per_disease/.
  2. joint    Optuna trials of the network on top, started from stage 1's per-disease model
              with its PRS weights fixed (--init_wide_from, --wide_freeze).

With --wide on the run's model is the joint one, whatever its tuning AUROC, and the linear model
alone is recorded as an ablation ("linear_only"); --wide only runs stage 1 alone. README.md lists
what the run's folder holds; rerunning the same command continues the search.
"""
import contextlib
import copy
import fcntl
import gc
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
DEFAULT_PRS_GRID = "0.003,0.01,0.03,0.1,0.3,1,3,10,30"   # prior variances of the linear stage: methods.md section 8
LINEAR_LR = 1e-2       # the linear stage's covariate weights and intercepts (methods.md section 8)


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
    if stage == "linear":                                # methods.md sections 6 and 8
        cfg.wide, cfg.annot = "only", False
        cfg.prs_grid = base.prs_grid if getattr(base, "prs_grid", None) else DEFAULT_PRS_GRID
        cfg.batch_size, cfg.accumulate = max(int(base.batch_size), 256), 1
        cfg.lr, cfg.lr_schedule, cfg.warmup_steps = LINEAR_LR, "cosine", 0
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
    """A stage's study. Each worker's sampler is seeded with the run's seed plus its index (with
    one seed for all, every worker would start with the same random settings)."""
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
    pairs = ([x.strip() for x in row.split(",")] for row in rows)
    return [idx for idx, uuid in pairs if uuid not in busy]


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
    import torch                                         # torch and the training stack load only where used
    import tune_optuna
    from . import train
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
        out = train.run_training(cfg, trial=trial)
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


def claim_trial(study, lock_path, n_total):
    """The next trial of a stage, or None once ``n_total`` are finished or running. Counted and
    asked under a file lock shared by the stage's workers, so that together they start exactly
    ``n_total`` (failed trials do not count: fail_stale queues their re-runs, which go first)."""
    with open(lock_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            taken = len(study.get_trials(deepcopy=False, states=FINISHED + (TrialState.RUNNING,)))
            return study.ask() if taken < int(n_total) else None
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def worker(base, stage, n_total, worker_index=0, init=None):
    """Run trials of a stage, taking the next one whenever free (methods.md section 8), until
    the stage has ``n_total`` finished or running."""
    if int(n_total) <= 0:
        return
    study = open_stage_study(base, stage, worker_index)
    lock_path = os.path.join(run_root(base), stage, ".claim.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    while True:
        trial = claim_trial(study, lock_path, n_total)
        if trial is None:
            return
        try:
            value = objective(trial, base, stage, init)
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
        except BaseException:
            study.tell(trial, state=TrialState.FAIL)
            raise
        else:
            study.tell(trial, value)
        gc.collect()


def worker_from_env(base):
    """A worker process started by tune_run (main_multilabel.py --tune_worker)."""
    env = os.environ
    if env["WIDEDEEPGP_TUNE_STAGE"] == "linear":
        return linear_run(base)
    worker(base, env["WIDEDEEPGP_TUNE_STAGE"], int(env["WIDEDEEPGP_TUNE_N"]), int(env.get("WIDEDEEPGP_TUNE_WORKER", 0)),
           env.get("WIDEDEEPGP_WIDE_INIT") or None)


def _spawn_worker(argv, script, log_path, gpu, stage, n_total, index=0, init=None):
    """A worker process (``script`` --tune_worker) on one GPU, appending its output to ``log_path``."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WIDEDEEPGP_TUNE_STAGE=stage,
               WIDEDEEPGP_TUNE_N=str(n_total), WIDEDEEPGP_TUNE_WORKER=str(index), WIDEDEEPGP_WIDE_INIT=init or "")
    with open(log_path, "a") as log:
        return subprocess.Popen([sys.executable, script, *argv, "--tune_worker"], env=env, stdout=log,
                                stderr=subprocess.STDOUT)


def linear_dir(base):
    return os.path.join(run_root(base), "linear")


def linear_run(base):
    """Stage 1: the grid run, resumed from its last checkpoint if an earlier attempt left one."""
    from . import train
    cfg = stage_cfg(base, "linear")
    cfg.resume = train.last_checkpoint(os.path.join(linear_dir(base), "checkpoints")) is not None
    train.run_training(cfg)


def linear_stage(base, argv, script, gpus):
    """Stage 1 on the first of ``gpus`` (in a worker process) or here; nothing to do if done."""
    if not os.path.exists(os.path.join(linear_dir(base), "results.json")):
        if gpus:
            log_dir = os.path.join(run_root(base), "workers")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"linear_gpu{gpus[0]}.log")
            code = _spawn_worker(argv, script, log_path, gpus[0], "linear", 1).wait()
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
    """Trials still marked running when a search starts were left by a crash: mark them failed
    and queue each again with its settings, carrying on from its last checkpoint in its own
    folder when it wrote one (resume_from; methods.md section 7)."""
    from . import train
    stale = study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
    for t in stale:
        study.tell(t.number, state=TrialState.FAIL)
        if not t.params:
            continue
        attrs = {"resumes_trial": t.number}
        d = t.user_attrs.get("run_dir")
        try:
            if d and train.last_checkpoint(os.path.join(d, "checkpoints")):
                attrs["resume_from"] = d
        except ValueError:                               # an ambiguous checkpoint folder: start afresh
            pass
        study.enqueue_trial(dict(t.params), user_attrs=attrs)
    return len(stale)


def run_workers(base, argv, script, stage, n_total, remaining, gpus, init):
    """The trials of a stage: in this process without GPUs, else one worker per GPU (at most one
    per trial still to run), each taking trials until the stage has ``n_total``."""
    if not gpus:
        return worker(base, stage, n_total, 0, init)
    k = min(len(gpus), remaining)
    log_dir = os.path.join(run_root(base), stage, "workers")
    os.makedirs(log_dir, exist_ok=True)
    procs = [_spawn_worker(argv, script, os.path.join(log_dir, f"gpu{gpus[i]}.log"), gpus[i], stage, n_total, i, init)
             for i in range(k)]
    codes = [p.wait() for p in procs]
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


@contextlib.contextmanager
def _run_lock(root):
    """One search per run at a time."""
    path = os.path.join(root, ".search.lock")
    with open(path, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"another search of {os.path.dirname(path)} is running")
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


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
    import torch
    root = run_root(base)
    os.makedirs(root, exist_ok=True)
    with _run_lock(root):
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
            if getattr(base, "tune_first_given", False) and not study.trials:
                # the settings given on the command line (the classic DeepGP's published ones) are tried first
                study.enqueue_trial({"lr": float(base.lr), "l2": float(base.l2), "dropout": float(base.dropout)},
                                    user_attrs={"given": True})
            stale = fail_stale(study)
            if stale:
                print(f"{stage}: {stale} trial(s) left running by an earlier attempt marked failed", flush=True)
            remaining = n - len(study.get_trials(deepcopy=False, states=FINISHED))
            if remaining > 0:                            # GPUs picked per stage: the rule applies when trials start
                run_workers(base, argv, script, stage, n, remaining, pick_gpus(spec), init)
            summaries[stage] = stage_summary(base, stage)
            if "best_trial" not in summaries[stage]:
                raise RuntimeError(f"no trial of stage {stage} in {root} completed; see the trials' folders")
        summary = choose(base, summaries)
    print(json.dumps({k: summary[k] for k in ("chosen_stage", "best_trial_dir", "best_val_auroc_mean", "best_params")},
                     indent=1), flush=True)
    return summary


def choose(base, summaries):
    """The run's model is the last stage's best trial: with the linear path on, the network on
    top of the linear model whatever its tuning AUROC, with the linear model alone recorded as an
    ablation ("linear_only"; methods.md section 8). Writes the run's results.json and deletes the
    weights of every other trial but stage 1's per-disease model."""
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
