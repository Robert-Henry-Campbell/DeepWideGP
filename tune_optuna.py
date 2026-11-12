# -*- coding: utf-8 -*-
"""Hyper-parameter search for the multi-disease model with Optuna.

Start one worker per GPU; all workers of a study share one journal file and so one search:

    CUDA_VISIBLE_DEVICES=2 python tune_optuna.py --study tune1 --n_trials 20 -- \
        --label <dataset> --rd_mode <variant> --final_pool atten --d_model 32 --batch_size 16 ...
    CUDA_VISIBLE_DEVICES=3 python tune_optuna.py --study tune1 --n_trials 20 -- <same arguments>

Everything after "--" is passed to the training code as for main_multilabel.py and fixes what
is not searched. Each trial is an ordinary run under runs/<study>/trial_NNNN/ (TensorBoard,
metrics, snapshots), trained by widedeepgp.train.run_training, the same function that trains
the final model.

Searched (--space optim): learning rate, weight decay, embedding dropout, accumulation, warm-up.
With --space full, also the class-weight exponent and cap, the sampler and the head's width.
Model width, depth, the sequence block and annotations are experimental axes: they are set on
the command line, never searched.

The value maximised is the best mean validation AUROC of the trial. After each validation pass
the value is reported, and successive halving stops the trials that trail (``--min_validations``
passes before a trial may be stopped, keeping one in ``--reduction`` at each rung).

To look at a study without writing to its journal, open it with ``load_study``.
"""
import argparse
import json
import os
import sys
import time

import torch

_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from main_multilabel import parse_args
from widedeepgp.train import run_training


def suggest(trial, cfg, space):
    cfg.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    cfg.l2 = trial.suggest_float("l2", 1e-6, 1e-2, log=True)
    cfg.dropout = trial.suggest_float("dropout", 0.0, 0.4)
    cfg.accumulate = trial.suggest_categorical("accumulate", [1, 2, 4])
    cfg.warmup_steps = trial.suggest_categorical("warmup_steps", [0, 200, 1000])
    if space == "full":
        cfg.gamma = trial.suggest_categorical("gamma", [0.0, 0.5, 1.0])
        cfg.w_max = trial.suggest_categorical("w_max", [5.0, 20.0, 100.0])
        cfg.sampler = trial.suggest_categorical("sampler", ["rotation", "shuffled"])
        cfg.head_hidden = trial.suggest_categorical("head_hidden", [128, 256, 512])


def open_study(name, run_dir, min_validations=2, reduction=3, seed=None):
    os.makedirs(os.path.join(run_dir, name), exist_ok=True)
    storage = JournalStorage(JournalFileBackend(os.path.join(run_dir, name, "optuna_journal.log")))
    return optuna.create_study(
        study_name=name, storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=min_validations, reduction_factor=reduction))


def load_study(name, run_dir):
    """An existing study, opened read-only to inspect it. open_study is for workers: opening a
    study that exists with create_study still appends a record to the journal."""
    path = os.path.join(run_dir, name, "optuna_journal.log")
    if not os.path.exists(path):                  # the journal backend would create an empty one
        raise FileNotFoundError(f"no study journal at {path}")
    return optuna.load_study(study_name=name, storage=JournalStorage(JournalFileBackend(path)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    base = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--n_trials", type=int, default=10, help="trials this worker runs")
    ap.add_argument("--space", choices=["optim", "full"], default="optim")
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--min_validations", type=int, default=2)
    ap.add_argument("--reduction", type=int, default=3)
    ap.add_argument("--sampler_seed", type=int, default=None)
    a = ap.parse_args(own)
    study = open_study(a.study, a.run_dir, a.min_validations, a.reduction, a.sampler_seed)

    def objective(trial):
        cfg = parse_args(base)                       # the defaults of main_multilabel.py
        cfg.run_dir, cfg.experiment, cfg.exp_name = a.run_dir, a.study, f"trial_{trial.number:04d}"
        suggest(trial, cfg, a.space)
        try:
            out = run_training(cfg, trial=trial)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("out of GPU memory")
        trial.set_user_attr("run_dir", out["run_dir"])
        trial.set_user_attr("epoch_equivalents", out["epoch_equivalents"])
        trial.set_user_attr("train_seconds", out["train_seconds"])
        return out["best_val_auroc_mean"]

    study.optimize(objective, n_trials=a.n_trials, gc_after_trial=True)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    summary = {"study": a.study, "trials": len(study.trials), "complete": len(done), "pruned": len(pruned),
               "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    if done:
        summary.update(best_value=study.best_value, best_params=study.best_params, best_trial=study.best_trial.number,
                       best_run_dir=study.best_trial.user_attrs.get("run_dir"))
    with open(os.path.join(a.run_dir, a.study, "best.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
