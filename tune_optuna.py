# -*- coding: utf-8 -*-
"""Stand-alone hyper-parameter search for the multi-disease model with Optuna (the search inside
every run, widedeepgp/tuning.py, uses suggest, open_study and load_study from here).

Start one worker per GPU; all workers of a study share one journal file and so one search:

    CUDA_VISIBLE_DEVICES=2 python tune_optuna.py --study tune1 --n_trials 20 -- \
        --label <dataset> --rd_mode <variant> --final_pool atten --d_model 32 --batch_size 16 ...
    CUDA_VISIBLE_DEVICES=3 python tune_optuna.py --study tune1 --n_trials 20 -- <same arguments>

Everything after "--" is passed to the training code as for main_multilabel.py and fixes what
is not searched. Each trial is an ordinary run, runs/<study>/trial_NNNN/. Searched: learning
rate, weight decay, embedding dropout, accumulation and warm-up. The value maximised is the
trial's best mean validation AUROC; successive halving stops trailing trials (after
--min_validations passes, keeping one in three at each rung).
"""
import argparse
import json
import os
import sys
import time

import optuna
import torch
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState

from main_multilabel import parse_args        # which also lets torch.load read Lightning checkpoints
from widedeepgp.train import run_training


def suggest(trial, cfg, space):
    """Set the searched settings on cfg; space "deep" (a run's network stage) searches the first three only."""
    cfg.lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    cfg.l2 = trial.suggest_float("l2", 1e-6, 1e-2, log=True)
    cfg.dropout = trial.suggest_float("dropout", 0.0, 0.4)
    if space != "deep":
        cfg.accumulate = trial.suggest_categorical("accumulate", [1, 2, 4])
        cfg.warmup_steps = trial.suggest_categorical("warmup_steps", [0, 200, 1000])


def open_study(name, run_dir, min_validations=2, reduction=3, seed=None, n_startup_trials=10, prune=True):
    os.makedirs(os.path.join(run_dir, name), exist_ok=True)
    storage = JournalStorage(JournalFileBackend(os.path.join(run_dir, name, "optuna_journal.log")))
    pruner = (optuna.pruners.SuccessiveHalvingPruner(min_resource=min_validations, reduction_factor=reduction)
              if prune else optuna.pruners.NopPruner())
    return optuna.create_study(
        study_name=name, storage=storage, direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True, constant_liar=True, seed=seed,
                                           n_startup_trials=n_startup_trials),
        pruner=pruner)


def load_study(name, run_dir):
    """An existing study, read only (opening it with open_study appends a record to its journal)."""
    path = os.path.join(run_dir, name, "optuna_journal.log")
    if not os.path.exists(path):                  # the journal backend would create an empty one
        raise FileNotFoundError(f"no study journal at {path}")
    return optuna.load_study(study_name=name, storage=JournalStorage(JournalFileBackend(path)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    cut = argv.index("--") if "--" in argv else len(argv)
    own, base = argv[:cut], argv[cut + 1:]              # this worker's arguments, the training arguments
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", required=True)
    ap.add_argument("--n_trials", type=int, default=10, help="trials this worker runs")
    ap.add_argument("--run_dir", default="runs")
    ap.add_argument("--min_validations", type=int, default=2)
    a = ap.parse_args(own)
    study = open_study(a.study, a.run_dir, a.min_validations, seed=1)      # the project's seed

    def objective(trial):
        cfg = parse_args(base)
        cfg.run_dir, cfg.experiment, cfg.exp_name = a.run_dir, a.study, f"trial_{trial.number:04d}"
        suggest(trial, cfg, "optim")
        try:
            out = run_training(cfg, trial=trial)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise optuna.TrialPruned("out of GPU memory")
        for key in ("run_dir", "epoch_equivalents", "train_seconds"):
            trial.set_user_attr(key, out[key])
        return out["best_val_auroc_mean"]

    study.optimize(objective, n_trials=a.n_trials, gc_after_trial=True)
    states = [t.state for t in study.trials]
    summary = {"study": a.study, "trials": len(states), "complete": states.count(TrialState.COMPLETE),
               "pruned": states.count(TrialState.PRUNED), "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    if TrialState.COMPLETE in states:
        best = study.best_trial
        summary.update(best_value=best.value, best_params=best.params, best_trial=best.number,
                       best_run_dir=best.user_attrs.get("run_dir"))
    with open(os.path.join(a.run_dir, a.study, "best.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
