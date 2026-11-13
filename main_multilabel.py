# -*- coding: utf-8 -*-
"""Train the multi-disease DeepGP (WideDeepGP).

    python main_multilabel.py --data_dir <data_dir> --label <dataset> --rd_mode <variant> \
        --exp_name <run> --final_pool atten --d_model 32 --e_layers 1 --batch_size 16 ...

A run tunes itself (widedeepgp/tuning.py), on the free GPUs minus one: first the linear model
alone (PRS and covariates) at every prior variance of --prs_grid in one pass, each disease
keeping its best, then the deep network on top of that PRS (--tune_trials settings, 12 by
default), keeping whichever is better on the tuning set. --tune_trials 0 trains the given
settings once, on the GPU this process sees. The linear PRS path is on by default (--wide on;
'off' is the network alone, 'only' the PRS and covariates).

The dataset is <data_dir>/<label>/<rd_mode>/ in the genotype-store layout; when its store.json
names "labels", every named label is predicted (or the subset given with --labels). SNPs are
standardised with statistics of the training split unless --snp_stats says otherwise. The
single-disease script, main_genome.py, is unchanged. See methods.md for what the model does and
widedeepgp/train.py for what a run writes.
"""
import os
import sys
import time

import torch

# PyTorch >= 2.6 loads checkpoints with weights_only=True by default; Lightning checkpoints hold
# plain Python objects as well (same workaround as main_genome.py).
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

from args_generator import args_initial
from widedeepgp.train import run_training

_UNSET = "<not given>"
SEED = 1                                      # the project's single seed (the pilot, its control and tuning used it)
# what main_multilabel.py sets unless the command line says otherwise (the shared parser's own
# defaults are kept for main_genome.py)
DEFAULTS = {"--seed": str(SEED), "--wide": "on", "--wide_kappa": "0.1", "--tune_trials": "12",
            "--tune_gpus": "auto", "--val_min_participants": "20000", "--val_max_epoch_equiv": "1",
            "--min_participants": "128000", "--snapshots_per_decade": "6", "--snapshots_max": "40"}


def parse_args(argv=None):
    """The parser of args_generator.py with the multi-disease defaults: datasets under
    pukb/genes, attention pooling, SNP statistics of the training split, and DEFAULTS (seed 1,
    the linear PRS path with its weights settling in about 10 epoch-equivalents, tuning in every
    run, validation and budget floors for small training sets, at most 40 snapshots). The parser's own defaults are kept for main_genome.py."""
    argv = sys.argv[1:] if argv is None else list(argv)
    lead = ["--snp_stats", _UNSET]
    for flag, value in DEFAULTS.items():
        lead += [flag, value]
    cfg = args_initial(lead + argv)                      # what argv gives comes later and wins
    if cfg.snp_stats == _UNSET:
        cfg.snp_stats = "train"
    if cfg.data_dir is None:
        cfg.data_dir = "pukb/genes"
    if cfg.final_pool == "concat":            # the parser's default, which the model does not implement
        cfg.final_pool = "atten"
    return cfg


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    cfg = parse_args(argv)
    if cfg.exp_name is None:
        cfg.exp_name = time.strftime("run_%Y%m%d_%H%M%S")
        argv = argv + ["--exp_name", cfg.exp_name]      # so that a search's workers share the run's folder
    if cfg.tune_worker:
        from widedeepgp import tuning
        return tuning.worker_from_env(cfg)
    if cfg.tune_trials > 0:
        from widedeepgp import tuning
        return tuning.tune_run(cfg, argv, os.path.abspath(__file__))
    out = run_training(cfg)
    print(f"\nrun directory: {out['run_dir']}")
    print(f"best mean validation AUROC {out['best_val_auroc_mean']:.4f} at step {out['best_step']} "
          f"({out['best_samples_seen']} participants seen); trained {out['epoch_equivalents']:.2f} "
          f"epoch-equivalents in {out['train_seconds'] / 60:.1f} min"
          f"{' (stopped early)' if out['stopped_early'] else ''}")
    print(f"{'label':<14}{'AUROC':>8}{'AP':>8}{'BCE':>8}")
    for name, m in out["val"].items():
        print(f"{name:<14}{m['auroc']:>8.4f}{m['ap']:>8.4f}{m['bce']:>8.4f}")
    return out


if __name__ == "__main__":
    main()
