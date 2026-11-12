# -*- coding: utf-8 -*-
"""Train the multi-disease DeepGP (WideDeepGP).

    CUDA_VISIBLE_DEVICES=<gpu> python main_multilabel.py --label <dataset> --rd_mode <variant> \
        --exp_name <run> --final_pool atten --d_model 32 --e_layers 1 --batch_size 16 ...

The dataset is <data_dir>/<label>/<rd_mode>/ in the genotype-store layout; when its store.json
names "labels", every named label is predicted (or the subset given with --labels). SNPs are
standardised with statistics of the training split unless --snp_stats says otherwise. The
single-disease script, main_genome.py, is unchanged. See methods.md for what the model does and
widedeepgp/train.py for what a run writes.
"""
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


def parse_args(argv=None):
    """The parser of args_generator.py with the multi-disease defaults: datasets under
    pukb/genes, attention pooling, seed 1, and SNP statistics of the training split unless
    --snp_stats is given (the parser's own defaults, kept for main_genome.py, are seed 33 and
    all store samples)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    # a --snp_stats or --seed in argv comes later and wins
    cfg = args_initial(["--snp_stats", _UNSET, "--seed", str(SEED)] + argv)
    if cfg.snp_stats == _UNSET:
        cfg.snp_stats = "train"
    if cfg.data_dir is None:
        cfg.data_dir = "pukb/genes"
    if cfg.final_pool == "concat":            # the parser's default, which the model does not implement
        cfg.final_pool = "atten"
    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    if cfg.exp_name is None:
        cfg.exp_name = time.strftime("run_%Y%m%d_%H%M%S")
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
