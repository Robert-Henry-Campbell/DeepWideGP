"""Train the multi-disease DeepGP (WideDeepGP).

    python main_multilabel.py --data_dir <data_dir> --label <dataset> --rd_mode <variant> \
        --exp_name <run> --final_pool atten --d_model 32 --e_layers 1 --batch_size 16 ...

A run tunes itself on the free GPUs minus one (widedeepgp/tuning.py; methods.md section 8). By
default (--input_mode blocks, --wide off) its model is one network trained end to end from dosages:
a per-SNP table pooled by LD block (--partition), then a mixer over the block tokens. The linear
PRS path (--wide on, or only) is kept as an option and is never the reported model.
--tune_trials 0 trains the given settings once, on the GPU this process sees. The dataset is
<data_dir>/<label>/<rd_mode>/ in the genotype-store layout; README.md says what a run writes.
"""
import os
import sys
import time

import torch

# torch >= 2.6 loads only tensors by default; Lightning checkpoints also hold Python objects (as main_genome.py)
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

from args_generator import args_initial
from widedeepgp.train import run_training

# this script's defaults; the shared parser keeps main_genome.py's
DEFAULTS = {"--snp_stats": "train", "--data_dir": "pukb/genes", "--seed": "1",    # the project's single seed
            "--input_mode": "blocks", "--lr_schedule": "cosine", "--warmup_steps": "200",
            "--tune_trials": "12", "--tune_gpus": "auto",
            "--val_min_participants": "20000", "--val_max_epoch_equiv": "1", "--min_participants": "128000",
            "--snapshots_per_decade": "6", "--snapshots_max": "40"}


def parse_args(argv=None):
    """The shared parser with DEFAULTS given first, so that the command line wins."""
    argv = sys.argv[1:] if argv is None else list(argv)
    cfg = args_initial([s for pair in DEFAULTS.items() for s in pair] + argv)
    if cfg.final_pool == "concat":            # the parser's default, which the model does not implement
        cfg.final_pool = "mean" if cfg.input_mode == "blocks" else "atten"
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
