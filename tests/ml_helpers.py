"""Shared helpers for the multi-disease tests (tests/test_ml_*.py)."""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if os.path.join(REPO, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "scripts"))


def have_fork():
    """True where the Vision-Mamba fork of mamba_ssm is installed (the published block)."""
    try:
        import inspect

        import mamba_ssm
        return "bimamba_type" in inspect.signature(mamba_ssm.Mamba.__init__).parameters
    except Exception:
        return False


def have_mamba3():
    """True where upstream mamba_ssm with Mamba3 is installed (the second environment)."""
    try:
        from mamba_ssm import Mamba3  # noqa: F401
        return True
    except Exception:
        return False


def make_cfg(data_dir, label, rd_mode="store", **over):
    """An argument namespace as main_multilabel.py would build it, with small test defaults."""
    from args_generator import args_initial
    cfg = args_initial(["--label", label, "--rd_mode", rd_mode, "--dm", "snps_covs", "--snp_embed", "cov",
                        "--final_pool", "atten", "--batch_size", "16", "--d_model", "16", "--n_heads", "4",
                        "--num_workers", "0"])
    cfg.data_dir = data_dir
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def synthetic_dataset(tmp_path, name="multi", n=1200, snps=132, seed=3):
    """A small six-trait dataset in the store layout; returns (data_dir, truth)."""
    from generate_synthetic_multilabel import generate
    data_dir = str(tmp_path / "genes")
    _, truth = generate(name, n_participants=n, n_snps=snps, base_dir=data_dir, seed=seed)
    return data_dir, truth


def random_labels(n=4000, prevalences=(0.3, 0.1, 0.05, 0.02), shared=0.5, seed=0, missing=None):
    """A label matrix with co-occurring labels; ``missing`` = column index made not applicable for half."""
    from scipy.stats import norm
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((n, 1))
    liab = np.sqrt(shared) * c + np.sqrt(1 - shared) * rng.standard_normal((n, len(prevalences)))
    y = (liab > norm.isf(np.asarray(prevalences))).astype(np.float32)
    if missing is not None:
        y[rng.random(n) < 0.5, missing] = np.nan
    return y
