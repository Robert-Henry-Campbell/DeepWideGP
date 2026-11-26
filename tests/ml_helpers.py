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


DATA = "/mnt/sdg/robert/deepwidegp/data"
PRIVE = f"{DATA}/reference/partitions/prive431"
HAVE_PRIVE = os.path.exists(f"{PRIVE}/partition.npz") and os.path.exists(f"{DATA}/datasets/ard30/hm3plus/store.json")


def prive_block_lengths():
    """The SNP counts of the 431 Privé blocks of the HapMap3+ store, chromosome by chromosome (no participant is read)."""
    import json
    from genostore.datamodule import default_regions, load_partition
    from genostore.store import GenoStore
    from widedeepgp.train import block_lengths
    with open(f"{DATA}/datasets/ard30/hm3plus/store.json") as f:
        store = GenoStore(json.load(f)["store"])
    return block_lengths(load_partition(PRIVE, store, default_regions(store)))


def ld_top_rung_model(data_dir, mixer, **over):
    """The LD-block classic model (methods.md 4.13) at the top rung's shape: one sequence per Privé block of the
    1,258,200 HapMap3+ SNPs, 30 outputs, 12 covariates, the published width 32 and one layer. Its settings come from the
    synthetic dataset at ``data_dir``; of the project's data only the store's variant list and the partition are read."""
    from widedeepgp.losses import label_counts
    from widedeepgp.model import WideDeepGP, positions_digest
    from widedeepgp.train import build_data, model_hparams
    cfg = make_cfg(data_dir, "multi", input_mode="snps", partition="uniform:2", mixer=mixer, d_model=32, n_heads=4,
                   e_layers=1, input_mask=0.0, **over)
    hp = model_hparams(cfg, build_data(cfg))
    pos = [np.arange(n) for c in prive_block_lengths() for n in c]
    y = np.zeros((10, 30), dtype=np.float32)
    y[:2] = 1
    hp.update(seq_len=[len(p) for p in pos], pos_digest=positions_digest(pos), n_cov=12,
              label_names=[f"d{i}" for i in range(30)], label_counts=label_counts(y))
    return WideDeepGP(hp, pos)


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
