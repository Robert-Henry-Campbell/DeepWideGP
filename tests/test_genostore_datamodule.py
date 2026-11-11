"""The store-backed data module must hand the model the same batches as the legacy pickle loader.

Run:  python -m pytest tests/test_genostore_datamodule.py -q        (needs torch + lightning)
"""
import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from genostore import format as gf
from genostore import torch_decode as td
from genostore.store import GenoStore
from genostore.from_pickles import convert as convert_pickles

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
LEGACY = os.path.join(REPO, "pukb", "genes", "1causal_10case", "ld_all")


def make_configs(label, rd_mode, data_dir, **over):
    old = sys.argv
    sys.argv = ["test", "--label", label, "--rd_mode", rd_mode, "--dm", "snps_covs", "--snp_embed", "cov",
                "--final_pool", "atten", "--batch_size", "16", "--d_model", "16", "--num_workers", "0"]
    try:
        from args_generator import args_initial
        cfg = args_initial()
    finally:
        sys.argv = old
    cfg.data_dir = data_dir
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.parametrize("device", DEVICES)
def test_torch_decode_matches_numpy(tmp_path, device):
    rng = np.random.default_rng(3)
    n, block = 29, 64
    sizes = {"1": 7, "2": 64, "3": 64 * 3 + 5, "4": 64 * 2}
    store = str(tmp_path / "s")
    gf.create_store(store, [f"s{i}" for i in range(n)], block_variants=block)
    truth = {}
    for c, m in sizes.items():
        q = rng.choice(np.array([0, 100, 200], dtype=np.uint8), size=(n, m), p=[0.5, 0.3, 0.2])
        exc = rng.random((n, m)) < 0.08
        q[exc] = rng.integers(1, 200, size=int(exc.sum()), dtype=np.uint8)
        q[rng.random((n, m)) < 0.01] = gf.MISSING
        truth[c] = q
        gf.write_chrom_from_array(store, c, q, block_variants=block)
    gs = GenoStore(store)
    t = lambda a: (torch.from_numpy(np.ascontiguousarray(a)) if a is not None else torch.empty(0, dtype=torch.uint8)).to(device)
    for c, m in sizes.items():
        for _ in range(30):
            rows = rng.integers(0, n, size=int(rng.integers(1, 7)))
            v0 = int(rng.integers(0, m)); v1 = int(rng.integers(v0 + 1, m + 1))
            r = gs.read_raw(c, rows, v0, v1)
            q = td.decode_segment(t(r.full), t(r.full_ovr), t(r.tail), t(r.tail_ovr), block, m % block, v0 % block, v1 - v0)
            assert q.dtype == torch.uint8 and np.array_equal(q.cpu().numpy(), truth[c][rows][:, v0:v1])
    # float conversion: standardised, missing -> 0
    q = torch.tensor([[0, 100, 200, 255, 37]], dtype=torch.uint8, device=device)
    mean = torch.full((5,), 0.5, device=device); inv = torch.full((5,), 2.0, device=device)
    x = td.to_model_input(q, mean, inv)
    assert x.shape == (1, 5, 1) and torch.allclose(x[0, :, 0].cpu(), torch.tensor([-1.0, 1.0, 3.0, 0.0, -0.26]), atol=1e-6)
    assert torch.allclose(td.to_model_input(q)[0, :, 0].cpu(), torch.tensor([0.0, 1.0, 2.0, 0.0, 0.37]), atol=1e-6)


@pytest.mark.skipif(not os.path.exists(os.path.join(LEGACY, "snp_train.pkl")), reason="synthetic pickles not present")
@pytest.mark.parametrize("device", DEVICES)
def test_datamodule_matches_legacy_loader(tmp_path, device):
    from utils import SNPPCACHRDataModule
    from genostore.datamodule import SNPStoreDataModule, is_store_dataset

    # the same synthetic dataset, once as pickles and once converted to a store
    data_dir = str(tmp_path / "genes")
    os.makedirs(os.path.join(data_dir, "syn"))
    os.symlink(LEGACY, os.path.join(data_dir, "syn", "ld_all"))
    worst = convert_pickles(LEGACY, os.path.join(data_dir, "syn", "store"))
    assert worst == 0.0                                          # hard calls are stored exactly
    cfg_old = make_configs("syn", "ld_all", data_dir)
    cfg_new = make_configs("syn", "store", data_dir, snp_stats="train")
    assert not is_store_dataset(cfg_old) and is_store_dataset(cfg_new)
    old, new = SNPPCACHRDataModule(cfg_old), SNPStoreDataModule(cfg_new)

    # what main_genome.py reads from the module
    assert [len(g) for g in old.genes_chr] == [len(g) for g in new.genes_chr]
    assert len(old.genes) == len(new.genes)
    assert all(np.array_equal(a, b) for a, b in zip(old.pos_chr, new.pos_chr))
    assert list(np.asarray(new.genes_chr[0])) == list(old.genes_chr[0])

    old.setup("fit"); old.setup("test"); new.setup("fit"); new.setup("test")
    pairs = [(old.snp_train, new.snp_train), (old.snp_val, new.snp_val), (old.snp_test, new.snp_test)]
    for ds_old, ds_new in pairs:
        assert len(ds_old) == len(ds_new)
        idx = np.linspace(0, len(ds_old) - 1, 64).astype(int)
        batch = {k: (v.to(device) if torch.is_tensor(v) else [[[t.to(device) for t in seg] for seg in seq] for seq in v])
                 for k, v in ds_new[idx].items()}
        x_new, y_new = new.decode_batch(batch)
        assert len(x_new) == 23 and x_new[0].dtype == torch.float32
        assert torch.equal(y_new.cpu(), ds_old.y[idx])                      # same people, same order
        assert torch.allclose(x_new[-1].cpu(), ds_old.c[idx], atol=1e-6)    # covariates
        for c in range(22):
            ref = ds_old.X[c][idx]
            ok = torch.isfinite(ref)                                        # legacy gives inf/NaN where std == 0
            assert x_new[c].shape == ref.shape
            assert torch.allclose(x_new[c].cpu()[ok], ref[ok], atol=2e-5), f"chromosome {c + 1}"
            assert torch.all(x_new[c].cpu()[~ok] == 0)

    # identical shuffling: with the same seed both loaders draw the same people into each batch
    # (the shuffle seed is drawn on the first next(), so each iterator is consumed separately)
    def first_batches(dm, k=3):
        torch.manual_seed(123)
        it = iter(dm.train_dataloader())
        return [next(it) for _ in range(k)]

    for (xo, yo), bn in zip(first_batches(old), first_batches(new)):
        xn, yn = new.on_after_batch_transfer(bn, 0)
        assert torch.equal(yo, yn) and torch.allclose(xo[-1], xn[-1], atol=1e-6)
        assert all(torch.allclose(a, b, atol=2e-5, equal_nan=True) for a, b in zip(xo[:22], xn[:22]))

    # default statistics (stored with the genotypes, all samples) are close to the train-only ones
    new_default = SNPStoreDataModule(make_configs("syn", "store", data_dir))
    for a, b in zip(new.snp_mean, new_default.snp_mean):
        assert np.abs(a - b).max() < 0.05
