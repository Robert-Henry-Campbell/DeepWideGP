"""Inputs of the classic DeepGP ladder runs (main_genome.py on store datasets):
- store.json "snps": the loader keeps exactly the listed SNPs of the full rows, turned by their signs, with
  statistics from the whole chromosome (shared through "stats_dir"), and refuses malformed lists;
- a sign of -1 gives the standardised dosage of the other allele (what recoding to 2 - dosage gives);
- balanced_one_disease: half cases in every batch, every non-case exactly once per epoch, cases recycled.

Run:  python -m pytest tests/test_classic_ladder_inputs.py -q"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from genostore import format as gf
from genostore.datamodule import SNPStoreDataModule
from widedeepgp.samplers import balanced_one_disease

SIZES = {"1": 40, "2": 70, "3": 33}
N = 60
SPLITS = (("train", slice(0, 40)), ("val", slice(40, 50)), ("test", slice(50, 60)))


def make_configs(label, rd_mode, data_dir, **over):
    old = sys.argv
    sys.argv = ["test", "--label", label, "--rd_mode", rd_mode, "--dm", "snps_covs", "--snp_embed", "cov",
                "--batch_size", "8", "--num_workers", "0", "--snp_stats", "train"]
    try:
        from args_generator import args_initial
        cfg = args_initial()
    finally:
        sys.argv = old
    cfg.data_dir = data_dir
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def build_store(tmp_path, name="store", turn=None):
    """N people x SIZES SNPs (hard calls, imputed values, missing); turn: {chrom: bool per SNP} stored as 2 - dosage."""
    rng = np.random.default_rng(5)
    store = str(tmp_path / name)
    gf.create_store(store, [f"s{i}" for i in range(N)], block_variants=16)
    for c, m in SIZES.items():
        q = rng.choice(np.array([0, 100, 200], dtype=np.uint8), size=(N, m), p=[0.4, 0.4, 0.2])
        exc = rng.random((N, m)) < 0.1
        q[exc] = rng.integers(1, 200, size=int(exc.sum()), dtype=np.uint8)
        q[rng.random((N, m)) < 0.02] = gf.MISSING
        if turn is not None:
            sub = q[:, turn[c]]
            ok = sub != gf.MISSING
            sub[ok] = 200 - sub[ok]
            q[:, turn[c]] = sub
        gf.write_chrom_from_array(store, c, q, block_variants=16, pos=np.arange(m) * 1000 + 17)
    return store


def build_dataset(tmp_path, store, name, snps=None, **spec):
    d = tmp_path / "data" / "D" / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"IID": [f"s{i}" for i in range(N)], "D": (rng.random(N) < 0.4).astype(int),
                       "c1": rng.normal(size=N), "c2": rng.normal(size=N)})
    for split, rows in SPLITS:
        df.iloc[rows].to_csv(d / f"samples_{split}.tsv", sep="\t", index=False)
    js = {"store": store, "labels": ["D"], "covariates": ["c1", "c2"], **spec}
    if snps is not None:
        np.savez(d / "snps.npz", **snps)
        js["snps"] = "snps.npz"
    (d / "store.json").write_text(json.dumps(js))
    return str(tmp_path / "data")


def snp_list(keep, sign=None):
    out = {f"idx_{c}": np.asarray(i, dtype=np.int64) for c, i in keep.items()}
    if sign is not None:
        out.update({f"sign_{c}": np.asarray(s, dtype=np.float32) for c, s in sign.items()})
    return out


def decode(dm, split, idx=None):
    ds = {"train": dm.snp_train, "val": dm.snp_val}[split]
    idx = np.arange(len(ds)) if idx is None else idx
    return dm.decode_batch(ds[idx])


def test_list_keeps_and_turns_the_listed_snps(tmp_path):
    rng = np.random.default_rng(1)
    keep = {c: np.sort(rng.choice(m, size=m // 3, replace=False)) for c, m in SIZES.items()}
    sign = {c: np.where(rng.random(len(keep[c])) < 0.5, -1.0, 1.0) for c in SIZES}
    store = build_store(tmp_path)
    data = build_dataset(tmp_path, store, "full")
    build_dataset(tmp_path, store, "list", snps=snp_list(keep, sign), pos_mode="pos")
    full = SNPStoreDataModule(make_configs("D", "full", data))
    sub = SNPStoreDataModule(make_configs("D", "list", data))
    # what main_genome.py reads: lengths, IDs, base-pair positions of the kept SNPs
    assert sub.seq_len == [len(keep[c]) for c in SIZES] and len(sub.genes) == sum(sub.seq_len)
    assert [list(g) for g in sub.genes_chr] == [[f"chr{c}_snp{j}" for j in keep[c]] for c in SIZES]
    assert all(np.array_equal(p, keep[c] * 1000 + 17) for p, c in zip(sub.pos_chr, SIZES))
    full.setup("fit"); sub.setup("fit")
    for split in ("train", "val"):
        xf, yf = decode(full, split)
        xs, ys = decode(sub, split)
        assert torch.equal(yf, ys) and torch.equal(xf[-1], xs[-1])          # same people, labels, covariates
        for k, c in enumerate(SIZES):
            want = xf[k][:, keep[c]] * torch.from_numpy(sign[c].astype(np.float32))[:, None]
            assert xs[k].shape == (len(ys), len(keep[c]), 1) and torch.equal(xs[k], want)


def test_index_positions_without_pos_mode(tmp_path):
    store = build_store(tmp_path)
    data = build_dataset(tmp_path, store, "list", snps=snp_list({c: [0, 3, 9] for c in SIZES}))
    sub = SNPStoreDataModule(make_configs("D", "list", data))
    assert all(np.array_equal(p, np.arange(3)) for p in sub.pos_chr)


def test_sign_gives_the_other_alleles_dosage(tmp_path):
    rng = np.random.default_rng(3)
    turn = {c: rng.random(m) < 0.5 for c, m in SIZES.items()}
    data = build_dataset(tmp_path, build_store(tmp_path), "signed",
                         snps=snp_list({c: np.arange(m) for c, m in SIZES.items()},
                                       {c: np.where(turn[c], -1.0, 1.0) for c in SIZES}))
    build_dataset(tmp_path, build_store(tmp_path, "recoded", turn), "recoded")
    a = SNPStoreDataModule(make_configs("D", "signed", data))
    b = SNPStoreDataModule(make_configs("D", "recoded", data))
    a.setup("fit"); b.setup("fit")
    for split in ("train", "val"):
        xa, _ = decode(a, split)
        xb, _ = decode(b, split)
        for k in range(len(SIZES)):
            assert torch.allclose(xa[k], xb[k], atol=1e-5)


def test_statistics_come_from_the_whole_chromosome_and_are_shared(tmp_path):
    store = build_store(tmp_path)
    shared = tmp_path / "stats"
    shared.mkdir()
    data = build_dataset(tmp_path, store, "full")
    build_dataset(tmp_path, store, "l1", snps=snp_list({"1": [0, 5], "2": [1], "3": [2, 3]}), stats_dir=str(shared))
    build_dataset(tmp_path, store, "l2", snps=snp_list({"1": [7], "2": [0, 9], "3": [4]}), stats_dir=str(shared))
    full = SNPStoreDataModule(make_configs("D", "full", data))
    l1 = SNPStoreDataModule(make_configs("D", "l1", data))
    cached = sorted(os.listdir(shared))
    assert len(cached) == 1 and cached[0].startswith("snp_stats_train_")
    l2 = SNPStoreDataModule(make_configs("D", "l2", data))
    assert sorted(os.listdir(shared)) == cached                             # reused, not written again
    assert not [f for f in os.listdir(os.path.join(data, "D", "l2")) if f.startswith("snp_stats")]
    for dm, keep in ((l1, {"1": [0, 5], "2": [1], "3": [2, 3]}), (l2, {"1": [7], "2": [0, 9], "3": [4]})):
        for k, c in enumerate(SIZES):
            assert np.array_equal(dm.snp_mean[k], full.snp_mean[k][keep[c]])
            assert np.array_equal(dm.snp_inv_std[k], full.snp_inv_std[k][keep[c]])


@pytest.mark.parametrize("bad, match", [
    ({"idx_1": [3, 1]}, "sorted, distinct"),
    ({"idx_1": [1, 1]}, "sorted, distinct"),
    ({"idx_1": [0, 40]}, "sorted, distinct"),
    ({"idx_1": []}, "sorted, distinct"),
    ({"sign_1": [1.0, 0.5]}, r"\+1 or -1"),
    ({"sign_1": [1.0]}, r"\+1 or -1"),
])
def test_malformed_lists_are_refused(tmp_path, bad, match):
    snps = {**snp_list({c: [0, 2] for c in SIZES}), **{k: np.asarray(v) for k, v in bad.items()}}
    data = build_dataset(tmp_path, build_store(tmp_path), "bad", snps=snps)
    with pytest.raises(ValueError, match=match):
        SNPStoreDataModule(make_configs("D", "bad", data))


def test_lists_need_every_chromosome_whole_and_standardised_dosages(tmp_path):
    store = build_store(tmp_path)
    data = build_dataset(tmp_path, store, "nochr3", snps=snp_list({"1": [0], "2": [1]}))
    with pytest.raises(KeyError, match="idx_3"):
        SNPStoreDataModule(make_configs("D", "nochr3", data))
    build_dataset(tmp_path, store, "regions", snps=snp_list({c: [0] for c in SIZES}),
                  regions=[[["1", 0, 10]], [["2", 0, 70]], [["3", 0, 33]]])
    with pytest.raises(ValueError, match="whole chromosome"):
        SNPStoreDataModule(make_configs("D", "regions", data))
    build_dataset(tmp_path, store, "vocab", snps=snp_list({c: [0] for c in SIZES}, {c: [-1.0] for c in SIZES}))
    with pytest.raises(ValueError, match="standardised"):
        SNPStoreDataModule(make_configs("D", "vocab", data, snp_embed="vocab"))


@pytest.mark.parametrize("n_case, n_ctrl", [(10, 96), (7, 100), (150, 64)])
def test_balanced_one_disease_epochs(n_case, n_ctrl):
    y = np.r_[np.ones(n_case), np.zeros(n_ctrl)][np.random.default_rng(0).permutation(n_case + n_ctrl)]
    s = balanced_one_disease(y, 64, seed=1)
    assert len(s) == -(-n_ctrl // 32)
    ctrl = np.flatnonzero(y == 0)
    seen_ctrl = []
    for epoch in range(3):
        s.set_epoch(epoch)
        batches = list(s)
        assert len(batches) == len(s)
        for b in batches:
            assert len(b) == 64 and int(y[b].sum()) == 32                # half cases in every batch
        seen = np.concatenate([b[y[b] == 0] for b in batches])
        seen_ctrl.append(seen)
    allc = np.concatenate(seen_ctrl)
    # non-cases come round in passes: every one once per pass, in order of passes across epochs
    passes = [allc[i:i + n_ctrl] for i in range(0, len(allc) - n_ctrl + 1, n_ctrl)]
    assert passes and all(np.array_equal(np.sort(p), ctrl) for p in passes)
    s.set_epoch(1)
    again = list(s)
    s.set_epoch(1)
    assert all(np.array_equal(a, b) for a, b in zip(again, list(s)))      # a pure function of (seed, epoch)


def test_balanced_one_disease_refuses_more_than_one_label():
    with pytest.raises(ValueError, match="one label"):
        balanced_one_disease(np.zeros((10, 2)), 64)
