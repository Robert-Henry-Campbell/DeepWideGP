"""Held-out groups through the store data module: a split the dataset lacks comes from the sibling dataset
HELD_OUT, in that dataset's own store, prepared with the training statistics of the dataset that reads it.

Run:  CUDA_VISIBLE_DEVICES= python -m pytest tests/test_genostore_heldout.py -q
"""
import hashlib
import os
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from ml_helpers import make_cfg, synthetic_dataset


def held_out_copy(data_dir, split="heldx", prefix="H", ids_from_table=True):
    """A sibling dataset whose group is the tuning people again under new IDs, in a copy of the store (same
    genotypes, other sample names): what the held-out path reads must then equal the tuning set exactly."""
    src = os.path.join(data_dir, "multi", "store")
    dst = os.path.join(data_dir, "multi", "hm3plus_asian_black")
    shutil.copytree(os.path.join(src, "genostore"), os.path.join(dst, "genostore"))
    path = os.path.join(dst, "genostore", "samples.txt")
    with open(path) as f:
        names = [line.rstrip("\n") for line in f if line.strip()]
    with open(path, "w") as f:
        f.write("".join(f"{prefix}{s}\n" for s in names))
    shutil.copy(os.path.join(src, "store.json"), dst)
    val = pd.read_csv(os.path.join(src, "samples_val.tsv"), sep="\t", dtype={"IID": str})
    if ids_from_table:
        val = val.assign(IID=prefix + val["IID"])
    val.to_csv(os.path.join(dst, f"samples_{split}.tsv"), sep="\t", index=False)
    return dst, val["IID"].to_numpy(str)


def decoded(dm, loader):
    out = []
    for batch in loader:
        x_all, y = dm.decode_batch(batch)[:2]
        out.append(([t.clone() for t in x_all], y.clone()))
    return out


def same(a, b):
    return len(a) == len(b) and all(len(xa) == len(xb) and all(torch.equal(p, q) for p, q in zip(xa, xb))
                                    and torch.equal(ya, yb) for (xa, ya), (xb, yb) in zip(a, b))


@pytest.mark.parametrize("snp_stats", ["train", "store"])
def test_a_held_out_group_is_prepared_exactly_like_the_tuning_set(tmp_path, snp_stats):
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path)
    d, ids = held_out_copy(data_dir)
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi", snp_stats=snp_stats))
    dm.setup("fit")
    # genotypes standardised and covariates scaled with the reading dataset's training statistics, labels as given
    assert same(decoded(dm, dm.val_dataloader()), decoded(dm, dm.eval_dataloder("heldx")))
    assert np.array_equal(dm.iids("heldx"), ids)
    val_ids = pd.read_csv(os.path.join(data_dir, "multi", "store", "samples_val.tsv"), sep="\t", dtype={"IID": str})["IID"]
    assert np.array_equal(dm.iids("val"), val_ids.to_numpy(str))
    log = open(os.path.join(d, "access.log")).read().splitlines()
    assert len(log) == 2 and all("\tsplit=heldx\t" in line and dm.dir in line for line in log)   # eval_dataloder, iids


def test_the_held_out_store_is_read_not_the_datasets_own(tmp_path):
    """Same names, other genotypes in the copy: the held-out batches follow the copy."""
    from genostore.datamodule import SNPStoreDataModule
    from genostore.store import GenoStore
    data_dir, _ = synthetic_dataset(tmp_path)
    d, _ = held_out_copy(data_dir)
    main = GenoStore(os.path.join(data_dir, "multi", "store", "genostore"))
    copy_dir = os.path.join(d, "genostore")
    ref = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    before = decoded(ref, ref.eval_dataloder("heldx"))
    import json
    for c in main.chroms:                   # a one-block chromosome without overrides: its rows are per person
        info = json.load(open(os.path.join(copy_dir, f"chr{c}.json")))
        if info["n_blocks"] == 1 and info["ovr_bytes"] == 0:
            break
    base = os.path.join(copy_dir, f"chr{c}.base.bin")
    rows = np.fromfile(base, dtype=np.uint8).reshape(main.n_samples, -1)
    rows[::-1].tofile(base)                 # the people's genotype rows in reverse order
    assert not np.array_equal(rows, rows[::-1])
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    after = decoded(dm, dm.eval_dataloder("heldx"))
    assert not same(before, after)


def test_a_held_out_store_with_other_variants_is_refused(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path)
    d, _ = held_out_copy(data_dir)
    vpath = os.path.join(d, "genostore", "chr1.variants.tsv")
    lines = open(vpath).read().splitlines(keepends=True)
    i = next(k for k, line in enumerate(lines) if not line.startswith("#"))
    p = lines[i].rstrip("\n").split("\t")
    p[3], p[4] = p[4], p[3]                 # the alleles of one variant swapped
    lines[i] = "\t".join(p) + "\n"
    open(vpath, "w").write("".join(lines))
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    with pytest.raises(ValueError, match="other variants"):
        dm.eval_dataloder("heldx")
    assert not os.path.exists(os.path.join(d, "access.log"))


def test_the_datasets_own_table_wins_and_is_not_logged(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path)
    d, _ = held_out_copy(data_dir)
    own = os.path.join(data_dir, "multi", "store")
    shutil.copy(os.path.join(own, "samples_val.tsv"), os.path.join(own, "samples_heldx.tsv"))
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    dm.setup("fit")
    assert same(decoded(dm, dm.val_dataloader()), decoded(dm, dm.eval_dataloder("heldx")))
    assert "store_dir" not in dm._read_table("heldx")
    assert not os.path.exists(os.path.join(d, "access.log"))


def test_missing_tables_and_unknown_people_are_refused(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path)
    held_out_copy(data_dir, ids_from_table=False)       # the table names people the copied store calls H...
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    with pytest.raises(FileNotFoundError, match="neither in"):
        dm.eval_dataloder("nowhere")
    with pytest.raises(KeyError, match=r"IDs are not in the store"):
        dm.eval_dataloder("heldx")


def test_a_scorer_can_point_at_another_held_out_dataset(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path)
    d, ids = held_out_copy(data_dir)
    elsewhere = str(tmp_path / "elsewhere")
    shutil.move(d, elsewhere)
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    with pytest.raises(FileNotFoundError):
        dm.eval_dataloder("heldx")
    dm.held_out_dir = elsewhere
    assert np.array_equal(dm.iids("heldx"), ids)


def test_held_out_people_get_the_pseudo_fold(tmp_path):
    """A fold-routed first layer: a group that --folds does not list gets sha256("<IID>\\t<seed>") mod k."""
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path)
    _, ids = held_out_copy(data_dir)
    tr = pd.read_csv(os.path.join(data_dir, "multi", "store", "samples_train.tsv"), sep="\t", dtype={"IID": str})["IID"]
    folds = str(tmp_path / "folds.npz")
    np.savez(folds, k=np.array(5), seed=np.array(1), iid_train=tr.to_numpy(str),
             fold_train=(np.arange(len(tr)) % 5).astype(np.int8))
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi", folds=folds))
    want = [int.from_bytes(hashlib.sha256(f"{s}\t1".encode()).digest()[:8], "big") % 5 for s in ids]
    assert dm._read_table("heldx")["fold"].tolist() == want
    x_all = dm.decode_batch(next(iter(dm.eval_dataloder("heldx"))))[0]
    assert x_all[-1].tolist() == want[:len(x_all[-1])]            # the fold travels last, as for the tuning set
