"""Label matrices and the multi-disease data contract.  python -m pytest tests/test_ml_labels.py -q"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from ml_helpers import make_cfg, random_labels, synthetic_dataset

from widedeepgp import labels as L


def test_check_labels():
    y = random_labels(500)
    L.check_labels(y)
    y_nan = y.copy()
    y_nan[:50, 1] = np.nan
    L.check_labels(y_nan)                                   # "not applicable" is allowed
    bad = y.copy()
    bad[3, 2] = 2
    with pytest.raises(ValueError, match="0, 1 or empty"):
        L.check_labels(bad, ["a", "b", "c", "d"])
    none = y.copy()
    none[:, 3] = 0
    with pytest.raises(ValueError, match="cannot be learned"):
        L.check_labels(none)
    with pytest.raises(ValueError, match="2-D"):
        L.check_labels(y[:, 0])


def test_prevalence_and_weights():
    y = random_labels(6000, missing=1)
    pi = L.prevalence(y)
    ok = ~np.isnan(y[:, 1])
    assert np.isclose(pi[1], y[ok, 1].mean()) and np.isclose(pi[0], y[:, 0].mean())
    assert np.allclose(L.class_weights(pi, gamma=0), 1.0)
    assert np.allclose(L.class_weights(pi, gamma=1, w_max=1e9), (1 - pi) / pi)
    assert np.allclose(L.class_weights(pi, gamma=0.5, w_max=1e9), np.sqrt((1 - pi) / pi))
    assert L.class_weights(np.array([0.001]), gamma=1, w_max=20)[0] == 20


def test_sampling_rate():
    y = random_labels(8000, missing=2)
    om = L.sampling_rate(y)
    assert np.isclose(om.mean(), 1.0)                       # by construction
    rc, rn = L.rotation_rates(y)
    k = int(np.flatnonzero((np.nan_to_num(y) == 0).all(1) & ~np.isnan(y).any(1))[0])   # a person with no disease
    assert np.isclose(om[k], rn.mean())
    k = int(np.flatnonzero(np.isnan(y[:, 2]))[0])           # label 2 does not apply: it adds nothing
    yk = np.nan_to_num(y[k])
    expect = (yk * rc + (1 - yk) * rn)
    expect[2] = 0
    assert np.isclose(om[k], expect.mean())
    assert om.max() > 1.5 > om.min()                        # multimorbid people are drawn more often


def test_named_columns_only(tmp_path):
    """Only the columns store.json names are used; an unnamed disease column never becomes an input."""
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path)
    d = os.path.join(data_dir, "multi", "store")
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    assert dm.multilabel and dm.n_labels == 6 and dm.n_cov == 8
    assert dm.label_names[0] == "shared_a" and dm.cov_names == ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5", "pc6"]
    tables = {s: pd.read_csv(os.path.join(d, f"samples_{s}.tsv"), sep="\t") for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):                      # the validation file is used as given
        assert dm.splits[s]["y"].shape == (len(tables[s]), 6) and dm.splits[s]["cov"].shape == (len(tables[s]), 8)
        assert np.array_equal(dm.splits[s]["y"], tables[s][dm.label_names].to_numpy(dtype=np.float32))
    cov = tables["train"][dm.cov_names].to_numpy()
    assert np.allclose(dm.splits["val"]["cov"],
                       (tables["val"][dm.cov_names].to_numpy() - cov.mean(0)) / cov.std(0))

    # a subset of labels: the others are ignored, not turned into covariates
    sub = SNPStoreDataModule(make_cfg(data_dir, "multi", labels="solo_e,shared_a"))
    assert sub.label_names == ["solo_e", "shared_a"] and sub.n_cov == 8
    assert np.array_equal(sub.splits["test"]["y"], tables["test"][["solo_e", "shared_a"]].to_numpy(dtype=np.float32))
    assert np.allclose(sub.splits["train"]["cov"], dm.splits["train"]["cov"])
    with pytest.raises(KeyError, match="not labels of this dataset"):
        SNPStoreDataModule(make_cfg(data_dir, "multi", labels="age"))

    # an extra, unnamed column in the tables changes nothing
    for s, t in tables.items():
        t.assign(leak=t["shared_a"]).to_csv(os.path.join(d, f"samples_{s}.tsv"), sep="\t", index=False)
    again = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    assert again.n_cov == 8 and np.allclose(again.splits["train"]["cov"], dm.splits["train"]["cov"])


def test_bad_tables_are_rejected(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=66)
    d = os.path.join(data_dir, "multi", "store")
    path = os.path.join(d, "samples_train.tsv")
    good = pd.read_csv(path, sep="\t")

    t = good.copy()
    t.loc[5, "solo_d"] = 2
    t.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="0, 1 or empty"):
        SNPStoreDataModule(make_cfg(data_dir, "multi"))

    t = good.copy()
    t["null_f"] = t["null_f"].astype(float)
    t.loc[:20, "null_f"] = np.nan                           # empty label = not applicable: accepted
    t.to_csv(path, sep="\t", index=False)
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    assert np.isnan(dm.splits["train"]["y"][:21, 5]).all() and not np.isnan(dm.splits["train"]["y"][21:, 5]).any()

    t = good.copy()
    t.loc[2, "pc3"] = np.nan
    t.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError, match="missing values"):
        SNPStoreDataModule(make_cfg(data_dir, "multi"))

    good.drop(columns=["pc6"]).to_csv(path, sep="\t", index=False)
    with pytest.raises(KeyError, match="lacks the columns"):
        SNPStoreDataModule(make_cfg(data_dir, "multi"))
    good.to_csv(path, sep="\t", index=False)

    spec_path = os.path.join(d, "store.json")
    spec = json.load(open(spec_path))
    json.dump(dict(spec, covariates=spec["covariates"] + ["shared_a"]), open(spec_path, "w"))
    with pytest.raises(ValueError, match="both label and covariate"):
        SNPStoreDataModule(make_cfg(data_dir, "multi"))


def test_batches_carry_their_target(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    from widedeepgp.samplers import RotationBatchSampler
    data_dir, _ = synthetic_dataset(tmp_path)
    dm = SNPStoreDataModule(make_cfg(data_dir, "multi"))
    y_train = dm.splits["train"]["y"]
    dm.train_sampler = RotationBatchSampler(y_train, batch_size=16, steps_per_epoch=12, seed=1)
    dm.setup("fit")
    seen = []
    for batch in dm.train_dataloader():
        x_all, y, target = dm.on_after_batch_transfer(batch, 0)
        t = int(target)
        assert len(x_all) == 23 and x_all[0].shape[0] == 16 and x_all[-1].shape == (16, 8) and y.shape == (16, 6)
        assert int(y[:, t].sum()) == 8                       # half the batch are cases of the target
        seen.append(t)
    assert sorted(seen[:6]) == list(range(6)) and len(seen) == 12
    x_all, y = dm.on_after_batch_transfer(next(iter(dm.val_dataloader())), 0)      # evaluation batches: no target
    assert y.shape[1] == 6


def test_from_pickles_label_matrix(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    from genostore.from_pickles import convert
    rng = np.random.default_rng(0)
    src = tmp_path / "genes" / "leg" / "ld_all"
    os.makedirs(src)
    n = {"train": 60, "test": 20}
    sizes = [3] * 22
    dump = lambda name, obj: pickle.dump(obj, open(src / f"{name}.pkl", "wb"))
    dump("genes", [np.array([f"rs{c}_{j}" for j in range(m)]) for c, m in enumerate(sizes)])
    dump("pos", [np.arange(m) for m in sizes])
    for s in n:
        dump(f"snp_{s}", [rng.integers(0, 3, size=(n[s], m)).astype(float) for m in sizes])
        lab = rng.integers(0, 2, size=(n[s], 3))
        dump(f"label_{s}", [lab for _ in sizes])
        dump(f"covar_{s}", rng.standard_normal((n[s], 4)))
    json.dump(["t2d", "chd", "af"], open(src / "label_names.json", "w"))
    assert convert(str(src), str(tmp_path / "genes" / "leg" / "store")) == 0.0
    dm = SNPStoreDataModule(make_cfg(str(tmp_path / "genes"), "leg"))
    assert dm.label_names == ["t2d", "chd", "af"] and dm.cov_names == ["cov0", "cov1", "cov2", "cov3"]
    assert dm.splits["test"]["y"].shape == (20, 3) and len(dm.splits["train"]["y"]) + len(dm.splits["val"]["y"]) == 60
    assert isinstance(dm.splits["train"]["y"], np.ndarray) and torch.is_tensor(torch.from_numpy(dm.splits["train"]["y"]))
