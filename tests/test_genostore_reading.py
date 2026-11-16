"""The reading side of genostore, checked against known answers on synthetic data (CPU):
the raw regions the store hands out, the torch decoder, the store-backed data module (refusals,
scaling, the cache of training statistics, batch order) and the polite prefetcher.

    python -m pytest tests/test_genostore_reading.py -q
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler

from ml_helpers import make_cfg

from genostore import format as gf
from genostore import prefetch as P
from genostore import torch_decode as td
from genostore.store import GenoStore, RawRegion

W = 8                                         # variants per block in these stores
MB = 1024 * 1024
REQ = 1280 * 1024


def make_q(rng, n, m, p_exc, p_missing):
    """Mostly whole dosages (0/100/200), some fractional ones and some missing."""
    q = rng.choice(np.array([0, 100, 200], dtype=np.uint8), size=(n, m), p=[0.55, 0.25, 0.20])
    exc = rng.random((n, m)) < p_exc
    q[exc] = rng.integers(1, 200, size=int(exc.sum()), dtype=np.uint8)
    q[rng.random((n, m)) < p_missing] = gf.MISSING
    return q


def exception_bytes(q, rows, blocks, m):
    """What the store must return for these rows and blocks: each exception's byte, block by
    block, row by row, variant by variant."""
    out = [q[r, b * W:min((b + 1) * W, m)] for b in blocks for r in rows]
    out = [v[(v != 0) & (v != 100) & (v != 200)] for v in out]
    return np.concatenate(out).astype(np.uint8) if out else np.empty(0, np.uint8)


@pytest.fixture
def store(tmp_path):
    """Chromosomes of awkward sizes: the narrow last block only, one full block, full blocks and a
    last block, and one with no exception byte at all."""
    rng = np.random.default_rng(1)
    n, path = 23, str(tmp_path / "s")
    gf.create_store(path, [f"p{i}" for i in range(n)], block_variants=W)
    truth = {}
    for c, m, p_exc in (("1", 5, 0.1), ("2", W, 0.1), ("3", 3 * W + 7, 0.1), ("4", 20, 0.0)):
        truth[c] = make_q(rng, n, m, p_exc, 0.02 if p_exc else 0.0)
        gf.write_chrom_from_array(path, c, truth[c], block_variants=W)
    return GenoStore(path), truth


# ---- the store ---------------------------------------------------------------------------------
def test_raw_regions_hold_the_blocks_and_exception_bytes_of_the_rows_read(store):
    gs, truth = store
    for c, q in truth.items():
        m = q.shape[1]
        n_full, tail_width = m // W, m % W
        for rows, v0, v1 in (([4, 0, 4, 19], 0, m), ([7], m - 1, m), ([2, 3], 1, max(2, m // 2)), ([9], 0, 1)):
            r = gs.read_raw(c, rows, v0, v1)
            b0, b1 = v0 // W, (v1 - 1) // W
            full_blocks = list(range(b0, min(b1 + 1, n_full)))
            assert isinstance(r, RawRegion)
            assert (r.skip, r.n, r.tail_width) == (v0 - b0 * W, v1 - v0, tail_width)
            if full_blocks:
                assert r.full.dtype == np.uint8 and r.full.shape == (len(full_blocks), len(rows), W // 4)
                assert r.full.flags["C_CONTIGUOUS"]
            else:
                assert r.full is None
            assert r.full_ovr.dtype == np.uint8
            assert np.array_equal(r.full_ovr, exception_bytes(q, rows, full_blocks, m))
            if b1 >= n_full:                                        # reaches the narrow last block
                assert r.tail.shape == (len(rows), (tail_width + 3) // 4) and r.tail.flags["C_CONTIGUOUS"]
                assert np.array_equal(r.tail_ovr, exception_bytes(q, rows, [n_full], m))
            else:
                assert r.tail is None and len(r.tail_ovr) == 0
            assert r.tail_ovr.dtype == np.uint8
            assert np.array_equal(gs.decode(r), q[rows, v0:v1])
            assert np.array_equal(gs.read_q(c, rows, v0, v1), q[rows, v0:v1])


def test_a_batch_of_no_rows_reads_nothing(store):
    gs, truth = store
    r = gs.read_raw("3", np.array([], dtype=np.int64))
    assert r.full.shape == (3, 0, W // 4) and r.tail.shape == (0, 2)
    assert r.full_ovr.dtype == r.tail_ovr.dtype == np.uint8 and len(r.full_ovr) == len(r.tail_ovr) == 0
    assert gs.read_q("3", []).shape == (0, 31)


def test_read_dosage_is_q_over_100_with_nan_where_missing(store):
    gs, truth = store
    for c, q in truth.items():
        d = gs.read_dosage(c, np.arange(q.shape[0]))
        miss = q == gf.MISSING
        assert d.dtype == np.float32 and np.array_equal(np.isnan(d), miss)
        assert np.array_equal(d[~miss], q[~miss].astype(np.float32) / np.float32(100))
    assert not np.isnan(gs.read_dosage("4", [0, 1])).any()                   # nothing missing there


def test_bad_variant_ranges_are_refused(store):
    gs, _ = store
    for v0, v1 in ((0, 32), (-1, 3), (4, 4), (31, 32)):
        with pytest.raises(IndexError, match=rf"chr3: bad variant range \[{v0}, {v1}\) for 31 variants"):
            gs.read_raw("3", [0], v0, v1)


def test_store_size_on_disk(store):
    gs, _ = store
    per = [gs.nbytes(c) for c in gs.chroms]
    assert per == [gs.info[c]["base_bytes"] + gs.info[c]["ovr_bytes"] for c in gs.chroms]
    assert gs.nbytes() == sum(per) and gs.nbytes(3) == gs.nbytes("3")
    assert gs.info["4"]["ovr_bytes"] == 0                                    # no exception file to map


def test_memory_maps_are_opened_once_per_process(store):
    gs, truth = store
    gs.read_q("3", [0])
    maps = gs._maps["3"]
    gs.read_q("3", [1], 4, 9)
    assert gs._maps["3"] is maps
    gs._pid = -1                                       # what a forked DataLoader worker sees
    assert np.array_equal(gs.read_q("3", [2]), truth["3"][[2]])
    assert gs._maps["3"] is not maps and gs._pid == os.getpid()


# ---- the torch decoder ---------------------------------------------------------------------------
def test_unpack_codes_matches_the_numpy_codec():
    packed = np.random.default_rng(2).integers(0, 256, size=(3, 5, 7), dtype=np.uint8)
    codes = td.unpack_codes(torch.from_numpy(packed))
    assert codes.dtype == torch.uint8
    assert np.array_equal(codes.numpy(), ((packed[..., None] >> gf.SHIFTS) & 3).reshape(3, 5, 28))


@pytest.mark.parametrize("w", [1, 4, 13, 28])
def test_decode_packed_restores_q_at_any_width(w):
    q = make_q(np.random.default_rng(w), 6, w, 0.2, 0.05)
    packed, ovr, _ = gf.encode_block(q)
    tp, to = torch.from_numpy(packed), torch.from_numpy(ovr)
    assert np.array_equal(td.decode_packed(tp, to, w).numpy(), q)
    every = td.decode_packed(tp, to)                                         # every 2-bit slot of the bytes
    pad = 4 * packed.shape[1] - w
    assert every.shape == (6, w + pad) and np.array_equal(every[:, :w].numpy(), q) and not every[:, w:].any()
    assert torch.equal(td.decode_packed(tp, to, w + pad + 5), every)
    with pytest.raises(ValueError, match=f"exception count mismatch: {len(ovr)} code-3 slots but {len(ovr) + 1} bytes"):
        td.decode_packed(tp, torch.cat([to, torch.tensor([9], dtype=torch.uint8)]), w)


def test_decode_segment_joins_blocks_and_drops_the_skipped_variants(store):
    gs, truth = store
    t = lambda a: torch.empty(0, dtype=torch.uint8) if a is None else torch.from_numpy(np.ascontiguousarray(a))
    for c, q in truth.items():
        m = q.shape[1]
        for v0, v1 in ((0, m), (1, m), (0, 1), (m - 1, m), (3, max(4, m - 2))):
            r = gs.read_raw(c, [5, 1, 1], v0, v1)
            out = td.decode_segment(t(r.full), t(r.full_ovr), t(r.tail), t(r.tail_ovr), W, m % W, v0 % W, v1 - v0)
            assert out.dtype == torch.uint8 and np.array_equal(out.numpy(), q[[5, 1, 1], v0:v1])


def test_model_input_is_the_standardised_dosage_with_missing_at_zero():
    q = torch.tensor([[0, 100, 200, 37], [255, 255, 50, 0]], dtype=torch.uint8)
    x = td.to_model_input(q)
    assert x.shape == (2, 4, 1) and x.dtype == torch.float32
    assert torch.equal(x[..., 0], torch.tensor([[0.0, 1.0, 2.0, 0.37], [0.0, 0.0, 0.5, 0.0]], dtype=torch.float32))
    mean, inv = torch.tensor([1.0, 0.5, 1.5, 0.2]), torch.tensor([2.0, 0.0, 1.0, 4.0])
    expect = (q.to(torch.float32) * 0.01 - mean) * inv
    expect[q == 255] = 0.0
    assert torch.equal(td.to_model_input(q, mean, inv)[..., 0], expect)
    whole = q[:1]                                                            # nothing missing
    assert torch.equal(td.to_model_input(whole, mean, inv)[..., 0], (whole.to(torch.float32) * 0.01 - mean) * inv)
    assert td.MISSING == gf.MISSING == 255


# ---- the data module -----------------------------------------------------------------------------
N, SIZES = 60, {"1": 5, "2": W, "3": 31}


@pytest.fixture
def data(tmp_path):
    """One shared store; dataset "one" with a single unnamed label (every other column a
    covariate), dataset "many" with named labels and covariates, a tuning table and annotations."""
    rng = np.random.default_rng(7)
    ids = [f"id{i:03d}" for i in range(N)]
    genes = tmp_path / "genes"
    gf.create_store(str(genes / "shared"), ids, block_variants=W)
    q = {c: make_q(rng, N, m, 0.1, 0.03) for c, m in SIZES.items()}
    for c, m in SIZES.items():
        gf.write_chrom_from_array(str(genes / "shared"), c, q[c], block_variants=W, pos=np.arange(m) * 10 + 1)
    perm = rng.permutation(N)
    rows = {"train": np.sort(perm[:36]), "val": np.sort(perm[36:46]), "test": np.sort(perm[46:]),
            "ext": np.sort(perm[10:22])}
    cov = np.round(rng.normal(size=(N, 2)), 3)
    lab = (rng.random((N, 2)) < 0.4).astype(int)

    def table(d, split, rows_, cols):
        t = pd.DataFrame({"IID": [ids[i] for i in rows_]})
        for name, values in cols.items():
            t[name] = values[rows_]
        t.to_csv(d / f"samples_{split}.tsv", sep="\t", index=False)

    one = genes / "one" / "store"
    one.mkdir(parents=True)
    for s in ("train", "test", "ext"):
        r = np.sort(np.concatenate([rows["train"], rows["val"]])) if s == "train" else rows[s]
        table(one, s, r, {"label": lab[:, 0], "c1": cov[:, 0], "c2": cov[:, 1]})
    json.dump({"store": "../../shared"}, open(one / "store.json", "w"))
    many = genes / "many" / "store"
    many.mkdir(parents=True)
    for s in ("train", "val", "test", "ext"):
        table(many, s, rows[s], {"a": lab[:, 0], "b": lab[:, 1], "c1": cov[:, 0], "c2": cov[:, 1],
                                 "junk": rng.normal(size=N)})
    np.savez(many / "annotations.npz", **{f"chr{c}": rng.normal(size=(m, 3)) for c, m in SIZES.items()})
    json.dump({"store": "../../shared", "labels": ["a", "b"], "covariates": ["c1", "c2"],
               "annotations": {"file": "annotations.npz"}}, open(many / "store.json", "w"))
    return {"dir": str(genes), "q": q, "rows": rows, "cov": cov, "lab": lab, "one": one, "many": many}


def module(data, name, **over):
    from genostore.datamodule import SNPStoreDataModule
    return SNPStoreDataModule(make_cfg(data["dir"], name, **over))


def set_label(path, i, value):
    t = pd.read_csv(path, sep="\t", dtype={"IID": str})
    t.loc[i, "label"] = value
    t.to_csv(path, sep="\t", index=False)


def test_a_single_label_must_be_coded_0_1_in_every_table(data, capsys):
    one = data["one"]

    def refused(split, make):
        values = np.unique(pd.read_csv(one / f"samples_{split}.tsv", sep="\t")["label"]).tolist()
        with pytest.raises(ValueError) as e:
            make()
        assert str(e.value) == f"{split} labels must be encoded as 0/1, found {values}"

    set_label(one / "samples_ext.tsv", 3, 7)
    dm = module(data, "one", ve="ext")                                       # the external table is read later
    refused("ext", lambda: dm.eval_dataloder("ext"))
    dm.setup("test")
    refused("ext", dm.test_dataloader)
    assert capsys.readouterr().out == "Validating with external cohort: ext\n"
    set_label(one / "samples_test.tsv", 2, 2)
    refused("test", lambda: module(data, "one"))
    set_label(one / "samples_train.tsv", 0, -1)                              # both bad: train is reported
    refused("train", lambda: module(data, "one"))
    cont = module(data, "one", cont_pheno=True)                              # a continuous phenotype: scaled, not checked
    y = cont.splits["train"]["y"]
    assert abs(float(y.mean())) < 1e-9 and abs(float(y.std()) - 1) < 1e-9


def test_without_scaling_genotypes_and_covariates_are_passed_as_they_are(data):
    dm = module(data, "many", scaling=False)
    assert dm.snp_mean is None and dm.snp_inv_std is None and getattr(dm, "cov_cm", None) is None
    tr = data["rows"]["train"]
    assert np.array_equal(dm.splits["train"]["cov"], data["cov"][tr])
    dm.setup("fit")
    x_all, y = dm.decode_batch(dm._dataset(dm.splits["train"])[[0, 3, 3]])
    for k, c in enumerate(SIZES):
        q = torch.from_numpy(data["q"][c][tr[[0, 3, 3]]])
        assert torch.equal(x_all[k][..., 0], (q.to(torch.float32) * 0.01).masked_fill(q == 255, 0.0))
    assert torch.equal(x_all[-1], torch.from_numpy(data["cov"][tr[[0, 3, 3]]].astype(np.float32)))
    assert torch.equal(y, torch.from_numpy(data["lab"][tr[[0, 3, 3]]].astype(np.float32)))


def test_training_statistics_are_computed_once_and_cached_per_rows_and_sequences(data):
    dm = module(data, "many", snp_stats="train")
    tr = data["rows"]["train"]
    h = hashlib.sha256(tr.astype(np.int64).tobytes())
    h.update(json.dumps([[[c, 0, m]] for c, m in SIZES.items()]).encode())
    name = f"snp_stats_train_{h.hexdigest()[:16]}.npz"
    assert [f for f in os.listdir(data["many"]) if f.startswith("snp_stats")] == [name]
    z = np.load(data["many"] / name)
    assert list(z.keys()) == ["mean0", "mean1", "mean2", "std0", "std1", "std2"]
    for k, c in enumerate(SIZES):
        d = data["q"][c][tr].astype(np.float64)
        d[d == 255] = np.nan
        d *= 0.01
        assert np.allclose(z[f"mean{k}"], np.nanmean(d, 0), atol=1e-12)
        assert np.allclose(z[f"std{k}"], np.nanstd(d, 0), atol=1e-12)
        assert dm.snp_mean[k].dtype == np.float32 and np.array_equal(dm.snp_mean[k], z[f"mean{k}"].astype(np.float32))
    np.savez(data["many"] / name, **{f"mean{k}": np.full(m, 0.5) for k, m in enumerate(SIZES.values())},
             **{f"std{k}": np.full(m, 2.0) for k, m in enumerate(SIZES.values())})
    again = module(data, "many", snp_stats="train")                          # read from the cache, not recomputed
    assert all(np.all(m == 0.5) for m in again.snp_mean) and all(np.all(s == 0.5) for s in again.snp_inv_std)
    mean, std = dm.compute_snp_stats(data["rows"]["val"], chunk=3, device="cpu")
    for k, c in enumerate(SIZES):
        d = data["q"][c][data["rows"]["val"]].astype(np.float64)
        d[d == 255] = np.nan
        assert np.allclose(mean[k], np.nanmean(d * 0.01, 0), atol=1e-12) and np.allclose(std[k], np.nanstd(d * 0.01, 0), atol=1e-12)


def test_training_batches_follow_torch_random_sampler_and_hold_the_standardised_store_rows(data):
    dm = module(data, "many")                                                # statistics stored with the genotypes
    gs = GenoStore(dm.store_dir)
    for k, c in enumerate(SIZES):
        mean, std, _ = gs.stats(c)
        assert np.array_equal(dm.snp_mean[k], mean.astype(np.float32))
        assert np.array_equal(dm.snp_inv_std[k], np.where(std > 0, 1.0 / np.where(std > 0, std, 1.0), 0.0).astype(np.float32))
    tr = data["rows"]["train"]
    cov_tr = data["cov"][tr]

    class Index(Dataset):
        def __len__(self):
            return len(tr)

        def __getitem__(self, b):
            return np.asarray(b)

    torch.manual_seed(5)
    order = [b.numpy() for b in DataLoader(Index(), sampler=BatchSampler(RandomSampler(Index()), 16, False), batch_size=None)]
    dm.setup("fit")
    torch.manual_seed(5)
    batches = list(dm.train_dataloader())
    assert [len(b["y"]) for b in batches] == [16, 16, 4] and len(order) == 3
    for idx, batch in zip(order, batches):
        x_all, y = dm.on_after_batch_transfer(batch, 0)
        for k, c in enumerate(SIZES):
            q = torch.from_numpy(gs.read_q(c, tr[idx]))
            expect = q.to(torch.float32) * 0.01
            expect = (expect - torch.from_numpy(dm.snp_mean[k])) * torch.from_numpy(dm.snp_inv_std[k])
            assert torch.equal(x_all[k][..., 0], expect.masked_fill(q == 255, 0.0))
        cov = (cov_tr[idx] - cov_tr.mean(0)) / cov_tr.std(0)
        assert torch.equal(x_all[-1], torch.from_numpy(cov.astype(np.float32)))
        assert torch.equal(y, torch.from_numpy(data["lab"][tr[idx]].astype(np.float32)))
    dm.eval_batch_size = 7
    sizes = [len(b["y"]) for b in dm.val_dataloader()]
    assert sizes == [7, 3]                                                   # the tuning set in table order


def test_dataset_items_are_encoded_batches(data):
    dm = module(data, "many")
    ds = dm._dataset(dm.splits["val"])
    item = ds[[3, 1, 1]]
    assert list(item) == ["geno", "y", "cov"]
    assert [[len(seg) for seg in seq] for seq in item["geno"]] == [[4], [4], [4]]
    full, full_ovr, tail, tail_ovr = item["geno"][0][0]                     # chromosome 1: 5 variants, no full block
    assert full.numel() == 0 and full.dtype == torch.uint8 and tail.shape == (3, 2)
    full, full_ovr, tail, tail_ovr = item["geno"][1][0]                     # chromosome 2: one full block, no tail
    assert full.shape == (1, 3, 2) and tail.numel() == 0 and tail.dtype == torch.uint8
    assert all(t.dtype == torch.uint8 for seq in item["geno"] for seg in seq for t in seg)
    assert item["y"].dtype == torch.float32 and item["cov"].dtype == torch.float32
    with_target = ds[(1, [0, 2])]
    assert list(with_target) == ["geno", "y", "cov", "target"] and with_target["target"].item() == 1
    assert with_target["target"].dtype == torch.int64
    assert ds[np.array([[1], [2]])]["y"].shape == (2, 2)
    x_all, y = dm.decode_batch(ds[[3, 1, 1]])
    assert [x.shape for x in x_all[:3]] == [(3, 5, 1), (3, W, 1), (3, 31, 1)]
    q = dm.decode_q(ds[[3, 1, 1]]["geno"], "cpu")
    rows = data["rows"]["val"][[3, 1, 1]]
    assert all(np.array_equal(q[k].numpy(), data["q"][c][rows]) for k, c in enumerate(SIZES))


def test_regions_given_in_the_configuration_replace_those_of_store_json(data):
    regions = [[["3", 2, 20], ["1", 0, 5]], [["2", 1, 3]]]
    dm = module(data, "many", geno_regions=regions, pos_mode="pos")
    assert dm.regions == regions and dm.seq_len == [23, 2]
    gs = GenoStore(dm.store_dir)
    ids = list(gs.variants("3")["id"][2:20]) + list(gs.variants("1")["id"][0:5])
    assert list(np.asarray(dm.genes_chr[0])) == ids
    assert np.array_equal(dm.pos_chr[0], np.concatenate([np.arange(2, 20) * 10 + 1, np.arange(5) * 10 + 1]))
    annot, names = dm.annotations()
    z = np.load(data["many"] / "annotations.npz")
    assert names == ["a0", "a1", "a2"] and annot[1].dtype == np.float32
    assert np.array_equal(annot[0], np.concatenate([z["chr3"][2:20], z["chr1"][0:5]]).astype(np.float32))
    np.savez(data["many"] / "annotations.npz", **{f"chr{c}": np.zeros((m - (c == "1"), 3)) for c, m in SIZES.items()})
    with pytest.raises(ValueError, match="annotations for chr1 have 4 rows, the store has 5"):
        dm.annotations()


# ---- the prefetcher ------------------------------------------------------------------------------
class Clock:
    """Stands in for the time module: sleeping and reading advance it instead of waiting."""

    def __init__(self):
        self.now, self.slept = 1000.0, []

    def time(self):
        return self.now

    perf_counter = time

    def sleep(self, s):
        self.slept.append(s)
        self.now += s

    def strftime(self, fmt):
        return "12:00:00"


class Disk:
    """Stands in for the os module: records read-ahead requests and blocking reads (0.4 s each)."""

    def __init__(self, clock=None, fail_at=None):
        self.calls, self.clock, self.fail_at = [], clock, fail_at

    def posix_fadvise(self, fd, off, n, advice):
        if advice != os.POSIX_FADV_WILLNEED:
            return os.posix_fadvise(fd, off, n, advice)
        self.calls.append(("willneed", off, n))

    def pread(self, fd, n, off):
        self.calls.append(("pread", n, off))
        if self.clock is not None:
            self.clock.now += 0.4
        if self.fail_at is not None and sum(c[0] == "pread" for c in self.calls) >= self.fail_at:
            raise OSError(5, "simulated read error")
        return b"\0" * n

    def __getattr__(self, name):
        return getattr(os, name)


def burst_calls(pos, n):
    """The read-ahead requests and the blocking read of one burst of n bytes at pos."""
    return [("willneed", off, min(REQ, pos + n - off)) for off in range(pos, pos + n, REQ)] + [("pread", 1, pos + n - 1)]


@pytest.fixture
def fake_disk(tmp_path, monkeypatch):
    """A 7 MB sparse file, a fake clock and a fake disk; probes see the scripted byte counters."""
    path = tmp_path / "f.pgen"
    with open(path, "wb") as f:
        f.truncate(7 * MB + 5)
    clock = Clock()
    disk = Disk(clock)
    monkeypatch.setattr(P, "time", clock)
    monkeypatch.setattr(P, "os", disk)
    counters = []
    monkeypatch.setattr(P, "disk_read_bytes", lambda dev: counters.pop(0) if counters else 0)
    return str(path), clock, disk, counters


def run(pf):
    pf.start()
    pf.join(30)
    assert not pf.is_alive()
    return pf


def test_prefetch_without_a_measurable_disk_reads_in_bursts(fake_disk, monkeypatch):
    path, clock, disk, _ = fake_disk
    monkeypatch.setattr(P, "disk_of", lambda p: None)                        # no probes at all
    size = 7 * MB + 5
    pf = run(P.Prefetcher(path, burst_reqs=2, ride_cache=False))
    assert disk.calls == burst_calls(0, 2 * REQ) + burst_calls(2 * REQ, 2 * REQ) + burst_calls(4 * REQ, size - 4 * REQ)
    assert (pf.pos, pf.bytes_done, pf.bytes_from_cache, pf.t_wait_idle, pf.error) == (size, size, 0, 0.0, None)
    assert pf.t_io == pytest.approx(1.2) and clock.slept == [] and pf.last_progress == clock.now
    disk.calls.clear()                                                       # start is aligned down, end is kept
    pf = run(P.Prefetcher(path, start=3 * MB + 17, end=5 * MB + 1, ride_cache=False))
    assert disk.calls == burst_calls(2 * REQ, 5 * MB + 1 - 2 * REQ)
    assert (pf.pos, pf.bytes_done) == (5 * MB + 1, 5 * MB + 1 - 2 * REQ)
    assert pf.wait_until(10 * MB) is True


def test_prefetch_idle_policy_waits_for_the_others_to_stop(fake_disk, monkeypatch):
    path, clock, disk, counters = fake_disk
    monkeypatch.setattr(P, "disk_of", lambda p: "sdx")
    counters += [0, 10 ** 6, 10 ** 6, 10 ** 6]                               # busy (4 MB/s), then quiet
    logs = []
    pf = run(P.Prefetcher(path, ride_cache=False, log=logs.append, log_every=1.0))
    size = 7 * MB + 5
    assert clock.slept == [0.25, 5.0, 0.25] and pf.t_wait_idle == 5.0 and pf.others_bps == 0.0
    assert disk.calls == burst_calls(0, size) and pf.bytes_done == size
    assert logs == ["prefetch f.pgen: disk busy (others reading 4.0 MB/s); waiting, at 0.00/0.01 GB (0.00 GB taken from cache)",
                    "prefetch f.pgen: 0.01/0.01 GB (1.2 MB/s over the last 6s; 0.00 GB taken from cache; "
                    "waited for the disk 0.1 min in total)"]


def test_prefetch_share_policy_keeps_its_share_of_disk_time(fake_disk, monkeypatch):
    path, clock, disk, counters = fake_disk
    monkeypatch.setattr(P, "disk_of", lambda p: "sdx")
    counters += [i * 10 ** 6 for i in range(6)]                              # every probe busy
    pf = run(P.Prefetcher(path, burst_reqs=2, policy="share", duty=0.25, ride_cache=False))
    size = 7 * MB + 5
    assert disk.calls == burst_calls(0, 2 * REQ) + burst_calls(2 * REQ, 2 * REQ) + burst_calls(4 * REQ, size - 4 * REQ)
    assert clock.slept == pytest.approx([0.25, 1.2] * 3)                    # 0.4 s of I/O, then 3x that asleep
    assert pf.t_wait_idle == 0.0 and pf.bytes_done == size


def test_prefetch_rides_pages_another_job_has_cached(fake_disk, monkeypatch):
    path, clock, disk, _ = fake_disk
    monkeypatch.setattr(P, "disk_of", lambda p: None)
    asked, answers = [], [3 * MB + 100, 1000]

    class Probe:
        supported = True

        def __init__(self, p, *a):
            asked.append(("open", os.path.basename(p), a))

        def cached(self, off, n):
            asked.append(("cached", off, n))
            return answers.pop(0)

        def close(self):
            asked.append(("close",))

    monkeypatch.setattr(P, "CacheProbe", Probe)
    size = 7 * MB + 5
    pf = run(P.Prefetcher(path))
    assert asked == [("open", "f.pgen", ()), ("cached", 0, size), ("cached", 3 * MB, size - 3 * MB), ("close",)]
    assert disk.calls == burst_calls(3 * MB, size - 3 * MB)                  # whole 4 KB pages from the cache
    assert (pf.bytes_done, pf.bytes_from_cache) == (size, 3 * MB)
    answers[:] = [size]
    pf = run(P.Prefetcher(path))
    assert (pf.bytes_done, pf.bytes_from_cache) == (size, size)


def test_prefetch_follows_its_reader_and_stops(fake_disk, monkeypatch):
    path, _, disk, _ = fake_disk
    monkeypatch.setattr(P, "time", __import__("time"))                       # real waits here
    monkeypatch.setattr(P, "disk_of", lambda p: None)
    pf = P.Prefetcher(path, burst_reqs=1, max_ahead=2 * REQ, ride_cache=False)
    assert pf.wait_until(REQ) is True                                        # not started: nothing to wait for
    pf.start()
    assert pf.wait_until(2 * REQ, timeout=20) is True
    pf.set_consumed(REQ)
    assert pf.wait_until(3 * REQ, timeout=20) is True
    pf.set_consumed(0)                                                       # never goes back
    pf.stop()
    pf.join(20)
    assert not pf.is_alive() and pf.pos == 3 * REQ and pf.consumed == REQ
    assert [c for c in disk.calls if c[0] == "pread"] == [("pread", 1, (k + 1) * REQ - 1) for k in range(3)]
    stopped = P.Prefetcher(path, ride_cache=False)
    stopped.stop()
    run(stopped)
    assert stopped.bytes_done == 0 and len(disk.calls) == 6


def test_prefetch_errors_reach_the_reader(fake_disk, monkeypatch):
    path, clock, disk, _ = fake_disk
    monkeypatch.setattr(P, "disk_of", lambda p: None)
    disk.fail_at = 2
    pf = run(P.Prefetcher(path, burst_reqs=1, ride_cache=False))
    assert isinstance(pf.error, OSError) and pf.bytes_done == REQ
    with pytest.raises(OSError, match="simulated read error"):
        pf.wait_until(7 * MB)
    with pytest.raises(ValueError, match="policy must be 'idle' or 'share'"):
        P.Prefetcher(path, policy="fast")
    with pytest.raises(FileNotFoundError):
        P.Prefetcher(path + ".missing", policy="fast")                       # the file is looked at first


def test_disk_busy_compares_others_reads_with_the_threshold(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(P, "time", clock)
    monkeypatch.setattr(P, "disk_of", lambda p: "sdx")
    for counters, kw, busy in (([1000, 1000 + 10 ** 6], {}, True), ([1000, 1001], {}, False),
                               ([None, 5], {}, False), ([5, None], {}, False),
                               ([0, 100], {"window_s": 0.001, "threshold_bps": 10 ** 4}, True)):
        monkeypatch.setattr(P, "disk_read_bytes", lambda dev, c=list(counters): c.pop(0))
        assert P.disk_busy("/x/y.pgen", **kw) is busy
    assert clock.slept == [0.5] * 4 + [0.001]


def test_cache_probe_counts_cached_bytes_up_to_the_end_of_the_file(tmp_path):
    path = tmp_path / "cached.bin"
    path.write_bytes(os.urandom(3 * MB + 123))                               # just written: in the page cache
    size = 3 * MB + 123
    probe = P.CacheProbe(str(path))
    try:
        got = probe.cached(0, size)
        if probe.supported:                                                  # preadv2 with RWF_NOWAIT works here
            assert got == size and probe.cached(5, 1000) == 1000
            assert probe.cached(size - 10, 100) == 10 and probe.cached(size + 5, 10) == 0
        assert probe.cached(0, 0) == 0
    finally:
        probe.close()
    assert P.disk_read_bytes(None) is None and P.disk_read_bytes("no_such_disk") is None
    assert P.disk_of(str(tmp_path / "missing")) is None


def test_prefetch_command_line(fake_disk, monkeypatch, capsys):
    path, clock, disk, _ = fake_disk
    monkeypatch.setattr(P, "disk_of", lambda p: None)
    monkeypatch.setattr(P, "_preadv2", None)                                 # no cache riding: every byte is read
    assert P.main([path, "--burst", "2"]) == 0
    assert capsys.readouterr().out == f"12:00:00 done {path}: 0.01 GB in 1s (6.1 MB/s); io 1s, waited for idle disk 0s\n"
    with pytest.raises(SystemExit):
        P.main([path, "--policy", "fast"])
