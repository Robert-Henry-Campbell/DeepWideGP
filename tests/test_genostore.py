"""Tests for the disk-backed genotype store (numpy-only part).

Run:  python -m pytest tests/test_genostore.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genostore import format as gf
from genostore.store import GenoStore


def make_q(rng, n, m, p_exc=0.04, p_missing=0.0):
    """q matrix shaped like imputed data: mostly 0/100/200 plus a few fractional dosages."""
    q = rng.choice(np.array([0, 100, 200], dtype=np.uint8), size=(n, m), p=[0.55, 0.25, 0.20])
    exc = rng.random((n, m)) < p_exc
    q[exc] = rng.integers(1, 200, size=int(exc.sum()), dtype=np.uint8)
    if p_missing:
        q[rng.random((n, m)) < p_missing] = gf.MISSING
    return q


def test_quantize_dosage():
    d = np.array([0.0, 1.0, 2.0, 0.004, 0.006, 0.37, 1.996, -9.0, np.nan], dtype=np.float32)
    q = gf.quantize_dosage(d)
    assert q.tolist() == [0, 100, 200, 0, 1, 37, 200, 255, 255]
    assert gf.quantize_dosage(np.array([[0.0, 0.5, 2.0]], dtype=np.float32)).tolist() == [[0, 50, 200]]
    assert gf.quantize_dosage(np.array([2.3, 0.2], dtype=np.float64)).tolist() == [200, 20]   # clipped
    assert gf.quantize_dosage(np.empty((0, 3), np.float32)).shape == (0, 3)


def test_threaded_encoder_matches_serial():
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.default_rng(7)
    q = make_q(rng, 1000, 4099, p_exc=0.06, p_missing=0.01)
    a = gf.encode_block(q, slab=64)
    with ThreadPoolExecutor(4) as pool:
        b = gf.encode_block(q, slab=37, pool=pool)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert np.array_equal(gf.decode_packed(b[0], b[1], q.shape[1]), q)


@pytest.mark.parametrize("n,width", [(1, 1), (3, 5), (7, 64), (5, 63), (4, 16384), (2, 16385 % 16384 + 3)])
def test_block_codec_roundtrip(n, width):
    rng = np.random.default_rng(n * 1000 + width)
    q = make_q(rng, n, width, p_exc=0.1, p_missing=0.02)
    packed, ovr, counts = gf.encode_block(q, slab=2)
    assert packed.shape == (n, (width + 3) // 4)
    assert counts.sum() == len(ovr) == int((gf.CODE_OF[q] == 3).sum())
    assert np.array_equal(gf.decode_packed(packed, ovr, width), q)


def test_codec_extremes():
    rng = np.random.default_rng(0)
    for q in (np.zeros((3, 10), np.uint8),                       # no exceptions at all
              rng.integers(1, 100, size=(3, 10), dtype=np.uint8),  # every entry an exception
              np.full((2, 9), gf.MISSING, np.uint8)):              # all missing
        packed, ovr, _ = gf.encode_block(q)
        assert np.array_equal(gf.decode_packed(packed, ovr, q.shape[1]), q)
    packed, ovr, _ = gf.encode_block(np.zeros((2, 8), np.uint8))
    with pytest.raises(ValueError):
        gf.decode_packed(packed, np.array([5], np.uint8), 8)       # stray exception byte is detected


@pytest.mark.parametrize("block", [8, 64, 16384])
def test_store_roundtrip_and_regions(tmp_path, block):
    rng = np.random.default_rng(block)
    n = 37
    sizes = {"1": 5, "2": block, "3": block * 3 + 7, "10": block * 2, "X": 1}
    store = str(tmp_path / "s")
    gf.create_store(store, [f"id{i}" for i in range(n)], block_variants=block)
    truth = {}
    for c, m in sizes.items():
        truth[c] = make_q(rng, n, m, p_exc=0.0 if c == "10" else 0.05, p_missing=0.01 if c == "3" else 0.0)
        info = gf.write_chrom_from_array(store, c, truth[c], block_variants=block)
        assert info["complete"] and info["n_variants"] == m
    gs = GenoStore(store)
    assert gs.chroms == ["1", "2", "3", "10", "X"]             # natural order
    assert gs.n_samples == n and list(gs.samples[:2]) == ["id0", "id1"]
    for c, m in sizes.items():
        assert np.array_equal(gs.read_q(c, np.arange(n)), truth[c])
        for _ in range(25):                                      # random rows (unsorted, repeated) x random ranges
            rows = rng.integers(0, n, size=int(rng.integers(1, 9)))
            v0 = int(rng.integers(0, m)); v1 = int(rng.integers(v0 + 1, m + 1))
            assert np.array_equal(gs.read_q(c, rows, v0, v1), truth[c][rows][:, v0:v1])
        # statistics written at conversion time match a direct computation
        mean, std, cnt = gs.stats(c)
        t = truth[c].astype(np.float64); ok = truth[c] != gf.MISSING
        t[~ok] = np.nan
        with np.errstate(all="ignore"):
            exp_mean, exp_std = np.nanmean(t, axis=0) / 100, np.nanstd(t, axis=0) / 100
        has = ok.sum(axis=0) > 0
        assert np.allclose(mean[has], exp_mean[has], atol=1e-9) and np.allclose(std[has], exp_std[has], atol=1e-7)
        assert np.array_equal(cnt, ok.sum(axis=0))
    d = gs.read_dosage("3", np.arange(n))
    assert np.isnan(d).sum() == int((truth["3"] == gf.MISSING).sum())
    with pytest.raises(IndexError):
        gs.read_raw("1", [0], 0, sizes["1"] + 1)


def test_store_refuses_different_samples(tmp_path):
    store = str(tmp_path / "s")
    gf.create_store(store, ["a", "b", "c"])
    gf.create_store(store, ["a", "b", "c"])                       # same samples: fine
    with pytest.raises(ValueError):
        gf.create_store(store, ["a", "b", "d"])


def test_writer_resume_discards_partial_block(tmp_path):
    rng = np.random.default_rng(5)
    n, block, m = 11, 16, 16 * 4 + 5
    q = make_q(rng, n, m, p_exc=0.2)
    store = str(tmp_path / "s")
    gf.create_store(store, list(range(n)), block_variants=block)
    w = gf.ChromWriter(store, "1", n, m, block, drop_cache=False)
    for _ in range(2):
        v0, v1 = w.block_range(w.next_block)
        w.append_block(q[:, v0:v1])
    for fh in w.f.values():                                       # simulate a crash mid-block: junk appended
        fh.write(b"\x07" * 13); fh.flush()
    del w
    w = gf.ChromWriter(store, "1", n, m, block, drop_cache=False)  # resume
    assert w.next_block == 2
    while not w.complete:
        v0, v1 = w.block_range(w.next_block)
        w.append_block(q[:, v0:v1])
    w.finalize()
    assert np.array_equal(GenoStore(store).read_q("1", np.arange(n)), q)
