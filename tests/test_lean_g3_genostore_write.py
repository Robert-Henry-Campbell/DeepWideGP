"""Known answers for the writing side of the genotype store (CPU, no pgenlib needed).

Pins what a store conversion writes (bytes, JSON keys and their order, refusals, log lines that
scripts read) and how the command line hands its work to convert_chrom, so a refactor of
genostore/format.py, convert.py or from_pickles.py cannot change them unnoticed.

Run:  python -m pytest tests/test_lean_g3_genostore_write.py -q
"""
import hashlib
import json
import os
import pickle
import re
import stat
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genostore import convert as gc
from genostore import format as gf
from genostore.from_pickles import convert as convert_pickles
from genostore.from_pickles import main as from_pickles_main
from genostore.store import GenoStore


# ---- quantisation and the block codec -----------------------------------------------------
def test_quantize_dosage_is_exact_round_half_up_on_the_pgenlib_grid():
    k = np.arange(32769)                                  # pgenlib dosages are k / 16384, 0 <= k <= 32768
    want = ((200 * k + 16384) // 32768).astype(np.uint8)  # floor(100 k / 16384 + 1/2), in integers
    d = (k / 16384).astype(np.float32)
    assert np.array_equal(gf.quantize_dosage(d), want)
    assert np.array_equal(gf.quantize_dosage(d.astype(np.float64)), want)
    assert np.array_equal(gf.quantize_dosage(np.repeat(d, 2)[::2]), want)        # strided input
    assert np.array_equal(gf.quantize_dosage(d.reshape(1, -1)), want.reshape(1, -1))
    # with a missing value the other path runs and must give the same answers
    mixed = np.concatenate([d, np.array([np.nan, -9.0, 2.5, -0.004, -1.0, 3.0], np.float32)])
    before = mixed.copy()
    q = gf.quantize_dosage(mixed)
    assert q.dtype == np.uint8 and np.array_equal(q[:-6], want)
    assert q[-6:].tolist() == [255, 255, 200, 0, 255, 200]
    assert np.array_equal(mixed, before, equal_nan=True)                         # input untouched


@pytest.mark.parametrize("shape", [(0,), (0, 3), (4, 0)])
def test_quantize_dosage_keeps_empty_shapes(shape):
    q = gf.quantize_dosage(np.empty(shape, np.float32))
    assert q.shape == shape and q.dtype == np.uint8


def test_encode_block_bit_layout_and_exception_order():
    q = np.array([[0, 100, 200, 37, 255, 0],
                  [200, 1, 0, 0, 100, 199]], dtype=np.uint8)
    # codes 0 1 2 3 | 3 0 (padding 0 0) and 2 3 0 0 | 1 3; variant j sits at bit 2 * (j % 4) of byte j // 4
    want = [[0 | 1 << 2 | 2 << 4 | 3 << 6, 3], [2 | 3 << 2, 1 | 3 << 2]]
    for kw in ({}, {"slab": 1}):
        packed, ovr, counts = gf.encode_block(q, **kw)
        assert packed.tolist() == want and ovr.tolist() == [37, 255, 1, 199] and counts.tolist() == [2, 2]
        assert (packed.dtype, ovr.dtype, counts.dtype) == (np.uint8, np.uint8, np.int64)
    assert np.array_equal(gf.decode_packed(packed, ovr, 6), q)
    assert np.array_equal(gf.decode_packed(packed, ovr, 8), gf.decode_packed(packed, ovr))
    assert gf.decode_packed(packed, ovr).tolist() == [r + [0, 0] for r in q.tolist()]   # padding decodes as 0
    assert np.array_equal(gf.decode_packed(packed[None], ovr, 6), q[None])
    with pytest.raises(ValueError, match="exception count mismatch: 4 code-3 slots but 3 bytes"):
        gf.decode_packed(packed, ovr[:3], 6)
    packed, ovr, counts = gf.encode_block(np.zeros((0, 5), np.uint8))
    assert packed.shape == (0, 2) and ovr.shape == (0,) and ovr.dtype == np.uint8 and counts.shape == (0,)


def test_encode_block_column_slices_and_odd_widths():
    rng = np.random.default_rng(1)
    big = rng.choice(np.array([0, 100, 200, 7, 255], np.uint8), size=(9, 40))
    for v0, v1 in ((0, 40), (3, 40), (5, 6), (1, 34)):
        q = big[:, v0:v1]                                              # not contiguous
        packed, ovr, counts = gf.encode_block(q, slab=4)
        assert np.array_equal(gf.decode_packed(packed, ovr, v1 - v0), q)
        assert counts.tolist() == [int(((r != 0) & (r != 100) & (r != 200)).sum()) for r in q]


def test_samples_digest_is_sha256_of_the_ids_one_per_line():
    assert gf.samples_digest(["a", "b"]) == hashlib.sha256(b"a\nb\n").hexdigest()[:16]
    assert gf.samples_digest([1, 22]) == gf.samples_digest(["1", "22"])
    assert gf.samples_digest(str(i) for i in (1, 22)) == gf.samples_digest(["1", "22"])
    assert gf.samples_digest([]) == hashlib.sha256(b"").hexdigest()[:16]
    assert gf.samples_digest(["\u00e9"]) == hashlib.sha256("\u00e9\n".encode()).hexdigest()[:16]


# ---- what a store holds, byte for byte ----------------------------------------------------
GOLDEN = {                                  # sha256[:16] of each file, written by the code of d019bb9
    "chr1.base.bin": "82a37ae6a7765bac", "chr1.ovr.bin": "03c449030599ede3",
    "chr1.ovr_ptr.bin": "1d1a10b423948875", "chr1.pos.npy": "97a3122bb9b5ff26",
    "chr1.stats.npy": "c4344ef9248fedd9", "chr1.variants.tsv": "46b8da0c4cb13c3d",
    "chr2.base.bin": "dc24b0a9367a3c8b", "chr2.ovr.bin": "fd57bfdc286642e5",
    "chr2.ovr_ptr.bin": "250892a2dc81e45d", "chr2.pos.npy": "3e6f830f62f573fe",
    "chr2.stats.npy": "f0f6cfcf794b9b53", "chr2.variants.tsv": "5bb3d67b072e47cb",
    "samples.txt": "64aefc03236f4505",
}
CHROM_KEYS = ["chrom", "n_samples", "n_variants", "block_variants", "n_blocks", "base_bytes", "ovr_bytes",
              "ovr_block_start", "n_override", "n_missing", "elapsed_s", "bytes_per_genotype", "provenance",
              "complete"]
PROGRESS_KEYS = ["n_samples", "n_variants", "block_variants", "blocks_done", "base_bytes", "ovr_bytes",
                 "ovr_block_start", "n_override", "n_missing", "elapsed_s"]
META_KEYS = ["format", "version", "encoding", "dosage_scale", "missing", "block_variants", "n_samples",
             "samples_sha256_16", "created", "provenance"]


def small_q():
    rng = np.random.default_rng(1)
    q = rng.choice(np.array([0, 100, 200], np.uint8), size=(13, 70), p=[0.5, 0.3, 0.2])
    exc = rng.random(q.shape) < 0.1
    q[exc] = rng.integers(1, 200, size=int(exc.sum()), dtype=np.uint8)
    q[rng.random(q.shape) < 0.03] = gf.MISSING
    return q


def write_small_store(path):
    q = small_q()
    gf.create_store(path, [f"id{i}" for i in range(13)], block_variants=16, provenance={"test": 1})
    gf.write_chrom_from_array(path, "1", q, block_variants=16)
    gf.write_chrom_from_array(path, "2", q[:, 3:8], [f"2\t{10 * j}\trs{j}\tA\tC" for j in range(5)], 16,
                              {"src": "x"}, pos=np.arange(5) * 3)
    return q


def test_store_files_are_byte_stable(tmp_path):
    store = str(tmp_path / "s")
    q = write_small_store(store)
    names = sorted(os.listdir(store))
    assert names == sorted(list(GOLDEN) + ["meta.json", "chr1.json", "chr2.json",
                                           "chr1.progress.json", "chr2.progress.json"])
    got = {f: hashlib.sha256(open(os.path.join(store, f), "rb").read()).hexdigest()[:16] for f in GOLDEN}
    assert got == GOLDEN
    js = {f: json.load(open(os.path.join(store, f))) for f in names if f.endswith(".json")}
    assert list(js["meta.json"]) == META_KEYS
    assert all(list(js[f"chr{c}.json"]) == CHROM_KEYS for c in (1, 2))
    assert all(list(js[f"chr{c}.progress.json"]) == PROGRESS_KEYS for c in (1, 2))
    assert js["chr1.json"]["ovr_block_start"] == [0, 18, 36, 64, 98, 104]
    assert {k: js["chr1.json"][k] for k in ("n_blocks", "base_bytes", "ovr_bytes", "n_override", "n_missing")} == \
        {"n_blocks": 5, "base_bytes": 234, "ovr_bytes": 104, "n_override": 104, "n_missing": 21}
    assert js["chr1.json"]["bytes_per_genotype"] == 338 / (13 * 70) and js["chr1.json"]["provenance"] == {}
    assert js["chr2.json"]["provenance"] == {"src": "x"} and js["chr2.json"]["complete"] is True
    assert js["meta.json"]["samples_sha256_16"] == GOLDEN["samples.txt"]
    assert js["meta.json"]["provenance"] == {"test": 1}
    # the statistics are the integer sums, exactly
    ok = q != gf.MISSING
    qz = np.where(ok, q, 0).astype(np.float64)
    want = np.stack([qz.sum(0), (qz * qz).sum(0), ok.sum(0)])
    stats = np.load(os.path.join(store, "chr1.stats.npy"))
    assert stats.dtype == np.float64 and np.array_equal(stats, want)
    assert np.array_equal(np.load(os.path.join(store, "chr2.pos.npy")), np.arange(5) * 3)
    assert np.array_equal(np.load(os.path.join(store, "chr1.pos.npy")), np.arange(1, 71))
    assert open(os.path.join(store, "chr1.variants.tsv")).read().splitlines()[:2] == \
        ["#CHROM\tPOS\tID\tREF\tALT", "1\t1\tchr1_snp0\tA\tG"]
    assert np.array_equal(GenoStore(store).read_q("2", np.arange(13)), q[:, 3:8])


def test_writer_refusals_and_resume_state(tmp_path):
    store = str(tmp_path / "s")
    gf.create_store(store, ["a", "b"], block_variants=8)
    with pytest.raises(ValueError, match="already holds a store with different n_samples: 2 vs 3"):
        gf.create_store(store, ["a", "b", "c"], block_variants=8)
    with pytest.raises(ValueError, match="different samples_sha256_16"):
        gf.create_store(store, ["a", "c"], block_variants=8)
    with pytest.raises(ValueError, match="different block_variants: 8 vs 16384"):
        gf.create_store(store, ["a", "b"])
    assert gf.create_store(store, ["a", "b"], 8, {"other": 1})["provenance"] == {}    # the existing meta
    w = gf.ChromWriter(store, "1", 2, 10, 8, drop_cache=False)
    assert (w.next_block, w.n_blocks, w.complete, w.block_range(1)) == (0, 2, False, (8, 10))
    with pytest.raises(ValueError, match=r"block 0 must have shape \(2, 8\), got \(2, 7\)"):
        w.append_block(np.zeros((2, 7), np.uint8))
    with pytest.raises(RuntimeError, match="chr1: only 0/2 blocks written"):
        w.finalize()
    w.append_block(np.full((2, 8), 100, np.uint8))
    with pytest.raises(ValueError, match="progress file for chr1 does not match this conversion"):
        gf.ChromWriter(store, "1", 2, 11, 8, drop_cache=False)
    w = gf.ChromWriter(store, "1", 2, 10, 8, drop_cache=False)                     # resumes after block 0
    assert w.next_block == 1 and w.st["ovr_block_start"] == [0, 0]
    w.append_block(np.array([[255, 7], [0, 200]], np.uint8))
    assert w.complete
    with pytest.raises(ValueError, match="chr1: 9 variant lines for 10 variants"):
        w.finalize([f"1\t{j}\tv{j}\tA\tC" for j in range(9)])
    with pytest.raises(ValueError, match="chr1: 3 positions for 10 variants"):
        w.finalize(pos=np.arange(3))
    info = w.finalize(pos=np.arange(10))
    assert (info["n_override"], info["n_missing"], info["ovr_block_start"]) == (2, 1, [0, 0, 2])
    assert np.load(gf.chrom_paths(store, "1")["pos"]).tolist() == list(range(10))
    fresh = gf.ChromWriter(store, "1", 2, 10, 8, resume=False, drop_cache=False)  # starts again
    assert fresh.next_block == 0 and os.path.getsize(fresh.p["base"]) == 0


def test_store_files_are_owner_only_even_when_rewritten(tmp_path):
    store = str(tmp_path / "s")
    old = os.umask(0o002)
    try:
        write_small_store(store)
        p = gf.chrom_paths(store, "1")["base"]
        os.chmod(p, 0o664)
        gf.write_chrom_from_array(store, "1", small_q(), block_variants=16)            # rewrites in place
    finally:
        os.umask(old)
    assert stat.S_IMODE(os.stat(store).st_mode) == 0o700
    assert {stat.S_IMODE(os.stat(os.path.join(store, f)).st_mode) for f in os.listdir(store)} == {0o600}


# ---- legacy pickles -----------------------------------------------------------------------
def write_pickles(src, x_train, labels=None, names=None):
    os.makedirs(src)
    dump = lambda name, obj: pickle.dump(obj, open(os.path.join(src, name + ".pkl"), "wb"))
    dump("genes", [np.array(["rsA", "rsB", "rsC"]), np.array(["rsD"])])
    dump("pos", [np.array([5, 9, 12]), np.array([7])])
    dump("snp_train", [x_train, np.array([[1.0], [2.0], [0.0]])])
    dump("snp_test", [np.array([[1.0, 1.0, 1.0], [0.25, 2.0, 0.0]]), np.array([[0.0], [1.0]])])
    lab = labels or {"train": np.array([0, 1, 0]), "test": np.array([1, 0])}
    for s in ("train", "test"):
        dump(f"label_{s}", [lab[s], lab[s]])
    dump("covar_train", np.array([[1.5, 2.0], [0.0, -1.0], [3.0, 4.0]]))
    dump("covar_test", np.array([[0.5, 0.5], [1.0, 2.0]]))
    for f, v in (names or {}).items():
        json.dump(v, open(os.path.join(src, f), "w"))


def test_from_pickles_writes_known_tables(tmp_path, capsys):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    write_pickles(src, np.array([[0.0, 1.0, 2.0], [2.0, 0.5, 1.0], [0.0, 0.0, 0.0]]))
    assert from_pickles_main(["--src", src, "--dst", dst]) == 0
    assert capsys.readouterr().out == f"wrote {dst}; largest |stored - original| genotype value: 0.000000\n"
    assert sorted(os.listdir(dst)) == ["genostore", "samples_test.tsv", "samples_train.tsv", "store.json"]
    assert open(os.path.join(dst, "store.json")).read() == '{\n "store": "genostore",\n "pos_mode": "pos"\n}'
    assert open(os.path.join(dst, "samples_train.tsv")).read() == (
        "IID\tlabel\tcov0\tcov1\ntr0000000\t0\t1.5\t2.0\ntr0000001\t1\t0.0\t-1.0\ntr0000002\t0\t3.0\t4.0\n")
    assert open(os.path.join(dst, "samples_test.tsv")).read() == (
        "IID\tlabel\tcov0\tcov1\nte0000000\t1\t0.5\t0.5\nte0000001\t0\t1.0\t2.0\n")
    gs = GenoStore(os.path.join(dst, "genostore"))
    assert list(gs.samples) == ["tr0000000", "tr0000001", "tr0000002", "te0000000", "te0000001"]
    assert gs.w == gf.BLOCK_VARIANTS and gs.meta["provenance"] == {"legacy_pickles": os.path.abspath(src)}
    assert gs.read_q("1", np.arange(5)).tolist() == [[0, 100, 200], [200, 50, 100], [0, 0, 0],
                                                      [100, 100, 100], [25, 200, 0]]
    assert gs.read_q("2", np.arange(5)).ravel().tolist() == [100, 200, 0, 0, 100]
    assert list(gs.variants("1")["id"]) == ["rsA", "rsB", "rsC"] and list(gs.variants("1")["pos"]) == [0, 1, 2]
    assert list(gs.positions("1")) == [5, 9, 12] and list(gs.positions("2")) == [7]


def test_from_pickles_label_matrix_and_out_of_range_values(tmp_path, capsys):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    labels = {"train": np.array([[0, 1], [1, 1], [0, 0]]), "test": np.array([[1, 0], [0, 1]])}
    write_pickles(src, np.array([[0.0, 1.0, 2.5], [2.0, 0.5, 1.0], [0.0, 0.0, 0.0]]), labels,
                  {"label_names.json": ["t2d", "chd"], "covar_names.json": ["age", "sex"]})
    assert convert_pickles(src, dst) == 0.5                                          # 2.5 is stored as 2
    assert json.load(open(os.path.join(dst, "store.json"))) == \
        {"store": "genostore", "pos_mode": "pos", "labels": ["t2d", "chd"], "covariates": ["age", "sex"]}
    assert open(os.path.join(dst, "samples_test.tsv")).read() == (
        "IID\tt2d\tchd\tage\tsex\nte0000000\t1\t0\t0.5\t0.5\nte0000001\t0\t1\t1.0\t2.0\n")
    assert from_pickles_main(["--src", src, "--dst", str(tmp_path / "dst2")]) == 1
    assert capsys.readouterr().out.endswith(
        "value: 0.500000  <-- values are not dosages in [0, 2]; this store cannot represent them\n")
    json.dump(["only_one"], open(os.path.join(src, "covar_names.json"), "w"))
    with pytest.raises(ValueError, match="covar_names.json has 1 names for 2 columns"):
        convert_pickles(src, str(tmp_path / "dst3"))


# ---- convert.py helpers that need no pgenlib ----------------------------------------------
def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    return str(path)


def test_read_psam_and_pvar(tmp_path):
    assert gc.read_psam_ids(write(tmp_path / "a.psam", "#FID\tIID\tSEX\n1\t10\tNA\n2\t20\tNA\n")) == ["10", "20"]
    assert gc.read_psam_ids(write(tmp_path / "b.psam", "#IID\tSEX\n10\tF\n\n20\tM\n")) == ["10", "20"]
    assert gc.read_psam_ids(write(tmp_path / "c.fam", "1 10 0 0 1 -9\n2 20 0 0 2 -9\n")) == ["10", "20"]
    pvar = write(tmp_path / "a.pvar", "##fileformat=PVARv1.0\n#CHROM\tID\tPOS\tALT\tREF\tQUAL\n"
                                     "1\trs1\t5\tG\tA\t.\n1\trs2\t9\tT\tC\t.\n")
    assert gc.read_pvar_lines(pvar) == ["1\t5\trs1\tA\tG", "1\t9\trs2\tC\tT"]
    assert gc.read_pvar_lines(pvar, 1) == ["1\t5\trs1\tA\tG"]
    with pytest.raises(ValueError, match="no #CHROM header line"):
        gc.read_pvar_lines(write(tmp_path / "b.pvar", "1\t5\trs1\tA\tG\n"))


def fake_pgen(path, mode, fmt, m, n, rest=b""):
    with open(path, "wb") as f:
        f.write(b"\x6c\x1b" + bytes([mode]) + m.to_bytes(4, "little") + n.to_bytes(4, "little") + bytes([fmt]) + rest)
    return str(path)


def test_pgen_header_and_record_offsets(tmp_path):
    p = fake_pgen(tmp_path / "x.pgen", 0x02, 0x40, 5, 3, b"\0" * 100)
    assert gc.pgen_header(p) == {"mode": 2, "n_variants": 5, "n_samples": 3, "fmt": 64}
    offs, exact = gc.pgen_variant_offsets(p)                     # not the layout parsed: interpolated
    assert not exact and offs.dtype == np.uint64 and offs.tolist() == [12, 32, 52, 72, 92, 112]
    # one variant block: 8-bit record types, 1-byte record lengths 4, 5 and 6, then the records
    index = (26).to_bytes(8, "little") + b"\x01\x01\x01" + bytes([4, 5, 6])
    offs, exact = gc.pgen_variant_offsets(fake_pgen(tmp_path / "y.pgen", 0x10, 0x04, 3, 2, index + b"\0" * 15))
    assert exact and offs.tolist() == [26, 30, 35, 41]
    # the same with one allele-count byte per variant and per-variant non-ref flags
    index = (30).to_bytes(8, "little") + b"\x01\x01\x01" + bytes([4, 5, 6]) + b"\x02\x02\x02" + b"\0"
    offs, exact = gc.pgen_variant_offsets(fake_pgen(tmp_path / "z.pgen", 0x10, 0xD4, 3, 2, index + b"\0" * 15))
    assert exact and offs.tolist() == [30, 34, 39, 45]
    offs, exact = gc.pgen_variant_offsets(fake_pgen(tmp_path / "w.pgen", 0x10, 0x04, 3, 2, index + b"\0" * 9))
    assert not exact                                             # the index disagrees with the file
    with pytest.raises(ValueError, match="not a .pgen file"):
        gc.pgen_header(write(tmp_path / "t.txt", "not a pgen at all"))


def test_read_extract_and_select_variants(tmp_path):
    assert gc.read_extract(write(tmp_path / "a.tsv", "CHROM\tPOS\tID\textra\n2\t107\trs2_1\tx\n1\t100\trs1_0\tx\n")) \
        == {"2": {"id": ["rs2_1"], "pos": [107]}, "1": {"id": ["rs1_0"], "pos": [100]}}
    assert gc.read_extract(write(tmp_path / "b.txt", "chr pos\n1 100\n\n1 107\n")) == \
        {"1": {"id": None, "pos": [100, 107]}}
    assert gc.read_extract(write(tmp_path / "c.txt", "#chromosome\tid\n1\trs1\n\t\n")) == \
        {"1": {"id": ["rs1"], "pos": None}}
    with pytest.raises(ValueError, match=r"the header must name CHROM and ID \(or POS\); found \['CHROM', 'REF'\]"):
        gc.read_extract(write(tmp_path / "d.txt", "CHROM\tREF\n1\tA\n"))
    pvar = ["1\t100\trs1_0\tA\tG", "1\t107\trs1_1\tA\tG", "1\t114\trs1_2\tA\tG"]
    got = gc.select_variants(pvar, {"id": ["rs1_2", "rs1_0"], "pos": None}, "1", "f")
    assert got.dtype == np.uint32 and got.tolist() == [0, 2]
    assert gc.select_variants(pvar, {"id": None, "pos": [114, 107]}, "1", "f").tolist() == [1, 2]
    assert gc.select_variants(pvar, {"id": ["rs1_1"], "pos": [107]}, "1", "f").tolist() == [1]
    with pytest.raises(ValueError, match="chr1: 1 variants have a different position in the .pvar"):
        gc.select_variants(pvar, {"id": ["rs1_1"], "pos": [108]}, "1", "f")
    with pytest.raises(ValueError, match=r"chr1: 1 of 2 variants in f are not in the .pvar, e.g. \['nope'\]"):
        gc.select_variants(pvar, {"id": ["rs1_1", "nope"], "pos": None}, "1", "f")
    with pytest.raises(ValueError, match="chr1: 1 requested variants are ambiguous in the .pvar"):
        gc.select_variants(pvar + ["1\t120\trs1_0\tC\tT"], {"id": ["rs1_0"], "pos": None}, "1", "f")
    with pytest.raises(ValueError, match="chr1: f lists some variants more than once"):
        gc.select_variants(pvar, {"id": ["rs1_0", "rs1_0"], "pos": None}, "1", "f")


def test_choose_samples(tmp_path, capsys):
    ids = [str(10 * i) for i in range(10)]
    assert gc.choose_samples(ids) is None and gc.choose_samples(ids, 10, 1) is None
    got = gc.choose_samples(ids, 3, 1)
    assert got.dtype == np.uint32 and got.tolist() == np.sort(np.random.default_rng(1).choice(10, 3, replace=False)).tolist()
    keep = write(tmp_path / "keep.txt", "# a comment\n1 30\n\n2 10\nx 999\n")
    assert gc.choose_samples(ids, 3, 1, keep).tolist() == [1, 3]                      # file order does not matter
    assert capsys.readouterr().out.endswith(f"WARNING: 1 IDs in {keep} are not in the .psam\n")


# ---- the command line, with the .pgen part replaced -------------------------------------
@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A .psam per chromosome and a convert_chrom that records what it was given."""
    calls = []

    def fake(pgen, pvar, store_dir, chrom, *a, **k):
        calls.append((pgen, pvar, store_dir, chrom, a, k))
        return fake.status

    fake.status = "done"
    monkeypatch.setattr(gc, "convert_chrom", fake)
    for c in ("1", "2"):
        write(tmp_path / f"T_chr{c}.psam", "#FID\tIID\n" + "".join(f"{i}\t{100 + i}\n" for i in range(5)))
        write(tmp_path / f"T_chr{c}.pvar", "#CHROM\tPOS\tID\tREF\tALT\n" +
              "".join(f"{c}\t{10 * j}\trs{c}_{j}\tA\tG\n" for j in range(6)))
    common = ["--pgen-dir", str(tmp_path), "--pattern", "T_chr{chrom}", "--out", str(tmp_path / "out")]
    return calls, fake, common, tmp_path


def test_cli_passes_its_settings_to_convert_chrom(cli, capsys):
    calls, _, common, d = cli
    out = str(d / "out")
    assert gc.main(common + ["--chroms", "1", "2", "--keep-n", "3", "--seed", "1"]) == 0
    rows = np.sort(np.random.default_rng(1).choice(5, 3, replace=False))
    assert [c[3] for c in calls] == ["1", "2"]
    pgen, pvar, store_dir, chrom, a, k = calls[0]
    assert (pgen, pvar, store_dir) == (str(d / "T_chr1.pgen"), str(d / "T_chr1.pvar"), out)
    assert a[0].tolist() == rows.tolist() and a[1:] == (5, 16384, 256, "idle", 0.5, 1536 << 20, True, None, None)
    assert k == {"threads": 8, "stall_timeout": None, "var_idx": None, "extract_src": None}
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["provenance"] == {"psam": str(d / "T_chr1.psam"), "n_raw_samples": 5,
                                  "raw_samples_sha256_16": gf.samples_digest([str(100 + i) for i in range(5)]),
                                  "subset": {"keep_n": 3, "seed": 1}}
    assert open(os.path.join(out, "samples.txt")).read() == "".join(f"{100 + i}\n" for i in rows)
    assert np.load(os.path.join(out, "sample_rows.npy")).tolist() == rows.tolist()
    assert f"store {out}: 3 samples (of 5), block 16384" in capsys.readouterr().out


def test_cli_extract_and_settings(cli):
    calls, _, common, d = cli
    ex = write(d / "keep.tsv", "CHROM\tID\n2\trs2_4\n1\trs1_5\n1\trs1_0\n2\trs2_1\n")
    assert gc.main(common + ["--chroms", "2", "1", "--extract", ex, "--block-variants", "64", "--chunk", "5",
                             "--disk-policy", "share", "--duty", "0.3", "--max-ahead-gb", "0.5", "--no-prefetch",
                             "--threads", "3", "--max-variants", "1", "--time-budget-s", "1000"]) == 0
    assert [c[3] for c in calls] == ["2", "1"]
    (*_, a2, k2), (*_, a1, k1) = calls
    assert a2[0] is None and a2[1:9] == (5, 64, 5, "share", 0.3, 1 << 29, False, 1)
    assert len(a2) == 10 and a2[9] > 0 and a2[9] == a1[9]                            # one deadline for the run
    assert k2["var_idx"].tolist() == [1, 4] and k1["var_idx"].tolist() == [0, 5]
    assert k1["threads"] == 3 and k1["stall_timeout"] is None and k1["extract_src"] == os.path.abspath(ex)
    assert json.load(open(d / "out" / "meta.json"))["provenance"]["subset"] == "all"
    assert not os.path.exists(d / "out" / "sample_rows.npy")


def test_cli_keep_file_and_anonymised_subsets(cli):
    calls, _, common, d = cli
    keep = write(d / "keep.txt", "103\n100\n")
    assert gc.main(common + ["--chroms", "1", "--keep-file", keep]) == 0
    assert calls[-1][4][0].tolist() == [0, 3]
    meta = json.load(open(d / "out" / "meta.json"))
    assert meta["provenance"]["subset"] == {"keep_file": keep} and meta["n_samples"] == 2
    assert np.load(d / "out" / "sample_rows.npy").tolist() == [0, 3]
    common[-1] = str(d / "anon")
    assert gc.main(common + ["--chroms", "1", "--keep-n", "2", "--seed", "1", "--anonymize-ids"]) == 0
    assert open(d / "anon" / "samples.txt").read() == "anon000000\nanon000001\n"
    assert json.load(open(d / "anon" / "meta.json"))["provenance"]["subset"] == {"keep_n": 2, "anonymized": True}
    assert not os.path.exists(d / "anon" / "sample_rows.npy")


def test_cli_refusals(cli):
    calls, _, common, d = cli
    ex = write(d / "keep.tsv", "CHROM\tID\n1\trs1_0\n")
    old = os.umask(0o022)
    try:
        with pytest.raises(SystemExit, match="--extract cannot be combined with --opportunistic"):
            gc.main(common + ["--chroms", "1", "--extract", ex, "--opportunistic"])
        assert os.umask(0o022) == 0o022                                             # restored after a refusal
        with pytest.raises(SystemExit, match=f"{re.escape(ex)} lists no variants on chr2"):
            gc.main(common + ["--chroms", "1", "2", "--extract", ex])
        with pytest.raises(SystemExit, match="--anonymize-ids is meant for small subsets; combine it with --keep-n"):
            gc.main(common + ["--chroms", "1", "--anonymize-ids"])
        write(d / "T_chr2.psam", "#FID\tIID\n" + "".join(f"{i}\t{104 - i}\n" for i in range(5)))
        with pytest.raises(ValueError, match="chr2: .psam sample order differs from the one this store was built from"):
            gc.main(common + ["--chroms", "1", "2"])
    finally:
        os.umask(old)
    assert [c[3] for c in calls] == ["1"]


def test_cli_resumable_exit_and_skips_finished_chromosomes(cli, capsys):
    calls, fake, common, d = cli
    out = str(d / "out")
    fake.status = "deadline"
    assert gc.main(common + ["--chroms", "1", "2"]) == 3
    assert [c[3] for c in calls] == ["1"]
    assert "not finished: chr1,2 (rerun the same command to continue)" in capsys.readouterr().out
    gf.write_chrom_from_array(out, "1", np.zeros((5, 3), np.uint8))                 # chromosome 1 is complete
    fake.status = "done"
    assert gc.main(common + ["--chroms", "1", "2"]) == 0
    assert [c[3] for c in calls] == ["1", "2"]
    gf.write_chrom_from_array(out, "2", np.zeros((5, 3), np.uint8))
    assert gc.main(common + ["--chroms", "1", "2", "--time-budget-s", "0"]) == 0     # nothing left to do
    assert len(calls) == 2


def test_verify_reports_mismatches_in_the_line_scripts_read(tmp_path, capsys):
    store = str(tmp_path / "s")
    rng = np.random.default_rng(1)
    gf.create_store(store, [f"s{i}" for i in range(6)], block_variants=8)
    qs = {}
    for c in ("1", "2", "3"):
        qs[c] = rng.choice(np.array([0, 100, 200, 3], np.uint8), size=(6, 20))
        gf.write_chrom_from_array(store, c, qs[c], block_variants=8)
    rows = np.array([0, 5])
    for c in ("1", "2"):
        np.save(os.path.join(store, f"chr{c}.audit_rows.npy"), rows)
        audit = qs[c][rows].copy()
        if c == "2":
            audit[1, 7] ^= 1
        np.save(os.path.join(store, f"chr{c}.audit.npy"), audit)
    capsys.readouterr()
    assert gc.verify_store(store, ["1"])
    line = capsys.readouterr().out.strip()
    assert re.fullmatch(r"\d\d:\d\d:\d\d chr1: 2 audit rows x 20 variants -> 0 mismatches; "
                        r"mean ALT dosage \d\.\d{4}, monomorphic variants \d+", line), line
    assert not gc.verify_store(store)
    out = capsys.readouterr().out
    assert "chr2: 2 audit rows x 20 variants -> 1 mismatches;" in out and "chr3: no audit file, skipped" in out
    assert gc.main(["verify", "--out", store, "--chroms", "1", "3"]) == 0
    assert gc.main(["verify", "--out", store]) == 1
