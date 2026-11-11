"""PGEN -> store conversion tests. Needs pgenlib (skipped otherwise).

Run in any env that has pgenlib + numpy:  python -m pytest tests/test_genostore_pgen.py -q
(or without pytest:  python tests/test_genostore_pgen.py)
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pgenlib
except ImportError:                                  # pragma: no cover
    pgenlib = None
    try:
        import pytest
        pytestmark = pytest.mark.skip(reason="pgenlib not installed")
    except ImportError:
        pass

from genostore import format as gf
from genostore import convert as gc
from genostore.store import GenoStore


def write_pgen(d, chrom, n, m, seed, p_frac=0.05, p_missing=0.0):
    """A small imputed-looking dataset: mostly whole dosages, some fractional, optional missing."""
    rng = np.random.default_rng(seed)
    dos = rng.choice(np.array([0.0, 1.0, 2.0], dtype=np.float32), size=(m, n), p=[0.55, 0.25, 0.2])
    frac = rng.random((m, n)) < p_frac
    dos[frac] = rng.uniform(0, 2, size=int(frac.sum())).astype(np.float32)
    if p_missing:
        dos[rng.random((m, n)) < p_missing] = -9.0
    prefix = os.path.join(d, f"T_chr{chrom}")
    w = pgenlib.PgenWriter((prefix + ".pgen").encode(), n, variant_ct=m, nonref_flags=False, dosage_present=True)
    for v in range(m):
        w.append_dosages(dos[v])
    w.close()
    with open(prefix + ".psam", "w") as f:
        f.write("#FID\tIID\tSEX\n" + "".join(f"{1000 + i}\t{1000 + i}\tNA\n" for i in range(n)))
    with open(prefix + ".pvar", "w") as f:
        f.write("#CHROM\tPOS\tID\tREF\tALT\n" + "".join(f"{chrom}\t{100 + 7 * v}\trs{chrom}_{v}\tA\tG\n" for v in range(m)))
    # what pgenlib itself returns (the file stores dosages at 1/16384 precision)
    r = pgenlib.PgenReader((prefix + ".pgen").encode())
    back = np.empty((m, n), dtype=np.float32)
    r.read_dosages_range(0, m, back)
    r.close()
    return prefix, back


def run(argv):
    return gc.main(argv)


def test_convert_all_and_subset_and_resume():
    with tempfile.TemporaryDirectory() as d:
        n = 150
        truth = {}
        for c, m, pm in (("1", 1000, 0.0), ("2", 257, 0.01), ("3", 64, 0.0)):
            _, back = write_pgen(d, c, n, m, seed=int(c), p_missing=pm)
            truth[c] = gf.quantize_dosage(back).T                        # [n, m]
        common = ["--pgen-dir", d, "--pattern", "T_chr{chrom}", "--block-variants", "64", "--chunk", "50"]

        # all samples, three chromosomes
        s_all = os.path.join(d, "store_all")
        assert run(common + ["--chroms", "1", "2", "3", "--out", s_all]) == 0
        gs = GenoStore(s_all)
        assert gs.chroms == ["1", "2", "3"] and gs.n_samples == n
        assert list(gs.samples[:3]) == ["1000", "1001", "1002"]
        for c in truth:
            assert np.array_equal(gs.read_q(c, np.arange(n)), truth[c])
            assert gs.variants(c)["pos"][1] == 107 and gs.variants(c)["id"][0] == f"rs{c}_0"
        assert (truth["2"] == gf.MISSING).any() and gs.info["2"]["n_missing"] == int((truth["2"] == gf.MISSING).sum())
        assert gc.verify_store(s_all)

        # random subset of samples
        s_sub = os.path.join(d, "store_sub")
        assert run(common + ["--chroms", "1", "2", "--out", s_sub, "--keep-n", "40", "--seed", "3"]) == 0
        gsub = GenoStore(s_sub)
        rows = np.load(os.path.join(s_sub, "sample_rows.npy"))
        assert len(rows) == 40 and np.all(np.diff(rows) > 0)
        assert list(gsub.samples) == [str(1000 + i) for i in rows]
        for c in ("1", "2"):
            assert np.array_equal(gsub.read_q(c, np.arange(40)), truth[c][rows])
        assert gc.verify_store(s_sub)

        # interrupted conversion resumes and ends identical
        s_res = os.path.join(d, "store_resume")
        assert run(common + ["--chroms", "1", "--out", s_res, "--time-budget-s", "0"]) == 3
        assert not os.path.exists(os.path.join(s_res, "chr1.json"))
        assert run(common + ["--chroms", "1", "--out", s_res, "--max-variants", "1000", "--time-budget-s", "0"]) == 3
        assert run(common + ["--chroms", "1", "--out", s_res]) == 0
        assert np.array_equal(GenoStore(s_res).read_q("1", np.arange(n)), truth["1"])
        assert run(common + ["--chroms", "1", "--out", s_res]) == 0     # idempotent

        # opportunistic mode: files just written are in the page cache, so they are "readable now"
        s_opp = os.path.join(d, "store_opp")
        assert run(common + ["--chroms", "3", "1", "2", "--out", s_opp, "--opportunistic", "--keep-n", "25",
                             "--seed", "1", "--anonymize-ids"]) == 0
        g = GenoStore(s_opp)
        assert g.chroms == ["1", "2", "3"] and list(g.samples[:2]) == ["anon000000", "anon000001"]
        assert not os.path.exists(os.path.join(s_opp, "sample_rows.npy"))
        rows25 = np.sort(np.random.default_rng(1).choice(n, size=25, replace=False))
        for c in truth:
            assert np.array_equal(g.read_q(c, np.arange(25)), truth[c][rows25])

        # --max-variants prefix
        s_pre = os.path.join(d, "store_prefix")
        assert run(common + ["--chroms", "1", "--out", s_pre, "--max-variants", "130"]) == 0
        assert np.array_equal(GenoStore(s_pre).read_q("1", np.arange(n)), truth["1"][:, :130])


def test_record_offsets_are_exact():
    with tempfile.TemporaryDirectory() as d:
        prefix, _ = write_pgen(d, "9", 60, 500, seed=9)
        offs, exact = gc.pgen_variant_offsets(prefix + ".pgen")
        assert exact and len(offs) == 501
        assert int(offs[-1]) == os.path.getsize(prefix + ".pgen") and np.all(np.diff(offs.astype(np.int64)) > 0)


if __name__ == "__main__":
    test_record_offsets_are_exact()
    test_convert_all_and_subset_and_resume()
    print("pgen conversion tests passed")
