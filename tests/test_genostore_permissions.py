"""Genotype stores hold participant-level data: they are created owner-only.

Run:  python -m pytest tests/test_genostore_permissions.py -q   (CPU)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_genotype_stores_are_their_owners_only(tmp_path, monkeypatch):
    """Stores hold participant-level data: directories 0700 and files 0600 whatever the umask,
    whether written through genostore.format or by the conversion command."""
    import stat
    from genostore import convert as gc
    from genostore import format as gf
    mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)

    def written_as_convert_chrom_writes(pgen, pvar, store_dir, chrom, *a, **k):   # no .pgen needed
        np.save(os.path.join(store_dir, f"chr{chrom}.audit_rows.npy"), np.arange(2))
        return "done"

    monkeypatch.setattr(gc, "convert_chrom", written_as_convert_chrom_writes)
    (tmp_path / "T_chr1.psam").write_text("#FID\tIID\n" + "".join(f"{i}\t{i}\n" for i in range(5)))
    old = os.umask(0o002)                                   # group- and world-readable by default
    try:
        lib = str(tmp_path / "lib")
        gf.create_store(lib, ["a", "b", "c"], block_variants=8)
        gf.write_chrom_from_array(lib, "1", np.zeros((3, 20), np.uint8), block_variants=8)
        cli = str(tmp_path / "cli")
        assert gc.main(["--pgen-dir", str(tmp_path), "--pattern", "T_chr{chrom}", "--chroms", "1", "--out", cli,
                        "--keep-n", "3"]) == 0
        assert os.umask(0o002) == 0o002                     # the command gave the caller's umask back
    finally:
        os.umask(old)
    for d in (lib, cli):
        files = {f: oct(mode(os.path.join(d, f))) for f in os.listdir(d)}
        assert mode(d) == 0o700 and len(files) >= 3 and set(files.values()) == {"0o600"}, files
    assert {"samples.txt", "meta.json", "sample_rows.npy", "chr1.audit_rows.npy"} <= set(os.listdir(cli))
