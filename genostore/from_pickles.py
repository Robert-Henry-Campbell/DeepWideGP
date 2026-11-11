# -*- coding: utf-8 -*-
"""Convert a legacy pickle dataset into the store layout (numpy only).

    python -m genostore.from_pickles --src pukb/genes/<label>/ld_all --dst pukb/genes/<label>/store
    python main_genome.py --label <label> --rd_mode store ...      # now trains through the store

Writes <dst>/genostore/ (the genotypes), <dst>/samples_{train,test}.tsv and <dst>/store.json.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

from . import format as gf


def _load(src, name):
    with open(os.path.join(src, name + ".pkl"), "rb") as f:
        return pickle.load(f)


def convert(src, dst, store_name="genostore", block_variants=gf.BLOCK_VARIANTS):
    genes, pos = _load(src, "genes"), _load(src, "pos")
    snp = {s: _load(src, f"snp_{s}") for s in ("train", "test")}
    lab = {s: np.asarray(_load(src, f"label_{s}")[0]) for s in ("train", "test")}
    cov = {s: np.asarray(_load(src, f"covar_{s}")) for s in ("train", "test")}
    n = {s: len(lab[s]) for s in ("train", "test")}
    ids = {"train": [f"tr{i:07d}" for i in range(n["train"])], "test": [f"te{i:07d}" for i in range(n["test"])]}
    os.makedirs(dst, exist_ok=True)
    store = os.path.join(dst, store_name)
    gf.create_store(store, ids["train"] + ids["test"], block_variants, {"legacy_pickles": os.path.abspath(src)})
    worst = 0.0
    for c in range(len(genes)):
        x = np.concatenate([np.asarray(snp["train"][c]), np.asarray(snp["test"][c])], axis=0)
        if x.ndim == 3:
            x = x[:, :, 0]
        q = gf.quantize_dosage(x.astype(np.float32))
        ok = q != gf.MISSING
        if ok.any():
            worst = max(worst, float(np.abs(q[ok] / gf.DOSAGE_SCALE - x[ok]).max()))
        lines = (f"{c + 1}\t{j}\t{genes[c][j]}\tN\tN" for j in range(x.shape[1]))
        gf.write_chrom_from_array(store, str(c + 1), q, lines, block_variants, pos=np.asarray(pos[c]))
    for s in ("train", "test"):
        with open(os.path.join(dst, f"samples_{s}.tsv"), "w") as f:
            f.write("\t".join(["IID", "label"] + [f"cov{k}" for k in range(cov[s].shape[1])]) + "\n")
            for i in range(n[s]):
                f.write("\t".join([ids[s][i], repr(lab[s][i].item())] + [repr(float(v)) for v in cov[s][i]]) + "\n")
    with open(os.path.join(dst, "store.json"), "w") as f:
        json.dump({"store": store_name, "pos_mode": "pos"}, f, indent=1)
    return worst


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args(argv)
    worst = convert(a.src, a.dst)
    print(f"wrote {a.dst}; largest |stored - original| genotype value: {worst:.6f}"
          + ("" if worst <= 0.005 + 1e-6 else "  <-- values are not dosages in [0, 2]; this store cannot represent them"))
    return 0 if worst <= 0.005 + 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
