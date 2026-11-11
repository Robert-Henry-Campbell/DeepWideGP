"""Generate a synthetic multi-disease dataset in the genotype-store layout.

Six traits on a liability scale, chosen so that mistakes in the multi-disease training code
show up as a failed check rather than as a slightly worse number:

    shared_a  30%   share one set of causal SNPs (genetic correlation ``rho``) and
    shared_b  10%   each has causal SNPs of its own; shared_a also depends on a
    shared_c   5%   non-genetic factor
    solo_d    15%   causal SNPs of its own only
    solo_e     2%   causal SNPs of its own only; rare, so imbalance handling is exercised
    null_f    10%   NO genetic signal; depends on the same non-genetic factor as shared_a,
                    so the two co-occur. Its AUROC must stay at 0.5: anything else means the
                    covariates, the sampler or the loss manufactured signal.

Per-SNP annotations: channel ``causal`` marks every causal SNP; the other three carry no
information. Covariates (age, sex, six "PCs") have no effect on any trait.

Writes <base_dir>/<name>/<rd_mode>/ with the genotype store, samples_{train,val,test}.tsv,
store.json (naming labels and covariates), labels.json, annotations.npz and truth.json (the
simulation's own AUROC ceiling per trait).

    python scripts/generate_synthetic_multilabel.py --name multi6 --n_participants 20000
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genostore import format as gf

TRAITS = [  # name, prevalence, heritability, share of the non-genetic factor, uses the shared causal set
    ("shared_a", 0.30, 0.50, 0.20, True),
    ("shared_b", 0.10, 0.50, 0.00, True),
    ("shared_c", 0.05, 0.50, 0.00, True),
    ("solo_d", 0.15, 0.50, 0.00, False),
    ("solo_e", 0.02, 0.50, 0.00, False),
    ("null_f", 0.10, 0.00, 0.30, False),
]
COVARIATES = ["age", "sex", "pc1", "pc2", "pc3", "pc4", "pc5", "pc6"]
ANNOTATIONS = ["causal", "noise_binary", "noise_continuous", "maf"]


def _auc(score, y):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def generate(name, n_participants=20000, n_snps=2200, base_dir="pukb/genes", rd_mode="store", seed=7,
             n_shared_causal=30, n_own_causal=20, rho=0.6, frac_val=0.15, frac_test=0.15, block_variants=64):
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    n_chr = 22
    per_chr = [n_snps // n_chr + (1 if i < n_snps % n_chr else 0) for i in range(n_chr)]
    m = sum(per_chr)
    maf = rng.uniform(0.05, 0.5, size=m)
    geno = rng.binomial(2, maf, size=(n_participants, m)).astype(np.int8)
    z = (geno - geno.mean(0)) / geno.std(0)

    # causal SNPs: one shared set, plus a disjoint set per heritable trait
    heritable = [t for t in TRAITS if t[2] > 0]
    need = n_shared_causal + n_own_causal * len(heritable)
    if need > m // 2:                                       # tiny test datasets: keep at most half the SNPs causal
        shrink = (m // 2) / need
        n_shared_causal = max(2, int(n_shared_causal * shrink))
        n_own_causal = max(2, int(n_own_causal * shrink))
    picks = rng.choice(m, size=n_shared_causal + n_own_causal * len(heritable), replace=False)
    shared_idx = np.sort(picks[:n_shared_causal])
    own_idx = {t[0]: np.sort(picks[n_shared_causal + i * n_own_causal: n_shared_causal + (i + 1) * n_own_causal])
               for i, t in enumerate(heritable)}

    def std(v):
        return (v - v.mean()) / v.std()

    g_shared = std(z[:, shared_idx] @ rng.standard_normal(n_shared_causal))
    c = rng.standard_normal(n_participants)                 # non-genetic factor shared by shared_a and null_f
    labels = np.zeros((n_participants, len(TRAITS)), dtype=np.int8)
    genetic = np.zeros((n_participants, len(TRAITS)))
    for j, (tname, k, h2, c2, uses_shared) in enumerate(TRAITS):
        g = np.zeros(n_participants)
        if h2 > 0:
            g_own = std(z[:, own_idx[tname]] @ rng.standard_normal(n_own_causal))
            g = std(np.sqrt(rho) * g_shared + np.sqrt(1 - rho) * g_own) if uses_shared else g_own
        liab = np.sqrt(h2) * g + np.sqrt(c2) * c + np.sqrt(1 - h2 - c2) * rng.standard_normal(n_participants)
        labels[:, j] = liab > norm.isf(k)
        genetic[:, j] = g

    cov = rng.standard_normal((n_participants, len(COVARIATES)))
    cov[:, 0] = rng.normal(55, 8, size=n_participants)
    cov[:, 1] = rng.binomial(1, 0.5, size=n_participants)

    causal = np.zeros(m, dtype=bool)
    causal[shared_idx] = True
    for v in own_idx.values():
        causal[v] = True
    annot = np.stack([causal.astype(np.float32), rng.binomial(1, 0.2, size=m).astype(np.float32),
                      rng.standard_normal(m).astype(np.float32), maf.astype(np.float32)], axis=1)

    # splits
    perm = rng.permutation(n_participants)
    n_test = int(round(frac_test * n_participants))
    n_val = int(round(frac_val * n_participants))
    split = {"test": np.sort(perm[:n_test]), "val": np.sort(perm[n_test:n_test + n_val]),
             "train": np.sort(perm[n_test + n_val:])}

    out = os.path.join(base_dir, name, rd_mode)
    os.makedirs(out, exist_ok=True)
    store = os.path.join(out, "genostore")
    ids = [f"s{i:07d}" for i in range(n_participants)]
    gf.create_store(store, ids, block_variants, {"synthetic": "generate_synthetic_multilabel.py", "seed": seed})
    start, annot_by_chr = 0, {}
    for ci, mc in enumerate(per_chr):
        q = (geno[:, start:start + mc].astype(np.uint8) * 100)          # dosage 0/1/2 -> 0/100/200
        lines = (f"{ci + 1}\t{j}\trs_syn_{start + j}\tA\tG" for j in range(mc))
        gf.write_chrom_from_array(store, str(ci + 1), q, lines, block_variants, pos=np.arange(mc))
        annot_by_chr[f"chr{ci + 1}"] = annot[start:start + mc]
        start += mc
    np.savez(os.path.join(out, "annotations.npz"), **annot_by_chr)
    names = [t[0] for t in TRAITS]
    for s, rows in split.items():
        with open(os.path.join(out, f"samples_{s}.tsv"), "w") as f:
            f.write("\t".join(["IID"] + names + COVARIATES) + "\n")
            for i in rows:
                f.write("\t".join([ids[i]] + [str(int(v)) for v in labels[i]] + [repr(float(v)) for v in cov[i]]) + "\n")
    with open(os.path.join(out, "store.json"), "w") as f:
        json.dump({"store": "genostore", "pos_mode": "index", "labels": names, "covariates": COVARIATES,
                   "annotations": {"file": "annotations.npz", "names": ANNOTATIONS}}, f, indent=1)
    tr = split["train"]
    with open(os.path.join(out, "labels.json"), "w") as f:
        json.dump([{"label": t[0], "definition": "synthetic liability-threshold trait",
                    "cases_train": int(labels[tr, j].sum()), "prevalence_train": float(labels[tr, j].mean())}
                   for j, t in enumerate(TRAITS)], f, indent=1)
    truth = {"seed": seed, "n_participants": n_participants, "n_snps": m, "rho": rho,
             "traits": [{"label": t[0], "prevalence": t[1], "h2": t[2], "nongenetic_share": t[3], "uses_shared_set": t[4],
                         "oracle_auroc": {s: (_auc(genetic[rows, j], labels[rows, j]) if t[2] > 0 else 0.5)
                                          for s, rows in split.items()}}
                        for j, t in enumerate(TRAITS)],
             "shared_causal": shared_idx.tolist(), "own_causal": {k: v.tolist() for k, v in own_idx.items()},
             "co_occurrence_shared_a_null_f": float(np.corrcoef(labels[:, 0], labels[:, 5])[0, 1])}
    with open(os.path.join(out, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)
    return out, truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="multi6")
    ap.add_argument("--n_participants", type=int, default=20000)
    ap.add_argument("--n_snps", type=int, default=2200)
    ap.add_argument("--base_dir", default="pukb/genes")
    ap.add_argument("--rd_mode", default="store")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out, truth = generate(a.name, a.n_participants, a.n_snps, a.base_dir, a.rd_mode, a.seed)
    print(f"wrote {out}")
    for t in truth["traits"]:
        print(f"  {t['label']:9s} prevalence {t['prevalence']:.2f}  h2 {t['h2']:.2f}  "
              f"AUROC ceiling on test {t['oracle_auroc']['test']:.3f}")
    print(f"  correlation of shared_a and null_f labels: {truth['co_occurrence_shared_a_null_f']:.3f}")
