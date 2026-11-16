"""Check a multi-disease run on the synthetic six-trait dataset against what the simulation allows.

    python scripts/check_synthetic_run.py runs/synthetic/multi6 pukb/genes/multi6/store/truth.json

Prints each trait's validation AUROC beside the simulation's own ceiling and exits non-zero if
a band is missed. The bands catch a model that does not learn, a rare trait that is ignored and
a null trait that is "predicted"; they are not a test of small differences in accuracy. They
come from five training seeds of the smoke run (multilabel_synthetic.sh) in
multilabel_synthetic_seeds.csv (methods.md section 12): a heritable trait, and the mean over
heritable traits, must reach the lowest seed minus the larger of MARGIN and SD_MULTIPLE seed
standard deviations; the null trait must stay in NULL_BAND. A run with annotation vectors is
held to the annotated seeds' bands.
"""
import csv
import json
import os
import statistics
import sys

SEEDS_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multilabel_synthetic_seeds.csv")
MARGIN, SD_MULTIPLE = 0.04, 2.5
NULL_TRAIT, NULL_BAND = "null_f", (0.40, 0.60)
MEAN = "mean over heritable traits"


def bands(annot=False):
    """{trait: lowest acceptable validation AUROC}, plus the same for the mean over heritable traits."""
    arm = "annotated" if annot else "plain"
    by_trait, by_seed = {}, {}
    with open(SEEDS_TABLE) as f:
        for r in csv.DictReader(f):
            if r["arm"] != arm or r["trait"] == NULL_TRAIT:
                continue
            by_trait.setdefault(r["trait"], []).append(float(r["val_auroc"]))
            by_seed.setdefault(r["seed"], []).append(float(r["val_auroc"]))
    by_trait[MEAN] = [sum(v) / len(v) for v in by_seed.values()]
    floor = lambda v: min(v) - max(MARGIN, SD_MULTIPLE * statistics.stdev(v))
    return {k: int(floor(v) * 100) / 100.0 for k, v in by_trait.items()}        # rounded down to 0.01


def check(results, truth, verbose=True, annot=False):
    """results: results.json of a run; truth: truth.json of the dataset. Returns a list of failures."""
    ceiling = {t["label"]: t["oracle_auroc"]["val"] for t in truth["traits"]}
    low = bands(annot)
    fails, herit = [], []
    if verbose:
        print(f"{'trait':<10}{'AUROC':>8}{'ceiling':>9}{'band':>16}")
    for name, m in results["val"].items():
        a = m["auroc"]
        if name == NULL_TRAIT:
            ok, band = NULL_BAND[0] <= a <= NULL_BAND[1], f"{NULL_BAND[0]:.2f}-{NULL_BAND[1]:.2f}"
        else:
            ok, band = a >= low[name], f">= {low[name]:.2f}"
            herit.append(a)
        if verbose:
            print(f"{name:<10}{a:>8.3f}{ceiling[name]:>9.3f}{band:>16}{'' if ok else '   FAIL'}")
        if not ok:
            fails.append(f"{name}: AUROC {a:.3f} outside {band}")
    mean = sum(herit) / len(herit)
    if mean < low[MEAN]:
        fails.append(f"mean AUROC of the heritable traits {mean:.3f} < {low[MEAN]:.2f}")
    if verbose:
        print(f"{MEAN} {mean:.3f} (>= {low[MEAN]:.2f}; {'with' if annot else 'no'} annotation vectors)")
    return fails


if __name__ == "__main__":
    run, truth_path = sys.argv[1], sys.argv[2]
    annot = bool(json.load(open(f"{run}/config.json"))["args"].get("annot"))
    fails = check(json.load(open(f"{run}/results.json")), json.load(open(truth_path)), annot=annot)
    print("PASS" if not fails else "FAIL: " + "; ".join(fails))
    sys.exit(1 if fails else 0)
