"""Score the snapshots of a multi-disease run on one split.

    python score_snapshots.py --run runs/<experiment>/<run>                       # every snapshot, validation set
    python score_snapshots.py --run ... --snapshots best --split test --unlock_test --save_scores

Writes, in the run directory, scores_<split>.csv (one row per snapshot, label and metric: AUROC,
average precision and population-scale cross-entropy, with what the snapshot cost, plus rows for
a covariates-only logistic model) and scores_<split>_summary.csv (the last snapshot against the
covariates-only model).

The test split is locked: scoring it needs --unlock_test, and every access is appended to
test_access.log in the run directory. Model selection belongs on the validation split.

With --save_scores, per-person population-scale logits and labels of each scored snapshot are
written to --scores_dir (default: scores/ beside the dataset), owner-only. These are participant-level
data: a folder in the run directory or in a git work tree is refused.
"""
import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

_torch_load = torch.load                      # Lightning checkpoints hold Python objects (as main_multilabel.py)
torch.load = lambda *a, **k: _torch_load(*a, **{**k, "weights_only": False})

from widedeepgp import metrics as M
from widedeepgp.train import build_data, last_checkpoint, load_model, predict_split, train_labels

FIELDS = ["snapshot", "step", "samples_seen", "epoch_equiv", "train_seconds", "split", "label", "metric", "value",
          "n", "n_cases"]


def covariates_only(dm, split, names):
    """A logistic model per label on the (standardised) covariates, fitted on the training split."""
    from sklearn.linear_model import LogisticRegression
    xtr, ytr = dm.splits["train"]["cov"], train_labels(dm)
    xte = dm.splits[split]["cov"]
    yte = np.asarray(dm.splits[split]["y"], dtype=np.float32)
    yte = yte[:, None] if yte.ndim < 2 else yte
    logits = np.zeros(yte.shape, dtype=np.float64)
    for j in range(yte.shape[1]):
        ok = ~np.isnan(ytr[:, j])
        logits[:, j] = LogisticRegression(max_iter=2000).fit(xtr[ok], ytr[ok, j]).decision_function(xte)
    return M.per_label_metrics(logits, yte, None, names)


def pick_snapshots(run, which):
    """[(file relative to the run, its row of snapshots.csv)]. "best" is the checkpoint that
    results.json names once the run has finished, "last" the one a resume would continue from:
    a run copied and resumed before RunCheckpoint existed has best-v1.ckpt and last-v1.ckpt beside
    stale best.ckpt and last.ckpt."""
    table = []
    path = os.path.join(run, "snapshots.csv")
    if os.path.exists(path):
        with open(path) as f:
            table = list(csv.DictReader(f))
    if which == "all":
        return [(r["file"], r) for r in table]
    if which not in ("best", "last"):
        want = {int(s) for s in which.split(",")}
        return [(r["file"], r) for r in table if int(r["step"]) in want]
    rel = os.path.join("checkpoints", which + ".ckpt")
    results = os.path.join(run, "results.json")
    if which == "last":
        latest = last_checkpoint(os.path.join(run, "checkpoints"))
        if latest:
            rel = os.path.relpath(latest, run)
    elif os.path.exists(results):
        with open(results) as f:                 # a path inside the run, which may have moved since
            rel = os.path.join("checkpoints", os.path.basename(json.load(f)["best_checkpoint"]))
    return [(rel, {"snapshot": which, "step": "", "samples_seen": "", "epoch_equiv": "", "train_seconds": ""})]


def chosen_trial(run):
    """The folder to score: for a run that tuned itself, the chosen trial its results.json names."""
    path = os.path.join(run, "results.json")
    if os.path.exists(path) and not os.path.exists(os.path.join(run, "config.json")):
        with open(path) as f:
            r = json.load(f)
        if r.get("tuning"):
            return os.path.join(run, r["best_trial_dir"])    # relative to the run's folder
    return run


def refuse_shared_place(path, run_dir):
    """Per-person scores are participant-level: never in the run directory or a git work tree
    (CLAUDE.md section 3)."""
    real, run = os.path.realpath(path), os.path.realpath(run_dir)
    if real == run or real.startswith(run + os.sep):
        raise SystemExit(f"--scores_dir {path} is inside the run directory; per-person scores belong under data/")
    p = real
    while p != os.path.dirname(p):
        if os.path.exists(os.path.join(p, ".git")):
            raise SystemExit(f"--scores_dir {path} is inside the git repository {p}; per-person scores belong under data/")
        p = os.path.dirname(p)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (holds config.json), or a run that tuned itself")
    ap.add_argument("--split", default="val", help="val, test, train, or the name of an extra evaluation table")
    ap.add_argument("--snapshots", default="all", help="all, best, last, or comma-separated optimiser steps")
    ap.add_argument("--unlock_test", action="store_true", help="required to score the test split; the access is logged")
    ap.add_argument("--save_scores", action="store_true", help="also write per-person scores (participant-level)")
    ap.add_argument("--scores_dir", default=None, help="where per-person scores go (default: scores/ beside the dataset)")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    a.run = chosen_trial(a.run)

    with open(os.path.join(a.run, "config.json")) as f:
        config = json.load(f)
    snaps = pick_snapshots(a.run, a.snapshots)
    if not snaps:
        raise SystemExit("no snapshots match")
    scores_dir = None
    if a.save_scores:
        base = a.scores_dir or os.path.join(config["dataset_dir"], "scores")
        scores_dir = os.path.join(base, config["args"]["experiment"], config["args"]["exp_name"])
        refuse_shared_place(scores_dir, a.run)
    if a.split == "test":
        if not a.unlock_test:
            raise SystemExit("the test split is locked: pass --unlock_test once model selection is finished")
        with open(os.path.join(a.run, "test_access.log"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tsnapshots={a.snapshots}\tn={len(snaps)}\n")
    cfg = SimpleNamespace(**config["args"])
    if a.batch_size:
        cfg.eval_batch_size = a.batch_size
    dm = build_data(cfg)
    names = config["model"]["label_names"]

    rows = []
    for rel, info in snaps:
        model, ckpt = load_model(os.path.join(a.run, rel), dm, a.device)
        logits, y = predict_split(model, dm, a.split, a.device)
        shift = model.loss_fn.logit_shift.cpu().numpy()
        m = M.per_label_metrics(logits, y, shift, names)
        tag = info.get("snapshot") or os.path.splitext(os.path.basename(rel))[0]
        meta = dict(snapshot=tag, step=info["step"] or ckpt.get("global_step", ""),
                    samples_seen=info["samples_seen"] or ckpt.get("samples_seen", ""),
                    epoch_equiv=info["epoch_equiv"], train_seconds=info["train_seconds"], split=a.split)
        rows += M.to_rows(m, **meta)
        print(f"{tag}: mean AUROC {m['auroc_mean']:.4f}, mean cross-entropy {m['bce_mean']:.4f}", flush=True)
        if scores_dir:
            os.makedirs(scores_dir, mode=0o700, exist_ok=True)
            sp = dm.splits[a.split] if a.split in dm.splits else dm._read_table(a.split, dm._row_of)
            iid = np.asarray(dm.store.samples)[sp["rows"]].astype(str)   # text, so the file loads without pickle
            path = os.path.join(scores_dir, f"{a.split}_{tag}.npz")
            np.savez(path, iid=iid, labels=np.array(names),
                     logit_population=(logits - shift).astype(np.float32), y=y.astype(np.float32))
            os.chmod(path, 0o600)
        del model
    if a.split in dm.splits and dm.splits["train"]["cov"].shape[1]:
        base_m = covariates_only(dm, a.split, names)
        rows += M.to_rows(base_m, snapshot="covariates_only", step="", samples_seen="", epoch_equiv="",
                          train_seconds="", split=a.split)
        with open(os.path.join(a.run, f"scores_{a.split}_summary.csv"), "w", newline="") as f:
            w = csv.writer(f)                    # tag and m are the last snapshot scored
            w.writerow(["snapshot", "label", "n", "n_cases", "auroc", "auroc_covariates_only", "auroc_gain",
                        "bce", "bce_covariates_only"])
            for j, name in enumerate(names):
                w.writerow([tag, name, int(m["n"][j]), int(m["n_cases"][j]), f"{m['auroc'][j]:.6f}",
                            f"{base_m['auroc'][j]:.6f}", f"{m['auroc'][j] - base_m['auroc'][j]:.6f}",
                            f"{m['bce'][j]:.6f}", f"{base_m['bce'][j]:.6f}"])
        print(f"covariates only: mean AUROC {base_m['auroc_mean']:.4f}")
    out = os.path.join(a.run, f"scores_{a.split}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(snaps)} snapshots)")
    return rows


if __name__ == "__main__":
    main()
