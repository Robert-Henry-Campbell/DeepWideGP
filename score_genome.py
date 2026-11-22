"""Per-person scores of the classic DeepGP ladder runs (main_genome.py): the tuning set and held-out cohorts.

    python score_genome.py --rung hm3plus_n13199 --disease I10 [--cohort asian --cohort black] [--device cuda]
    python score_genome.py --rung hm3plus_n13199 --disease I10 --check_only      # the tuning-set check alone
    python score_genome.py --all                                                 # every finished cell

A cell's settings are rebuilt as the ladder's scheduler launched main_genome.py (ARGS, read from schedule.py) and must
equal those saved in the run's best checkpoint, whose weights are loaded. The tuning set is scored first, through the
loader the cohorts use: its AUROC must reproduce results.json and its probabilities preds_val.pkl, within TOL, or
nothing is written and the call exits non-zero. A cohort is a held-out group of --held_out_dir, prepared with the
run's own training statistics; no metric is computed on it. Outputs that exist are kept unless --force.

Outputs, participant-level and owner-only: <out>/<rung>/scores/classic_deepgp/ladder/<disease>/<split>.npz with
iid [n], labels [1], y [n, 1] and logits [n, 1] (float32, the input of the model's final sigmoid: p = sigmoid(logit)).
"""
import argparse
import ast
import glob
import json
import os
import pickle

import numpy as np
import torch
from pytorch_lightning.utilities import move_data_to_device
from sklearn.metrics import roc_auc_score

from args_generator import args_initial
from main_genome import build, results_folder
from score_snapshots import refuse_shared_place

LADDER = "/mnt/sdg/robert/deepwidegp/data/classic_deepgp_ladder"
SCHEDULER = "/mnt/sdg/robert/deepwidegp/runs/classic_deepgp_ladder/schedule.py"
HELD_OUT = "/mnt/sdg/robert/deepwidegp/data/datasets/ard30/hm3plus_asian_black"
OUT = "/mnt/sdg/robert/deepwidegp/data/datasets/ard30"
TOL = 1e-5      # float32 on one GPU model repeats a run's numbers; a wrong checkpoint or preparation moves them far more


def scheduler_value(path, name):
    """A literal of the ladder's schedule.py (ARGS, RUNGS), read without running it."""
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{path} sets no {name}")


def cell_configs(rung, disease, ladder, args):
    """A run's settings, as the scheduler's launch() passed them to main_genome.py."""
    return args_initial(["--label", disease, "--labels", disease, "--rd_mode", rung, "--data_dir", f"{ladder}/datasets",
                         *args])


def _same(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def differing(saved, rebuilt):
    """Settings saved with a run that the rebuilt ones do not reproduce. Options added to the parser since are not in
    the checkpoint and are ignored: main_genome.py reads none of them, and the tuning check would show it if it did."""
    new = vars(rebuilt)
    return sorted(k for k, v in vars(saved).items() if k not in new or not _same(v, new[k]))


def count(n):
    """A count for stdout; 1 to 9 is written <10 (CLAUDE.md section 3)."""
    return "<10" if 0 < n < 10 else f"{n:,}"


@torch.no_grad()
def predict(model, dm, loader, device):
    """(y, p, logit) of everyone in a loader, in its order: p is the model's output, logit the input of its sigmoid."""
    ys, ps, logits = [], [], []
    hook = model.combine.register_forward_hook(lambda mod, inp, out: logits.append(out.float().cpu()))
    try:
        for batch in loader:
            x_all, y = dm.decode_batch(move_data_to_device(batch, device))    # as on_after_batch_transfer in training
            ps.append(model(x_all).float().cpu())
            ys.append(y.float().cpu())
    finally:
        hook.remove()
    return [torch.cat(t).numpy() for t in (ys, ps, logits)]


def tuning_check(y, p, logit, record, preds):
    """The tuning set's AUROC, and what of it does not reproduce the run's results.json and preds_val.pkl."""
    y0, p0 = (np.asarray(a, dtype=np.float32) for a in preds)
    auc = roc_auc_score(y[:, 0], p[:, 0])
    if y0.shape != y.shape or not np.array_equal(y0, y, equal_nan=True):
        return auc, ["its people or labels differ from preds_val.pkl"]
    bad, off = [], int((np.abs(p - p0) > TOL).sum())
    if off:
        bad.append(f"{count(off)} of {count(len(p))} probabilities differ from preds_val.pkl by more than {TOL:g} "
                   f"(mean difference {np.abs(p - p0).mean():.1e})")
    for what, s in (("probabilities", p), ("logits", logit)):
        a = roc_auc_score(y[:, 0], s[:, 0])
        if not abs(a - record["val_auroc"]) <= TOL:
            bad.append(f"the AUROC of the {what}, {a:.6f}, is not the recorded {record['val_auroc']:.6f}")
    return auc, bad


def write_scores(out, split, **arrays):
    """<out>/<split>.npz, owner-only and whole: written under a temporary name, then renamed into place."""
    old = os.umask(0o077)                       # folders made here are 0700
    try:
        os.makedirs(out, exist_ok=True)
    finally:
        os.umask(old)
    os.chmod(out, 0o700)
    path = os.path.join(out, f"{split}.npz")
    tmp = f"{path}.{os.getpid()}.tmp"
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)
    return path


def score_cell(rung, disease, a):
    """Check one finished cell on its tuning set, then write the per-person scores it lacks."""
    work = os.path.join(a.ladder, "work", rung, disease)
    out = os.path.join(a.out, rung, "scores", "classic_deepgp", "ladder", disease)
    refuse_shared_place(out, work)
    todo = [s for s in ["val", *a.cohort] if a.force or not os.path.exists(os.path.join(out, f"{s}.npz"))]
    if not (todo or a.check_only):
        print(f"{rung}/{disease}: every output exists; skipped")
        return
    configs = cell_configs(rung, disease, a.ladder, scheduler_value(a.scheduler, "ARGS"))
    dm, model = build(configs)
    res = os.path.join(work, results_folder(configs))
    with open(os.path.join(res, "results.json")) as f:
        record = json.load(f)
    ckpt = torch.load(os.path.join(work, record["best_checkpoint"]), map_location="cpu", weights_only=False)
    diff = differing(ckpt["hyper_parameters"]["configs"], configs)
    if diff:
        raise SystemExit(f"{rung}/{disease}: the rebuilt settings {diff} differ from the checkpoint's; nothing written")
    model.load_state_dict(ckpt["state_dict"])
    model.to(a.device).eval()
    dm.held_out_dir = a.held_out_dir
    y, p, logit = predict(model, dm, dm.eval_dataloder("val"), a.device)
    with open(os.path.join(res, "preds_val.pkl"), "rb") as f:
        auc, bad = tuning_check(y, p, logit, record, pickle.load(f))
    print(f"{rung}/{disease}: tuning set, {count(len(y))} people, {count(int(np.nansum(y)))} cases: AUROC {auc:.6f}, "
          f"recorded {record['val_auroc']:.6f}: " + ("; ".join(bad) if bad else f"reproduced within {TOL:g}"), flush=True)
    if bad:
        raise SystemExit(f"{rung}/{disease}: the tuning set does not reproduce the run's record; nothing written")
    if a.check_only:
        return
    for split in todo:
        if split != "val":
            y, _, logit = predict(model, dm, dm.eval_dataloder(split), a.device)
        iid = np.asarray(dm.iids(split)).astype(str)          # text, so the file loads without pickle
        if len(iid) != len(y):
            raise SystemExit(f"{rung}/{disease}: {split}: the loader and its table disagree on the people")
        path = write_scores(out, split, iid=iid, labels=np.array([disease]), y=y, logits=logit)
        print(f"  {split}: {count(len(y))} people, {count(int(np.nansum(y)))} cases -> {path}", flush=True)


def finished_cells(scheduler):
    """(rung, disease) of each ladder run its scheduler recorded as ended well (results/ beside it), rungs in order."""
    rungs = scheduler_value(scheduler, "RUNGS")
    cells = [os.path.basename(p)[:-len(".json")].split("__")
             for p in glob.glob(os.path.join(os.path.dirname(scheduler), "results", "*__*.json"))]
    return sorted(((r, d) for r, d in cells if r in rungs), key=lambda c: (rungs.index(c[0]), c[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung")
    ap.add_argument("--disease")
    ap.add_argument("--all", action="store_true", help="every finished cell (the scheduler's results/)")
    ap.add_argument("--cohort", action="append", help="a held-out group, repeatable (default: asian and black)")
    ap.add_argument("--check_only", action="store_true", help="score the tuning set and check it; write nothing")
    ap.add_argument("--force", action="store_true", help="rewrite outputs that exist")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ladder", default=LADDER, help="the ladder's data folder (datasets/, work/)")
    ap.add_argument("--scheduler", default=SCHEDULER, help="the ladder's schedule.py")
    ap.add_argument("--held_out_dir", default=HELD_OUT, help="the dataset of the held-out groups")
    ap.add_argument("--out", default=OUT, help="the folder of the rungs' datasets")
    a = ap.parse_args(argv)
    a.cohort = a.cohort or ["asian", "black"]
    if (a.all and (a.rung or a.disease)) or (not a.all and not (a.rung and a.disease)):
        ap.error("give --rung and --disease, or --all")
    for rung, disease in finished_cells(a.scheduler) if a.all else [(a.rung, a.disease)]:
        score_cell(rung, disease, a)


if __name__ == "__main__":
    main()
