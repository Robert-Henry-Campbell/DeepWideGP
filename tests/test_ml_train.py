"""Training runs end to end: schedule, resume, snapshot scoring, the tuner, and the two
learning checks (one label against the published band; six synthetic traits against the
simulation).

Needs a GPU. The two learning checks take a few minutes each:
    python -m pytest tests/test_ml_train.py -q                 # everything
    python -m pytest tests/test_ml_train.py -q -m "not slow"   # the quick ones
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from ml_helpers import REPO, have_fork, make_cfg, synthetic_dataset

gpu = pytest.mark.skipif(not (torch.cuda.is_available() and have_fork()),
                         reason="needs a GPU and the Vision-Mamba fork of mamba_ssm (the published block)")


def test_schedule():
    from widedeepgp.callbacks import snapshot_steps
    from widedeepgp.train import schedule
    cfg = make_cfg("x", "x", batch_size=16, accumulate=4, val_per_epoch=4.0, max_epoch_equiv=2.0)
    s = schedule(cfg, 1000)                                  # 63 batches per epoch-equivalent
    assert s["batches_per_epoch_equiv"] == 63 and s["batches_per_validation"] % 4 == 0
    assert s["validations"] * s["batches_per_validation"] >= 2 * 63
    assert s["optimizer_steps"] == s["validations"] * s["batches_per_validation"] // 4
    steps = snapshot_steps(1000, per_decade=8, every=250)
    assert steps[0] == 1 and steps[-1] == 1000 and {250, 500, 750}.issubset(steps)
    assert len([x for x in snapshot_steps(1000, per_decade=8) if 100 <= x < 1000]) == 8   # eight per tenfold increase
    assert steps == sorted(set(steps))


def _tiny(tmp_path, data_dir, name, **over):
    return make_cfg(data_dir, "multi", run_dir=str(tmp_path / "runs"), experiment="t", exp_name=name,
                    steps_per_epoch_equiv=12, val_per_epoch=2, stop_patience=0, snapshots_per_decade=4, **over)


@gpu
def test_a_resumed_run_continues_the_same_batches(tmp_path):
    import pytorch_lightning as pl
    from widedeepgp.train import run_training

    class Targets(pl.Callback):
        def __init__(self):
            self.seen = []

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.seen.append((int(batch[2]), float(batch[1].sum()), float(batch[0][0].sum())))

    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    straight = Targets()
    a = run_training(_tiny(tmp_path, data_dir, "straight", max_epoch_equiv=2.0), extra_callbacks=[straight])
    first, second = Targets(), Targets()
    run_training(_tiny(tmp_path, data_dir, "resumed", max_epoch_equiv=1.0), extra_callbacks=[first])
    b = run_training(_tiny(tmp_path, data_dir, "resumed", max_epoch_equiv=2.0, resume=True), extra_callbacks=[second])
    assert len(straight.seen) == 24 and len(first.seen) == 12 and len(second.seen) == 12
    assert first.seen + second.seen == straight.seen          # same targets, same people, same genotypes
    assert b["samples_seen"] == a["samples_seen"] == 24 * 16
    assert [h["validation"] for h in b["history"]] == [1, 2, 3, 4]
    with open(os.path.join(b["run_dir"], "snapshots.csv")) as f:
        rows = list(csv.DictReader(f))
    assert [int(r["step"]) for r in rows] == sorted({int(r["step"]) for r in rows}) and int(rows[-1]["step"]) == 24
    assert json.load(open(os.path.join(b["run_dir"], "config.json")))["resumed"] is True


def _tensorboard_tags(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tags = set()
    for name in os.listdir(os.path.join(run, "tb")):
        if name.startswith("events.out.tfevents"):
            ea = EventAccumulator(os.path.join(run, "tb", name), size_guidance={"scalars": 0})
            ea.Reload()
            tags |= set(ea.Tags()["scalars"])
    return tags


@gpu
def test_scoring_snapshots_and_the_test_lock(tmp_path):
    import score_snapshots
    from widedeepgp.train import run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    out = run_training(_tiny(tmp_path, data_dir, "s", max_epoch_equiv=1.0, accumulate=2))
    run = out["run_dir"]
    assert _tensorboard_tags(run) >= {"train/loss", "train/lr", "train/grad_norm", "perf/samples_per_s",
                                      "val/auroc_mean", "val_auroc/shared_a", "train_loss/solo_e", "cases_seen/null_f",
                                      "epoch_equiv"}
    assert out["gpu_peak_train_gib"] > 0 and out["seconds_per_batch"] > 0
    rows = score_snapshots.main(["--run", run, "--split", "val", "--snapshots", "best"])
    mean = [r for r in rows if r["snapshot"] == "best" and r["label"] == "MEAN" and r["metric"] == "auroc"][0]
    assert abs(mean["value"] - out["best_val_auroc_mean"]) < 1e-6          # a reloaded snapshot scores the same
    assert any(r["snapshot"] == "covariates_only" for r in rows)
    with open(os.path.join(run, "scores_val_summary.csv")) as f:
        summary = list(csv.DictReader(f))
    assert len(summary) == 6 and abs(float(summary[0]["auroc_gain"]) -
                                     (float(summary[0]["auroc"]) - float(summary[0]["auroc_covariates_only"]))) < 1e-5
    all_rows = score_snapshots.main(["--run", run, "--split", "val"])
    assert len({r["snapshot"] for r in all_rows}) == len(os.listdir(os.path.join(run, "snapshots"))) + 1

    with pytest.raises(SystemExit, match="locked"):
        score_snapshots.main(["--run", run, "--split", "test", "--snapshots", "best"])
    assert not os.path.exists(os.path.join(run, "test_access.log"))
    scores = str(tmp_path / "scores")
    score_snapshots.main(["--run", run, "--split", "test", "--snapshots", "best", "--unlock_test", "--save_scores",
                          "--scores_dir", scores])
    assert len(open(os.path.join(run, "test_access.log")).read().strip().splitlines()) == 1
    z = np.load(os.path.join(scores, "t", "s", "test_best.npz"))
    assert z["logit_population"].shape == z["y"].shape == (len(z["iid"]), 6) and str(z["iid"][0]).startswith("s")
    assert not any(f.endswith(".npz") for f in os.listdir(run))            # nothing per-person in the run directory


@gpu
def test_the_tuner_runs_shared_trials_and_prunes(tmp_path):
    import optuna
    from widedeepgp.callbacks import OptunaPruning
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    base = ["--study", "tune", "--n_trials", "2", "--run_dir", str(tmp_path / "runs"), "--min_validations", "1", "--",
            "--data_dir", data_dir, "--label", "multi", "--rd_mode", "store", "--final_pool", "atten", "--d_model", "16",
            "--n_heads", "4", "--batch_size", "16", "--num_workers", "0", "--steps_per_epoch_equiv", "8",
            "--val_per_epoch", "2", "--max_epoch_equiv", "1.5", "--stop_patience", "0"]
    env = dict(os.environ, PYTHONPATH=REPO)
    workers = [subprocess.Popen([sys.executable, os.path.join(REPO, "tune_optuna.py")] + base, cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    logs = [w.communicate(timeout=900)[0] for w in workers]
    assert all(w.returncode == 0 for w in workers), logs[0][-3000:]
    import tune_optuna
    study = tune_optuna.open_study("tune", str(tmp_path / "runs"))
    states = [t.state.name for t in study.trials]
    assert len(states) == 4 and set(states) <= {"COMPLETE", "PRUNED"} and "COMPLETE" in states   # two workers, one study
    assert 0.0 < study.best_value <= 1.0 and {"lr", "l2", "dropout", "accumulate", "warmup_steps"} <= set(study.best_params)
    assert os.path.exists(os.path.join(str(tmp_path / "runs"), "tune", "trial_0000", "metrics.csv"))

    class Trial:                                             # the callback stops a run when the pruner says so
        def __init__(self):
            self.reported = []

        def report(self, value, step):
            self.reported.append((step, value))

        def should_prune(self):
            return len(self.reported) >= 2

    class Tr:
        sanity_checking = False
        callback_metrics = {"val/auroc_mean": torch.tensor(0.6)}
        global_step = 10

    cb, tr = OptunaPruning(Trial()), Tr()
    cb.on_validation_end(tr, None)
    cb.on_validation_end(tr, None)                           # the same validation again (a resumed run): not counted
    assert len(cb.trial.reported) == 1
    tr.global_step = 20
    with pytest.raises(optuna.TrialPruned):
        cb.on_validation_end(tr, None)
    assert cb.trial.reported == [(1, pytest.approx(0.6)), (2, pytest.approx(0.6))]


@gpu
@pytest.mark.slow
def test_one_label_lands_in_the_published_band(tmp_path):
    """The new pipeline with one label on the published smoke dataset, default objective."""
    from widedeepgp import metrics as M
    from widedeepgp.train import predict_split, run_training
    if not os.path.exists(os.path.join(REPO, "pukb", "genes", "1causal_10case", "store", "store.json")):
        pytest.skip("pukb/genes/1causal_10case/store is not built (python -m genostore.from_pickles ...)")
    cfg = make_cfg(os.path.join(REPO, "pukb", "genes"), "1causal_10case", run_dir=str(tmp_path / "runs"),
                   experiment="t", exp_name="one", snp_stats="train", dropout=0.2, lr=7e-4, l2=1e-4, num_workers=4,
                   val_per_epoch=1.0, max_epoch_equiv=3.0, stop_patience=0)
    out = run_training(cfg)
    logits, y = predict_split(out["model"], out["datamodule"], "test")
    auc = M.per_label_metrics(logits, y)["auroc"][0]
    assert 0.72 <= auc <= 0.82, auc                          # the band of tests/test_synthetic_1causal_10case.py


@gpu
@pytest.mark.slow
@pytest.mark.parametrize("annot", [False, True], ids=["plain", "annotated"])
def test_six_synthetic_traits(tmp_path, annot):
    """Heritable traits are learned, the rare one included; the null trait is not 'predicted';
    with annotation vectors that mark the causal SNPs the model reaches the higher bands of
    the annotated seeds. The dataset and settings are those of scripts/multilabel_synthetic.sh."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from check_synthetic_run import bands, check
    from widedeepgp.train import run_training
    data_dir, truth = synthetic_dataset(tmp_path, name="multi6", n=12000, snps=220, seed=7)
    cfg = make_cfg(data_dir, "multi6", run_dir=str(tmp_path / "runs"), experiment="t", exp_name="six",
                   snp_stats="train", dropout=0.2, lr=7e-4, l2=1e-4, num_workers=4, val_per_epoch=2.0,
                   max_epoch_equiv=3.0, stop_patience=0, annot=annot)
    out = run_training(cfg)
    fails = check(out, truth, verbose=False, annot=annot)
    assert not fails, fails
    plain, annotated = bands(False), bands(True)
    assert all(annotated[k] >= plain[k] for k in plain)      # annotations that mark causal SNPs must not cost accuracy
