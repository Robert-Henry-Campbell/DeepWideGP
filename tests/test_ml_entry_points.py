"""The entry points and scripts, pinned for the cleanup of 16 Nov 2026 (main_multilabel.py,
tune_optuna.py, score_snapshots.py, args_generator.py and scripts/), all on the CPU.

Training is the linear path alone (--wide only, no encoder) or a stand-in for run_training; the
generators and the schematic are checked against the committed synthetic data and figure.
    python -m pytest tests/test_ml_entry_points.py -q
"""
import csv
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ml_helpers import REPO, make_cfg, synthetic_dataset

# main_multilabel.py's own defaults, spelled out
GIVEN = ["--snp_stats", "train", "--data_dir", "pukb/genes", "--seed", "1", "--input_mode", "blocks",
         "--lr", "1e-4", "--l2", "1e-3", "--dropout", "0.5", "--batch_size", "64", "--lr_schedule", "plateau",
         "--warmup_steps", "0", "--input_mask", "0.4", "--stop_patience", "16", "--stop_burnin", "2000",
         "--tune_min_validations", "8", "--tune_trials", "12", "--tune_gpus", "auto",
         "--val_min_participants", "20000", "--val_max_epoch_equiv", "1", "--min_participants", "128000",
         "--snapshots_per_decade", "6", "--snapshots_max", "40"]


# ---- main_multilabel.py ----

@pytest.mark.parametrize("argv", [
    [], ["--label", "x"],
    ["--label", "x", "--seed", "7", "--snp_stats=store", "--data_dir", "d", "--final_pool", "last"],
    ["--wide", "on", "--tune_trials", "0", "--final_pool", "concat", "--tune_gpus", "1,2", "--prs_grid", "none"],
    ["--input_mode", "snps", "--final_pool", "concat"],
    ["--min_participants", "0", "--val_min_participants", "0", "--exp_name", "r", "--resume", "--data_dir=e"]])
def test_parse_args_is_the_shared_parser_with_the_multi_disease_defaults(argv):
    import main_multilabel
    from args_generator import args_initial
    got = vars(main_multilabel.parse_args(argv))
    want = vars(args_initial(GIVEN + argv))
    if want["final_pool"] == "concat":                   # the parser's default, given or not: mean (blocks) or atten
        want["final_pool"] = "mean" if want["input_mode"] == "blocks" else "atten"
    if want["input_mode"] != "blocks" and not any(a.startswith("--input_mask") for a in argv):
        want["input_mask"] = 0.0                         # the masking default is a blocks-mode default
    assert list(got) == list(want)                       # the order of config.json's and search.json's keys
    assert got == want


def test_what_main_multilabel_changes_in_the_shared_defaults(monkeypatch):
    import main_multilabel
    from args_generator import args_initial
    mine, shared = vars(main_multilabel.parse_args([])), vars(args_initial([]))
    assert {k: (shared[k], mine[k]) for k in shared if shared[k] != mine[k]} == {
        "snp_stats": ("store", "train"), "data_dir": (None, "pukb/genes"), "seed": (33, 1),
        "input_mode": ("snps", "blocks"), "l2": (1e-5, 1e-3), "dropout": (0.25, 0.5), "lr_schedule": ("constant", "plateau"),
        "input_mask": (0.0, 0.4), "stop_patience": (8, 16), "stop_burnin": (0, 2000), "tune_min_validations": (2, 8),
        "tune_trials": (0, 12), "tune_gpus": (None, "auto"),
        "val_min_participants": (0, 20000), "val_max_epoch_equiv": (0.0, 1.0), "min_participants": (0, 128000),
        "snapshots_per_decade": (8, 6), "snapshots_max": (0, 40), "final_pool": ("concat", "mean")}
    monkeypatch.setattr(sys, "argv", ["main_multilabel.py", "--label", "y", "--seed", "5"])
    cfg = main_multilabel.parse_args()                   # the command line when no argv is given
    assert (cfg.label, cfg.seed, cfg.snp_stats, cfg.data_dir) == ("y", 5, "train", "pukb/genes")


def test_main_trains_once_and_reports(monkeypatch, capsys):
    import main_multilabel
    out = {"run_dir": "/runs/t/r", "best_val_auroc_mean": 0.612345, "best_step": 40, "best_samples_seen": 640,
           "epoch_equivalents": 1.5, "train_seconds": 90.0, "stopped_early": True,
           "val": {"a": {"auroc": 0.6, "ap": 0.25, "bce": 0.3},
                   "a_long_label_name": {"auroc": 0.71234, "ap": 0.1, "bce": 0.29}}}
    seen = []
    monkeypatch.setattr(main_multilabel, "run_training", lambda cfg: seen.append(cfg) or out)
    assert main_multilabel.main(["--label", "x", "--tune_trials", "0", "--exp_name", "r"]) is out
    assert seen[0].exp_name == "r" and seen[0].tune_trials == 0 and seen[0].wide == "off"
    assert capsys.readouterr().out == (
        "\nrun directory: /runs/t/r\n"
        "best mean validation AUROC 0.6123 at step 40 (640 participants seen); trained 1.50 epoch-equivalents in "
        "1.5 min (stopped early)\n"
        "label            AUROC      AP     BCE\n"
        "a               0.6000  0.2500  0.3000\n"
        "a_long_label_name  0.7123  0.1000  0.2900\n")
    out["stopped_early"] = False
    main_multilabel.main(["--label", "x", "--tune_trials", "0"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[2].endswith("epoch-equivalents in 1.5 min") and re.fullmatch(r"run_\d{8}_\d{6}", seen[1].exp_name)


def test_main_hands_a_search_to_the_tuner(monkeypatch):
    import main_multilabel
    from widedeepgp import tuning
    calls = []
    monkeypatch.setattr(tuning, "tune_run", lambda cfg, argv, script: calls.append(("run", cfg, argv, script)) or "ran")
    monkeypatch.setattr(tuning, "worker_from_env", lambda cfg: calls.append(("worker", cfg)) or "worked")
    monkeypatch.setattr(main_multilabel, "run_training", lambda cfg: pytest.fail("trained outside the search"))
    argv = ["--label", "x", "--wide", "off"]
    assert main_multilabel.main(argv) == "ran"
    _, cfg, passed, script = calls[-1]
    assert re.fullmatch(r"run_\d{8}_\d{6}", cfg.exp_name) and cfg.tune_trials == 12
    assert passed == argv + ["--exp_name", cfg.exp_name]           # so that the workers share the run's folder
    assert argv == ["--label", "x", "--wide", "off"]                 # the caller's list is left alone
    assert script == os.path.abspath(main_multilabel.__file__)
    assert main_multilabel.main(argv + ["--exp_name", "r"]) == "ran" and calls[-1][2] == argv + ["--exp_name", "r"]
    assert main_multilabel.main(argv + ["--exp_name", "r", "--tune_worker"]) == "worked"
    assert calls[-1][0] == "worker" and calls[-1][1].tune_worker and calls[-1][1].exp_name == "r"
    assert main_multilabel.main(argv + ["--tune_worker", "--tune_trials", "0"]) == "worked"   # a worker never trains alone
    assert len(calls) == 4


# ---- tune_optuna.py ----

def test_the_search_spaces():
    import optuna
    import tune_optuna
    F, C = optuna.distributions.FloatDistribution, optuna.distributions.CategoricalDistribution
    deep = {"lr": F(1e-4, 3e-3, log=True), "l2": F(1e-6, 1e-2, log=True), "dropout": F(0.0, 0.4)}
    optim = dict(deep, accumulate=C([1, 2, 4]), warmup_steps=C([0, 200, 1000]))
    for space, want in (("deep", deep), ("optim", optim)):
        trial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1)).ask()
        cfg = SimpleNamespace()
        tune_optuna.suggest(trial, cfg, space)
        assert list(trial.distributions) == list(want) and trial.distributions == want   # names, ranges, draw order
        assert vars(cfg) == trial.params and list(vars(cfg)) == list(want)


def test_a_study_s_sampler_pruner_and_journal(tmp_path):
    import optuna
    import tune_optuna
    s = tune_optuna.open_study("s", str(tmp_path), 3, 4, seed=5, n_startup_trials=7)
    assert os.path.isfile(tmp_path / "s" / "optuna_journal.log") and s.study_name == "s"
    assert s.direction == optuna.study.StudyDirection.MAXIMIZE and isinstance(s.sampler, optuna.samplers.TPESampler)
    assert (s.sampler._multivariate, s.sampler._constant_liar, s.sampler._n_startup_trials) == (True, True, 7)
    assert isinstance(s.pruner, optuna.pruners.SuccessiveHalvingPruner)
    assert (s.pruner._min_resource, s.pruner._reduction_factor) == (3, 4)
    d = tune_optuna.open_study("d", str(tmp_path))
    assert d.sampler._n_startup_trials == 10 and (d.pruner._min_resource, d.pruner._reduction_factor) == (2, 3)
    assert isinstance(tune_optuna.open_study("n", str(tmp_path), prune=False).pruner, optuna.pruners.NopPruner)

    def first(name, seed):
        trial = tune_optuna.open_study(name, str(tmp_path), seed=seed).ask()
        tune_optuna.suggest(trial, SimpleNamespace(), "deep")
        return trial.params

    assert first("a", 1) == first("b", 1) != first("c", 2)        # the seed sets the draws
    s.ask()
    assert len(tune_optuna.open_study("s", str(tmp_path)).trials) == 1   # an existing study is continued
    assert len(tune_optuna.load_study("s", str(tmp_path)).trials) == 1


def test_the_stand_alone_tuner(tmp_path, monkeypatch, capsys):
    import optuna
    import torch
    import tune_optuna
    seen = []

    def fake_run_training(cfg, trial=None):
        seen.append(cfg)
        if trial.number == 1:
            raise torch.cuda.OutOfMemoryError("fake")
        return {"run_dir": f"/r/{cfg.exp_name}", "epoch_equivalents": 1.0 + trial.number, "train_seconds": 2.5,
                "best_val_auroc_mean": 0.5 + 0.01 * trial.number}

    monkeypatch.setattr(tune_optuna, "run_training", fake_run_training)
    runs = str(tmp_path / "runs")
    out = tune_optuna.main(["--study", "st", "--n_trials", "3", "--run_dir", runs, "--", "--label", "x", "--seed", "3"])
    assert [c.exp_name for c in seen] == ["trial_0000", "trial_0001", "trial_0002"]
    for c in seen:                                       # main_multilabel.py's defaults, then the given arguments
        assert (c.run_dir, c.experiment, c.seed, c.snp_stats, c.wide, c.tune_trials) == (runs, "st", 3, "train", "off", 12)
        assert {"lr", "l2", "dropout", "accumulate", "warmup_steps"} <= set(vars(c))
    with open(os.path.join(runs, "st", "best.json")) as f:
        best = json.load(f)
    assert list(best) == ["study", "trials", "complete", "pruned", "written", "best_value", "best_params", "best_trial",
                          "best_run_dir"]
    assert {k: best[k] for k in ("study", "trials", "complete", "pruned", "best_trial", "best_run_dir")} == {
        "study": "st", "trials": 3, "complete": 2, "pruned": 1, "best_trial": 2, "best_run_dir": "/r/trial_0002"}
    assert best["best_value"] == pytest.approx(0.52) and re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", best["written"])
    assert out == best and capsys.readouterr().out.endswith(json.dumps(out, indent=1) + "\n")
    study = tune_optuna.load_study("st", runs)
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                                              optuna.trial.TrialState.COMPLETE]
    assert study.trials[0].user_attrs == {"run_dir": "/r/trial_0000", "epoch_equivalents": 1.0, "train_seconds": 2.5}
    assert best["best_params"] == study.trials[2].params and list(best["best_params"]) == [
        "lr", "l2", "dropout", "table_kappa", "table_decay", "accumulate", "warmup_steps"]   # the table's: blocks mode
    ref = tune_optuna.open_study("ref", str(tmp_path), seed=1).ask()   # the sampler has the project's seed
    tune_optuna.suggest(ref, SimpleNamespace(input_mode="blocks"), "optim")
    assert ref.params == study.trials[0].params
    def out_of_memory(cfg, trial=None):
        raise torch.cuda.OutOfMemoryError("fake")

    monkeypatch.setattr(tune_optuna, "run_training", out_of_memory)
    none = tune_optuna.main(["--study", "oom", "--n_trials", "1", "--run_dir", runs])
    assert list(none) == ["study", "trials", "complete", "pruned", "written"] and none["pruned"] == 1


# ---- score_snapshots.py ----

@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A finished run of the linear path alone, with its dataset."""
    from widedeepgp.train import run_training
    tmp = tmp_path_factory.mktemp("score")
    data_dir, _ = synthetic_dataset(tmp, n=600, snps=66)
    cfg = make_cfg(data_dir, "multi", wide="only", lr=3e-3, run_dir=str(tmp / "runs"), experiment="t", exp_name="s",
                   steps_per_epoch_equiv=12, val_per_epoch=2, stop_patience=0, snapshots_per_decade=4,
                   max_epoch_equiv=1.0)
    out = run_training(cfg)
    return {"dir": out["run_dir"], "data": os.path.join(data_dir, "multi", "store"), "tmp": tmp,
            "best": out["best_val_auroc_mean"], "labels": list(out["val"])}


def _read(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_scoring_the_best_snapshot(run, capsys):
    import score_snapshots
    rows = score_snapshots.main(["--run", run["dir"], "--snapshots", "best"])
    printed = capsys.readouterr().out.splitlines()
    mean = [r for r in rows if r["snapshot"] == "best" and r["label"] == "MEAN" and r["metric"] == "auroc"][0]
    assert abs(mean["value"] - run["best"]) < 1e-6
    assert [r["snapshot"] for r in rows] == ["best"] * 21 + ["covariates_only"] * 21     # 6 labels and MEAN, 3 metrics
    assert {r["split"] for r in rows} == {"val"} and rows[0]["step"] != "" and rows[0]["epoch_equiv"] == ""
    assert all(r[k] == "" for r in rows[21:] for k in ("step", "samples_seen", "epoch_equiv", "train_seconds"))
    table = _read(os.path.join(run["dir"], "scores_val.csv"))
    assert table[0] == score_snapshots.FIELDS == ["snapshot", "step", "samples_seen", "epoch_equiv", "train_seconds",
                                                 "split", "label", "metric", "value", "n", "n_cases"]
    assert table[1:] == [[str(r[k]) for k in table[0]] for r in rows]
    summary = _read(os.path.join(run["dir"], "scores_val_summary.csv"))
    assert summary[0] == ["snapshot", "label", "n", "n_cases", "auroc", "auroc_covariates_only", "auroc_gain", "bce",
                          "bce_covariates_only"]
    assert [r[:2] for r in summary[1:]] == [["best", name] for name in run["labels"]]
    for r in summary[1:]:
        assert all(re.fullmatch(r"-?\d+\.\d{6}|nan", v) for v in r[4:])
        assert "nan" in r[4:7] or abs(float(r[6]) - (float(r[4]) - float(r[5]))) <= 1.5e-6
    by = {(r["snapshot"], r["label"], r["metric"]): r["value"] for r in rows}
    assert printed[0] == f"best: mean AUROC {by['best', 'MEAN', 'auroc']:.4f}, mean cross-entropy {by['best', 'MEAN', 'bce']:.4f}"
    assert printed[1] == f"covariates only: mean AUROC {by['covariates_only', 'MEAN', 'auroc']:.4f}"
    assert printed[2] == f"wrote {os.path.join(run['dir'], 'scores_val.csv')} (1 snapshots)"


def test_scoring_every_snapshot_or_chosen_steps(run, capsys):
    import score_snapshots
    with open(os.path.join(run["dir"], "snapshots.csv")) as f:
        table = list(csv.DictReader(f))
    assert len(table) >= 2
    rows = score_snapshots.main(["--run", run["dir"], "--split", "val"])
    tags = [os.path.splitext(os.path.basename(r["file"]))[0] for r in table]
    assert list(dict.fromkeys(r["snapshot"] for r in rows)) == tags + ["covariates_only"]
    first = [r for r in rows if r["snapshot"] == tags[0]][0]
    assert (first["step"], first["samples_seen"], first["epoch_equiv"], first["train_seconds"]) == (
        table[0]["step"], table[0]["samples_seen"], table[0]["epoch_equiv"], table[0]["train_seconds"])
    summary = _read(os.path.join(run["dir"], "scores_val_summary.csv"))
    assert {r[0] for r in summary[1:]} == {tags[-1]}                # the last snapshot scored
    steps = ",".join(r["step"] for r in table[::2])
    rows = score_snapshots.main(["--run", run["dir"], "--snapshots", steps])
    assert list(dict.fromkeys(r["snapshot"] for r in rows)) == tags[::2] + ["covariates_only"]
    last = score_snapshots.main(["--run", run["dir"], "--snapshots", "last", "--batch_size", "7"])
    assert last[0]["snapshot"] == "last" and last[0]["step"] != ""
    with pytest.raises(SystemExit, match="no snapshots match"):
        score_snapshots.main(["--run", run["dir"], "--snapshots", "999999"])
    assert "wrote" in capsys.readouterr().out


def test_the_test_split_is_locked_and_per_person_scores_stay_out_of_the_run(run, tmp_path):
    import score_snapshots
    d = str(tmp_path / "s")
    shutil.copytree(run["dir"], d)
    with pytest.raises(SystemExit, match="the test split is locked: pass --unlock_test once model selection is finished"):
        score_snapshots.main(["--run", d, "--split", "test", "--snapshots", "best"])
    assert not os.path.exists(os.path.join(d, "test_access.log"))
    scores = str(tmp_path / "scores")
    for _ in range(2):
        score_snapshots.main(["--run", d, "--split", "test", "--snapshots", "best", "--unlock_test", "--save_scores",
                              "--scores_dir", scores])
    with open(os.path.join(d, "test_access.log")) as f:
        log = f.read().splitlines()
    assert len(log) == 2 and all(re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\tsnapshots=best\tn=1", x) for x in log)
    z = np.load(os.path.join(scores, "t", "s", "test_best.npz"))
    assert sorted(z.files) == ["iid", "labels", "logit_population", "y"] and z["iid"].dtype.kind == "U"
    assert list(z["labels"]) == run["labels"] and z["logit_population"].dtype == z["y"].dtype == np.float32
    assert z["logit_population"].shape == z["y"].shape == (len(z["iid"]), 6) and str(z["iid"][0]).startswith("s")
    assert not any(f.endswith(".npz") for f in os.listdir(d))         # nothing per person in the run directory
    assert os.path.isfile(os.path.join(d, "scores_test_summary.csv"))


def test_per_person_scores_go_beside_the_dataset_by_default(run, tmp_path):
    import score_snapshots
    d = str(tmp_path / "s")
    shutil.copytree(run["dir"], d)
    data = run["data"]
    shutil.copy(os.path.join(data, "samples_test.tsv"), os.path.join(data, "samples_extra.tsv"))
    try:
        rows = score_snapshots.main(["--run", d, "--split", "extra", "--snapshots", "best", "--save_scores"])
        z = np.load(os.path.join(data, "scores", "t", "s", "extra_best.npz"))
        assert oct(os.stat(os.path.join(data, "scores", "t", "s")).st_mode & 0o777) == oct(0o700 & ~_umask())
    finally:
        os.remove(os.path.join(data, "samples_extra.tsv"))
        shutil.rmtree(os.path.join(data, "scores"), ignore_errors=True)
    assert len(z["iid"]) == len(z["y"]) > 0 and z["logit_population"].shape == z["y"].shape
    assert {r["snapshot"] for r in rows} == {"best"} and {r["split"] for r in rows} == {"extra"}   # no covariate model
    assert not os.path.exists(os.path.join(d, "scores_extra_summary.csv"))
    assert os.path.isfile(os.path.join(d, "scores_extra.csv"))


def _umask():
    mask = os.umask(0)
    os.umask(mask)
    return mask


def test_which_snapshots_and_which_folder(tmp_path):
    from score_snapshots import chosen_trial, pick_snapshots
    run = tmp_path / "run"
    os.makedirs(run / "checkpoints")
    assert pick_snapshots(str(run), "all") == [] and pick_snapshots(str(run), "5") == []
    with open(run / "snapshots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "batches", "samples_seen", "epoch_equiv", "train_seconds", "file"])
        for s in (1, 2, 10):
            w.writerow([s, s, 16 * s, s / 10, s, f"snapshots/step_{s:08d}.ckpt"])
    assert [p for p, _ in pick_snapshots(str(run), "all")] == [f"snapshots/step_{s:08d}.ckpt" for s in (1, 2, 10)]
    picked = pick_snapshots(str(run), "10,1")
    assert [p for p, _ in picked] == ["snapshots/step_00000001.ckpt", "snapshots/step_00000010.ckpt"]
    assert picked[0][1]["samples_seen"] == "16"
    assert pick_snapshots(str(run), "best") == [(os.path.join("checkpoints", "best.ckpt"), {
        "snapshot": "best", "step": "", "samples_seen": "", "epoch_equiv": "", "train_seconds": ""})]
    assert pick_snapshots(str(run), "last")[0][0] == os.path.join("checkpoints", "last.ckpt")
    with pytest.raises(ValueError):
        pick_snapshots(str(run), "best,2")
    assert chosen_trial(str(run)) == str(run)                           # no results.json
    with open(run / "results.json", "w") as f:
        json.dump({"tuning": True, "best_trial_dir": os.path.join("joint", "trial_0003")}, f)
    assert chosen_trial(str(run)) == os.path.join(str(run), "joint", "trial_0003")
    (run / "config.json").write_text("{}")
    assert chosen_trial(str(run)) == str(run)                           # a run folder of its own
    os.remove(run / "config.json")
    with open(run / "results.json", "w") as f:
        json.dump({"tuning": False, "best_trial_dir": "x"}, f)
    assert chosen_trial(str(run)) == str(run)


# ---- scripts/check_synthetic_run.py ----

def test_the_bands_of_the_synthetic_check():
    from check_synthetic_run import MEAN, bands
    assert bands(False) == {"shared_a": 0.65, "shared_b": 0.67, "shared_c": 0.68, "solo_d": 0.62, "solo_e": 0.52,
                            MEAN: 0.66}
    assert bands(True) == {"shared_a": 0.75, "shared_b": 0.73, "shared_c": 0.77, "solo_d": 0.66, "solo_e": 0.74,
                           MEAN: 0.73}


def _results(**auroc):
    return {"val": {k: {"auroc": v} for k, v in auroc.items()}}


TRUTH = {"traits": [{"label": k, "oracle_auroc": {"val": 0.8}} for k in
                    ("shared_a", "shared_b", "shared_c", "solo_d", "solo_e", "null_f")]}
GOOD = dict(shared_a=0.70, shared_b=0.70, shared_c=0.70, solo_d=0.70, solo_e=0.60, null_f=0.50)


def test_the_synthetic_check(capsys):
    from check_synthetic_run import check
    assert check(_results(**GOOD), TRUTH) == []
    assert capsys.readouterr().out.splitlines() == [
        "trait        AUROC  ceiling            band",
        "shared_a     0.700    0.800         >= 0.65",
        "shared_b     0.700    0.800         >= 0.67",
        "shared_c     0.700    0.800         >= 0.68",
        "solo_d       0.700    0.800         >= 0.62",
        "solo_e       0.600    0.800         >= 0.52",
        "null_f       0.500    0.800       0.40-0.60",
        "mean over heritable traits 0.680 (>= 0.66; no annotation vectors)"]
    bad = dict(GOOD, solo_e=0.51, null_f=0.61)
    assert check(_results(**bad), TRUTH, verbose=False) == [
        "solo_e: AUROC 0.510 outside >= 0.52", "null_f: AUROC 0.610 outside 0.40-0.60"]
    assert capsys.readouterr().out == ""
    low = dict(GOOD, shared_a=0.66, shared_b=0.67, shared_c=0.68, solo_d=0.62, solo_e=0.53)
    assert check(_results(**low), TRUTH, verbose=False) == ["mean AUROC of the heritable traits 0.632 < 0.66"]
    assert check(_results(**GOOD), TRUTH, annot=True) == [
        "shared_a: AUROC 0.700 outside >= 0.75", "shared_b: AUROC 0.700 outside >= 0.73",
        "shared_c: AUROC 0.700 outside >= 0.77", "solo_e: AUROC 0.600 outside >= 0.74",
        "mean AUROC of the heritable traits 0.680 < 0.73"]
    assert capsys.readouterr().out.splitlines()[1:] == [
        "shared_a     0.700    0.800         >= 0.75   FAIL",
        "shared_b     0.700    0.800         >= 0.73   FAIL",
        "shared_c     0.700    0.800         >= 0.77   FAIL",
        "solo_d       0.700    0.800         >= 0.66",
        "solo_e       0.600    0.800         >= 0.74   FAIL",
        "null_f       0.500    0.800       0.40-0.60",
        "mean over heritable traits 0.680 (>= 0.73; with annotation vectors)"]
    assert check(_results(null_f=0.4, **{k: v for k, v in GOOD.items() if k != "null_f"}), TRUTH, verbose=False) == []


def test_the_synthetic_check_from_the_command_line(tmp_path):
    run = tmp_path / "run"
    os.makedirs(run)
    (tmp_path / "truth.json").write_text(json.dumps(TRUTH))
    script = os.path.join(REPO, "scripts", "check_synthetic_run.py")
    for annot, results, code, last in ((False, GOOD, 0, "PASS"), (True, GOOD, 1, "FAIL: shared_a: AUROC 0.700"),
                                       (False, dict(GOOD, null_f=0.3), 1, "FAIL: null_f: AUROC 0.300 outside 0.40-0.60")):
        (run / "config.json").write_text(json.dumps({"args": {"annot": annot}}))
        (run / "results.json").write_text(json.dumps(_results(**results)))
        p = subprocess.run([sys.executable, script, str(run), str(tmp_path / "truth.json")], capture_output=True,
                           text=True)
        assert p.returncode == code and p.stdout.splitlines()[-1].startswith(last), p.stdout + p.stderr


# ---- the generators and the schematic ----

@pytest.mark.parametrize("name,cases", [("1causal_10case", 800), ("1causal_50case", 4000)])
def test_the_single_disease_generator_reproduces_the_committed_data(tmp_path, capsys, name, cases):
    from generate_synthetic import generate_dataset
    generate_dataset(name, 8000, 100, cases, 1, base_dir=str(tmp_path))
    out = os.path.join(str(tmp_path), name, "ld_all")
    committed = os.path.join(REPO, "pukb", "genes", name, "ld_all")
    assert sorted(os.listdir(out)) == sorted(os.listdir(committed)) == sorted(
        ["genes.pkl", "pos.pkl", "snp_train.pkl", "snp_test.pkl", "label_train.pkl", "label_test.pkl",
         "covar_train.pkl", "covar_test.pkl"])
    for f in os.listdir(out):
        with open(os.path.join(out, f), "rb") as a, open(os.path.join(committed, f), "rb") as b:
            assert a.read() == b.read(), f
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"Dataset '{name}' created in {out}" and lines[1:3] == [
        "  8000 participants, 100 SNPs across 22 chromosomes", f"  {cases} cases / {8000 - cases} controls"]
    assert re.fullmatch(r"  1 causal SNP\(s\) on chromosome 1 at indices \[\d\]", lines[3])
    assert lines[4:] == ["  Train: 6400, Test: 1600", "  SNPs per chromosome: " + str([5] * 12 + [4] * 10)]


def test_the_single_disease_generator_with_ties_and_its_command_line(tmp_path):
    import pickle
    script = os.path.join(REPO, "scripts", "generate_synthetic.py")
    p = subprocess.run([sys.executable, script, "--name", "tiny", "--n_participants", "40", "--n_snps", "30",
                        "--n_cases", "7", "--n_causal", "2", "--rd_mode", "m", "--base_dir", str(tmp_path),
                        "--seed", "3"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    d = tmp_path / "tiny" / "m"
    labels = pickle.load(open(d / "label_train.pkl", "rb")) + pickle.load(open(d / "label_test.pkl", "rb"))
    assert len(labels) == 44 and int(labels[0].sum() + labels[22].sum()) == 7
    assert all(x is not labels[0] and np.array_equal(x, labels[0]) for x in labels[1:22])   # 22 arrays, not one
    snp = pickle.load(open(d / "snp_train.pkl", "rb"))
    assert [a.shape for a in snp] == [(32, 2)] * 8 + [(32, 1)] * 14 and snp[0].dtype == np.float64
    genes = pickle.load(open(d / "genes.pkl", "rb"))
    assert list(genes[0]) == ["rs_syn_0", "rs_syn_1"] and genes[21][0] == "rs_syn_29" and genes[0].dtype == "<U8"
    cov = pickle.load(open(d / "covar_train.pkl", "rb"))
    assert cov.shape == (32, 8) and set(np.unique(cov[:, 1])) <= {0.0, 1.0}
    big = subprocess.run([sys.executable, script, "--name", "x", "--n_participants", "10", "--n_cases", "11",
                          "--base_dir", str(tmp_path)], capture_output=True, text=True)
    assert big.returncode != 0 and "IndexError" in big.stderr          # more cases than participants


def _digest(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


@pytest.mark.skipif(np.__version__ != "1.26.4", reason="the digests are of numpy 1.26.4's random streams")
def test_the_multi_disease_generator(tmp_path):
    from generate_synthetic_multilabel import generate
    out, truth = generate("m", n_participants=600, n_snps=66, base_dir=str(tmp_path), seed=3)
    assert out == os.path.join(str(tmp_path), "m", "store")
    assert {f: _digest([os.path.join(out, f)]) for f in sorted(os.listdir(out)) if f != "genostore"} == DIGESTS
    store = os.path.join(out, "genostore")
    assert _digest(sorted(os.path.join(store, f) for f in os.listdir(store) if f.endswith((".bin", ".npy", ".tsv"))
                          or f == "samples.txt")) == STORE_DIGEST
    with open(os.path.join(out, "truth.json")) as f:
        assert json.dumps(json.load(f)) == json.dumps(truth)       # NaN where a split has no case of solo_e
    assert [t["label"] for t in truth["traits"]] == ["shared_a", "shared_b", "shared_c", "solo_d", "solo_e", "null_f"]
    assert truth["traits"][5]["oracle_auroc"] == {"test": 0.5, "val": 0.5, "train": 0.5}
    p = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "generate_synthetic_multilabel.py"), "--name", "c",
                        "--n_participants", "600", "--n_snps", "66", "--base_dir", str(tmp_path), "--seed", "3"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout.splitlines()[0] == f"wrote {os.path.join(str(tmp_path), 'c', 'store')}"
    assert len(p.stdout.splitlines()) == 8 and p.stdout.splitlines()[1].startswith("  shared_a  prevalence 0.30  h2 0.50")
    assert _digest([os.path.join(str(tmp_path), "c", "store", "truth.json")]) == DIGESTS["truth.json"]


# of the code of d019bb9 (numpy 1.26.4)
DIGESTS = {"annotations.npz": "23b2527d4a9b4369", "labels.json": "5fd9bd20a75bc892", "samples_test.tsv": "563086fe1193ab17",
           "samples_train.tsv": "91f0581dcea08cba", "samples_val.tsv": "7a75fb6c0656be0e",
           "store.json": "d876efa01e4b69a5", "truth.json": "25d133c89f4a1706"}
STORE_DIGEST = "c0a0f6c32029ebab"


def _png_software(path):
    data, i = open(path, "rb").read(), 8
    while i < len(data):
        n, kind = struct.unpack(">I", data[i:i + 4])[0], data[i + 4:i + 8]
        if kind == b"tEXt" and data[i + 8:i + 8 + n].startswith(b"Software\x00"):
            return data[i + 17:i + 8 + n].decode()
        i += 12 + n
    return None


def test_the_schematic_reproduces_the_committed_figure(tmp_path, monkeypatch):
    import matplotlib
    import make_architecture_figure as fig
    monkeypatch.setattr(fig, "OUT", str(tmp_path))
    fig.main()
    name = "widedeepgp_architecture"
    for suffix in ("_data.csv", "_legend.md"):
        with open(os.path.join(str(tmp_path), name + suffix), "rb") as a, \
                open(os.path.join(REPO, "figures", name + suffix), "rb") as b:
            assert a.read() == b.read(), suffix
    committed = os.path.join(REPO, "figures", name + ".png")
    if _png_software(committed) == f"Matplotlib version{matplotlib.__version__}, https://matplotlib.org/":
        with open(os.path.join(str(tmp_path), name + ".png"), "rb") as a, open(committed, "rb") as b:
            assert a.read() == b.read()
    assert os.path.getsize(os.path.join(str(tmp_path), name + ".pdf")) > 0


def test_the_run_recipes_parse():
    for f in sorted(os.listdir(os.path.join(REPO, "scripts"))):
        if f.endswith(".sh"):
            p = subprocess.run(["bash", "-n", os.path.join(REPO, "scripts", f)], capture_output=True, text=True)
            assert p.returncode == 0, (f, p.stderr)
