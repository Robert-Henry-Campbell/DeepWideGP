"""The linear PRS path (--wide) and the search inside every run (widedeepgp/tuning.py).

Everything runs on the CPU except the tests marked gpu (the Mamba kernels): the linear path
alone (--wide only) has no encoder. The learning check is marked slow (a few minutes on the CPU).
    python -m pytest tests/test_ml_wide.py -q
"""
import json
import os

import numpy as np
import pytest
import torch

from ml_helpers import have_fork, make_cfg, synthetic_dataset

gpu = pytest.mark.skipif(not (torch.cuda.is_available() and have_fork()),
                         reason="needs a GPU and the Vision-Mamba fork of mamba_ssm (the published block)")


def _build(data_dir, **over):
    from genostore.datamodule import SNPStoreDataModule
    from widedeepgp.train import build_model, schedule, train_labels
    cfg = make_cfg(data_dir, "multi", **over)
    dm = SNPStoreDataModule(cfg)
    return cfg, dm, build_model(cfg, dm, schedule(cfg, len(train_labels(dm))))


def _batch(dm):
    return dm.decode_batch(next(iter(dm._loader(dm._dataset(dm.splits["train"]), False))))


def test_the_linear_path_is_a_prs_plus_covariates(tmp_path):
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, dm, model = _build(data_dir, wide="only")
    x_all, y = _batch(dm)[:2]
    assert model.n_snps == 154 and sum(w.numel() for w in model.prs) == 154 * model.n_labels
    pi, c1 = model.loss_fn.pi.double(), model.loss_fn.c1.double()
    pw = pi * c1 / (pi * c1 + 1 - pi)
    assert torch.allclose(model.intercept.double(), torch.log(pw / (1 - pw)), atol=1e-6)   # the weighted base rate
    with torch.no_grad():
        assert torch.equal(model(x_all), model.intercept.expand(len(y), -1))     # the PRS and covariates start at zero
        for w in model.prs:
            w.normal_()
        model.cov_linear.normal_()
        model.intercept.normal_()
        prs = sum(x_all[i].squeeze(-1) @ model.prs[i] for i in range(model.n_seq)) / np.sqrt(model.n_snps)
        assert torch.allclose(model(x_all), model.intercept + prs + x_all[-1] @ model.cov_linear, atol=1e-5)
        assert torch.allclose(model.prs_scores(x_all), prs, atol=1e-5)


def test_the_ridge_penalty(tmp_path):
    from widedeepgp.model import prs_penalty
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, _, model = _build(data_dir, wide="only", prs_var=0.2)
    n = int(model.hp["label_counts"]["n"])
    pi, c1 = model.loss_fn.pi.double(), model.loss_fn.c1.double()
    pw = pi * c1 / (pi * c1 + 1 - pi)
    expect = (pw * (1 - pw)) / (pi * (1 - pi)) / (2 * n * 0.2)
    assert torch.allclose(model.prs_lambda.double(), expect, rtol=1e-6)
    order = torch.argsort(pi)
    assert model.prs_lambda[order[0]] > model.prs_lambda[order[-1]]           # the rarest label is shrunk most
    assert torch.allclose(prs_penalty(model.loss_fn, n, 0.8), model.prs_lambda / 4)
    with torch.no_grad():
        for w in model.prs:
            w.fill_(0.5)
    sq = torch.full((model.n_labels,), 0.25 * model.n_snps)
    assert torch.allclose(model.penalty(), (model.prs_lambda * sq).mean())



def test_the_penalty_correction_is_the_curvature_ratio():
    """At the base-rate solution a label's class-weighted loss is r = c/Z^2 times as curved as the
    unweighted likelihood (section 5.2 of methods.md), the factor the ridge penalty carries."""
    import math
    from widedeepgp.losses import WideLoss
    from widedeepgp.model import curvature_ratios
    g = torch.Generator().manual_seed(1)
    n, m = 20000, 4
    x = torch.randn(n, m, generator=g)
    for prevalence in (0.04, 0.1, 0.3):
        y = (torch.rand(n, 1, generator=g) < prevalence).float()
        cases = float(y.sum())
        loss = WideLoss(counts={"cases": [cases], "non_cases": [n - cases], "n": n}, sampler="shuffled")
        pi, c = float(loss.pi[0]), float(loss.c1[0])
        pw = pi * c / (pi * c + 1 - pi)
        weighted = lambda w: loss((math.log(pw / (1 - pw)) + x @ w).unsqueeze(1), y, 0)[0]
        plain = lambda w: torch.nn.functional.binary_cross_entropy_with_logits(math.log(pi / (1 - pi)) + x @ w, y[:, 0])
        hw = torch.autograd.functional.hessian(weighted, torch.zeros(m)).diagonal().sum()
        hp = torch.autograd.functional.hessian(plain, torch.zeros(m)).diagonal().sum()
        assert float(hw / hp) == pytest.approx(float(curvature_ratios(loss)[0]), rel=0.05)

def test_the_optimiser_groups(tmp_path):
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    for mode, n_groups in (("only", 2), ("on", 3)):
        _, _, model = _build(data_dir, wide=mode, lr=3e-4, lr_wide=2e-2, l2=1e-4)
        opt = model.configure_optimizers()
        groups = (opt["optimizer"] if isinstance(opt, dict) else opt).param_groups
        assert len(groups) == n_groups
        if mode == "on":                                 # the deep network first: RunLog reports its rate
            assert (groups[0]["lr"], groups[0]["weight_decay"]) == (3e-4, 1e-4)
        assert (groups[-2]["lr"], groups[-2]["weight_decay"]) == (3e-4, 0.0)      # covariate weights, intercepts
        assert {id(p) for p in groups[-1]["params"]} == {id(p) for p in model.prs}
        assert (groups[-1]["lr"], groups[-1]["weight_decay"]) == (2e-2, 0.0)      # the PRS: own rate, ridge in the loss
        assert sum(p.numel() for g in groups for p in g["params"]) == sum(p.numel() for p in model.parameters())


def test_older_weights_load_with_the_path_at_zero(tmp_path):
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, _, old = _build(data_dir, wide="off")
    _, _, new = _build(data_dir, wide="on")
    left = new.load_weights(old.state_dict())
    assert left and all(k.startswith(("prs.", "cov_linear", "intercept")) for k in left)
    assert all(float(w.abs().sum()) == 0 for w in new.prs) and float(new.cov_linear.abs().sum()) == 0
    assert float(new.intercept.abs().sum()) == 0                     # the old head holds the intercepts
    a, b = old.state_dict(), new.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if not k.startswith("loss_fn."))


def test_the_path_starts_afresh_for_other_diseases(tmp_path):
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, _, ab = _build(data_dir, wide="only", labels="solo_d,null_f")
    _, _, ba = _build(data_dir, wide="only", labels="null_f,solo_d")
    with torch.no_grad():
        for w in ab.prs:
            w.fill_(1.0)
    left = ba.load_weights(ab.state_dict(), allow_new_head=True, source_labels=ab.label_names)
    assert {k.split(".")[0] for k in left} == {"prs", "cov_linear", "intercept"}
    assert all(float(w.abs().sum()) == 0 for w in ba.prs)


def test_combinations_that_are_refused(tmp_path):
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    with pytest.raises(ValueError, match="every output learns"):
        _build(data_dir, wide="on", loss_outputs="target")
    with pytest.raises(ValueError, match="wide must be one of"):
        _build(data_dir, wide="sideways")


def test_validation_interval_and_budget_floors():
    from widedeepgp.train import schedule
    cfg = make_cfg("x", "x", batch_size=16, accumulate=4, val_per_epoch=4.0, max_epoch_equiv=30.0)
    assert schedule(cfg, 2160)["batches_per_validation"] == 36       # a quarter of 135, in whole optimiser steps
    full = schedule(cfg, 215872)
    cfg.val_min_participants, cfg.min_participants = 20000, 400000
    s = schedule(cfg, 2160)
    assert s["batches_per_validation"] * 16 >= 20000 and s["batches_per_validation"] % 4 == 0
    assert s["validations"] * s["batches_per_validation"] * 16 >= 400000
    assert schedule(cfg, 215872) == full                             # at full size the floors do nothing
    cfg.val_max_epoch_equiv = 1.0                                    # but never fewer than one pass per epoch-equivalent
    assert schedule(cfg, 2160)["batches_per_validation"] == 136      # 135 batches, in whole optimiser steps


def test_snapshot_cap():
    from widedeepgp.callbacks import snapshot_steps
    steps = snapshot_steps(100000, per_decade=6, every=3000)
    capped = snapshot_steps(100000, per_decade=6, every=3000, max_count=40)
    assert len(steps) > 40 >= len(capped) and 100000 in capped
    assert set(snapshot_steps(100000, per_decade=6)) <= set(capped)  # every log-spaced step is kept


def test_the_multi_disease_defaults():
    import main_multilabel
    from args_generator import args_initial
    cfg = main_multilabel.parse_args(["--label", "x"])
    assert (cfg.wide, cfg.wide_kappa, cfg.prs_grid, cfg.tune_trials, cfg.tune_gpus, cfg.val_min_participants,
            cfg.val_max_epoch_equiv, cfg.min_participants, cfg.snapshots_per_decade, cfg.snapshots_max, cfg.seed) == \
        ("off", 0.03, None, 12, "auto", 20000, 1.0, 128000, 6, 40, 1)       # the ridge path is off by default (16 Nov 2026)
    cfg = main_multilabel.parse_args(["--label", "x", "--wide", "on", "--tune_trials", "0"])
    assert (cfg.wide, cfg.tune_trials) == ("on", 0)
    legacy = args_initial(["--label", "x", "--rd_mode", "x"])       # the published script's parser is unchanged
    assert (legacy.wide, legacy.tune_trials, legacy.gpus, legacy.snapshots_per_decade) == ("off", 0, 1, 8)


def test_a_run_tunes_itself(tmp_path):
    import main_multilabel
    import score_snapshots
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    argv = ["--data_dir", data_dir, "--label", "multi", "--rd_mode", "store", "--d_model", "16", "--n_heads", "4",
            "--num_workers", "0", "--batch_size", "16", "--run_dir", str(tmp_path / "runs"), "--experiment", "t",
            "--exp_name", "r", "--wide", "only", "--input_mode", "snps", "--prs_grid", "0.1,1,10", "--max_epoch_equiv", "4",
            "--val_min_participants", "0", "--min_participants", "0", "--stop_patience", "0"]
    root = tmp_path / "runs" / "t" / "r"
    main_multilabel.main(argv)
    with open(root / "results.json") as f:
        r = json.load(f)
    lin = r["stages"]["linear"]
    assert r["tuning"] and r["chosen_stage"] == "linear" and lin["n_complete"] == 1
    assert r["best_trial_dir"] == os.path.join("linear", "per_disease")
    grid_run = json.load(open(root / "linear" / "results.json"))
    cfg = json.load(open(root / "linear" / "config.json"))
    assert cfg["args"]["wide"] == "only" and cfg["args"]["batch_size"] == 256 and cfg["model"]["prs_grid"] == [0.1, 1, 10]
    assert cfg["model"]["lr_schedule"] == "cosine" and cfg["model"]["lr_wide"] > 0
    rows = grid_run["per_disease"]
    assert {x["label"] for x in rows} == set(cfg["model"]["label_names"]) and all(x["prs_var"] in (0.1, 1, 10) for x in rows)
    for x in rows:                                       # scoring the chosen model again gives what it was chosen by
        assert x["val_auroc_rescored"] == pytest.approx(x["val_auroc"], abs=1e-6)
    metrics = __import__("pandas").read_csv(root / "linear" / "metrics.csv")
    auc = metrics[metrics.metric == "auroc"]
    for x in rows:                                       # each disease's choice is the best over passes and variances
        mine = auc[auc.label.str.startswith(x["label"] + "@")]
        assert x["val_auroc"] == pytest.approx(mine.value.max(), abs=1e-9)
    assert r["best_val_auroc_mean"] == pytest.approx(np.mean([x["val_auroc"] for x in rows]), abs=1e-6)
    assert not os.path.isdir(root / "linear" / "checkpoints")       # the grid's weights are deleted,
    assert os.path.isfile(root / "linear" / "per_disease" / "checkpoints" / "best.ckpt")   # the chosen model kept
    scored = score_snapshots.main(["--run", str(root), "--split", "val", "--snapshots", "best"])
    mean = [x for x in scored if x["snapshot"] == "best" and x["label"] == "MEAN" and x["metric"] == "auroc"][0]
    assert abs(mean["value"] - r["best_val_auroc_mean"]) < 1e-6      # scoring the run scores its chosen model
    main_multilabel.main(argv)                                       # rerunning continues: nothing left to do
    assert json.load(open(root / "results.json"))["best_val_auroc_mean"] == r["best_val_auroc_mean"]
    import shutil                                                    # a run folder that moved is read where it is
    moved = tmp_path / "runs" / "t" / "moved"
    shutil.copytree(root / "linear", moved / "linear")
    shutil.rmtree(root / "linear")
    main_multilabel.main([a if a != "r" else "moved" for a in argv])
    m = json.load(open(moved / "results.json"))
    assert m["best_val_auroc_mean"] == r["best_val_auroc_mean"]
    assert m["stages"]["linear"]["best_checkpoint"] == str(moved / "linear" / "per_disease" / "checkpoints" / "best.ckpt")
    with pytest.raises(ValueError, match="other arguments"):         # but not with other arguments
        main_multilabel.main(argv + ["--prs_var", "3.0"])


def test_a_resumed_run_scores_the_whole_tuning_set(tmp_path):
    """A resumed run repeats the validation pass its checkpoint was written in, and Lightning
    restores that pass part-way through (on 13 Nov a resumed grid run took every disease's choice
    from one batch of the tuning set). Every pass must be scored on the whole tuning set, so the
    repeat changes nothing: the choices still rescore exactly and the best checkpoint is a pass of
    metrics.csv."""
    import pandas as pd
    import pytorch_lightning as pl
    from widedeepgp.train import run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    scored = []

    class Count(pl.Callback):
        def on_validation_end(self, trainer, pl_module):
            scored.append(int(max(pl_module.last_val_metrics["n"])))

    cfg = make_cfg(data_dir, "multi", wide="only", prs_grid=[0.1, 1, 10], batch_size=16, eval_batch_size=16,
                   run_dir=str(tmp_path / "runs"), experiment="t", exp_name="r", val_per_epoch=1, max_epoch_equiv=2,
                   stop_patience=0, lr=3e-3)
    run_training(cfg, extra_callbacks=[Count()])
    cfg.max_epoch_equiv, cfg.resume = 3, True
    out = run_training(cfg, extra_callbacks=[Count()])
    n_val = len(out["datamodule"].splits["val"]["y"])
    assert len(scored) == 4 and set(scored) == {n_val}                # the repeated pass included
    for r in out["per_disease"]:
        assert r["val_auroc_rescored"] == pytest.approx(r["val_auroc"], abs=1e-9)
    m = pd.read_csv(tmp_path / "runs" / "t" / "r" / "metrics.csv")
    mean = m[(m.label == "MEAN") & (m.metric == "auroc")]
    assert out["best_step"] in set(mean.step) and out["best_val_auroc_mean"] == pytest.approx(mean.value.max(), abs=1e-9)


def test_each_grid_point_trains_as_a_model_of_its_own(tmp_path):
    """A grid column follows exactly the path a model with that prior variance alone would take
    (its penalty and learning rate), so the grid is K runs in one pass over the genotypes."""
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, dm, grid = _build(data_dir, wide="only", prs_var=1.0, prs_grid=[0.3, 3.0], wide_kappa=0.1)
    singles = [_build(data_dir, wide="only", prs_var=t, wide_kappa=0.1)[2] for t in (0.3, 3.0)]
    x_all, y = _batch(dm)[:2]
    opts = [m.configure_optimizers() for m in [grid] + singles]
    for _ in range(6):
        for m, opt in zip([grid] + singles, opts):
            def closure(m=m, opt=opt):                   # what training_step returns (summed over grid points)
                opt.zero_grad()
                yy = y.repeat(1, m.n_grid) if m.grid else y
                loss = (m.loss_fn(m(x_all), yy)[0] + m.penalty()) * m.n_grid
                loss.backward()
                return loss
            m.optimizer_step(0, 0, opt, closure)         # the model's own step (per-column rates)
    n = grid.n_labels
    for k, single in enumerate(singles):
        cols = slice(k * n, (k + 1) * n)
        for wg, ws in zip(grid.prs, single.prs):
            assert torch.allclose(wg[:, cols], ws, rtol=1e-4, atol=1e-7)
        assert torch.allclose(grid.intercept[cols], single.intercept, rtol=1e-5, atol=1e-7)
        assert torch.allclose(grid.cov_linear[:, cols], single.cov_linear, rtol=1e-4, atol=1e-7)
    with pytest.raises(ValueError, match="needs --wide only"):
        _build(data_dir, wide="on", prs_grid=[0.3, 3.0])


def test_the_run_reports_the_network_on_top_whatever_its_value(tmp_path):
    """Robert's decision of 15 Nov 2026: with the linear path on, the run's model is the network
    on top of the linear model even when the linear model alone scores higher; the linear model
    is recorded beside it as an ablation, and both keep their weights."""
    from types import SimpleNamespace
    from widedeepgp import tuning
    base = SimpleNamespace(run_dir=str(tmp_path), experiment="t", exp_name="r")
    root = tmp_path / "t" / "r"

    def trial(name, value):
        d = root / name
        (d / "checkpoints").mkdir(parents=True)
        (d / "checkpoints" / "best.ckpt").write_text("weights")
        (d / "results.json").write_text(json.dumps({"val": {"a": {"auroc": value}}, "val_mean": {"auroc": value}}))
        return str(d)

    lin = trial("linear/per_disease", 0.72)
    joint = [trial(f"joint/trial_{i:04d}", v) for i, v in enumerate((0.71, 0.715))]
    summaries = {
        "linear": {"best_value": 0.72, "best_trial_dir": lin, "best_checkpoint": lin + "/checkpoints/best.ckpt",
                   "best_params": {}, "train_seconds": 1.0,
                   "trials": [{"trial": 0, "run_dir": str(root / "linear")}]},
        "joint": {"best_value": 0.715, "best_trial_dir": joint[1], "best_checkpoint": joint[1] + "/checkpoints/best.ckpt",
                  "best_params": {"lr": 1e-3}, "train_seconds": 2.0,
                  "trials": [{"trial": i, "run_dir": d} for i, d in enumerate(joint)]},
    }
    out = tuning.choose(base, summaries)
    assert out["chosen_stage"] == "joint" and out["best_trial_dir"] == os.path.join("joint", "trial_0001")
    assert out["best_val_auroc_mean"] == 0.715                       # below the linear model alone, reported all the same
    assert out["linear_only"]["best_trial_dir"] == os.path.join("linear", "per_disease")
    assert out["linear_only"]["network_minus_linear_only"] == pytest.approx(-0.005)
    assert json.load(open(root / "results.json"))["chosen_stage"] == "joint"
    assert os.path.isdir(os.path.join(joint[1], "checkpoints")) and os.path.isdir(os.path.join(lin, "checkpoints"))
    assert not os.path.isdir(os.path.join(joint[0], "checkpoints"))  # the other trials' weights go


def test_a_search_replaces_trials_a_crash_left_running(tmp_path):
    import optuna
    import tune_optuna
    from widedeepgp.tuning import fail_stale
    study = tune_optuna.open_study("r", str(tmp_path))
    study.ask()                                                      # a trial that never finished
    assert fail_stale(study) == 1
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


def test_a_trial_a_crash_interrupted_carries_on_from_its_checkpoint(tmp_path, monkeypatch):
    """At full size a trial can run for days: after a crash, a search must not throw it away.
    The interrupted trial is queued again with its settings and continues from its last
    checkpoint in its own folder; one that had written no checkpoint starts again afresh."""
    from types import SimpleNamespace
    import optuna
    import tune_optuna
    from widedeepgp import train, tuning
    import main_multilabel
    base = main_multilabel.parse_args(["--label", "x", "--run_dir", str(tmp_path), "--experiment", "t",
                                       "--exp_name", "r", "--wide", "off"])
    study = tuning.open_stage_study(base, "deep")
    crashed = []
    for i, with_ckpt in enumerate((True, False)):           # two trials a crash left running
        t = study.ask()
        tune_optuna.suggest(t, SimpleNamespace(), "deep")
        d = tmp_path / "t" / "r" / "deep" / f"trial_{t.number:04d}"
        (d / "checkpoints").mkdir(parents=True)
        if with_ckpt:
            (d / "checkpoints" / "last.ckpt").write_text("state")
        t.set_user_attr("run_dir", str(d))
        crashed.append((dict(t.params), str(d), with_ckpt))
    calls = []

    def fake_run_training(cfg, trial=None):
        calls.append({"exp_name": cfg.exp_name, "resume": cfg.resume, "lr": cfg.lr, "l2": cfg.l2, "dropout": cfg.dropout})
        d = os.path.join(cfg.run_dir, cfg.experiment, cfg.exp_name)
        os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
        return {"best_val_auroc_mean": 0.6, "epoch_equivalents": 1.0, "train_seconds": 1.0,
                "best_checkpoint": os.path.join(d, "checkpoints", "best.ckpt")}

    monkeypatch.setattr(train, "run_training", fake_run_training)
    assert tuning.fail_stale(study) == 2
    assert [t.state for t in study.trials[:2]] == [optuna.trial.TrialState.FAIL] * 2
    tuning.worker(base, "deep", 2)
    for (params, d, with_ckpt), call in zip(crashed, calls):
        assert {k: call[k] for k in params} == pytest.approx(params)          # the same settings again
        if with_ckpt:                                                          # in its own folder, resumed
            assert call["resume"] and call["exp_name"] == os.path.relpath(d, str(tmp_path / "t"))
        else:                                                                  # afresh, in a new folder
            assert not call["resume"] and call["exp_name"] != os.path.relpath(d, str(tmp_path / "t"))
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(done) == 2 and done[0].user_attrs["resumes_trial"] == 0


def test_a_search_can_try_the_given_settings_first(tmp_path, monkeypatch):
    """--tune_first_given: the first trial of the network stage uses --lr, --l2 and --dropout as
    given (the classic DeepGP comparator's published settings), the rest are the search's."""
    import main_multilabel
    from widedeepgp import train, tuning
    seen = []

    def fake_run_training(cfg, trial=None):
        seen.append((cfg.lr, cfg.l2, cfg.dropout))
        d = os.path.join(cfg.run_dir, cfg.experiment, cfg.exp_name)
        os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
        with open(os.path.join(d, "results.json"), "w") as f:
            json.dump({"val": {}, "val_mean": {}}, f)
        return {"best_val_auroc_mean": 0.6 + 0.01 * len(seen), "epoch_equivalents": 1.0, "train_seconds": 1.0,
                "best_checkpoint": os.path.join(d, "checkpoints", "best.ckpt")}

    monkeypatch.setattr(train, "run_training", fake_run_training)
    argv = ["--label", "x", "--run_dir", str(tmp_path), "--experiment", "t", "--exp_name", "r", "--wide", "off",
            "--tune_trials", "3", "--tune_gpus", "none", "--tune_first_given", "--lr", "7e-4", "--l2", "1e-4",
            "--dropout", "0.2"]
    out = main_multilabel.main(argv)
    assert seen[0] == pytest.approx((7e-4, 1e-4, 0.2)) and len(seen) == 3
    assert len({tuple(x) for x in seen}) == 3 and out["chosen_stage"] == "deep"


def test_free_workers_take_the_next_trial(tmp_path, monkeypatch):
    """Workers claim trials one at a time until the stage has its total: a slow trial on one
    worker does not leave the others idle (rung 1 of the ladder, 16 Nov 2026), and however many
    workers race for the last trials, exactly the total is started."""
    import threading
    import time as _time
    import main_multilabel
    from widedeepgp import tuning
    base = main_multilabel.parse_args(["--label", "x", "--run_dir", str(tmp_path), "--experiment", "t",
                                       "--exp_name", "r", "--wide", "off"])
    ran = {}

    def fake_objective(trial, base_, stage, init):
        ran.setdefault(threading.current_thread().name, []).append(trial.number)
        _time.sleep(1.5 if trial.number == 0 else 0.02)          # the first trial is slow
        return 0.6

    monkeypatch.setattr(tuning, "objective", fake_objective)
    threads = [threading.Thread(target=tuning.worker, args=(base, "deep", 8, i), name=f"w{i}") for i in range(2)]
    for t in threads:
        t.start()
        _time.sleep(0.2)                                          # w0 claims trial 0 first
    for t in threads:
        t.join()
    assert sorted(n for v in ran.values() for n in v) == list(range(8))
    assert ran["w0"] == [0] and len(ran["w1"]) == 7               # the free worker took all the rest
    ran.clear()                                                   # a race for the last trials: no overshoot
    base.exp_name = "r2"
    threads = [threading.Thread(target=tuning.worker, args=(base, "deep", 5, i), name=f"x{i}") for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    study = tuning.open_stage_study(base, "deep")
    assert len(study.trials) == 5 and sum(len(v) for v in ran.values()) == 5


def test_the_pruner_sees_a_resumed_run_s_pass_numbers():
    from widedeepgp.callbacks import OptunaPruning

    class Trial:
        def __init__(self):
            self.reported = []

        def report(self, value, step):
            self.reported.append(step)

        def should_prune(self):
            return False

    class Tr:
        sanity_checking = False
        callback_metrics = {"val/auroc_mean": torch.tensor(0.6)}

    t = Trial()
    cb, tr = OptunaPruning(t, steps_per_pass=25), Tr()
    for step in (250, 275, 300):                          # a run resumed after pass 10 (25 steps a pass)
        tr.global_step = step
        cb.on_validation_end(tr, None)
    assert t.reported == [10, 11, 12]


def test_the_gpus_a_search_uses(monkeypatch):
    from widedeepgp import tuning
    assert tuning.pick_gpus(None) == tuning.pick_gpus("none") == tuning.pick_gpus("cpu") == []
    assert tuning.pick_gpus("1, 3") == ["1", "3"]
    free = iter([["4"], ["0", "2", "3"]])                           # one free GPU: wait; three: use two of them
    monkeypatch.setattr(tuning, "free_gpus", lambda: next(free))
    assert tuning.pick_gpus("auto", poll_seconds=0) == ["2", "3"]


def test_the_learning_rate_of_the_prs_follows_kappa(tmp_path):
    from widedeepgp.model import curvature_ratios, wide_learning_rate
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, _, model = _build(data_dir, wide="only", prs_var=2.0, wide_kappa=0.05, batch_size=16, accumulate=4)
    r = curvature_ratios(model.loss_fn)
    rbar = float(torch.exp(torch.log(r).mean()))
    assert model.hp["lr_wide"] == pytest.approx(0.05 * 2.0 * np.sqrt(64) / (rbar * np.sqrt(154)))
    assert wide_learning_rate(0.05, 2.0, 64, r, 4 * 154) == pytest.approx(model.hp["lr_wide"] / 2)   # 1/sqrt(M)
    _, _, given = _build(data_dir, wide="only", lr_wide=1e-3)
    assert given.hp["lr_wide"] == 1e-3                                # an explicit rate wins


def test_a_run_can_start_from_a_fixed_linear_path(tmp_path):
    from widedeepgp.train import load_wide_init
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, _, lin = _build(data_dir, wide="only")
    with torch.no_grad():
        for w in lin.prs:
            w.normal_()
        lin.cov_linear.normal_()
    path = str(tmp_path / "linear.ckpt")
    torch.save({"hyper_parameters": {"hp": lin.hp}, "state_dict": lin.state_dict()}, path)
    _, _, joint = _build(data_dir, wide="on", wide_freeze=True)
    load_wide_init(joint, path)
    for a, b in zip(lin.prs, joint.prs):
        assert torch.equal(a, b) and not b.requires_grad             # copied and fixed
    assert torch.equal(lin.cov_linear, joint.cov_linear) and torch.equal(lin.intercept, joint.intercept)
    assert joint.penalty() is None                                   # a fixed PRS carries no penalty
    groups = joint.configure_optimizers()
    groups = (groups["optimizer"] if isinstance(groups, dict) else groups).param_groups
    assert not any(p is w for g in groups for p in g["params"] for w in joint.prs)
    _, _, other = _build(data_dir, wide="on", labels="solo_d,null_f")
    with pytest.raises(ValueError, match="other label_names"):
        load_wide_init(other, path)


def test_resuming_with_the_path_switched_is_refused(tmp_path):
    from widedeepgp.train import check_resume_tables
    path = str(tmp_path / "old.ckpt")
    torch.save({"hyper_parameters": {"hp": {"pos_digest": "p", "seq_len": [3]}}, "state_dict": {}}, path)
    with pytest.raises(ValueError, match="need --wide off"):
        check_resume_tables(path, {"pos_digest": "p", "seq_len": [3], "wide": "on"})

@pytest.mark.slow
def test_the_linear_path_alone_learns_the_synthetic_traits(tmp_path):
    from widedeepgp.train import run_training
    data_dir, _ = synthetic_dataset(tmp_path)                       # 1,200 people, 132 SNPs, six traits
    cfg = make_cfg(data_dir, "multi", wide="only", lr=3e-3, lr_wide=1e-2, prs_var=0.3, run_dir=str(tmp_path / "runs"),
                   experiment="t", exp_name="w", val_per_epoch=2, max_epoch_equiv=8, stop_patience=0)
    au = {k: v["auroc"] for k, v in run_training(cfg)["val"].items()}
    assert np.mean([v for k, v in au.items() if k != "null_f"]) > 0.70
    assert 0.40 < au["null_f"] < 0.60


@gpu
def test_with_the_path_at_zero_the_network_is_unchanged(tmp_path):
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, dm, off = _build(data_dir, wide="off")
    _, _, on = _build(data_dir, wide="on")
    on.load_weights(off.state_dict())
    off, on = off.cuda().eval(), on.cuda().eval()
    x_all = [t.cuda() for t in _batch(dm)[0]]
    with torch.no_grad():
        assert torch.allclose(off(x_all).float(), on(x_all), atol=1e-6)


@gpu
def test_bf16_keeps_the_path_in_float32(tmp_path):
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    _, dm, model = _build(data_dir, wide="on", precision="bf16")
    model = model.cuda()
    x_all = [t.cuda() for t in _batch(dm)[0]]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert model.wide_logits(x_all).dtype == torch.float32
        assert model(x_all).dtype == torch.float32


@gpu
def test_a_run_with_the_path_trains_it(tmp_path):
    from widedeepgp.train import load_model, run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    cfg = make_cfg(data_dir, "multi", wide="on", lr_wide=1e-2, run_dir=str(tmp_path / "runs"), experiment="t",
                   exp_name="on", steps_per_epoch_equiv=12, val_per_epoch=2, max_epoch_equiv=1.0, stop_patience=0)
    out = run_training(cfg)
    model, _ = load_model(out["best_checkpoint"], out["datamodule"])
    assert np.isfinite(out["best_val_auroc_mean"]) and sum(float(w.abs().sum()) for w in model.prs) > 0
