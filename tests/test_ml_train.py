"""Training runs end to end: schedule, resume, snapshot scoring, the tuner, and the two
learning checks (one label against the published band; six synthetic traits against the
simulation).

The tests marked gpu need a GPU (the Mamba kernels); the others run on the CPU. The two learning
checks take a few minutes each:
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


def test_the_multi_disease_entry_points_scale_snps_on_the_training_split():
    import main_multilabel
    from args_generator import args_initial
    assert main_multilabel.parse_args(["--label", "x"]).snp_stats == "train"
    assert main_multilabel.parse_args(["--label", "x", "--snp_stats", "store"]).snp_stats == "store"
    assert main_multilabel.parse_args(["--snp_stats=store", "--label", "x"]).snp_stats == "store"
    assert main_multilabel.parse_args(["--label", "x"]).seed == 1                 # the project's single seed
    assert main_multilabel.parse_args(["--label", "x", "--seed", "7"]).seed == 7
    assert main_multilabel.parse_args(["--seed=7", "--label", "x"]).seed == 7
    from args_generator import args_initial
    assert args_initial(["--label", "x", "--rd_mode", "x"]).seed == 33                # published default kept
    cfg = main_multilabel.parse_args(["--label", "x"])
    assert cfg.final_pool == "atten" and cfg.data_dir == "pukb/genes"
    assert args_initial(["--label", "x"]).snp_stats == "store"         # main_genome.py keeps the parser's default


def test_a_genotype_only_model_is_refused(tmp_path):
    """WideDeepGP always uses the covariates (methods.md, section 11)."""
    from widedeepgp.train import run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=66)
    for over in ({"dm": "snps"}, {"use_cov": False}):
        with pytest.raises(ValueError, match="genotype-only model"):
            run_training(_tiny(tmp_path, data_dir, "x", **over))
    spec_path = os.path.join(data_dir, "multi", "store", "store.json")
    spec = json.load(open(spec_path))
    json.dump(dict(spec, covariates=[]), open(spec_path, "w"))
    with pytest.raises(ValueError, match="names no covariates"):
        run_training(_tiny(tmp_path, data_dir, "x"))
    assert not os.path.exists(tmp_path / "runs")                        # refused before anything was written


def test_best_is_the_checkpoint_results_json_names(tmp_path):
    """A run copied and then resumed by older code wrote best-v1.ckpt beside the stale best.ckpt."""
    from score_snapshots import pick_snapshots
    run = tmp_path / "run"
    os.makedirs(run / "checkpoints")
    assert pick_snapshots(str(run), "best")[0][0] == os.path.join("checkpoints", "best.ckpt")   # no results.json yet
    json.dump({"best_checkpoint": "/elsewhere/runs/r/checkpoints/best-v1.ckpt"}, open(run / "results.json", "w"))
    (rel, info), = pick_snapshots(str(run), "best")
    assert rel == os.path.join("checkpoints", "best-v1.ckpt") and info["snapshot"] == "best"
    assert pick_snapshots(str(run), "last")[0][0] == os.path.join("checkpoints", "last.ckpt")
    for name, when in (("last.ckpt", 1000), ("last-v1.ckpt", 2000)):    # "last" is where a resume would continue
        (run / "checkpoints" / name).write_bytes(b"")
        os.utime(run / "checkpoints" / name, (when, when))
    assert pick_snapshots(str(run), "last")[0][0] == os.path.join("checkpoints", "last-v1.ckpt")


def test_a_run_records_its_code_and_interpreter(tmp_path):
    from widedeepgp.train import code_version
    git = subprocess.run(["git", "-C", REPO, "rev-parse", "--show-toplevel", "HEAD"], capture_output=True, text=True)
    top_head = git.stdout.splitlines()                                    # a commit only at the top of a checkout,
    expected = (top_head[1] if git.returncode == 0 and len(top_head) == 2  # not for a copy inside another one
                and os.path.realpath(top_head[0]) == os.path.realpath(REPO) else None)
    here = code_version()
    assert {k: here[k] for k in ("code_dir", "python", "git_commit")} == {
        "code_dir": os.path.realpath(REPO), "python": sys.executable, "git_commit": expected}
    assert (here["git_dirty"] is None) if here["git_commit"] is None else (here["git_dirty"] in (True, False))
    copy = code_version(str(tmp_path))                                    # not a git checkout, e.g. a frozen copy
    assert copy["git_commit"] is None and copy["git_dirty"] is None
    repo = tmp_path / "repo"                                              # a checkout whose code is edited after its commit
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)
    run("init", "-q")
    (repo / "m.py").write_text("x = 1\n")
    (repo / "notes.txt").write_text("a\n")
    run("add", "m.py", "notes.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    assert code_version(str(repo))["git_dirty"] is False
    (repo / "notes.txt").write_text("b\n")                                # only .py files count
    assert code_version(str(repo))["git_dirty"] is False
    (repo / "m.py").write_text("x = 2\n")
    assert code_version(str(repo))["git_dirty"] is True


def test_resume_checks_every_table(tmp_path):
    """check_resume_tables branch by branch, on checkpoints that hold only the saved settings."""
    import itertools
    import warnings
    from widedeepgp.losses import WideLoss
    from widedeepgp.train import check_resume_tables
    hp = {"pos_digest": "p", "seq_len": [3, 4], "annot_dim": 0, "annot_digest": None, "label_names": ["a", "b"],
          "label_counts": {"cases": [1.0, 2.0], "non_cases": [9.0, 8.0], "n": 10}, "gamma": 0.5, "w_max": 20.0,
          "case_share": 0.5, "lr": 7e-4, "precision": "32", "total_optimizer_steps": 100, "grad_ckpt": False}
    names = itertools.count()

    def ckpt(drop=(), loss=None, **changes):
        saved = {k: v for k, v in {**hp, **changes}.items() if k not in drop}
        path = str(tmp_path / f"{next(names)}.ckpt")
        state = {f"loss_fn.{k}": v for k, v in loss.state_dict().items()} if loss is not None else {}
        torch.save({"hyper_parameters": {"hp": saved}, "state_dict": state}, path)
        return path

    with warnings.catch_warnings():
        warnings.simplefilter("error")                                   # the same tables: resumes without a word
        check_resume_tables(ckpt(), hp)
        check_resume_tables(ckpt(drop=("label_names", "label_counts")), hp)          # saved before these were checked
        check_resume_tables(ckpt(total_optimizer_steps=50, grad_ckpt=True), hp)      # trains longer, checkpointing on
    refused = [({"pos_digest": "q"}, "other SNP positions"), ({"seq_len": [3, 5]}, "other SNP positions"),
               ({"annot_dim": 5}, "annot_dim 5"), ({"label_names": ["b", "a"]}, "trained on the labels"),
               ({"label_counts": {**hp["label_counts"], "n": 11}}, "other label counts"),
               ({"gamma": 1.0}, r"other settings \(gamma 1.0, now 0.5\)"), ({"lr": 1e-3}, "other settings"),
               ({"precision": "bf16"}, "other settings")]
    for changes, message in refused:
        with pytest.raises(ValueError, match=message):
            check_resume_tables(ckpt(**changes), hp)
    loss = WideLoss(counts=hp["label_counts"])                           # the loss's constants, saved with the weights
    check_resume_tables(ckpt(loss=WideLoss(counts=hp["label_counts"])), hp, loss)
    with pytest.raises(ValueError, match="loss's rate_case differs"):  # saved before case_share was recorded
        check_resume_tables(ckpt(drop=("case_share",), loss=WideLoss(counts=hp["label_counts"], case_share=9 / 17)),
                            hp, loss)
    annotated = {**hp, "annot_dim": 5, "annot_digest": "d"}
    check_resume_tables(ckpt(annot_dim=5, annot_digest="d"), annotated)
    with pytest.raises(ValueError, match="annotation vectors differ"):
        check_resume_tables(ckpt(annot_dim=5, annot_digest="e"), annotated)
    for old in (ckpt(annot_dim=5, annot_digest=None), ckpt(drop=("annot_digest",), annot_dim=5)):
        with pytest.warns(UserWarning, match="holds no digest"):           # saved before the digest existed
            check_resume_tables(old, annotated)


def test_resume_keeps_the_batches_and_snp_scaling(tmp_path):
    """check_resume_args against the config.json a run wrote when it started."""
    from widedeepgp.train import check_resume_args
    cfg = make_cfg("x", "x", batch_size=16, accumulate=4, seed=1, snp_stats="train", max_epoch_equiv=6.0,
                   exp_name="first")
    path = tmp_path / "config.json"
    check_resume_args(str(path), cfg)                                    # no config.json yet: nothing to check
    path.write_text(json.dumps({"args": {k: v for k, v in vars(cfg).items()
                                         if isinstance(v, (str, int, float, bool, type(None)))}}))
    longer = make_cfg("y", "y", batch_size=16, accumulate=4, seed=1, snp_stats="train", max_epoch_equiv=30.0,
                      exp_name="copy", num_workers=4, eval_batch_size=64, stop_patience=20, grad_ckpt=True)
    check_resume_args(str(path), longer)                  # longer, under another name and path, run differently
    for changes, message in [({"batch_size": 18}, "--batch_size 16, now 18"), ({"seed": 2}, "--seed 1, now 2"),
                             ({"accumulate": 2}, "--accumulate"), ({"snp_stats": "store"}, "--snp_stats"),
                             ({"scaling": False}, "--scaling"), ({"val_per_epoch": 2.0}, "--val_per_epoch"),
                             ({"steps_per_epoch_equiv": 12}, "--steps_per_epoch_equiv")]:
        other = make_cfg("x", "x", **{**dict(batch_size=16, accumulate=4, seed=1, snp_stats="train"), **changes})
        with pytest.raises(ValueError, match=message):
            check_resume_args(str(path), other)


def test_a_resumed_run_keeps_its_own_stopping_patience():
    """Lightning would put back the checkpoint's patience; the run's --stop_patience wins."""
    from widedeepgp.callbacks import EarlyStoppingOnce
    first = EarlyStoppingOnce(monitor="m", mode="max", patience=8)
    first.wait_count, first._seen_step = 3, 40
    later = EarlyStoppingOnce(monitor="m", mode="max", patience=20)
    later.load_state_dict(first.state_dict())
    assert (later.patience, later.wait_count, later._seen_step) == (20, 3, 40)


def test_the_pass_a_resume_repeats_is_kept_for_results_but_not_logged_again(tmp_path):
    """The repeated pass (scored whole by the model) goes into metrics_at, so results.json need
    not score the tuning set again, but adds no metrics.csv row, history entry or count."""
    from types import SimpleNamespace
    from widedeepgp import metrics as M
    from widedeepgp.callbacks import RunLog
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=(50, 2)).astype(float)

    def metrics():
        return M.per_label_metrics(rng.normal(size=(50, 2)), y, None, ["a", "b"])

    first = RunLog(str(tmp_path), n_train=100, n_snps=10)
    first.on_fit_start(SimpleNamespace(), None)
    module = SimpleNamespace(last_val_metrics=metrics(), samples_seen=100, label_names=["a", "b"])
    first.on_validation_end(SimpleNamespace(global_step=12, sanity_checking=False, logger=None), module)
    later = RunLog(str(tmp_path), n_train=100, n_snps=10)            # the resumed process
    later.load_state_dict(first.state_dict())
    rows = open(tmp_path / "metrics.csv").read()
    module.last_val_metrics = repeated = metrics()
    later.on_validation_end(SimpleNamespace(global_step=12, sanity_checking=False, logger=None), module)
    assert later.metrics_at == {12: repeated}
    assert (later.n_validations, len(later.history), open(tmp_path / "metrics.csv").read()) == (1, 1, rows)


def test_a_resume_starts_from_the_latest_last_checkpoint(tmp_path):
    """The highest last-vN.ckpt, which must also be the newest of them."""
    from widedeepgp.train import last_checkpoint
    assert last_checkpoint(str(tmp_path / "absent")) is None and last_checkpoint(str(tmp_path)) is None
    for name, when in (("last.ckpt", 1000), ("best-v1.ckpt", 3000), ("last-v1.ckpt", 2000), ("best.ckpt", 500)):
        (tmp_path / name).write_bytes(b"")
        os.utime(tmp_path / name, (when, when))
    assert last_checkpoint(str(tmp_path)) == str(tmp_path / "last-v1.ckpt")
    os.utime(tmp_path / "last.ckpt", (2500, 2500))                       # copied without the files' times
    with pytest.raises(ValueError, match="cannot tell where the run"):
        last_checkpoint(str(tmp_path))


def test_run_training_resumes_from_the_latest_last_checkpoint(tmp_path):
    """The resume reads last-v1.ckpt, not the stale last.ckpt (refused here before any training)."""
    from widedeepgp.train import run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=66)
    ckpts = tmp_path / "runs" / "t" / "r" / "checkpoints"
    ckpts.mkdir(parents=True)
    for name, when in (("last.ckpt", 1000), ("last-v1.ckpt", 2000)):
        torch.save({"hyper_parameters": {"hp": {"pos_digest": name}}, "state_dict": {}}, str(ckpts / name))
        os.utime(ckpts / name, (when, when))
    with pytest.raises(ValueError, match=r"last-v1\.ckpt: it was trained on other SNP positions"):
        run_training(_tiny(tmp_path, data_dir, "r", resume=True))


def test_a_copied_run_directory_carries_on_with_its_best_and_last(tmp_path):
    """RunCheckpoint moves the saved paths into the copy when the copied files are there."""
    from widedeepgp.callbacks import RunCheckpoint
    old, new, partial = tmp_path / "old", tmp_path / "new", tmp_path / "partial"
    new.mkdir()
    partial.mkdir()
    for name in ("best.ckpt", "last.ckpt"):
        (new / name).write_bytes(b"")
    (partial / "last.ckpt").write_bytes(b"")                            # best.ckpt was not copied
    state = {"dirpath": str(old), "best_model_path": str(old / "best.ckpt"), "best_model_score": torch.tensor(0.9),
             "kth_best_model_path": str(old / "best.ckpt"), "kth_value": torch.tensor(0.9),
             "best_k_models": {str(old / "best.ckpt"): torch.tensor(0.9)}, "last_model_path": str(old / "last.ckpt")}
    cb = RunCheckpoint(dirpath=str(new), filename="best", monitor="m", mode="max", save_top_k=1, save_last=True)
    here = lambda name: os.path.join(cb.dirpath, name)
    cb.load_state_dict(dict(state))
    assert (cb.best_model_path, cb.kth_best_model_path, cb.last_model_path) == (
        here("best.ckpt"), here("best.ckpt"), here("last.ckpt"))
    assert float(cb.best_model_score) == pytest.approx(0.9) and list(cb.best_k_models) == [here("best.ckpt")]
    cb = RunCheckpoint(dirpath=str(partial), filename="best", monitor="m", mode="max", save_top_k=1, save_last=True)
    with pytest.warns(UserWarning, match="dirpath has changed"):        # files missing: Lightning's own behaviour
        cb.load_state_dict(dict(state))
    assert cb.best_model_score is None and cb.last_model_path == ""


def test_run_checkpoint_reads_the_state_earlier_runs_saved(tmp_path):
    """Every run of the pilot saved a plain ModelCheckpoint's state, under that class's key; and
    the same folder named another way (here a symlink) is resumed as in place."""
    from pytorch_lightning.callbacks import ModelCheckpoint
    from widedeepgp.callbacks import RunCheckpoint
    kw = dict(filename="best", monitor="m", mode="max", save_top_k=1, save_last=True,
              save_on_train_epoch_end=False, auto_insert_metric_name=False)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    for name in ("best.ckpt", "last.ckpt"):
        (run / "checkpoints" / name).write_bytes(b"")
    (tmp_path / "link").symlink_to(run)
    d = str(run / "checkpoints")
    state = {"dirpath": d, "best_model_path": os.path.join(d, "best.ckpt"), "best_model_score": torch.tensor(0.9),
             "kth_best_model_path": os.path.join(d, "best.ckpt"), "kth_value": torch.tensor(0.9),
             "best_k_models": {os.path.join(d, "best.ckpt"): torch.tensor(0.9)},
             "last_model_path": os.path.join(d, "last.ckpt")}
    assert RunCheckpoint(dirpath=d, **kw).state_key == ModelCheckpoint(dirpath=d, **kw).state_key
    cb = RunCheckpoint(dirpath=str(tmp_path / "link" / "checkpoints"), **kw)
    cb.load_state_dict(dict(state))
    assert float(cb.best_model_score) == pytest.approx(0.9)
    assert cb.last_model_path == os.path.join(cb.dirpath, "last.ckpt")


def test_inspecting_a_study_leaves_its_journal_alone(tmp_path):
    import optuna
    import tune_optuna
    runs = str(tmp_path / "runs")
    lr = optuna.distributions.FloatDistribution(1e-4, 1e-2, log=True)
    tune_optuna.open_study("s", runs).add_trial(optuna.trial.create_trial(params={"lr": 1e-3}, distributions={"lr": lr},
                                                                          value=0.61))
    journal = os.path.join(runs, "s", "optuna_journal.log")
    before = open(journal, "rb").read()
    study = tune_optuna.load_study("s", runs)
    assert len(study.trials) == 1 and study.best_value == pytest.approx(0.61) and study.best_params == {"lr": 1e-3}
    assert open(journal, "rb").read() == before                          # read, not written to
    with pytest.raises(FileNotFoundError):
        tune_optuna.load_study("other", runs)
    assert not os.path.exists(os.path.join(runs, "other"))


@gpu
def test_starting_from_the_weights_of_a_run_on_other_diseases(tmp_path, capsys):
    """--init_from keeps everything but the output layer when the source's diseases differ,
    here the same two in the other order (an output layer of the same shape)."""
    from widedeepgp.train import run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    src = run_training(_tiny(tmp_path, data_dir, "src", max_epoch_equiv=1.0, labels="solo_d,null_f"))
    best = os.path.join(src["run_dir"], "checkpoints", "best.ckpt")
    capsys.readouterr()
    run_training(_tiny(tmp_path, data_dir, "same", max_epoch_equiv=1.0, labels="solo_d,null_f", init_from=best))
    assert "not loaded: nothing" in capsys.readouterr().out
    run_training(_tiny(tmp_path, data_dir, "flipped", max_epoch_equiv=1.0, labels="null_f,solo_d", init_from=best))
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("initialised from")][0]
    assert "combine.layers.3.weight" in line and "combine.layers.3.bias" in line



@gpu
def test_a_copied_run_resumes_as_if_in_place(tmp_path):
    """A run directory copied under another name and resumed keeps its best checkpoint and best
    score, and goes on writing best.ckpt and last.ckpt."""
    import shutil
    from widedeepgp.train import run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    first = run_training(_tiny(tmp_path, data_dir, "orig", max_epoch_equiv=1.0))
    copy = os.path.join(os.path.dirname(first["run_dir"]), "copy")
    shutil.copytree(first["run_dir"], copy)
    out = run_training(_tiny(tmp_path, data_dir, "copy", max_epoch_equiv=2.0, resume=True))
    assert sorted(os.listdir(os.path.join(copy, "checkpoints"))) == ["best.ckpt", "last.ckpt"]
    assert os.path.realpath(os.path.dirname(out["best_checkpoint"])) == os.path.realpath(os.path.join(copy, "checkpoints"))
    assert out["best_val_auroc_mean"] >= first["best_val_auroc_mean"] - 1e-6

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
    config = json.load(open(os.path.join(b["run_dir"], "config.json")))
    assert config["resumed"] is True
    assert config["code_dir"] == os.path.realpath(REPO) and config["python"] == sys.executable and "git_commit" in config


@gpu
def test_a_resumed_run_reports_a_best_step_from_before_the_resume(tmp_path, monkeypatch):
    """results.json of a run whose best pass came before a resume: that step, with the metrics
    metrics.csv holds for it. The resumed run scores the tuning set once, whole, for the pass it
    repeats; that scoring serves results.json when the repeated pass is the best one, and an
    earlier best step is scored again from its checkpoint (this process did not make it)."""
    import pytorch_lightning as pl
    import widedeepgp.train as T

    class NoLearning(pl.Callback):                         # after the resume nothing can improve
        def on_train_start(self, trainer, pl_module):      # (called after the optimiser state is restored)
            for group in trainer.optimizers[0].param_groups:
                group["lr"] = 0.0

    scored = []                                            # (split, people scored) of each scoring
    predict = T.predict_split

    def record(*a, **k):
        logits, y = predict(*a, **k)
        scored.append((a[2], len(y)))
        return logits, y

    monkeypatch.setattr(T, "predict_split", record)
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    first = T.run_training(_tiny(tmp_path, data_dir, "r", max_epoch_equiv=1.0))
    assert scored == []                                    # its best pass was made by the same process
    out = T.run_training(_tiny(tmp_path, data_dir, "r", max_epoch_equiv=2.0, resume=True),
                         extra_callbacks=[NoLearning()])
    n_val = len(out["datamodule"].splits["val"]["y"])
    earlier_best = first["best_step"] < first["history"][-1]["step"]      # not the pass the resume repeats
    assert scored == [("val", n_val)] * (2 if earlier_best else 1) and len(out["history"]) == 4
    assert out["best_step"] == first["best_step"] <= first["history"][-1]["step"]
    with open(os.path.join(out["run_dir"], "metrics.csv")) as f:
        logged = {(r["label"], r["metric"]): float(r["value"]) for r in csv.DictReader(f)
                  if int(r["step"]) == out["best_step"]}
    assert len(logged) == 7 * 3                            # six labels and their mean, three metrics each
    with open(os.path.join(out["run_dir"], "results.json")) as f:          # the file downstream scripts read
        res = json.load(f)
    assert res["best_step"] == first["best_step"]
    got = {(name, k): m[k] for name, m in res["val"].items() for k in m}
    got.update({("MEAN", k): v for k, v in res["val_mean"].items()})
    assert set(got) == set(logged)
    assert all(np.isclose(got[key], logged[key], rtol=0, atol=1e-6, equal_nan=True) for key in got), (got, logged)


@gpu
def test_a_resume_with_rewritten_annotations_is_refused(tmp_path):
    """The annotation vectors are rebuilt from the dataset, not saved with the weights: a run
    must not resume over a rewritten annotation file."""
    from widedeepgp.train import run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    first = run_training(_tiny(tmp_path, data_dir, "a", max_epoch_equiv=1.0, annot=True))
    metrics, config = (os.path.join(first["run_dir"], f) for f in ("metrics.csv", "config.json"))
    before = {f: open(f).read() for f in (metrics, config)}
    path = os.path.join(data_dir, "multi", "store", "annotations.npz")
    original = dict(np.load(path))
    np.savez(path, **{k: v[:, ::-1].copy() for k, v in original.items()})   # the same channels in another order
    with pytest.raises(ValueError, match="annotation vectors differ"):
        run_training(_tiny(tmp_path, data_dir, "a", max_epoch_equiv=2.0, annot=True, resume=True))
    assert {f: open(f).read() for f in (metrics, config)} == before      # nothing was trained, logged or rewritten
    np.savez(path, **original)                                           # the file it was trained with
    for over, message in (({"batch_size": 18}, "started with other arguments"), ({"gamma": 1.0}, "other settings")):
        with pytest.raises(ValueError, match=message):                   # run_training applies the other checks too
            run_training(_tiny(tmp_path, data_dir, "a", max_epoch_equiv=2.0, annot=True, resume=True, **over))
    last = os.path.join(first["run_dir"], "checkpoints", "last.ckpt")
    saved = open(last, "rb").read()
    old = torch.load(last, map_location="cpu", weights_only=False)
    del old["hyper_parameters"]["hp"]["case_share"]                      # as saved before case_share was recorded
    torch.save(old, last)
    with pytest.raises(ValueError, match="loss's rate_case differs"):  # an odd batch: other sampling rates
        run_training(_tiny(tmp_path, data_dir, "a", max_epoch_equiv=2.0, annot=True, resume=True, batch_size=17))
    with open(last, "wb") as f:
        f.write(saved)
    assert {f: open(f).read() for f in (metrics, config)} == before
    run_training(_tiny(tmp_path, data_dir, "a", max_epoch_equiv=2.0, annot=True, resume=True))      # resumes
    assert json.load(open(config))["resumed"] is True
    assert len(open(metrics).read()) > len(before[metrics])


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
            "--val_per_epoch", "2", "--max_epoch_equiv", "1.5", "--stop_patience", "0",
            "--min_participants", "0", "--val_min_participants", "0"]   # the floors are for real training sets
    env = dict(os.environ, PYTHONPATH=REPO)
    workers = [subprocess.Popen([sys.executable, os.path.join(REPO, "tune_optuna.py")] + base, cwd=REPO, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    logs = [w.communicate(timeout=900)[0] for w in workers]
    assert all(w.returncode == 0 for w in workers), logs[0][-3000:]
    import tune_optuna
    journal = os.path.join(str(tmp_path / "runs"), "tune", "optuna_journal.log")
    before = open(journal, "rb").read()
    study = tune_optuna.load_study("tune", str(tmp_path / "runs"))
    states = [t.state.name for t in study.trials]
    assert len(states) == 4 and set(states) <= {"COMPLETE", "PRUNED"} and "COMPLETE" in states   # two workers, one study
    assert 0.0 < study.best_value <= 1.0 and {"lr", "l2", "dropout", "accumulate", "warmup_steps"} <= set(study.best_params)
    assert open(journal, "rb").read() == before            # looking at the study wrote nothing to it
    assert os.path.exists(os.path.join(str(tmp_path / "runs"), "tune", "trial_0000", "metrics.csv"))
    trial = json.load(open(os.path.join(str(tmp_path / "runs"), "tune", "trial_0000", "config.json")))
    assert trial["args"]["snp_stats"] == "train"           # the default of main_multilabel.py

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
