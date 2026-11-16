"""Behaviours of widedeepgp/train.py, tuning.py and callbacks.py that the other CPU tests do not
reach: the search's worker processes (a stand-in script plays main_multilabel.py --tune_worker),
its run lock and GPU discovery, the stage settings, the code version when git fails, which last
checkpoint a resume picks, and the callbacks' logs, snapshots, stopping and pruning, driven by
stand-ins for Lightning's trainer. Everything runs on the CPU in seconds.
    python -m pytest tests/test_ml_training_pieces.py -q
"""
import csv
import json
import os
import re
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

import ml_helpers  # noqa: F401  (puts the repository on sys.path)

# plays main_multilabel.py --tune_worker: prints what it was given; "fails" on GPUs 6 and 9; as the
# linear stage, leaves what a finished grid run leaves
FAKE_WORKER = '''import json, os, sys
keys = ("CUDA_VISIBLE_DEVICES", "WIDEDEEPGP_TUNE_STAGE", "WIDEDEEPGP_TUNE_N", "WIDEDEEPGP_TUNE_WORKER",
        "WIDEDEEPGP_WIDE_INIT")
print(json.dumps({"argv": sys.argv[1:], "env": {k: os.environ.get(k) for k in keys}}), flush=True)
code = 3 if os.environ["CUDA_VISIBLE_DEVICES"] in ("6", "9") else 0
if os.environ["WIDEDEEPGP_TUNE_STAGE"] == "linear" and code == 0:
    d = os.path.join(sys.argv[sys.argv.index("--root") + 1], "linear")
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
    json.dump({"best_val_auroc_mean": 0.61, "best_checkpoint": "/elsewhere/checkpoints/best.ckpt",
               "epoch_equivalents": 3.0, "train_seconds": 12.345}, open(os.path.join(d, "results.json"), "w"))
    json.dump({"model": {"prs_var": 0.3}}, open(os.path.join(d, "config.json"), "w"))
sys.exit(code)
'''


def _search(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER)
    base = types.SimpleNamespace(run_dir=str(tmp_path / "runs"), experiment="e", exp_name="r")
    return str(script), base, tmp_path / "runs" / "e" / "r"


def _worker_logs(folder):
    return {name: [json.loads(line) for line in open(os.path.join(folder, name)).read().splitlines()]
            for name in sorted(os.listdir(folder))}


def test_each_search_worker_gets_its_gpu_stage_and_log(tmp_path, capsys):
    from widedeepgp import tuning
    script, base, root = _search(tmp_path)
    tuning.run_workers(base, ["--a", "b"], script, "joint", 7, 2, ["5", "6", "7"], "/init.ckpt")
    logs = root / "joint" / "workers"
    assert capsys.readouterr().out == f"joint: worker exit codes [0, 3]; logs in {logs}\n"
    got = _worker_logs(logs)
    assert list(got) == ["gpu5.log", "gpu6.log"]                 # at most one worker per trial still to run
    for i, gpu in enumerate(("5", "6")):
        assert got[f"gpu{gpu}.log"] == [{"argv": ["--a", "b", "--tune_worker"], "env": {
            "CUDA_VISIBLE_DEVICES": gpu, "WIDEDEEPGP_TUNE_STAGE": "joint", "WIDEDEEPGP_TUNE_N": "7",
            "WIDEDEEPGP_TUNE_WORKER": str(i), "WIDEDEEPGP_WIDE_INIT": "/init.ckpt"}}]
    tuning.run_workers(base, ["--a", "b"], script, "joint", 7, 5, ["5"], None)
    assert capsys.readouterr().out == ""                         # every worker exited cleanly: nothing to say
    again = _worker_logs(logs)["gpu5.log"]
    assert len(again) == 2 and again[1]["env"]["WIDEDEEPGP_WIDE_INIT"] == ""   # appended; no starting model
    tuning.run_workers(base, [], script, "deep", 3, 0, ["8"], None)            # nothing left to run
    assert capsys.readouterr().out == "" and os.listdir(root / "deep" / "workers") == []


def test_the_linear_stage_runs_once_in_a_worker_on_the_first_gpu(tmp_path, capsys):
    from widedeepgp import tuning
    script, base, root = _search(tmp_path)
    argv = ["--root", str(root)]
    with pytest.raises(RuntimeError, match="did not finish"):    # the worker failed and left no results
        tuning.linear_stage(base, argv, script, ["9", "4"])
    assert capsys.readouterr().out == f"linear: worker exit code 3; log in {root / 'workers'}\n"
    summary = tuning.linear_stage(base, argv, script, ["3", "4"])
    (line,) = _worker_logs(root / "workers")["linear_gpu3.log"]
    assert line == {"argv": argv + ["--tune_worker"], "env": {
        "CUDA_VISIBLE_DEVICES": "3", "WIDEDEEPGP_TUNE_STAGE": "linear", "WIDEDEEPGP_TUNE_N": "1",
        "WIDEDEEPGP_TUNE_WORKER": "0", "WIDEDEEPGP_WIDE_INIT": ""}}
    lin = str(root / "linear")
    assert summary == {"stage": "linear", "n_trials": 1, "n_complete": 1, "n_pruned": 0, "train_seconds": 12.3,
                       "trials": [{"trial": 0, "state": "complete", "value": 0.61, "params": {"prs_var": 0.3},
                                   "run_dir": lin, "epoch_equivalents": 3.0, "train_seconds": 12.345}],
                       "best_trial": 0, "best_value": 0.61, "best_params": {"prs_var": 0.3}, "best_trial_dir": lin,
                       "best_checkpoint": os.path.join(lin, "checkpoints", "best.ckpt")}    # where it is now
    assert tuning.linear_stage(base, argv, script, ["5"]) == summary          # done: no second worker
    assert sorted(os.listdir(root / "workers")) == ["linear_gpu3.log", "linear_gpu9.log"]


def test_one_search_per_run_and_its_lock_is_released_after_an_error(tmp_path):
    import fcntl
    from widedeepgp import tuning
    base = types.SimpleNamespace(run_dir=str(tmp_path), experiment="e", exp_name="r", wide="off", tune_trials=1,
                                 tune_gpus="none", seed=1)
    root = tmp_path / "e" / "r"
    root.mkdir(parents=True)
    (root / "search.json").write_text(json.dumps({"seed": 2}))  # a search started with other arguments
    with pytest.raises(ValueError, match="other arguments: .*--seed 2 then, 1 now"):
        tuning.tune_run(base, [], "unused.py")
    with open(root / ".search.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)            # free again after the error
        with pytest.raises(RuntimeError, match=re.escape(f"another search of {root} is running")):
            tuning.tune_run(base, [], "unused.py")
    assert json.loads((root / "search.json").read_text()) == {"seed": 2}      # left as it was


def test_free_gpus_are_those_without_a_compute_process(monkeypatch):
    from widedeepgp import tuning
    answers = iter(["GPU-b\nGPU-b\n", "0, GPU-a\n1, GPU-b\n2, GPU-c\n"])   # two processes on GPU 1
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=next(answers)))
    assert tuning.free_gpus() == ["0", "2"]
    for error in (FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 60)):
        def fail(*a, error=error, **k):
            raise error
        monkeypatch.setattr(subprocess, "run", fail)
        assert tuning.free_gpus() == []                          # no nvidia-smi: no GPU


def test_the_stage_settings():
    import main_multilabel
    from widedeepgp import tuning
    base = main_multilabel.parse_args(["--label", "x", "--exp_name", "r", "--batch_size", "16", "--stop_patience", "8",
                                       "--resume", "--tune_trials", "5"])
    assert tuning.stages(base) == [("linear", 1), ("joint", 5)]
    lin = tuning.stage_cfg(base, "linear")
    assert (lin.wide, lin.annot, lin.prs_grid, lin.batch_size, lin.accumulate, lin.lr, lin.lr_schedule,
            lin.warmup_steps, lin.stop_patience, lin.val_per_epoch, lin.val_max_epoch_equiv, lin.snapshots_per_decade,
            lin.snapshots_max, lin.exp_name, lin.tune_trials, lin.tune_worker, lin.resume, lin.init_wide_from,
            lin.wide_freeze) == ("only", False, "0.003,0.01,0.03,0.1,0.3,1,3,10,30", 256, 1, 1e-2, "cosine", 0, 10,
                                 1.0, 0.0, 0, 1, os.path.join("r", "linear"), 0, False, False, None, False)
    joint = tuning.stage_cfg(base, "joint", "/lin/best.ckpt")
    assert (joint.wide, joint.init_wide_from, joint.wide_freeze, joint.prs_grid, joint.batch_size, joint.lr,
            joint.exp_name, joint.resume) == ("on", "/lin/best.ckpt", True, None, 16, base.lr, "r", False)
    base.wide, base.stop_patience, base.prs_grid = "off", 0, "0.1,1"
    deep = tuning.stage_cfg(base, "deep")
    assert (deep.wide, deep.prs_grid, deep.stop_patience, deep.init_wide_from) == ("off", None, 0, None)
    lin = tuning.stage_cfg(base, "linear")
    assert (lin.stop_patience, lin.prs_grid) == (0, "0.1,1")    # no early stopping asked for; the grid given
    assert tuning.stages(base) == [("deep", 5)]
    base.wide = "only"
    assert tuning.stages(base) == [("linear", 1)]
    assert base.resume is True and base.batch_size == 16         # the run's own settings are left alone


def test_a_worker_process_runs_what_its_environment_says(monkeypatch):
    from widedeepgp import tuning
    calls = []
    monkeypatch.setattr(tuning, "linear_run", lambda base: calls.append(("linear", base)))
    monkeypatch.setattr(tuning, "worker", lambda *a: calls.append(("worker",) + a))
    for k, v in (("WIDEDEEPGP_TUNE_STAGE", "linear"), ("WIDEDEEPGP_TUNE_N", "1"), ("WIDEDEEPGP_TUNE_WORKER", "0"),
                 ("WIDEDEEPGP_WIDE_INIT", "")):
        monkeypatch.setenv(k, v)
    assert tuning.worker_from_env("cfg") is None
    for k, v in (("WIDEDEEPGP_TUNE_STAGE", "joint"), ("WIDEDEEPGP_TUNE_N", "12"), ("WIDEDEEPGP_TUNE_WORKER", "3"),
                 ("WIDEDEEPGP_WIDE_INIT", "/lin/best.ckpt")):
        monkeypatch.setenv(k, v)
    assert tuning.worker_from_env("cfg") is None
    monkeypatch.setenv("WIDEDEEPGP_WIDE_INIT", "")
    monkeypatch.delenv("WIDEDEEPGP_TUNE_WORKER")
    tuning.worker_from_env("cfg")
    assert calls == [("linear", "cfg"), ("worker", "cfg", "joint", 12, 3, "/lin/best.ckpt"),
                     ("worker", "cfg", "joint", 12, 0, None)]


def test_the_code_version_keeps_the_commit_when_git_status_fails(tmp_path, monkeypatch):
    from widedeepgp.train import code_version
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True)

    git("init", "-q")
    (repo / "m.py").write_text("x = 1\n")
    git("add", "m.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    head = git("rev-parse", "HEAD").stdout.strip()
    real = subprocess.run

    def status_times_out(cmd, *a, **k):
        if "status" in cmd:
            raise subprocess.TimeoutExpired(cmd, 60)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", status_times_out)
    assert code_version(str(repo)) == {"code_dir": os.path.realpath(repo), "git_commit": head, "git_dirty": None,
                                       "python": sys.executable}

    def no_git(*a, **k):
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert code_version(str(repo)) == {"code_dir": os.path.realpath(repo), "git_commit": None, "git_dirty": None,
                                       "python": sys.executable}


def test_a_resume_ranks_last_checkpoints_by_version_number(tmp_path):
    from widedeepgp.train import last_checkpoint
    for name in ("best.ckpt", "last-vx.ckpt", "Last.ckpt", "last.ckpt.tmp", "last-v1.ckpt.part"):
        (tmp_path / name).write_bytes(b"")
    assert last_checkpoint(str(tmp_path)) is None                # none of these is a last checkpoint
    for i, name in enumerate(("last.ckpt", "last-v2.ckpt", "last-v10.ckpt")):
        (tmp_path / name).write_bytes(b"")
        os.utime(tmp_path / name, (1000 + i, 1000 + i))
    assert last_checkpoint(str(tmp_path)) == str(tmp_path / "last-v10.ckpt")   # by number: 10 after 2


def test_prs_grids_as_given_on_the_command_line():
    from widedeepgp.train import parse_grid
    assert parse_grid(None) is None and parse_grid([]) is None and parse_grid("none") is None
    assert parse_grid(" None ") is None and parse_grid("") is None and parse_grid("  ") is None
    assert parse_grid(" 3 , 0.3,") == [0.3, 3.0] and parse_grid("0.1,1,10,1") == [0.1, 1.0, 10.0]
    assert parse_grid((3, 1, 1)) == [1.0, 3.0] and parse_grid([0.1, "2"]) == [0.1, 2.0] and parse_grid(0.5) == [0.5]
    assert parse_grid(",,") == []                                # an empty list, not None
    with pytest.raises(ValueError):
        parse_grid("0.1,x")


def test_snapshot_steps_known_answers():
    from widedeepgp.callbacks import snapshot_steps
    assert snapshot_steps(0) == [] and snapshot_steps(5, per_decade=0) == [5]
    assert snapshot_steps(100, per_decade=2) == [1, 3, 10, 32, 100]
    assert snapshot_steps(10, per_decade=0, every=3) == [3, 6, 9, 10]
    assert snapshot_steps(20, per_decade=0, every=2, max_count=4) == [6, 12, 18, 20]   # every third regular step
    assert snapshot_steps(20, per_decade=0, every=2, max_count=1) == [20]


def test_snapshots_are_saved_once_at_their_steps(tmp_path):
    from widedeepgp.callbacks import Snapshots
    saved = []
    trainer = types.SimpleNamespace(global_step=0, fit_loop=types.SimpleNamespace(total_batch_idx=0),
                                    save_checkpoint=lambda path, weights_only: saved.append(
                                        (os.path.relpath(path, tmp_path), weights_only)))
    mod = types.SimpleNamespace(samples_seen=0)
    snaps = Snapshots(str(tmp_path), [2, 4.0], n_train=8)
    snaps.on_fit_start(trainer, mod)
    assert os.path.isdir(tmp_path / "snapshots")
    for step in range(1, 5):
        trainer.global_step, trainer.fit_loop.total_batch_idx, mod.samples_seen = step, step - 1, 4 * step
        snaps.on_train_batch_start(trainer, mod, None, step - 1)
        snaps.on_train_batch_end(trainer, mod, None, None, step - 1)
    names = [os.path.join("snapshots", f"step_00000000{s}.ckpt") for s in (2, 4)]
    assert saved == [(names[0], True), (names[1], True)]         # weights only
    with open(tmp_path / "snapshots.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(r["step"], r["batches"], r["samples_seen"], r["epoch_equiv"], r["file"]) for r in rows] == [
        ("2", "2", "8", "1.0", names[0]), ("4", "4", "16", "2.0", names[1])]
    assert snaps.train_seconds >= 0 and float(rows[-1]["train_seconds"]) == round(snaps.train_seconds, 2)
    resumed = Snapshots(str(tmp_path), [2, 4, 6], n_train=8)    # a resumed run keeps the table and the time
    resumed.load_state_dict(snaps.state_dict())
    resumed.on_fit_start(trainer, mod)
    for step in (4, 6):                                          # step 4 was saved before the resume
        trainer.global_step = step
        resumed.on_train_batch_end(trainer, mod, None, None, step - 1)
    assert saved[2:] == [(os.path.join("snapshots", "step_000000006.ckpt"), True)]
    assert resumed.train_seconds == snaps.train_seconds
    with open(tmp_path / "snapshots.csv") as f:
        assert [r["step"] for r in csv.DictReader(f)] == ["2", "4", "6"]


class _Loop:
    def __init__(self):
        self.accumulating, self._batches_that_stepped = False, 0

    def _should_accumulate(self):
        return self.accumulating


class _Logger:
    def __init__(self):
        self.calls = []

    def log_metrics(self, metrics, step):
        self.calls.append((metrics, step))


def _module(prs, frozen):
    """A stand-in for the model: a network, PRS weights if ``prs``, fixed gradients."""
    m = torch.nn.Module()
    m.net = torch.nn.Linear(3, 2)
    if prs:
        m.prs = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(4, 2)), torch.nn.Parameter(torch.ones(2, 2))])
    m.samples_seen, m.label_names, m.last_val_metrics, m.logged = 0, ["a", "b"], None, []
    m.log = lambda key, value: m.logged.append((key, value))
    m.log_dict = lambda d: m.logged.append(dict(d))
    m.net.weight.grad, m.net.bias.grad = torch.full((2, 3), 2.0), torch.full((2,), 1.0)     # squares sum to 26
    if prs and not frozen:
        m.prs[0].grad, m.prs[1].grad = torch.full((4, 2), 3.0), torch.full((2, 2), 0.5)     # squares sum to 73
    return m


def _val_metrics(mean):
    return {"labels": ["a", "b"], "auroc": [0.5, 0.7], "ap": [0.1, 0.2], "bce": [0.3, 0.4], "n": np.array([5, 5]),
            "n_cases": np.array([1, 2]), "auroc_mean": mean, "ap_mean": 0.15, "bce_mean": 0.35}


@pytest.mark.parametrize("prs, frozen", [(False, False), (True, False), (True, True)], ids=["network", "prs", "fixed"])
def test_the_run_log(tmp_path, prs, frozen):
    """Step metrics only on the batches Lightning writes, the PRS's learning rate and gradient norm
    apart from the network's, and one row set per validation pass (a repeated pass ignored)."""
    from widedeepgp.callbacks import RunLog
    mod = _module(prs, frozen)
    groups = [{"lr": 0.5, "params": [mod.net.weight, mod.net.bias]}]
    groups += [{"lr": 0.25, "params": list(mod.prs)}] if prs else []
    loop = _Loop()
    trainer = types.SimpleNamespace(fit_loop=types.SimpleNamespace(epoch_loop=loop), log_every_n_steps=2,
                                    optimizers=[types.SimpleNamespace(param_groups=groups)], global_step=0,
                                    sanity_checking=False, logger=_Logger())
    log = RunLog(str(tmp_path), n_train=10, n_snps=6)
    log.on_fit_start(trainer, mod)
    y = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    for b in range(6):                       # every second optimiser step is written (batches 1, 3, 5) ...
        loop._batches_that_stepped, loop.accumulating = b, b == 3          # ... but batch 3 only accumulates
        mod.samples_seen += 2
        log.on_train_batch_end(trainer, mod, None, (None, y), b)
        log.on_before_optimizer_step(trainer, mod, None)
    rates = {"train/lr": 0.5, **({"train/lr_wide": 0.25} if prs else {})}
    norms = [("train/grad_norm", pytest.approx(26 ** 0.5))]
    norms += [("train/grad_norm_wide", pytest.approx(0.0 if frozen else 73 ** 0.5))] if prs else []
    steps = [{k: v for k, v in x.items() if not k.startswith("perf/")} if isinstance(x, dict) else x
             for x in mod.logged]
    assert steps == [rates, *norms, rates, *norms]                       # batches 1 and 5
    perf = {"perf/samples_per_s", "perf/million_sample_snps_per_s"} | (
        {"perf/gpu_peak_gib"} if torch.cuda.is_available() else set())
    assert all(set(x) - set(rates) == perf for x in mod.logged if isinstance(x, dict))
    for step, m in ((4, _val_metrics(0.6)), (4, _val_metrics(0.7)), (8, None), (9, _val_metrics(0.65))):
        trainer.global_step, mod.last_val_metrics = step, m              # step 4 twice: a resumed run's repeat
        log.on_validation_end(trainer, mod)
    assert log.history == [(1, 4, 12, 0.6), (2, 9, 12, 0.65)] and sorted(log.metrics_at) == [4, 9]
    assert log.metrics_at[4]["auroc_mean"] == 0.6
    with open(tmp_path / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    assert [(r["validation"], r["step"], r["epoch_equiv"]) for r in rows] == [("1", "4", "1.2")] * 9 + [
        ("2", "9", "1.2")] * 9
    assert [(r["label"], r["metric"], r["value"]) for r in rows[:9:4]] == [("a", "auroc", "0.5"), ("b", "ap", "0.2"),
                                                                          ("MEAN", "bce", "0.35")]
    seen = {"cases_seen/a": 6.0, "cases_seen/b": 6.0}                   # a missing label counts as no case
    assert trainer.logger.calls == [(seen, 4), ({"epoch_equiv": 1.2}, 4), (seen, 9), ({"epoch_equiv": 1.2}, 9)]
    state = log.state_dict()
    assert state == {"n_validations": 2, "history": [(1, 4, 12, 0.6), (2, 9, 12, 0.65)], "last_step": 9,
                     "cases_seen": [6.0, 6.0]}
    resumed = RunLog(str(tmp_path), 10, 6)
    resumed.load_state_dict(json.loads(json.dumps(state)))              # as a checkpoint holds it
    trainer.global_step, mod.last_val_metrics = 9, _val_metrics(0.9)    # the pass a resumed run repeats
    resumed.on_validation_end(trainer, mod)
    assert resumed.history == log.history and resumed.metrics_at == {} and resumed.n_validations == 2
    assert resumed.cases_seen.dtype == torch.float64 and resumed.cases_seen.tolist() == [6.0, 6.0]


def test_early_stopping_counts_a_repeated_pass_once(monkeypatch):
    from pytorch_lightning.callbacks import EarlyStopping
    from widedeepgp.callbacks import EarlyStoppingOnce
    checks = []
    monkeypatch.setattr(EarlyStopping, "_run_early_stopping_check",
                        lambda self, trainer: checks.append(trainer.global_step))
    es = EarlyStoppingOnce(monitor="val/auroc_mean", mode="max", patience=3)
    trainer = types.SimpleNamespace(global_step=0)
    for step in (10, 10, 20, 20, 30):
        trainer.global_step = step
        es._run_early_stopping_check(trainer)
    assert checks == [10, 20, 30] and es.state_dict()["seen_step"] == 30


def test_the_pruner_rounds_to_the_nearest_pass_and_says_where_it_stopped():
    import optuna
    from widedeepgp.callbacks import OptunaPruning

    class Trial:
        def __init__(self):
            self.reported = []

        def report(self, value, step):
            self.reported.append((step, round(value, 4)))

        def should_prune(self):
            return len(self.reported) == 3

    tr = types.SimpleNamespace(sanity_checking=True, callback_metrics={"val/auroc_mean": torch.tensor(0.61234)},
                               global_step=3)
    cb = OptunaPruning(Trial(), steps_per_pass=7)
    cb.on_validation_end(tr, None)                               # Lightning's sanity check: not a pass
    tr.sanity_checking, tr.callback_metrics = False, {}
    cb.on_validation_end(tr, None)                               # nothing to report yet
    tr.callback_metrics = {"val/auroc_mean": torch.tensor(0.61234)}
    for step in (3, 10):                                         # 3/7 rounds to 0, reported as pass 1; 10/7 to 1
        tr.global_step = step
        cb.on_validation_end(tr, None)
    tr.global_step = 11                                          # 11/7 rounds to 2
    with pytest.raises(optuna.TrialPruned, match=re.escape("pruned at validation 2 (val/auroc_mean = 0.6123)")):
        cb.on_validation_end(tr, None)
    assert cb.trial.reported == [(1, 0.6123), (1, 0.6123), (2, 0.6123)]
    counted = OptunaPruning(Trial())                             # without steps per pass, passes are counted
    for step in (5, 5, 9):                                       # (the repeat of step 5 once)
        tr.global_step = step
        counted.on_validation_end(tr, None)
    assert [s for s, _ in counted.trial.reported] == [1, 2]
