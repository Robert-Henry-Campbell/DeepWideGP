"""score_genome.py, the scorer of the classic DeepGP ladder: a run's settings rebuilt as its scheduler launched it,
the tuning-set check that guards every write, owner-only outputs in the shared layout, resumption, refusals, and
main_genome.build, which the scorer shares with the training script.

The CPU tests stand a stub in for the network (its Mamba kernels need CUDA): the same last layer (combine) and
sigmoid (opt), on the data module's real batches. The GPU test trains a small classic model with main_genome.py and
re-scores it to its record, with a held-out copy of the tuning people scored exactly like them.

Run:  CUDA_VISIBLE_DEVICES= python -m pytest tests/test_score_genome.py -q
GPU:  CUDA_VISIBLE_DEVICES=<gpu> python -m pytest tests/test_score_genome.py -q -k trained
"""
import copy
import json
import os
import pickle
import shutil
import stat
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from ml_helpers import REPO, have_fork
from test_genostore_heldout import held_out_copy

import main_genome            # importing it must not train: the script part runs only as __main__
import score_genome as sg

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
RUNG, DISEASE = "store", "shared_a"
ARGS = ["--dm", "snps_covs", "--exp_name", "final_model", "--snp_embed", "cov", "--final_pool", "atten",
        "--batch_size", "32", "--d_model", "16", "--e_layers", "1", "--dropout", "0.2", "--lr", "7e-4",
        "--n_heads", "4", "--n_fcn", "2", "--l2", "1e-4", "--patience", "10", "--cov_dim", "8",
        "--snp_stats", "train", "--balanced_batches", "--seed", "1", "--num_workers", "0"]


def ladder(tmp_path, n=600, snps=66):
    """The ladder's layout in miniature: <D>/datasets/<disease>/<rung> (the six-trait synthetic dataset, named after
    the disease as the ladder's are), <D>/work/<rung>/<disease>, a schedule.py whose ARGS are ARGS (running it would
    fail, as the real one does without its arguments), and a held-out copy of the tuning people that is not the
    dataset's sibling folder, so the scorer finds it only through --held_out_dir."""
    from generate_synthetic_multilabel import generate
    D = tmp_path / "ladder"
    generate("multi", n_participants=n, n_snps=snps, base_dir=str(D / "datasets"), seed=3)
    os.symlink("multi", D / "datasets" / DISEASE)
    held = str(tmp_path / "held_out")
    shutil.move(held_out_copy(str(D / "datasets"))[0], held)
    sched = tmp_path / "runs" / "schedule.py"
    sched.parent.mkdir()
    sched.write_text(f"import sys\nR, D, CODE = sys.argv[1:4]\nARGS = {ARGS!r}\nRUNGS = [{RUNG!r}]\n")
    (D / "work" / RUNG / DISEASE).mkdir(parents=True)
    return D, sched, held


class Stub(torch.nn.Module):
    """The classic network's interface: a last layer `combine` whose output goes through the sigmoid `opt`, here on
    each chromosome's mean standardised dosage and the covariates."""

    def __init__(self, n_cov):
        super().__init__()
        self.combine = torch.nn.Linear(22 + n_cov, 1)
        self.opt = torch.nn.Sigmoid()

    def features(self, x_all):
        return torch.cat([x.mean(1) for x in x_all[:-1]] + [x_all[-1]], 1)

    def forward(self, x_all):
        return self.opt(self.combine(self.features(x_all)))


def stub_build(configs):
    dm, _ = main_genome.build(configs)        # the real data module (and a real model, unused on the CPU)
    torch.manual_seed(0)                      # other weights than the run's: the scorer must load the checkpoint's
    return dm, Stub(dm.n_cov)


def fake_run(D):
    """What main_genome.py leaves for a finished cell, made with a stub: the best checkpoint (weights and settings),
    preds_val.pkl and results.json from the training code's own validation loader."""
    configs = sg.cell_configs(RUNG, DISEASE, str(D), ARGS)
    dm, _ = main_genome.build(configs)
    torch.manual_seed(5)
    stub = Stub(dm.n_cov).eval()
    dm.setup("fit")
    y, logit = [], []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            x_all, yb = dm.decode_batch(batch)
            y.append(yb)
            logit.append(stub.combine(stub.features(x_all)))
    y, logit = torch.cat(y).numpy(), torch.cat(logit).numpy()
    p = torch.sigmoid(torch.from_numpy(logit)).numpy()
    work = D / "work" / RUNG / DISEASE
    ck = work / "DeepGP/logs/DeepGP_snps_covs" / DISEASE / "final_model/checkpoints/epoch=8-step=300.ckpt"
    ck.parent.mkdir(parents=True)
    torch.save({"state_dict": stub.state_dict(), "hyper_parameters": {"configs": configs}}, ck)
    res = work / main_genome.results_folder(configs)
    res.mkdir(parents=True)
    with open(res / "preds_val.pkl", "wb") as f:
        pickle.dump((y, p), f)
    from sklearn.metrics import roc_auc_score
    with open(res / "results.json", "w") as f:
        json.dump({"val_auroc": float(roc_auc_score(y, p)), "best_checkpoint": str(ck.relative_to(work))}, f)
    return dict(y=y, p=p, logit=logit, res=res, ck=ck)


def score(D, sched, held, out, *extra):
    sg.main(["--rung", RUNG, "--disease", DISEASE, "--ladder", str(D), "--scheduler", str(sched), "--held_out_dir", held,
             "--out", str(out), "--cohort", "heldx", "--device", "cpu", *extra])
    return out / RUNG / "scores" / "classic_deepgp" / "ladder" / DISEASE


def val_ids(D):
    path = D / "datasets" / DISEASE / RUNG / "samples_val.tsv"
    return pd.read_csv(path, sep="\t", dtype={"IID": str})["IID"].to_numpy(str)


def test_settings_are_rebuilt_as_the_scheduler_launched_the_run(tmp_path):
    D, sched, _ = ladder(tmp_path)
    args = sg.scheduler_value(str(sched), "ARGS")                 # read, not run
    assert args == ARGS and sg.scheduler_value(str(sched), "RUNGS") == [RUNG]
    with pytest.raises(KeyError, match="sets no GATE"):
        sg.scheduler_value(str(sched), "GATE")
    c = sg.cell_configs(RUNG, DISEASE, str(D), args)
    assert (c.label, c.labels, c.rd_mode, c.data_dir) == (DISEASE, DISEASE, RUNG, f"{D}/datasets")   # schedule.launch
    assert (c.batch_size, c.d_model, c.lr, c.seed, c.snp_stats, c.balanced_batches, c.cov_dim, c.exp_name) == (
        32, 16, 7e-4, 1, "train", True, 8, "final_model")
    dm, model = main_genome.build(c)
    assert dm.dir == f"{D}/datasets/{DISEASE}/{RUNG}" and dm.label_names == [DISEASE]
    assert c.enc_len == 66 and c.enc_len_chr == [3] * 22 and len(c.pos) == 22 and dm.train_sampler is not None
    assert isinstance(model, main_genome.DeepGP.SNPCOV_Model_mamba)

    saved = copy.deepcopy(c)
    assert sg.differing(saved, c) == []
    newer = copy.deepcopy(c)
    newer.label_norm = "none"                    # an option added to the parser after the run: not in its checkpoint
    assert sg.differing(saved, newer) == []
    del newer.lr
    assert sg.differing(saved, newer) == ["lr"]
    other = copy.deepcopy(c)
    other.pos = [a.copy() for a in c.pos]
    other.pos[4][1] += 1
    other.d_model = 32
    assert sg.differing(saved, other) == ["d_model", "pos"]
    other.pos = c.pos[:-1]
    assert "pos" in sg.differing(saved, other)


@pytest.mark.skipif(not os.path.exists(sg.SCHEDULER), reason="the ladder's scheduler is not on this machine")
def test_the_ladders_own_scheduler_gives_the_published_settings():
    c = sg.cell_configs("hm3plus_n13199", "I10", sg.LADDER, sg.scheduler_value(sg.SCHEDULER, "ARGS"))
    assert (c.dm, c.d_model, c.e_layers, c.n_heads, c.batch_size, c.lr, c.l2, c.dropout, c.final_pool, c.cov_dim) == (
        "snps_covs", 32, 1, 4, 64, 7e-4, 1e-4, 0.2, "atten", 12)
    assert (c.seed, c.balanced_batches, c.snp_stats, c.exp_name, c.data_dir) == (
        1, True, "train", "final_model", f"{sg.LADDER}/datasets")
    assert sg.scheduler_value(sg.SCHEDULER, "RUNGS")[0] == "hm3plus_n13199"


def test_counts_of_one_to_nine_are_written_below_ten():
    assert [sg.count(n) for n in (0, 1, 9, 10, 20000)] == ["0", "<10", "<10", "10", "20,000"]


def test_a_reproduced_cell_writes_owner_only_scores_of_the_tuning_set_and_each_cohort(tmp_path, monkeypatch, capsys):
    D, sched, held = ladder(tmp_path)
    run = fake_run(D)
    monkeypatch.setattr(sg, "build", stub_build)
    out = tmp_path / "data"
    cell = score(D, sched, held, out)
    assert sorted(os.listdir(cell)) == ["heldx.npz", "val.npz"]              # no temporary file left
    for d in (cell, cell.parent, cell.parent.parent, cell.parent.parent.parent, out / RUNG, out):   # all made here
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
    v, h = (np.load(cell / f"{s}.npz", allow_pickle=False) for s in ("val", "heldx"))
    for name, z in (("val", v), ("heldx", h)):
        assert stat.S_IMODE(os.stat(cell / f"{name}.npz").st_mode) == 0o600
        assert sorted(z.files) == ["iid", "labels", "logits", "y"]
        assert z["iid"].dtype.kind == "U" and z["labels"].dtype.kind == "U" and z["labels"].tolist() == [DISEASE]
        assert z["y"].dtype == z["logits"].dtype == np.float32
        assert z["y"].shape == z["logits"].shape == (len(z["iid"]), 1) == (90, 1)
    ids = val_ids(D)
    assert np.array_equal(v["iid"], ids) and np.array_equal(h["iid"], np.char.add("H", ids))     # loader order
    assert np.array_equal(v["y"], run["y"])
    assert np.array_equal(v["logits"], run["logit"])           # the input of the sigmoid: p = sigmoid(logit)
    assert np.allclose(1 / (1 + np.exp(-v["logits"].astype(np.float64))), run["p"], rtol=0, atol=1e-7)
    # the same people under other names, in a copy of the store: prepared with the run's statistics, scored alike
    assert np.array_equal(h["logits"], v["logits"]) and np.array_equal(h["y"], v["y"])
    log = open(os.path.join(held, "access.log")).read().splitlines()
    assert log and all("\tsplit=heldx\t" in line for line in log)
    printed = capsys.readouterr().out
    assert "tuning set, 90 people" in printed and "reproduced within 1e-05" in printed and "heldx: 90 people" in printed
    metric_lines = [line for line in printed.splitlines() if "AUROC" in line]
    assert len(metric_lines) == 1 and "tuning set" in metric_lines[0]                  # no metric on a cohort


WHY = {"auroc": "the AUROC of the probabilities", "probabilities": "<10 of 90 probabilities differ from preds_val.pkl",
       "labels": "its people or labels differ", "people": "its people or labels differ",
       "weights": "90 of 90 probabilities differ", "settings": "settings ['lr'] differ", "positions": "settings ['pos'] differ"}


@pytest.mark.parametrize("tamper", list(WHY))
def test_a_cell_that_does_not_reproduce_its_record_writes_nothing(tmp_path, monkeypatch, capsys, tamper):
    D, sched, held = ladder(tmp_path)
    run = fake_run(D)
    y, p = run["y"].copy(), run["p"].copy()
    if tamper == "auroc":
        rec = json.load(open(run["res"] / "results.json"))
        rec["val_auroc"] += 2 * sg.TOL
        json.dump(rec, open(run["res"] / "results.json", "w"))
    elif tamper in ("weights", "settings", "positions"):
        ck = torch.load(run["ck"], weights_only=False)
        cfg = ck["hyper_parameters"]["configs"]
        if tamper == "weights":            # every logit moved alike: the AUROC holds, the probabilities do not
            ck["state_dict"]["combine.bias"] += 0.5
        elif tamper == "settings":
            cfg.lr = 1e-3
        else:
            cfg.pos = [cfg.pos[0] + 1] + list(cfg.pos[1:])
        torch.save(ck, run["ck"])
    else:
        if tamper == "probabilities":
            p[:3] += 1e-3
        elif tamper == "labels":
            y[int(np.argmax(y[:, 0] == 1))] = 0
        else:
            y, p = y[:-1], p[:-1]
        with open(run["res"] / "preds_val.pkl", "wb") as f:
            pickle.dump((y, p), f)
    monkeypatch.setattr(sg, "build", stub_build)
    with pytest.raises(SystemExit) as e:
        score(D, sched, held, tmp_path / "data")
    assert "nothing written" in str(e.value.code)                   # a message: the exit status is 1
    said = str(e.value.code) + capsys.readouterr().out
    assert WHY[tamper] in said
    if tamper == "weights":
        assert "AUROC of" not in said                                # an AUROC-only check would have passed it
    assert not (tmp_path / "data").exists()
    assert not os.path.exists(os.path.join(held, "access.log"))     # no held-out person was read


def test_differences_within_the_tolerance_pass(tmp_path, monkeypatch):
    D, sched, held = ladder(tmp_path)
    run = fake_run(D)
    rec = json.load(open(run["res"] / "results.json"))
    rec["val_auroc"] += sg.TOL / 2
    json.dump(rec, open(run["res"] / "results.json", "w"))
    p = run["p"].copy()
    p[:3] += sg.TOL / 2
    with open(run["res"] / "preds_val.pkl", "wb") as f:
        pickle.dump((run["y"], p), f)
    monkeypatch.setattr(sg, "build", stub_build)
    assert sorted(os.listdir(score(D, sched, held, tmp_path / "data"))) == ["heldx.npz", "val.npz"]


def test_check_only_writes_nothing_and_reads_no_held_out_person(tmp_path, monkeypatch, capsys):
    D, sched, held = ladder(tmp_path)
    fake_run(D)
    monkeypatch.setattr(sg, "build", stub_build)
    score(D, sched, held, tmp_path / "data", "--check_only")
    assert "reproduced within" in capsys.readouterr().out
    assert not (tmp_path / "data").exists() and not os.path.exists(os.path.join(held, "access.log"))


def test_outputs_that_exist_are_kept_unless_forced(tmp_path, monkeypatch, capsys):
    D, sched, held = ladder(tmp_path)
    fake_run(D)
    monkeypatch.setattr(sg, "build", stub_build)
    out = tmp_path / "data"
    cell = score(D, sched, held, out)
    first = os.stat(cell / "val.npz")
    os.remove(cell / "heldx.npz")
    score(D, sched, held, out)                         # checked again, then only the missing file is written
    again = os.stat(cell / "val.npz")
    assert (again.st_ino, again.st_mtime_ns) == (first.st_ino, first.st_mtime_ns) and (cell / "heldx.npz").exists()
    capsys.readouterr()
    monkeypatch.setattr(sg, "build", lambda c: pytest.fail("a cell with every output was loaded"))
    score(D, sched, held, out)
    assert "every output exists; skipped" in capsys.readouterr().out
    monkeypatch.setattr(sg, "build", stub_build)
    score(D, sched, held, out, "--force")
    assert os.stat(cell / "val.npz").st_ino != first.st_ino          # replaced (written beside the old file, renamed)
    assert sorted(os.listdir(cell)) == ["heldx.npz", "val.npz"]


def test_scores_are_refused_inside_the_run_or_a_git_work_tree(tmp_path, monkeypatch):
    D, sched, held = ladder(tmp_path)
    monkeypatch.setattr(sg, "build", lambda c: pytest.fail("loaded before the place was checked"))
    with pytest.raises(SystemExit, match="inside the run directory"):
        score(D, sched, held, D / "work" / RUNG / DISEASE / "scores")
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    with pytest.raises(SystemExit, match="inside the git repository"):
        score(D, sched, held, repo)
    assert not os.path.exists(repo / RUNG)


def test_all_scores_every_finished_cell_in_rung_order(tmp_path, monkeypatch):
    sched = tmp_path / "schedule.py"
    sched.write_text("RUNGS = ['r1', 'r2']\nARGS = []\n")
    (tmp_path / "results").mkdir()
    for name in ("r2__B.json", "r1__I84+K64.json", "r1__A.json", "gate__sim200.json", "notes.txt"):
        (tmp_path / "results" / name).write_text("{}")
    cells = [("r1", "A"), ("r1", "I84+K64"), ("r2", "B")]          # the gate cell and other files are not cells
    assert sg.finished_cells(str(sched)) == cells
    calls = []
    monkeypatch.setattr(sg, "score_cell", lambda r, d, a: calls.append((r, d, a.cohort, a.held_out_dir, a.out)))
    sg.main(["--all", "--scheduler", str(sched)])
    assert calls == [(r, d, ["asian", "black"], sg.HELD_OUT, sg.OUT) for r, d in cells]
    assert sg.HELD_OUT == "/mnt/sdg/robert/deepwidegp/data/datasets/ard30/hm3plus_asian_black"
    for bad in ([], ["--rung", "r1"], ["--all", "--rung", "r1", "--disease", "A"]):
        with pytest.raises(SystemExit):
            sg.main(bad + ["--scheduler", str(sched)])
    assert len(calls) == 3


def test_main_genome_build_is_the_scripts_own_sequence(tmp_path):
    """build() makes what the script made inline before it was shared: the same settings from the data module, the
    same balanced batches, and from the same seed the same initial weights."""
    from genostore.datamodule import SNPStoreDataModule
    from widedeepgp.samplers import balanced_one_disease
    D, _, _ = ladder(tmp_path)
    c1, c2 = (sg.cell_configs(RUNG, DISEASE, str(D), ARGS) for _ in range(2))
    torch.manual_seed(1)
    dm1, m1 = main_genome.build(c1)
    torch.manual_seed(1)
    dm2 = SNPStoreDataModule(c2)
    c2.enc_len = len(dm2.genes)
    c2.enc_len_chr = [len(snp_chr) for snp_chr in dm2.genes_chr]
    c2.pos = dm2.pos_chr
    dm2.train_sampler = balanced_one_disease(dm2.splits["train"]["y"], c2.batch_size, c2.seed)
    m2 = main_genome.with_full_val_auroc(main_genome.DeepGP.SNPCOV_Model_mamba)(c2)
    assert sg.differing(c1, c2) == [] == sg.differing(c2, c1)
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert list(s1) == list(s2) and all(torch.equal(s1[k], s2[k]) for k in s1)
    b1, b2 = list(dm1.train_sampler), list(dm2.train_sampler)
    assert len(b1) == len(b2) > 0 and all(np.array_equal(a, b) for a, b in zip(b1, b2))


@gpu
@pytest.mark.slow
@pytest.mark.skipif(not have_fork(), reason="needs the Vision-Mamba fork of mamba_ssm (the published block)")
def test_a_trained_classic_model_is_rescored_to_its_record_and_a_held_out_copy_scores_the_same(tmp_path):
    D, sched, held = ladder(tmp_path, n=1200, snps=132)
    work = D / "work" / RUNG / DISEASE
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    cmd = [sys.executable, os.path.join(REPO, "main_genome.py"), "--label", DISEASE, "--labels", DISEASE,
           "--rd_mode", RUNG, "--data_dir", f"{D}/datasets", *ARGS]                    # as schedule.launch runs it
    train = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
    assert train.returncode == 0, train.stdout[-3000:] + train.stderr[-3000:]
    res = work / "DeepGP/results/DeepGP_snps_covs" / DISEASE / "final_model"
    rec = json.load(open(res / "results.json"))
    sg.main(["--rung", RUNG, "--disease", DISEASE, "--ladder", str(D), "--scheduler", str(sched), "--held_out_dir", held,
             "--out", str(tmp_path / "data"), "--cohort", "heldx", "--device", "cuda"])
    cell = tmp_path / "data" / RUNG / "scores" / "classic_deepgp" / "ladder" / DISEASE
    v, h = (np.load(cell / f"{s}.npz", allow_pickle=False) for s in ("val", "heldx"))
    from sklearn.metrics import roc_auc_score
    assert abs(roc_auc_score(v["y"][:, 0], v["logits"][:, 0]) - rec["val_auroc"]) <= sg.TOL
    y0, p0 = pickle.load(open(res / "preds_val.pkl", "rb"))
    assert np.array_equal(v["y"], y0) and np.array_equal(v["iid"], val_ids(D))
    assert np.allclose(1 / (1 + np.exp(-v["logits"].astype(np.float64))), p0, rtol=0, atol=1e-6)
    assert np.allclose(h["logits"], v["logits"], rtol=0, atol=1e-6) and np.array_equal(h["y"], v["y"])
