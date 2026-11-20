"""The overnight tweaks of 20 November 2026 (blocks mode): per-disease loss normalisation (--label_norm entropy),
one small head per disease (--per_disease_head), the covariate skip (--cov_skip), the first-layer cache kept on disk
(--first_layer_cache) and the paper's and optimiser search spaces (--tune_space paper | optimizer). CPU.
    python -m pytest tests/test_ml_tweaks.py -q
"""
import hashlib
import json
import os
import stat
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ml_helpers import make_cfg, synthetic_dataset

CHROMS = 22


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return synthetic_dataset(tmp_path_factory.mktemp("d"))[0]


def _build(data_dir, **over):
    from widedeepgp.train import build_data, build_model
    base = dict(input_mode="blocks", partition="uniform:2", mixer="transformer", final_pool="mean", n_heads=2, d_model=8,
                embed_dim=4, lr_schedule="constant", warmup_steps=0, input_mask=0.0, stop_burnin=0)
    cfg = make_cfg(data_dir, "multi", **{**base, **over})
    dm = build_data(cfg)
    return cfg, dm, build_model(cfg, dm)


def _val_batch(dm, n=24):
    sp = dm.splits["val"]
    return dm.decode_batch(dm._dataset({k: v[:n] for k, v in sp.items()})[list(range(n))])


def _run_args(tmp_path, name, **over):
    args = ["--label", "multi", "--rd_mode", "store", "--run_dir", str(tmp_path), "--experiment", "t", "--exp_name", name,
            "--partition", "uniform:2", "--mixer", "transformer", "--d_model", "8", "--n_heads", "2", "--embed_dim", "4",
            "--batch_size", "16", "--num_workers", "0", "--max_epoch_equiv", "1", "--val_per_epoch", "2",
            "--min_participants", "0", "--val_min_participants", "0", "--stop_patience", "0", "--snapshots_per_decade", "2",
            "--tune_trials", "0", "--eval_batch_size", "64"]
    for k, v in over.items():
        args += [f"--{k}", str(v)]
    return args


def _folds_file(data_dir, path, k=2, seed=1):
    import pandas as pd
    iid = pd.read_csv(os.path.join(data_dir, "multi", "store", "samples_train.tsv"), sep="\t", dtype={"IID": str})["IID"].to_numpy(str)
    np.savez(path, seed=np.array(seed), k=np.array(k), iid_train=iid, fold_train=(np.arange(len(iid)) % k).astype(np.int8))
    return str(path)


def _frozen_init(dm, path, k=2, seed=5):
    g = torch.Generator().manual_seed(seed)
    state = {f"tables.{i}": torch.randn((k, 6, 4), generator=g) for i in range(CHROMS)}
    manifest = {"kind": "ridge", "partition_digest": dm.partition["digest"], "snp_code": "dosage", "embed_dim": 4, "n_folds": k}
    torch.save({"manifest": manifest, "state": state}, path)
    return state


# ---- the loss ------------------------------------------------------------------------------------
def test_entropy_normalisation_divides_each_labels_loss_by_its_null_entropy():
    from widedeepgp.losses import WideLoss
    y_train = (np.random.default_rng(0).random((5000, 3)) < np.array([0.3, 0.1, 0.02])).astype(np.float32)
    plain = WideLoss(y_train, gamma=0.0, sampler="shuffled")
    normed = WideLoss(y_train, gamma=0.0, sampler="shuffled", label_norm="entropy")
    logits, y = torch.randn(64, 3), torch.as_tensor(y_train[:64])
    _, a = plain(logits, y)
    loss, b = normed(logits, y)
    pi = y_train.mean(0)
    h = -(pi * np.log(pi) + (1 - pi) * np.log(1 - pi))
    assert np.allclose(b.numpy(), a.numpy() / h, rtol=1e-5) and loss.item() == pytest.approx(float(b.mean()), rel=1e-6)
    assert torch.allclose(plain.norm, torch.ones(3)) and normed.label_norm == "entropy"
    with pytest.raises(ValueError, match="label_norm"):
        WideLoss(y_train, label_norm="prevalence")
    # the rotation variant with one target output is scaled the same way
    rot = WideLoss(y_train, gamma=0.0, sampler="rotation", outputs="target", label_norm="entropy")
    rot0 = WideLoss(y_train, gamma=0.0, sampler="rotation", outputs="target")
    l1, _ = rot(logits, y, target=2)
    l0, _ = rot0(logits, y, target=2)
    assert l1.item() == pytest.approx(l0.item() / h[2], rel=1e-5)


def test_a_run_records_the_label_normalisation(data_dir, tmp_path):
    import main_multilabel
    out = main_multilabel.main(_run_args(tmp_path, "n", data_dir=data_dir, label_norm="entropy", sampler="rotation"))
    config = json.load(open(os.path.join(out["run_dir"], "config.json")))
    assert config["model"]["label_norm"] == "entropy" and out["best_val_auroc_mean"] > 0
    assert not torch.allclose(out["model"].loss_fn.norm, torch.ones(6))


# ---- the heads -----------------------------------------------------------------------------------
def test_the_per_disease_head_gives_each_disease_its_own_small_network(data_dir):
    from widedeepgp.model import PerDiseaseHead
    _, dm, model = _build(data_dir, per_disease_head=5, head_hidden=12)
    model.eval()
    head = model.combine.layers[-1]
    assert isinstance(head, PerDiseaseHead) and head.w1.shape == (6, 12, 5) and head.w2.shape == (6, 5)
    assert model.parameter_counts()["head"] == sum(p.numel() for p in model.combine.parameters())
    x_all, _ = _val_batch(dm)
    with torch.no_grad():
        out = model(x_all)
        caught = {}
        hook = model.combine.layers[-2].register_forward_hook(lambda m, i, o: caught.__setitem__("h", o))
        model(x_all)
        hook.remove()
        h = caught["h"]
        want = torch.stack([(torch.nn.functional.gelu(h @ head.w1[j] + head.b1[j]) * head.w2[j]).sum(-1) + head.b2[j]
                            for j in range(6)], 1)
    assert out.shape == (24, 6) and torch.allclose(out, want, atol=1e-6)
    with pytest.raises(ValueError, match="mean readout"):
        _build(data_dir, per_disease_head=5, final_pool="atten")


def test_the_covariate_skip_starts_at_the_covariates_only_model(data_dir):
    from sklearn.linear_model import LogisticRegression
    from widedeepgp.train import fit_cov_skip, train_labels
    _, dm, model = _build(data_dir, cov_skip="logistic")
    assert model.cov_skip.weight.shape == (6, dm.n_cov) and float(model.cov_skip.weight.abs().sum()) == 0
    fit_cov_skip(model, dm)
    model.eval()
    x_all, _ = _val_batch(dm)
    with torch.no_grad():
        out = model(x_all)
    x, y = dm.splits["train"]["cov"], train_labels(dm)
    want = np.stack([LogisticRegression(max_iter=2000).fit(x, y[:, j]).decision_function(dm.splits["val"]["cov"][:24])
                     for j in range(6)], 1)
    assert np.allclose(out.numpy(), want, atol=1e-4)                 # the head's output map is zero: the floor exactly
    assert model.parameter_counts()["head"] == sum(p.numel() for p in model.combine.parameters()) + 6 * dm.n_cov + 6
    _, _, zero = _build(data_dir, cov_skip="zero")
    assert float(zero.combine.layers[-1].weight.abs().sum()) > 0        # the zero variant leaves the head alone
    with pytest.raises(ValueError, match="covariates"):
        _build(data_dir, cov_skip="zero", dm="snps", use_cov=False)


def test_runs_train_with_the_per_disease_head_and_the_skip(data_dir, tmp_path):
    import main_multilabel
    out = main_multilabel.main(_run_args(tmp_path, "h", data_dir=data_dir, per_disease_head=4, cov_skip="logistic"))
    config = json.load(open(os.path.join(out["run_dir"], "config.json")))
    assert config["model"]["per_disease_head"] == 4 and config["model"]["cov_skip"] == "logistic"
    assert out["best_val_auroc_mean"] > 0.5
    ck = torch.load(out["best_checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
    assert "cov_skip.weight" in ck and "combine.layers.3.w1" in ck


# ---- the disk cache ------------------------------------------------------------------------------
def test_the_first_layer_cache_on_disk_is_written_once_loaded_after_and_owner_only(data_dir, tmp_path):
    import main_multilabel
    from widedeepgp import train
    folds = _folds_file(data_dir, tmp_path / "folds.npz")
    _, dm, _ = _build(data_dir, folds=folds)
    _frozen_init(dm, tmp_path / "init.pt")
    folder = tmp_path / "data" / "cache"

    def args(name, **over):
        return _run_args(tmp_path, name, data_dir=data_dir, folds=folds, init_encoder=str(tmp_path / "init.pt"), input_mask=0,
                         dropout=0.2, first_layer_cache=str(folder), **over) + ["--freeze_encoder"]

    first = main_multilabel.main(args("a"))
    c1 = json.load(open(os.path.join(first["run_dir"], "config.json")))
    files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    assert len(files) == 1 and c1["first_layer_cache_file"] == os.path.join(str(folder), files[0])
    assert stat.S_IMODE(os.stat(folder).st_mode) == 0o700 and stat.S_IMODE(os.stat(c1["first_layer_cache_file"]).st_mode) == 0o600
    z = torch.load(c1["first_layer_cache_file"], map_location="cpu", weights_only=True)
    assert z["train"].shape == (len(dm.splits["train"]["y"]), 66, 4) and z["val"].shape[0] == len(dm.splits["val"]["y"]) and "test" not in z

    calls = []
    orig = train.cache_first_layer
    train.cache_first_layer = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        second = main_multilabel.main(args("b"))
    finally:
        train.cache_first_layer = orig
    assert not calls                                                    # loaded, not computed again
    assert [h["val_auroc_mean"] for h in second["history"]] == pytest.approx([h["val_auroc_mean"] for h in first["history"]], abs=1e-6)
    assert len([f for f in os.listdir(folder) if f.endswith(".pt")]) == 1

    _frozen_init(dm, tmp_path / "init.pt", seed=6)                      # another table: another key, a second file
    main_multilabel.main(args("c"))
    assert len([f for f in os.listdir(folder) if f.endswith(".pt")]) == 2


# ---- the search spaces ---------------------------------------------------------------------------
def test_the_paper_and_optimizer_spaces():
    import optuna
    import tune_optuna
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    for _ in range(30):
        cfg = SimpleNamespace(input_mode="blocks", init_encoder="x.pt", freeze_encoder=True, lr=1e-4, l2=1e-3, dropout=0.2,
                              batch_size=64, e_layers=4, d_model=64, n_heads=4, n_fcn=2, mixer="transformer")
        tune_optuna.suggest(study.ask(), cfg, "paper")
        assert 1e-5 <= cfg.lr <= 1e-3 and 1e-5 <= cfg.l2 <= 1e-3 and 0.05 <= cfg.dropout <= 0.3
        assert cfg.batch_size in (32, 64, 128) and cfg.e_layers in (1, 2, 3) and cfg.d_model in (16, 32, 64)
        assert cfg.n_heads in (2, 4, 8) and cfg.n_fcn in (1, 2, 4)
        assert cfg.mixer == "transformer" if not torch.cuda.is_available() else cfg.mixer in ("transformer", "bimamba1")
        assert not hasattr(cfg, "table_kappa")                          # a frozen table is not searched
    cfg = SimpleNamespace(input_mode="blocks", init_encoder="x.pt", freeze_encoder=True, lr=1e-4, l2=1e-3, dropout=0.2,
                          lr_schedule="plateau", warmup_steps=0, batch_size=256)
    for _ in range(30):
        tune_optuna.suggest(study.ask(), cfg, "optimizer")
        assert 3e-5 <= cfg.lr <= 3e-3 and 1e-6 <= cfg.l2 <= 1e-2 and 0.0 <= cfg.dropout <= 0.4
        assert cfg.lr_schedule in ("plateau", "cosine", "constant") and cfg.warmup_steps in (0, 200, 1000)
        assert cfg.batch_size == 256 and not hasattr(cfg, "table_kappa")
    trained = SimpleNamespace(input_mode="blocks", init_encoder=None, freeze_encoder=False, lr=1e-4, l2=1e-3, dropout=0.2)
    tune_optuna.suggest(study.ask(), trained, "deep")
    assert 0.01 <= trained.table_kappa <= 0.03                          # a table that trains keeps its own search
    with pytest.raises(ValueError, match="space must be"):
        tune_optuna.suggest(study.ask(), trained, "everything")


def test_a_worker_of_a_paper_space_search_runs_a_trial(data_dir, tmp_path):
    """One worker of a --tune_space paper search (the queue runs several, then the summary) completes a trial whose
    settings come from the paper's grid."""
    import main_multilabel
    import tune_optuna
    args = _run_args(tmp_path, "s", data_dir=data_dir, tune_space="paper", max_epoch_equiv=0.5) + ["--tune_trials", "1", "--tune_gpus", "none"]
    out = main_multilabel.main(args)
    assert out["chosen_stage"] == "deep" and set(out["best_params"]) >= {"lr", "l2", "dropout", "batch_size", "e_layers", "d_model", "n_heads", "n_fcn", "mixer"}
    assert "table_kappa" in out["best_params"]                          # this search's table trains, so its pace is searched too
    trial_cfg = json.load(open(os.path.join(tmp_path, "t", "s", out["best_trial_dir"], "config.json")))["args"]
    assert trial_cfg["batch_size"] == out["best_params"]["batch_size"] and trial_cfg["d_model"] == out["best_params"]["d_model"]
    assert tune_optuna.load_study("deep", os.path.join(tmp_path, "t", "s")).trials[0].params["mixer"] == "transformer"   # the published block needs a GPU


# ---- within-block attention over the strongest SNPs -----------------------------------------------
def _one_copy_init(dm, path, de=4, seed=5):
    g = torch.Generator().manual_seed(seed)
    state = {f"tables.{i}": torch.randn((6, de), generator=g) for i in range(CHROMS)}
    manifest = {"kind": "pgi", "partition_digest": dm.partition["digest"], "snp_code": "dosage", "embed_dim": de, "n_folds": 1}
    torch.save({"manifest": manifest, "state": state}, path)
    return state


def test_block_attention_selects_the_largest_rows_in_genomic_order_and_adds_to_the_tokens(data_dir, tmp_path):
    from widedeepgp.blockattn import BlockAttention
    _, dm, model = _build(data_dir, partition="uniform:6", init_encoder=str(tmp_path / "i.pt"), freeze_encoder=True, block_attn=3,
                          block_attn_dim=8, block_attn_heads=2)
    state = _one_copy_init(dm, tmp_path / "i.pt")
    model.encoder.load_init(str(tmp_path / "i.pt"), model.hp["partition_digest"], "dosage", freeze=True)
    model.block_attn.select_from(model.encoder)
    ba = model.block_attn
    assert isinstance(ba, BlockAttention) and ba.rows.shape == (1, CHROMS, 3, 4) and ba.idx_4.shape == (1, 3)
    for i in range(CHROMS):                                              # the three largest rows of each block, in order
        norms = state[f"tables.{i}"].norm(dim=1)
        want = torch.topk(norms, 3).indices.sort().values
        assert torch.equal(getattr(ba, f"idx_{i}")[0], want) and torch.equal(ba.rows[0, i], state[f"tables.{i}"][want])
    model.eval()
    x_all, _ = _val_batch(dm)
    snps = x_all[:-1]
    sel = ba.select(snps)
    assert sel.shape == (24, CHROMS, 3)
    assert torch.equal(sel[:, 4], snps[4][:, ba.idx_4[0], 0].to(torch.bfloat16).float())
    with torch.no_grad():
        pooled, sel2, fold, rest = model.first_layer(x_all)
        assert torch.equal(sel, sel2) and fold is None and len(rest) == 1
        add = ba(sel, pooled)
        assert add.shape == (24, CHROMS, 8)
        out = model(x_all)
        ba.out.weight.zero_(); ba.out.bias.zero_()                       # the module off: the plain model's logits
        plain = model(x_all)
    assert out.shape == (24, 6) and not torch.allclose(out, plain)
    counts = model.parameter_counts()
    assert counts["block_attn"] == sum(p.numel() for p in ba.parameters()) > 0
    sd = model.snapshot(1)["state_dict"]
    assert "block_attn.rows" in sd and "block_attn.idx_4" in sd            # the selection travels with the weights
    with pytest.raises(ValueError, match="needs --init_encoder"):
        _build(data_dir, partition="uniform:6", block_attn=3)
    with pytest.raises(ValueError, match="smallest block"):
        _build(data_dir, partition="uniform:6", init_encoder=str(tmp_path / "i.pt"), freeze_encoder=True, block_attn=7)


def test_block_attention_on_a_fold_routed_table_uses_each_participants_copy_and_caches(data_dir, tmp_path):
    import main_multilabel
    from widedeepgp import train
    folds = _folds_file(data_dir, tmp_path / "f.npz", k=2)
    _, dm, model = _build(data_dir, partition="uniform:6", init_encoder=str(tmp_path / "i.pt"), freeze_encoder=True, block_attn=3,
                          block_attn_dim=8, block_attn_heads=2, folds=folds)
    g = torch.Generator().manual_seed(7)
    state = {f"tables.{i}": torch.randn((2, 6, 4), generator=g) for i in range(CHROMS)}
    torch.save({"manifest": {"kind": "ridge", "partition_digest": dm.partition["digest"], "snp_code": "dosage", "embed_dim": 4, "n_folds": 2},
                "state": state}, tmp_path / "i.pt")
    model.encoder.load_init(str(tmp_path / "i.pt"), model.hp["partition_digest"], "dosage", freeze=True)
    ba = model.block_attn.select_from(model.encoder)
    assert ba.rows.shape == (2, CHROMS, 3, 4)
    want = torch.topk(state["tables.4"].mean(0).norm(dim=1), 3).indices.sort().values      # chosen on the copies' mean
    assert torch.equal(ba.idx_4[0], want) and torch.equal(ba.rows[:, 4], state["tables.4"][:, want])
    model.eval()
    x_all, _ = _val_batch(dm)
    with torch.no_grad():
        pooled, sel, fold, rest = model.first_layer(x_all)
        assert fold.shape == (24,) and set(fold.tolist()) <= {0, 1}
        both = ba(sel, pooled, fold)
        copy0 = ba(sel, pooled, torch.zeros_like(fold))
        copy1 = ba(sel, pooled, torch.ones_like(fold))
    assert torch.allclose(both[fold == 0], copy0[fold == 0]) and torch.allclose(both[fold == 1], copy1[fold == 1])
    assert not torch.allclose(copy0, copy1)

    def args(name, cache=True):
        return _run_args(tmp_path, name, data_dir=data_dir, partition="uniform:6", init_encoder=str(tmp_path / "i.pt"), input_mask=0,
                         block_attn=3, block_attn_dim=8, block_attn_heads=2, folds=folds,
                         **({"first_layer_cache": str(tmp_path / "cache")} if cache else {})) + ["--freeze_encoder"]

    out = main_multilabel.main(args("fc"))
    orig = train.cache_first_layer
    train.cache_first_layer = lambda model, dm, device: None
    try:
        streamed = main_multilabel.main(args("fs", cache=False))
    finally:
        train.cache_first_layer = orig
    assert [h["val_auroc_mean"] for h in out["history"]] == pytest.approx([h["val_auroc_mean"] for h in streamed["history"]], abs=1e-6)


def test_a_block_attention_run_is_cached_equals_the_streamed_run_and_scores(data_dir, tmp_path):
    import main_multilabel
    import score_snapshots
    from widedeepgp import train
    _, dm, _ = _build(data_dir, partition="uniform:6")
    _one_copy_init(dm, tmp_path / "i.pt")
    folder = tmp_path / "cache"

    def args(name, **over):
        kw = dict(data_dir=data_dir, partition="uniform:6", init_encoder=str(tmp_path / "i.pt"), input_mask=0, block_attn=3,
                  block_attn_dim=8, block_attn_heads=2, first_layer_cache=str(folder), **{k: v for k, v in over.items() if k != "first_layer_cache"})
        cache = over.get("first_layer_cache", str(folder))
        if cache is None:
            kw.pop("first_layer_cache")
        else:
            kw["first_layer_cache"] = cache
        return _run_args(tmp_path, name, **kw) + ["--freeze_encoder"]

    out = main_multilabel.main(args("c"))
    config = json.load(open(os.path.join(out["run_dir"], "config.json")))
    assert config["model"]["block_attn"] == 3 and config["blocks"]["n_parameters_by_module"]["block_attn"] > 0
    z = torch.load(config["first_layer_cache_file"], map_location="cpu", weights_only=True)
    assert z["train_sel"].shape == (len(dm.splits["train"]["y"]), CHROMS, 3) and z["train_sel"].dtype == torch.bfloat16
    rows = score_snapshots.main(["--run", out["run_dir"], "--split", "val", "--snapshots", "best", "--device", "cpu"])
    rescored = {r["label"]: r["value"] for r in rows if r["snapshot"] == "best" and r["metric"] == "auroc" and r["label"] != "MEAN"}
    assert rescored == pytest.approx({k: v["auroc"] for k, v in out["val"].items()}, abs=1e-6)
    orig = train.cache_first_layer
    train.cache_first_layer = lambda model, dm, device: None               # the same run streamed
    try:
        streamed = main_multilabel.main(args("s", first_layer_cache=None))
    finally:
        train.cache_first_layer = orig
    assert [h["val_auroc_mean"] for h in out["history"]] == pytest.approx([h["val_auroc_mean"] for h in streamed["history"]], abs=1e-6)
    out2 = main_multilabel.main(args("c", max_epoch_equiv=2) + ["--resume"])
    assert out2["samples_seen"] > out["samples_seen"]
