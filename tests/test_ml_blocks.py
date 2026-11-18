"""Blocks mode (--input_mode blocks): the partition, the LD-block first layer, the transformer mixer, the
table's optimiser group and penalty, snapshots, a run, a search, a seed retrain and masked pretraining,
all on the CPU with the transformer mixer (bimamba1 needs the GPU: tests/test_ml_blocks_gpu.py).
    python -m pytest tests/test_ml_blocks.py -q
"""
import hashlib
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from ml_helpers import REPO, make_cfg, synthetic_dataset

SNPS, CHROMS = 132, 22                                   # the synthetic store: six SNPs per chromosome


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return synthetic_dataset(tmp_path_factory.mktemp("d"))[0]


def _cfg(data_dir, **over):
    base = dict(input_mode="blocks", partition="uniform:2", mixer="transformer", final_pool="mean", n_heads=2, d_model=8,
                embed_dim=4, lr_schedule="constant", warmup_steps=0, input_mask=0.0, stop_burnin=0)
    return make_cfg(data_dir, "multi", **{**base, **over})


def _build(data_dir, **over):
    from widedeepgp.train import build_data, build_model
    cfg = _cfg(data_dir, **over)
    dm = build_data(cfg)
    return cfg, dm, build_model(cfg, dm)


def _val_batch(dm, n=24):
    sp = dm.splits["val"]
    ds = dm._dataset({k: v[:n] for k, v in sp.items()})
    return dm.decode_batch(ds[list(range(n))])


def _write_partition(path, store, ids_per_chrom):
    """A partition.npz in the layout of the design document, from per-chromosome global block ids."""
    ids = np.concatenate([store.variants(c)["id"] for c in store.chroms])
    np.savez(path, chroms=np.array(store.chroms), n_variants=np.array([store.n_variants(c) for c in store.chroms]),
             variant_digest=np.array(hashlib.sha256("\n".join(ids).encode()).hexdigest()),
             n_blocks=np.int64(max(int(a.max()) for a in ids_per_chrom) + 1),
             **{f"block_chr{c}": np.asarray(a, dtype=np.int32) for c, a in zip(store.chroms, ids_per_chrom)})


# ---- the partition ------------------------------------------------------------------------------
def test_a_uniform_partition_is_contiguous_blocks_numbered_across_sequences(data_dir):
    _, dm, _ = _build(data_dir)
    p = dm.partition
    assert p["n_blocks"] == 3 * CHROMS and p["name"] == "uniform:2" and len(p["digest"]) == 16
    assert [a.tolist() for a in p["ids"][:2]] == [[0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5]]
    _, dm6, _ = _build(data_dir, partition="uniform:6")
    assert dm6.partition["n_blocks"] == CHROMS and dm6.partition["digest"] != p["digest"]
    _, dm4, _ = _build(data_dir, partition="uniform:4")                 # 4 + 2 SNPs per chromosome
    assert dm4.partition["ids"][1].tolist() == [2, 2, 2, 2, 3, 3]


def test_a_partition_file_is_checked_against_the_store(data_dir, tmp_path):
    from genostore.datamodule import load_partition, partition_digest
    _, dm, _ = _build(data_dir)
    store = dm.store
    good = [np.arange(6) // 2 + 3 * i for i in range(CHROMS)]
    _write_partition(tmp_path / "partition.npz", store, good)
    p = load_partition(str(tmp_path), store, dm.regions)
    assert p["name"] == tmp_path.name and p["digest"] == dm.partition["digest"]   # the same blocks as uniform:2
    assert all(np.array_equal(a, b) for a, b in zip(p["ids"], dm.partition["ids"]))
    # a region that keeps part of two chromosomes: the blocks present are renumbered from zero
    sub = load_partition(str(tmp_path / "partition.npz"), store, [[["1", 2, 6]], [["2", 0, 4]]])
    assert [a.tolist() for a in sub["ids"]] == [[0, 0, 1, 1], [2, 2, 3, 3]] and sub["n_blocks"] == 4
    assert sub["digest"] == partition_digest(sub["ids"]) != p["digest"]
    z = dict(np.load(tmp_path / "partition.npz"))
    for change, message in ((dict(n_variants=z["n_variants"] + 1), "variant counts"),
                            (dict(variant_digest=np.array("0" * 64)), "other variants"),
                            (dict(block_chr1=np.array([0, 1, 0, 1, 2, 2], dtype=np.int32)), "contiguous"),
                            (dict(block_chr2=np.array([2, 3, 3, 4, 4, 5], dtype=np.int32)), "contiguous")):
        np.savez(tmp_path / "bad.npz", **{**z, **change})
        with pytest.raises(ValueError, match=message):
            load_partition(str(tmp_path / "bad.npz"), store, dm.regions)
    with pytest.raises(ValueError, match="needs --partition"):
        from widedeepgp.train import build_data
        build_data(_cfg(data_dir, partition=None))


# ---- the first layer ----------------------------------------------------------------------------
@pytest.mark.parametrize("de", [1, 16, 64])
def test_pooling_is_the_segment_sum_of_dosage_scaled_vectors(data_dir, de):
    from widedeepgp.blockencoder import BlockEncoder
    _, dm, _ = _build(data_dir)
    enc = BlockEncoder(dm.partition, de, 8)
    g = torch.Generator().manual_seed(de)
    for t in enc.tables:
        t.data = torch.randn(t.shape, generator=g)
    x_all, _ = _val_batch(dm)
    pooled = enc.pool(x_all[:-1]).detach().numpy()
    assert pooled.shape == (24, 3 * CHROMS, de)
    k = 0
    for i, ids in enumerate(dm.partition["ids"]):
        x, w = x_all[i][:, :, 0].numpy(), enc.tables[i].detach().numpy()
        for b in np.unique(ids):
            s = ids == b
            np.testing.assert_allclose(pooled[:, k], (x[:, s] @ w[s]) / np.sqrt(s.sum()), rtol=1e-5, atol=1e-6)
            k += 1
    assert enc.hidden(enc.pool(x_all[:-1])).shape == (24, 3 * CHROMS, 4 * de)
    assert enc(x_all[:-1]).shape == (24, 3 * CHROMS, 8)


def test_the_genotype_code_interpolates_three_vectors_and_contains_the_dosage_code(data_dir):
    from widedeepgp.blockencoder import BlockEncoder, genotype_weights
    x = torch.tensor([0.0, 0.3, 1.0, 1.6, 2.0])
    assert torch.allclose(genotype_weights(x), torch.tensor([[1, 0, 0], [.7, .3, 0], [0, 1, 0], [0, .4, .6], [0, 0, 1.]]))
    _, dm, _ = _build(data_dir, snp_code="genotype")
    assert dm.snp_mean is None                                           # the raw dosages, not standardised ones
    enc3, enc1 = BlockEncoder(dm.partition, 4, 8, "genotype"), BlockEncoder(dm.partition, 4, 8, "dosage")
    for t3, t1 in zip(enc3.tables, enc1.tables):
        t1.data = torch.randn(t1.shape)
        t3.data = (t1.data[:, None, :] * torch.tensor([0.0, 1.0, 2.0])[None, :, None]).reshape(-1, 4)  # rows 0, v, 2v
    x_all, _ = _val_batch(dm)
    assert torch.allclose(enc3.pool(x_all[:-1]), enc1.pool(x_all[:-1]), atol=1e-5)
    assert sum(t.numel() for t in enc3.tables) == 3 * SNPS * 4
    with pytest.raises(ValueError, match="snp_code must be"):
        BlockEncoder(dm.partition, 4, 8, "haplotype")


def test_the_first_layer_contains_a_ridge_prs(data_dir):
    """At embed_dim 1 the pooled block values are the per-block sums of a ridge PRS, x @ w split by block and
    scaled by 1 / sqrt(block size), so the first layer contains a ridge; what the model predicts from them goes
    through the mixer and head."""
    from widedeepgp.blockencoder import BlockEncoder
    _, dm, _ = _build(data_dir, embed_dim=1)
    enc = BlockEncoder(dm.partition, 1, 8)
    w = np.random.default_rng(0).standard_normal(SNPS).astype(np.float32)
    with torch.no_grad():
        for i, t in enumerate(enc.tables):
            t.copy_(torch.from_numpy(w[6 * i:6 * i + 6])[:, None])
    x_all, _ = _val_batch(dm)
    x = np.concatenate([v[:, :, 0].numpy() for v in x_all[:-1]], axis=1)
    pooled = enc.pool(x_all[:-1]).detach().numpy()[:, :, 0]
    np.testing.assert_allclose(pooled.sum(1) * np.sqrt(2), x @ w, rtol=1e-4, atol=1e-5)   # two SNPs per block
    assert pooled.shape == (24, 3 * CHROMS)


def test_shapes_parameter_names_and_counts(data_dir):
    _, dm, model = _build(data_dir, e_layers=2)
    x_all, _ = _val_batch(dm)
    assert model(x_all).shape == (24, 6)
    names = {k.split(".")[0] for k in model.state_dict()}
    assert names == {"encoder", "layers", "norm_f", "covariate_encoder", "cov_norm", "combine", "loss_fn"}
    assert {k for k in model.state_dict() if k.startswith("encoder.") and not k.startswith("encoder.tables.")} == \
        {"encoder.block_emb.weight", "encoder.chrom_emb.weight", "encoder.mlp.0.weight", "encoder.mlp.0.bias",
         "encoder.mlp.2.weight", "encoder.mlp.2.bias"}
    assert [tuple(model.state_dict()[f"encoder.tables.{i}"].shape) for i in range(CHROMS)] == [(6, 4)] * CHROMS
    counts = model.parameter_counts()
    assert counts["table"] == SNPS * 4 and counts["identity"] == (3 * CHROMS + CHROMS) * 4
    assert counts["block_mlp"] == 4 * 16 + 16 + 16 * 8 + 8
    assert sum(counts.values()) == sum(p.numel() for p in model.parameters())
    assert counts["mixer"] == sum(p.numel() for p in model.layers.parameters()) + 2 * 8
    assert model.big_params() == list(model.encoder.tables)


def test_input_masking_hides_a_share_of_the_snps_in_training_only(data_dir):
    """With --input_mask 0.4 a training forward pass hides about 40% of the SNPs and rescales the rest, so the
    pooled vectors vary between passes but keep their expectation; evaluation uses every SNP; the SNP-token model
    and masks outside [0, 1) are refused."""
    from widedeepgp.model import WideDeepGP
    _, dm, model = _build(data_dir, input_mask=0.4, dropout=0.0)
    x_all, _ = _val_batch(dm)
    snps = x_all[:-1]
    torch.manual_seed(0)
    with torch.no_grad():
        for t in model.encoder.tables:
            t.normal_()                                                       # the table starts at zero otherwise
        full = model.encoder.pool(snps)
        masked = torch.stack([model.encoder.pool(snps, [torch.rand(x.shape[:2]) < 0.4 for x in snps]) / 0.6 for _ in range(500)])
    assert not torch.allclose(masked[0], masked[1])
    deviation = (masked.mean(0) - full).abs().mean()                            # the same in expectation
    assert deviation < 0.1 * full.abs().mean(), float(deviation)
    model.eval()
    with torch.no_grad():
        a, b = model(x_all), model(x_all)
    assert torch.equal(a, b)                                                  # no masking at evaluation
    model.train()
    with torch.no_grad():
        assert not torch.equal(model(x_all), model(x_all))                       # masked, so passes differ
    for change, message in ((dict(input_mask=1.0), "input_mask must be"), (dict(input_mode="snps", input_mask=0.4), "blocks-mode setting")):
        with pytest.raises(ValueError, match=message):
            WideDeepGP(dict(model.hp, **change), dm.pos_chr, blocks=dm.partition)


def test_the_plateau_schedule_watches_the_tuning_auroc(data_dir):
    _, dm, model = _build(data_dir, lr_schedule="plateau")
    out = model.configure_optimizers()
    sched = out["lr_scheduler"]
    assert isinstance(sched["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert (sched["monitor"], sched["interval"], sched["scheduler"].factor, sched["scheduler"].patience, sched["scheduler"].mode) == \
        ("val/auroc_mean", "epoch", 0.2, 10, "max")
    assert len(out["optimizer"].param_groups) == 2


def test_early_stopping_waits_for_the_burn_in(monkeypatch):
    from types import SimpleNamespace
    from pytorch_lightning.callbacks import EarlyStopping
    from widedeepgp.callbacks import EarlyStoppingOnce
    checks = []
    monkeypatch.setattr(EarlyStopping, "_run_early_stopping_check", lambda self, trainer: checks.append(trainer.global_step))
    stop = EarlyStoppingOnce(monitor="val/auroc_mean", mode="max", patience=1, burnin=100)
    for step in (10, 50, 99):
        stop._run_early_stopping_check(SimpleNamespace(global_step=step))
    assert checks == []                                                       # nothing counts before 100 steps
    stop._run_early_stopping_check(SimpleNamespace(global_step=120))
    stop._run_early_stopping_check(SimpleNamespace(global_step=120))        # a repeated pass counts once
    assert checks == [120] and stop.burnin == 100


def test_the_table_only_search_space():
    import optuna
    import tune_optuna
    from types import SimpleNamespace
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    cfg = SimpleNamespace(input_mode="blocks", lr=1e-4, l2=1e-3, dropout=0.5)
    tune_optuna.suggest(study.ask(), cfg, "table")
    assert (cfg.lr, cfg.l2, cfg.dropout) == (1e-4, 1e-3, 0.5)                  # untouched: the recipe's values stand
    assert 0.01 <= cfg.table_kappa <= 0.03 and 0.3 <= cfg.table_decay <= 3.0
    tune_optuna.suggest(study.ask(), cfg, "deep")
    assert cfg.lr != 1e-4 or cfg.l2 != 1e-3 or cfg.dropout != 0.5
    warm = SimpleNamespace(input_mode="blocks", init_encoder="init_ridge_d16.pt", lr=1e-4, l2=1e-3, dropout=0.5)
    for _ in range(20):                                                          # a loaded table: a slower, undecayed range
        tune_optuna.suggest(study.ask(), warm, "table")
        assert 3e-4 <= warm.table_kappa <= 1e-2 and 1e-3 <= warm.table_decay <= 3e-2
    assert (warm.lr, warm.l2, warm.dropout) == (1e-4, 1e-3, 0.5)


def test_a_transformer_block_keeps_the_block_interface():
    from widedeepgp.blocks import make_block
    block = make_block(8, "transformer", n_heads=2, dropout=0.0)
    h = torch.randn(3, 5, 8, requires_grad=True)
    out, residual = block(h, None)
    assert out.shape == h.shape and residual is None
    out.sum().backward()
    assert h.grad is not None and sum(p.numel() for p in block.parameters()) == 12 * 64 + 4 * 8 + 5 * 8 + 4 * 8
    with pytest.raises(ValueError, match="mixer must be one of"):
        make_block(8, "lstm")


def test_refusals(data_dir):
    from widedeepgp.model import WideDeepGP
    from widedeepgp.train import check_resume_tables, run_training
    _, dm, model = _build(data_dir)
    for change, message in ((dict(wide="on"), "no linear PRS path"), (dict(annot_dim=2), "no linear PRS path"),
                            (dict(final_pool="atten"), "pools the mixer's tokens by their mean"),
                            (dict(partition_digest="0" * 16), "does not match")):
        with pytest.raises(ValueError, match=message):
            WideDeepGP(dict(model.hp, **change), dm.pos_chr, blocks=dm.partition)
    with pytest.raises(ValueError, match="does not match"):
        WideDeepGP(model.hp, dm.pos_chr)                                 # no partition given
    with pytest.raises(ValueError, match="snp_code must be"):
        WideDeepGP(dict(model.hp, snp_code="haplotype"), dm.pos_chr, blocks=dm.partition)


def test_resume_and_init_from_refuse_another_partition(data_dir, tmp_path):
    from widedeepgp.train import check_resume_tables, run_training
    _, dm, model = _build(data_dir)
    other = _build(data_dir, partition="uniform:3")[2]
    torch.save(other.snapshot(0), tmp_path / "other.ckpt")
    with pytest.raises(ValueError, match="partition_digest"):
        check_resume_tables(str(tmp_path / "other.ckpt"), model.hp)
    cfg = _cfg(data_dir, run_dir=str(tmp_path), experiment="t", exp_name="r", init_from=str(tmp_path / "other.ckpt"),
               tune_trials=0)
    with pytest.raises(ValueError, match="another block partition"):
        run_training(cfg)


def test_the_table_has_its_own_optimiser_group(data_dir):
    """The table's learning rate is kappa sqrt(B) / sqrt(M); its AdamW group carries the per-step decay and the
    large epsilon; nothing else in the model shares them, and there is no loss penalty in blocks mode."""
    from widedeepgp.blockencoder import table_learning_rate
    cfg, dm, model = _build(data_dir, table_kappa=0.02, table_decay=2.0, table_eps=1e-4, batch_size=16, accumulate=2,
                            l2=1e-3, lr=1e-3)
    assert table_learning_rate(0.02, 32, SNPS) == pytest.approx(0.02 * np.sqrt(32) / np.sqrt(SNPS))
    assert model.hp["lr_table"] == pytest.approx(table_learning_rate(0.02, 32, SNPS))
    groups = model.configure_optimizers().param_groups
    assert [(g["lr"], g["weight_decay"], g["eps"], len(g["params"])) for g in groups] == \
        [(1e-3, 1e-3, 1e-8, len(list(model.parameters())) - CHROMS), (model.hp["lr_table"], 2.0, 1e-4, CHROMS)]
    assert model.penalty() is None
    for t in model.encoder.tables:
        t.requires_grad_(False)
    assert len(model.configure_optimizers().param_groups) == 1


def test_an_init_encoder_file_is_checked_loaded_and_may_be_frozen(data_dir, tmp_path):
    _, dm, model = _build(data_dir)
    enc = model.encoder
    state = {f"tables.{i}": torch.randn(6, 4) for i in range(CHROMS)}
    state["mlp.0.weight"] = torch.randn(16, 4)
    manifest = {"kind": "ridge", "partition_digest": dm.partition["digest"], "snp_code": "dosage", "embed_dim": 4}
    torch.save({"manifest": manifest, "state": state}, tmp_path / "init.pt")
    assert enc.load_init(str(tmp_path / "init.pt"), dm.partition["digest"], "dosage") == sorted(state)
    assert torch.equal(enc.tables[3], state["tables.3"]) and torch.equal(enc.mlp[0].weight, state["mlp.0.weight"])
    assert all(p.requires_grad for p in enc.parameters())
    enc.load_init(str(tmp_path / "init.pt"), dm.partition["digest"], "dosage", freeze=True)
    assert not enc.tables[0].requires_grad and not enc.mlp[0].weight.requires_grad and enc.mlp[0].bias.requires_grad
    for bad_manifest, message in ((dict(manifest, snp_code="genotype"), "another partition, SNP code, width or fold count"),
                                  (dict(manifest, embed_dim=16), "another partition, SNP code, width or fold count"),
                                  (dict(manifest, partition_digest="x"), "another partition, SNP code, width or fold count")):
        torch.save({"manifest": bad_manifest, "state": state}, tmp_path / "bad.pt")
        with pytest.raises(ValueError, match=message):
            enc.load_init(str(tmp_path / "bad.pt"), dm.partition["digest"], "dosage")
    torch.save({"manifest": manifest, "state": {"tables.0": torch.randn(7, 4)}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="misshaped"):
        enc.load_init(str(tmp_path / "bad.pt"), dm.partition["digest"], "dosage")
    torch.save({"manifest": manifest, "state": {"mlp.0.weight": torch.randn(16, 4)}}, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="no tables"):
        enc.load_init(str(tmp_path / "bad.pt"), dm.partition["digest"], "dosage")


# ---- runs ---------------------------------------------------------------------------------------
def _run_args(tmp_path, name, **over):
    args = ["--label", "multi", "--rd_mode", "store", "--run_dir", str(tmp_path), "--experiment", "t", "--exp_name", name,
            "--partition", "uniform:2", "--mixer", "transformer", "--d_model", "8", "--n_heads", "2", "--embed_dim", "4",
            "--batch_size", "16", "--num_workers", "0", "--max_epoch_equiv", "1", "--val_per_epoch", "2",
            "--min_participants", "0", "--val_min_participants", "0", "--stop_patience", "0", "--snapshots_per_decade", "2",
            "--tune_trials", "0", "--eval_batch_size", "64"]
    for k, v in over.items():
        args += [f"--{k}", str(v)]
    return args


def test_a_run_trains_snapshots_in_bfloat16_scores_and_resumes(data_dir, tmp_path):
    import main_multilabel
    import score_snapshots
    from widedeepgp.train import load_model
    out = main_multilabel.main(_run_args(tmp_path, "r", data_dir=data_dir))
    run = out["run_dir"]
    config = json.load(open(os.path.join(run, "config.json")))
    b = config["blocks"]
    assert (b["partition"], b["n_blocks"], b["n_tokens"], b["embed_dim"], b["snp_code"]) == ("uniform:2", 66, 67, 4, "dosage")
    assert b["n_embedding_parameters"] + b["n_non_embedding_parameters"] == config["n_parameters"] == out["n_parameters"]
    assert config["model"]["input_mode"] == "blocks" and config["model"]["final_pool"] == "mean"
    assert config["model"]["lr_schedule"] == "plateau" and config["model"]["input_mask"] == 0.4   # the recipe of Sigurdsson et al.
    assert config["n_encoder_parameters"] == b["n_parameters_by_module"]["mixer"] - 2 * 8
    snaps = sorted(os.listdir(os.path.join(run, "snapshots")))
    assert snaps and len(open(os.path.join(run, "snapshots.csv")).readlines()) == len(snaps) + 1
    ck = torch.load(os.path.join(run, "snapshots", snaps[-1]), map_location="cpu", weights_only=False)
    assert ck["state_dict"]["encoder.tables.0"].dtype == torch.bfloat16
    assert ck["state_dict"]["encoder.mlp.0.weight"].dtype == torch.float32 and ck["global_step"] > 0
    dm = out["datamodule"]
    model, ck2 = load_model(os.path.join(run, "snapshots", snaps[-1]), dm)
    assert model.encoder.tables[0].dtype == torch.float32 and ck2["samples_seen"] == ck["samples_seen"]
    rows = score_snapshots.main(["--run", run, "--split", "val", "--snapshots", "best", "--save_scores",
                                 "--scores_dir", str(tmp_path / "scores"), "--device", "cpu"])
    assert {r["metric"] for r in rows} == {"auroc", "ap", "bce"}
    z = np.load(os.path.join(tmp_path, "scores", "t", "r", "val_best.npz"))
    assert sorted(z.files) == ["iid", "labels", "logit_population", "y"] and z["logit_population"].shape == (len(z["iid"]), 6)
    summary = open(os.path.join(run, "scores_val_summary.csv")).readline().split(",")
    assert summary[:5] == ["snapshot", "label", "n", "n_cases", "auroc"]
    # a resume on the same data carries on (longer budget), with the same batches and settings checked
    out2 = main_multilabel.main(_run_args(tmp_path, "r", data_dir=data_dir, max_epoch_equiv=2) + ["--resume"])
    assert out2["samples_seen"] > out["samples_seen"] and len(out2["history"]) > len(out["history"])
    assert json.load(open(os.path.join(run, "config.json")))["resumed"] is True
    with pytest.raises(ValueError, match="other settings"):
        main_multilabel.main(_run_args(tmp_path, "r", data_dir=data_dir, embed_dim=8) + ["--resume"])


def test_a_search_records_the_new_scalars_and_a_retrain_reproduces_the_chosen_configuration(data_dir, tmp_path):
    import main_multilabel
    out = main_multilabel.main(_run_args(tmp_path, "s", data_dir=data_dir, tune_trials=1, tune_gpus="none"))
    assert out["chosen_stage"] == "deep" and set(out["best_params"]) == {"lr", "l2", "dropout", "table_kappa", "table_decay"}
    search = json.load(open(tmp_path / "t" / "s" / "search.json"))
    for key, value in (("input_mode", "blocks"), ("partition", "uniform:2"), ("snp_code", "dosage"), ("embed_dim", 4),
                       ("table_kappa", 0.03), ("table_decay", 1.0), ("table_eps", 2.5e-7), ("tune_min_validations", 8),
                       ("mixer", "transformer"), ("wide", "off")):
        assert search[key] == value
    trial = json.load(open(os.path.join(tmp_path / "t" / "s", out["best_trial_dir"], "config.json")))
    p = out["best_params"]
    again = main_multilabel.main(_run_args(tmp_path, "s_seed2", data_dir=data_dir, seed=2, lr=p["lr"], l2=p["l2"],
                                           dropout=p["dropout"], table_kappa=p["table_kappa"], table_decay=p["table_decay"]))
    retrain = json.load(open(os.path.join(again["run_dir"], "config.json")))
    assert retrain["model"] == trial["model"]                            # the same model settings, another seed
    changed = {k for k in trial["args"] if trial["args"][k] != retrain["args"][k]}
    assert changed == {"seed", "exp_name", "tune_gpus"}                 # a trial's own tune_trials is 0 already


def test_training_lowers_the_loss_on_the_synthetic_traits(tmp_path):
    """Mechanics only: three passes over 2,100 participants do not raise the training loss and the run records
    its passes. Whether the first layer learns per-SNP effects at biobank size is the E0 gate's question (design
    document, section 12): on 4,000 synthetic participants it did not, and Robert chose to keep the plain
    architecture and test it at scale (17 November 2026)."""
    import csv
    import main_multilabel
    data_dir, _ = synthetic_dataset(tmp_path, n=3000)
    out = main_multilabel.main(_run_args(tmp_path, "learn", data_dir=data_dir, max_epoch_equiv=3, val_per_epoch=1,
                                         dropout=0.1, embed_dim=8, d_model=16, n_heads=2, lr=1e-3))
    losses = {}
    with open(os.path.join(out["run_dir"], "metrics.csv")) as f:
        for r in csv.DictReader(f):
            if r["label"] == "MEAN" and r["metric"] == "bce":
                losses[int(r["validation"])] = float(r["value"])
    assert len(out["history"]) == 3 and 0.3 <= out["val"]["null_f"]["auroc"] <= 0.7
    assert min(losses.values()) <= losses[1]


def test_masked_pretraining_writes_a_file_a_run_starts_from(data_dir, tmp_path, capsys):
    import main_multilabel
    import pretrain_blocks
    out = str(tmp_path / "blocks" / "pretrained.pt")
    m = pretrain_blocks.main(["--out", out, "--epochs", "4", "--log_every", "10", "--lr", "3e-3", "--data_dir", data_dir,
                              "--label", "multi", "--rd_mode", "store", "--partition", "uniform:2", "--embed_dim", "4",
                              "--batch_size", "32", "--num_workers", "0"])
    f = torch.load(out, map_location="cpu", weights_only=False)
    assert f["manifest"]["kind"] == "masked" and f["manifest"]["embed_dim"] == 4 and f["manifest"]["snp_code"] == "dosage"
    assert set(f["state"]) == {f"tables.{i}" for i in range(CHROMS)} | {"block_emb.weight", "chrom_emb.weight",
                                                                        "mlp.0.weight", "mlp.0.bias"}
    assert m["source"]["loss_last"] < m["source"]["loss_first"]           # it learns the blocks' genotype structure
    assert m["source"]["accuracy_last"] >= m["source"]["majority_accuracy_last"] - 0.05
    assert oct(os.stat(out).st_mode)[-3:] == "600"
    run = main_multilabel.main(_run_args(tmp_path, "warm", data_dir=data_dir, init_encoder=out) + ["--freeze_encoder"])
    assert "first layer from" in capsys.readouterr().out
    config = json.load(open(os.path.join(run["run_dir"], "config.json")))
    assert config["blocks"]["init_encoder_sha256"] and config["model"]["freeze_encoder"] is True
    model = run["model"]
    assert torch.equal(model.encoder.tables[2].cpu(), f["state"]["tables.2"]) and not model.encoder.tables[2].requires_grad
    assert torch.equal(model.encoder.mlp[0].weight.cpu(), f["state"]["mlp.0.weight"])
    with pytest.raises(ValueError, match="another partition"):
        main_multilabel.main(_run_args(tmp_path, "cold", data_dir=data_dir, init_encoder=out, partition="uniform:3"))


def test_parse_args_defaults_to_blocks_mode_with_the_warm_up_and_cosine_recipe():
    import main_multilabel
    cfg = main_multilabel.parse_args(["--label", "x"])
    assert (cfg.input_mode, cfg.wide, cfg.lr_schedule, cfg.warmup_steps, cfg.final_pool, cfg.embed_dim, cfg.snp_code,
            cfg.table_kappa, cfg.table_decay, cfg.table_eps, cfg.partition, cfg.init_encoder, cfg.freeze_encoder) == \
        ("blocks", "off", "plateau", 0, "mean", 16, "dosage", 0.03, 1.0, 2.5e-7, None, None, False)
    assert (cfg.lr, cfg.l2, cfg.dropout, cfg.batch_size, cfg.input_mask, cfg.stop_patience, cfg.stop_burnin,
            cfg.tune_min_validations) == (1e-4, 1e-3, 0.5, 64, 0.4, 16, 2000, 8)
    snps = main_multilabel.parse_args(["--label", "x", "--input_mode", "snps"])
    assert (snps.final_pool, snps.input_mask) == ("atten", 0.0)                   # masking is a blocks-mode default
    assert main_multilabel.parse_args(["--label", "x", "--input_mode", "snps", "--input_mask", "0.2"]).input_mask == 0.2
    assert main_multilabel.parse_args(["--label", "x", "--final_pool", "atten"]).final_pool == "atten"


# ---- fold-routed tables -------------------------------------------------------------------------
def _folds_file(data_dir, path, k=3, seed=1):
    """A folds.npz as make_folds.py writes it, listing the training split only (the others take pseudo-folds)."""
    import pandas as pd
    iid = pd.read_csv(os.path.join(data_dir, "multi", "store", "samples_train.tsv"), sep="\t", dtype={"IID": str})["IID"].to_numpy(str)
    np.savez(path, seed=np.array(seed), k=np.array(k), iid_train=iid, fold_train=(np.arange(len(iid)) % k).astype(np.int8))
    return str(path)


def _pseudo_fold(iids, seed, k):
    return np.array([int.from_bytes(hashlib.sha256(f"{s}\t{seed}".encode()).digest()[:8], "big") % k for s in iids])


def test_fold_routed_pooling_uses_each_participants_own_table(data_dir):
    from widedeepgp.blockencoder import BlockEncoder
    _, dm, _ = _build(data_dir)
    enc = BlockEncoder(dm.partition, 4, 8, n_folds=3)
    g = torch.Generator().manual_seed(3)
    for t in enc.tables:
        assert t.shape[0] == 3
        t.data = torch.randn(t.shape, generator=g)
    x_all, _ = _val_batch(dm)
    fold = torch.arange(24) % 3
    pooled = enc.pool(x_all[:-1], fold=fold).detach().numpy()
    assert pooled.shape == (24, 3 * CHROMS, 4)
    for b in range(24):
        k = 0
        for i, ids in enumerate(dm.partition["ids"]):
            x, w = x_all[i][b, :, 0].numpy(), enc.tables[i][int(fold[b])].detach().numpy()
            for blk in np.unique(ids):
                s = ids == blk
                np.testing.assert_allclose(pooled[b, k], (x[s] @ w[s]) / np.sqrt(s.sum()), rtol=1e-5, atol=1e-6)
                k += 1
    masks = [torch.zeros(x.shape[:2], dtype=torch.bool) for x in x_all[:-1]]
    np.testing.assert_allclose(enc.pool(x_all[:-1], masks, fold).detach().numpy(), pooled, rtol=1e-6)
    with pytest.raises(ValueError, match="fold"):
        enc.pool(x_all[:-1])
    assert BlockEncoder(dm.partition, 4, 8).tables[0].dim() == 2                 # one copy: the shape as before


def test_the_data_module_routes_training_rows_by_file_and_the_rest_by_pseudo_fold(data_dir, tmp_path):
    import pandas as pd
    path = _folds_file(data_dir, tmp_path / "folds.npz", k=3, seed=7)
    cfg, dm, model = _build(data_dir, folds=path)
    assert dm.n_folds == 3 and model.encoder.k == 3 and model.hp["n_folds"] == 3 and model.hp["folds"] == path
    tr, va = dm.splits["train"], dm.splits["val"]
    assert np.array_equal(tr["fold"], np.arange(len(tr["rows"])) % 3)              # from the file, in table order
    iid_val = pd.read_csv(os.path.join(data_dir, "multi", "store", "samples_val.tsv"), sep="\t", dtype={"IID": str})["IID"].to_numpy(str)
    assert np.array_equal(va["fold"], _pseudo_fold(iid_val, 7, 3))                  # not listed: the comparators' pseudo-fold
    x_all, y = _val_batch(dm)
    assert len(x_all) == CHROMS + 2 and x_all[-1].dtype == torch.int64 and torch.equal(x_all[-1], torch.as_tensor(va["fold"][:24]))
    logits = model(x_all)
    assert logits.shape == (24, 6) and torch.isfinite(logits).all()
    # a file without the training split, or one missing training participants, is refused
    np.savez(tmp_path / "noval.npz", seed=np.array(1), k=np.array(3), iid_val=iid_val, fold_val=np.zeros(len(iid_val), np.int8))
    with pytest.raises(ValueError, match="training split"):
        _build(data_dir, folds=str(tmp_path / "noval.npz"))
    z = dict(np.load(path))
    z["iid_train"], z["fold_train"] = z["iid_train"][:-1], z["fold_train"][:-1]
    np.savez(tmp_path / "short.npz", **z)
    with pytest.raises(KeyError, match="1 of"):
        _build(data_dir, folds=str(tmp_path / "short.npz"))


def test_a_fold_routed_init_file_must_match_the_fold_count(data_dir, tmp_path):
    path = _folds_file(data_dir, tmp_path / "folds.npz", k=3)
    _, dm, model = _build(data_dir, folds=path)
    enc = model.encoder
    state = {f"tables.{i}": torch.randn(3, 6, 4) for i in range(CHROMS)}
    manifest = {"kind": "ridge", "partition_digest": dm.partition["digest"], "snp_code": "dosage", "embed_dim": 4, "n_folds": 3}
    torch.save({"manifest": manifest, "state": state}, tmp_path / "init3.pt")
    assert enc.load_init(str(tmp_path / "init3.pt"), dm.partition["digest"], "dosage", freeze=True) == sorted(state)
    assert torch.equal(enc.tables[5], state["tables.5"]) and not enc.tables[0].requires_grad
    one = {k: v[0] for k, v in state.items()}
    torch.save({"manifest": dict(manifest, n_folds=1), "state": one}, tmp_path / "init1.pt")
    with pytest.raises(ValueError, match="fold count"):
        enc.load_init(str(tmp_path / "init1.pt"), dm.partition["digest"], "dosage")
    _, _, plain = _build(data_dir)
    with pytest.raises(ValueError, match="fold count"):                              # and the other way round
        plain.encoder.load_init(str(tmp_path / "init3.pt"), dm.partition["digest"], "dosage")
    torch.save({"manifest": {k: v for k, v in manifest.items() if k != "n_folds"}, "state": one}, tmp_path / "old.pt")
    assert plain.encoder.load_init(str(tmp_path / "old.pt"), dm.partition["digest"], "dosage") == sorted(one)   # no key: one copy


def test_a_fold_routed_run_trains_snapshots_scores_and_resumes(data_dir, tmp_path):
    import main_multilabel
    import score_snapshots
    from widedeepgp.train import load_model
    path = _folds_file(data_dir, tmp_path / "folds.npz", k=2)
    out = main_multilabel.main(_run_args(tmp_path, "rf", data_dir=data_dir, folds=path))
    run = out["run_dir"]
    config = json.load(open(os.path.join(run, "config.json")))
    assert config["blocks"]["n_folds"] == 2 and config["model"]["n_folds"] == 2 and config["args"]["folds"] == path
    snaps = sorted(os.listdir(os.path.join(run, "snapshots")))
    ck = torch.load(os.path.join(run, "snapshots", snaps[-1]), map_location="cpu", weights_only=False)
    assert ck["state_dict"]["encoder.tables.0"].shape == (2, 6, 4) and ck["state_dict"]["encoder.tables.0"].dtype == torch.bfloat16
    model, _ = load_model(os.path.join(run, "snapshots", snaps[-1]), out["datamodule"])
    assert model.encoder.k == 2
    rows = score_snapshots.main(["--run", run, "--split", "val", "--snapshots", "best", "--device", "cpu"])
    assert {r["metric"] for r in rows} == {"auroc", "ap", "bce"}
    out2 = main_multilabel.main(_run_args(tmp_path, "rf", data_dir=data_dir, folds=path, max_epoch_equiv=2) + ["--resume"])
    assert out2["samples_seen"] > out["samples_seen"]
    with pytest.raises(ValueError, match="other settings"):
        main_multilabel.main(_run_args(tmp_path, "rf", data_dir=data_dir) + ["--resume"])      # one copy now: refused
