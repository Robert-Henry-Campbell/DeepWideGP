"""The model pieces on the CPU (widedeepgp/model.py, losses.py, labels.py, metrics.py, samplers.py,
blocks.py): what saved models depend on (digests, the layout and initial values of the weights,
load_weights), the validation pass with and without a grid, the per-disease model, the training
step, the learning-rate schedule, the samplers' seed keys and the metrics' layout. Pinned values
come from commit d019bb9.
    python -m pytest tests/test_ml_model_pieces.py -q
"""
import math
import warnings

import numpy as np
import pytest
import torch

from ml_helpers import have_fork, have_mamba3, make_cfg, random_labels, synthetic_dataset

fork = pytest.mark.skipif(not have_fork(), reason="the published block needs the Vision-Mamba fork of mamba_ssm")
DEEP = ["encoders", "dataEmbed", "norm_f", "mhsa", "covariate_encoder", "cov_norm", "combine"]
LAST = "combine.layers.3."                                   # the output layer (n_fcn 2)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return synthetic_dataset(tmp_path_factory.mktemp("lean"), n=600, snps=154)[0]


def _build(data_dir, seed=1, **over):
    from genostore.datamodule import SNPStoreDataModule
    from widedeepgp.train import build_model, schedule, train_labels
    cfg = make_cfg(data_dir, "multi", **over)
    dm = SNPStoreDataModule(cfg)
    torch.manual_seed(seed)
    return cfg, dm, build_model(cfg, dm, schedule(cfg, len(train_labels(dm))))


def _val_batch(dm, seed=0):
    """A batch of 16 tuning participants, with labels of our own: every disease has cases and non-cases."""
    x_all = dm.decode_batch(next(iter(dm._loader(dm._dataset(dm.splits["val"]), False))))[0]
    y = (np.random.default_rng(seed).random((len(x_all[0]), dm.n_labels)) < 0.4).astype(np.float32)
    y[0], y[1] = 1, 0
    return x_all, torch.from_numpy(y)


def _top_level(model):
    """The saved weights' top-level names, in the order a checkpoint holds them."""
    names = []
    for k in model.state_dict():
        if not names or names[-1] != k.split(".")[0]:
            names.append(k.split(".")[0])
    return names


def _randomise(model, seed=5):
    torch.manual_seed(seed)
    with torch.no_grad():
        for w in model.prs:
            w.normal_()
        model.cov_linear.normal_()


def _validate(model, x_all, y):
    with warnings.catch_warnings(), torch.no_grad():          # as Lightning validates
        warnings.simplefilter("ignore")                       # self.log without a trainer only warns
        model.validation_step((x_all, y), 0)
        model.on_validation_epoch_end()
    return model.last_val_metrics


# ---- what saved models depend on ---------------------------------------------------------------
def test_the_digests_saved_models_are_checked_against_do_not_change():
    from widedeepgp.model import annotations_digest, positions_digest
    pos = [np.arange(5), np.array([30, 10, 20]), np.arange(12)[::3]]            # a strided view too
    assert positions_digest(pos) == "9669aa191e2baa61"
    assert positions_digest([[int(v) for v in p] for p in pos]) == "9669aa191e2baa61"
    annot = [np.linspace(-1, 1, 6).reshape(3, 2), np.ones((2, 2))[:, ::-1] * np.array([0.5, -2.0])]
    assert annotations_digest(annot) == "309f26ef58edbff3"
    assert annotations_digest([a.reshape(-1) for a in annot]) == "b0f9c124fa05e5dc"   # the shapes count


@fork
def test_the_network_starts_from_the_published_weights_and_saves_no_tables(data_dir):
    from genostore.datamodule import SNPStoreDataModule
    from models import DeepGP
    from widedeepgp.train import build_model
    cfg = make_cfg(data_dir, "multi", labels="shared_a")
    dm = SNPStoreDataModule(cfg)
    cfg.enc_len_chr, cfg.enc_len, cfg.pos = list(dm.seq_len), int(sum(dm.seq_len)), dm.pos_chr
    torch.manual_seed(1)
    old = DeepGP.SNPCOV_Model_mamba(cfg)
    after_old = torch.get_rng_state()
    torch.manual_seed(1)
    new = build_model(cfg, dm)
    assert torch.equal(torch.get_rng_state(), after_old)                        # the same draws, no more
    old_sd, new_sd = old.state_dict(), new.state_dict()
    body = [k for k in new_sd if not k.startswith("loss_fn.")]
    assert body == [k for k in old_sd if not k.endswith("position_embedding.pe")]   # names and order
    assert all(torch.equal(new_sd[k], old_sd[k]) for k in body)
    tables = [n for n, _ in new.named_buffers() if n.endswith("position_embedding.pe")]
    assert len(tables) == 22 and not any(k.endswith("position_embedding.pe") for k in new_sd)
    for emb, published in zip(new.dataEmbed, old.dataEmbed):                    # kept, but not saved
        assert torch.equal(emb.position_embedding.pe, published.position_embedding.pe)


@fork
def test_saved_models_keep_their_layout_and_initial_weights(data_dir):
    off = _build(data_dir, wide="off")[2]
    assert _top_level(off) == DEEP + ["loss_fn"]
    on = _build(data_dir, wide="on", annot=True)[2]
    assert _top_level(on) == ["cov_linear", "intercept"] + DEEP + ["prs", "annot_gain", "annot_offset", "loss_fn"]
    a, b = off.state_dict(), on.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if not k.startswith(("loss_fn.", LAST)))   # same seed, same network
    assert all(float(b[k].abs().sum()) == 0 for k in b if k.startswith((LAST, "annot_", "prs.", "cov_linear")))
    assert not any(k.startswith("annot_") and "gain" not in k and "offset" not in k for k in b)   # matrices not saved


def test_the_linear_models_keep_their_layout(data_dir):
    grid = _build(data_dir, wide="only", prs_grid=[0.1, 1, 10])[2]
    best = ["best_auroc", "best_k", "best_step", "best_samples"] + [f"best_prs_{i}" for i in range(22)]
    assert _top_level(grid) == ["cov_linear", "intercept"] + best + ["best_cov", "best_intercept", "prs", "loss_fn"]
    assert [k for k in grid.state_dict() if k.startswith("loss_fn.")] == \
        [f"loss_fn.{n}" for n in ("pi", "c1", "z", "rate_case", "rate_ctrl", "logit_shift")]
    assert grid.prs_col_scale.tolist() == pytest.approx([0.1] * 6 + [1.0] * 6 + [10.0] * 6)       # tau_k / prs_var
    assert "prs_col_scale" not in grid.state_dict()
    one = _build(data_dir, wide="only")[2]
    assert _top_level(one) == ["cov_linear", "intercept", "prs", "loss_fn"] and one.prs_col_scale is None


@fork
def test_load_weights_refuses_and_reports_what_it_leaves_out(data_dir):
    six = _build(data_dir, wide="off")[2]
    two = _build(data_dir, wide="off", labels="solo_d,null_f")[2]
    with pytest.raises(ValueError, match="saved shape"):
        two.load_weights(six.state_dict())
    head = [LAST + "weight", LAST + "bias"]
    assert two.load_weights(six.state_dict(), allow_new_head=True) == head
    flipped = _build(data_dir, wide="off", labels="null_f,solo_d")[2]
    assert flipped.load_weights(two.state_dict(), allow_new_head=True, source_labels=["solo_d", "null_f"]) == head
    with pytest.raises(KeyError, match="weights not in this model"):
        two.load_weights(dict(two.state_dict(), stray=torch.zeros(1)))
    with pytest.raises(KeyError, match="weights missing from the source"):
        two.load_weights({k: v for k, v in two.state_dict().items() if not k.startswith("mhsa.")})
    published = dict(two.state_dict(), **{"dataEmbed.0.position_embedding.pe": torch.zeros(1, 3, 16)})
    assert two.load_weights(published) == []                                    # its tables are ignored


def test_a_grid_model_s_weights_go_into_a_model_without_a_grid_only_as_a_new_head(data_dir):
    grid = _build(data_dir, wide="only", prs_grid=[0.1, 1, 10])[2]
    one = _build(data_dir, wide="only")[2]
    with pytest.raises(ValueError, match="saved shape"):                        # its best_* are dropped, not refused
        one.load_weights(grid.state_dict())
    before = one.intercept.detach().clone()
    assert one.load_weights(grid.state_dict(), allow_new_head=True) == \
        ["cov_linear", "intercept"] + [f"prs.{i}" for i in range(22)]
    assert torch.equal(one.intercept, before)


def test_checkpoint_hooks(data_dir):
    model = _build(data_dir, wide="only")[2]
    ck = {"state_dict": {"dataEmbed.0.position_embedding.pe": torch.zeros(1), "intercept": torch.zeros(6)},
          "samples_seen": 48}
    model.on_load_checkpoint(ck)
    assert list(ck["state_dict"]) == ["intercept"] and model.samples_seen == 48
    model.on_load_checkpoint({})
    assert model.samples_seen == 0
    saved = {}
    model.samples_seen = 7
    model.on_save_checkpoint(saved)
    assert saved == {"samples_seen": 7}


@fork
def test_annotation_vectors_are_checked(data_dir):
    from widedeepgp.model import WideDeepGP
    from widedeepgp.train import load_annotations
    _, dm, model = _build(data_dir, wide="on", annot=True)
    annot = load_annotations(dm)[0]
    with pytest.raises(ValueError, match="one \\[L_i, A\\] array per sequence"):
        WideDeepGP(model.hp, dm.pos_chr)
    with pytest.raises(ValueError, match="annotations of sequence 0 have shape"):
        WideDeepGP(model.hp, dm.pos_chr, [a[:, :-1] for a in annot])
    with pytest.raises(ValueError, match="differ from the ones this model was trained with"):
        WideDeepGP(model.hp, dm.pos_chr, [a + 1.0 for a in annot])
    with pytest.warns(UserWarning, match="without a digest"):
        WideDeepGP(dict(model.hp, annot_digest=None), dm.pos_chr, annot)


def test_a_model_refuses_other_positions_and_settings(data_dir):
    from widedeepgp.model import WideDeepGP
    _, dm, model = _build(data_dir, wide="only")
    with pytest.raises(ValueError, match="SNP positions"):
        WideDeepGP(model.hp, [p + 1 for p in dm.pos_chr])
    for change, message in ((dict(final_pool="concat"), "final_pool must be one of"),
                            (dict(wide="sideways"), "wide must be one of"),
                            (dict(annot_dim=3), "--wide only has no deep network"),
                            (dict(loss_outputs="target"), "every output learns"),
                            (dict(wide="on", prs_grid=[1.0]), "needs --wide only"),
                            (dict(prs_grid=[0.1, 0.0]), "must be positive")):
        with pytest.raises(ValueError, match=message):
            WideDeepGP(dict(model.hp, **change), dm.pos_chr)


# ---- the linear path, validation and the per-disease model --------------------------------------
def test_the_linear_path_is_its_prs_plus_intercepts_and_covariates(data_dir):
    _, dm, model = _build(data_dir, wide="only")
    x_all, _ = _val_batch(dm)
    _randomise(model, 3)
    with torch.no_grad():
        prs = model.prs_scores(x_all)
        assert torch.equal(prs, sum(x_all[i].reshape(len(x_all[i]), -1) @ model.prs[i] for i in range(22))
                           * model.prs_scale)
        assert torch.equal(model.wide_logits(x_all), prs + model.intercept + x_all[-1] @ model.cov_linear)
        assert torch.equal(model(x_all), model.wide_logits(x_all))


def test_a_validation_pass_without_a_grid(data_dir):
    from widedeepgp import metrics as M
    _, dm, model = _build(data_dir, wide="only")
    model.on_validation_epoch_end()                                             # nothing scored: nothing kept
    assert model.last_val_metrics is None
    x_all, y = _val_batch(dm)
    _randomise(model)
    m = _validate(model, x_all, y)
    with torch.no_grad():
        want = M.per_label_metrics(model(x_all).numpy(), y.numpy(), model.loss_fn.logit_shift.numpy(),
                                   model.label_names)
    assert list(m) == ["auroc", "ap", "bce", "n", "n_cases", "auroc_mean", "ap_mean", "bce_mean", "labels"]
    for key in ("auroc", "ap", "bce", "n", "n_cases"):
        assert np.array_equal(m[key], want[key], equal_nan=True)
    assert all(m[k] == want[k] for k in ("auroc_mean", "ap_mean", "bce_mean", "labels"))


def test_a_validation_pass_scores_every_grid_point_and_keeps_each_disease_s_best(data_dir):
    from widedeepgp import metrics as M
    _, dm, model = _build(data_dir, wide="only", prs_grid=[0.1, 1, 10])
    x_all, y = _val_batch(dm)
    y[:, 0] = 0                                                                 # disease 0 cannot be scored
    _randomise(model)
    model.samples_seen = 64
    m = _validate(model, x_all, y)
    n, k = model.n_labels, model.n_grid
    with torch.no_grad():
        want = M.per_label_metrics(model(x_all).numpy(), np.tile(y.numpy(), (1, k)),
                                   model.loss_fn.logit_shift.numpy(), model.out_names)
    assert list(m) == ["auroc", "ap", "bce", "n", "n_cases", "auroc_mean", "ap_mean", "bce_mean", "labels", "grid_pick"]
    for key in ("auroc", "ap", "bce", "n", "n_cases"):
        assert np.array_equal(m[key], want[key], equal_nan=True)
    assert m["labels"] == [f"{name}@{t:g}" for t in (0.1, 1, 10) for name in model.label_names]
    auc = np.asarray(m["auroc"]).reshape(k, n)
    pick = [-1] + [int(np.nanargmax(auc[:, j])) for j in range(1, n)]
    assert m["grid_pick"] == pick
    assert m["auroc_mean"] == pytest.approx(np.mean([auc[pick[j], j] for j in range(1, n)]), abs=1e-12)
    assert model.best_k.tolist() == pick and model.best_step.tolist() == [-1] * n     # outside a trainer: step -1
    assert model.best_samples.tolist() == [0] + [64] * (n - 1) and float(model.best_auroc[0]) == -math.inf
    for j in range(1, n):
        c = pick[j] * n + j
        assert float(model.best_auroc[j]) == auc[pick[j], j]
        assert all(torch.equal(getattr(model, f"best_prs_{i}")[:, j], model.prs[i][:, c]) for i in range(22))
        assert torch.equal(model.best_cov[:, j], model.cov_linear[:, c])
        assert model.best_intercept[j] == model.intercept[c]
    before = {key: v.clone() for key, v in model.state_dict().items() if key.startswith("best_")}
    model.samples_seen = 128
    _validate(model, x_all, y)                                                  # the same pass again changes nothing
    assert all(torch.equal(model.state_dict()[key], v) for key, v in before.items())

    state, rows = model.per_disease_state()
    ref = 1                                                                     # the grid point nearest prs_var 1
    assert list(state) == [f"prs.{i}" for i in range(22)] + ["cov_linear", "intercept"]
    assert list(rows[0]) == ["label", "prs_var", "grid_index", "chosen", "step", "samples_seen", "val_auroc"]
    assert (rows[0]["prs_var"], rows[0]["grid_index"], rows[0]["chosen"], rows[0]["step"], rows[0]["samples_seen"]) \
        == (1.0, ref, False, -1, 0) and math.isnan(rows[0]["val_auroc"])
    assert [r["grid_index"] for r in rows[1:]] == pick[1:] and all(r["chosen"] for r in rows[1:])
    assert [r["prs_var"] for r in rows] == [model.grid[g] for g in [ref] + pick[1:]]
    assert [r["label"] for r in rows] == model.label_names and [r["samples_seen"] for r in rows[1:]] == [64] * (n - 1)
    assert [type(rows[1][key]) for key in rows[1]] == [str, float, int, bool, int, int, float]
    for j in range(n):
        c = (pick[j] if j else ref) * n + j
        assert all(torch.equal(state[f"prs.{i}"][:, j], model.prs[i][:, c]) for i in range(22))
        assert torch.equal(state["cov_linear"][:, j], model.cov_linear[:, c])
        assert state["intercept"][j] == model.intercept[c]
    assert not any(v.requires_grad for v in state.values())


@pytest.mark.parametrize("grid", [None, [0.3, 3.0]])
def test_the_training_step_returns_the_loss_with_its_penalty(data_dir, grid):
    _, dm, model = _build(data_dir, wide="only", prs_grid=grid)
    x_all, y = _val_batch(dm)
    _randomise(model)
    k = model.n_grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.training_step((x_all, y, torch.tensor(-1)), 0)
    loss = model.loss_fn(model(x_all), y.repeat(1, k), -1)[0] + model.penalty()
    assert torch.equal(out, loss * k if grid else loss)                         # with a grid, summed over grid points
    assert model.samples_seen == len(y)


def test_the_learning_rate_schedule(data_dir):
    from widedeepgp.model import WideDeepGP
    _, dm, model = _build(data_dir, wide="only")
    assert isinstance(model.configure_optimizers(), torch.optim.AdamW)          # constant: no scheduler
    hp = dict(model.hp, warmup_steps=3, lr_schedule="cosine", total_optimizer_steps=10)
    out = WideDeepGP(hp, dm.pos_chr).configure_optimizers()
    assert out["lr_scheduler"]["interval"] == "step"
    f = out["lr_scheduler"]["scheduler"].lr_lambdas[0]
    want = [1 / 3, 2 / 3, 1.0] + [0.5 * (1 + math.cos(math.pi * (s - 3) / 7)) for s in range(3, 11)] + [0.0]
    assert [f(s) for s in range(12)] == pytest.approx(want, abs=1e-15)
    f = WideDeepGP(dict(hp, lr_schedule="constant"), dm.pos_chr).configure_optimizers()["lr_scheduler"]["scheduler"]
    assert [f.lr_lambdas[0](s) for s in (0, 1, 2, 50)] == pytest.approx([1 / 3, 2 / 3, 1.0, 1.0])


# ---- samplers, loss, metrics and blocks ----------------------------------------------------------
def test_the_samplers_draw_from_their_seed_keys():
    """Every batch is a function of (seed, batch number) through fixed seed keys; other keys would
    change the batches of every run, resumed runs included."""
    from widedeepgp.samplers import NO_TARGET, RotationBatchSampler, ShuffledBatchSampler

    def passes(items, key, start, count):                   # endless passes, each its own permutation
        items = np.asarray(items)
        return [int(items[np.random.default_rng(key + (p // len(items),)).permutation(len(items))][p % len(items)])
                for p in range(start, start + count)]

    y = random_labels(400, prevalences=(0.3, 0.1, 0.05))
    s = RotationBatchSampler(y, batch_size=7, steps_per_epoch=5, seed=11)
    assert (s.n_case, s.n_ctrl) == (4, 3)                                       # round(3.5) = 4
    for g in range(60):
        rot, pos = divmod(g, 3)
        t = int(np.random.default_rng((11, 4, rot)).permutation(3)[pos])
        want = (passes(np.flatnonzero(y[:, t] == 1), (11, 2, t), rot * 4, 4)
                + passes(np.flatnonzero(y[:, t] == 0), (11, 3, t), rot * 3, 3))
        got_t, got = s.batch(g)
        assert got_t == t and got.tolist() == want
    sh = ShuffledBatchSampler(50, batch_size=8, steps_per_epoch=4, seed=2)
    for g in range(20):
        t, idx = sh.batch(g)
        assert t == NO_TARGET and idx.tolist() == passes(np.arange(50), (2, 1), g * 8, 8)


def test_the_loss_constants_and_refusals():
    from widedeepgp.losses import WideLoss
    y = random_labels(2000, missing=1)
    with pytest.raises(ValueError, match="outputs must be"):
        WideLoss(y, outputs="some")
    loss = WideLoss(y, case_share=0.4)
    assert [name for name, _ in loss.named_buffers()] == ["pi", "c1", "z", "rate_case", "rate_ctrl", "logit_shift"]
    n1, n0 = np.nansum(y, axis=0), np.nansum(1 - y, axis=0)
    pi = n1 / (n1 + n0)
    c = np.minimum(np.sqrt((1 - pi) / pi), 20.0)
    assert np.allclose(loss.z.numpy(), (n1 + n0) / 2000 * (pi * c + 1 - pi), rtol=1e-6)   # f_j: label 1 applies to half
    assert np.allclose(loss.rate_case.numpy(), 2000 * 0.4 / n1, rtol=1e-6)
    assert np.allclose(loss.rate_ctrl.numpy(), 2000 * 0.6 / n0, rtol=1e-6)
    assert loss.case_share == 0.4 and loss.n_labels == 4


def test_the_metrics_layout():
    from widedeepgp import metrics as M
    y = random_labels(300, missing=1)
    m = M.per_label_metrics(np.random.default_rng(0).standard_normal(y.shape), y, None, list("abcd"))
    assert list(m) == ["auroc", "ap", "bce", "n", "n_cases", "auroc_mean", "ap_mean", "bce_mean", "labels"]
    rows = M.to_rows(m, split="val", step=3)
    assert list(rows[0]) == ["split", "step", "label", "metric", "value", "n", "n_cases"]
    assert [(r["label"], r["metric"]) for r in rows[:3]] == [("a", "auroc"), ("a", "ap"), ("a", "bce")]
    assert [(r["label"], r["metric"]) for r in rows[-3:]] == [("MEAN", "auroc"), ("MEAN", "ap"), ("MEAN", "bce")]
    assert all(r["n"] == int(m["n"].max()) and r["n_cases"] == int(m["n_cases"].sum()) for r in rows[-3:])
    empty = M.to_rows(M.per_label_metrics(np.zeros((0, 0)), np.zeros((0, 0))))
    assert [(r["value"], r["n"]) for r in empty][0] == (0.5, 0)                 # nothing scored: still a row


def test_the_mamba3_settings_and_the_refused_wrappers():
    from widedeepgp.blocks import BiDirectional, mamba3_settings
    assert mamba3_settings(32) == dict(d_state=16, expand=2, headdim=64, is_mimo=False, chunk_size=64)
    assert (mamba3_settings(24, 8)["headdim"], mamba3_settings(24, 8)["d_state"]) == (16, 8)    # 48 = 3 x 16
    for bad in (2, 12):
        with pytest.raises(ValueError, match="power of two"):
            mamba3_settings(32, bad)
    with pytest.raises(ValueError, match="only Mamba-3"):
        BiDirectional(8, kind="mamba2")


@pytest.mark.skipif(not have_fork() or have_mamba3(), reason="needs the fork's environment, without Mamba-3")
def test_unknown_blocks_and_mamba3_outside_its_environment_are_refused():
    from widedeepgp.blocks import make_block
    with pytest.raises(ValueError, match="mixer must be one of \\('bimamba1', 'mamba3'\\)"):
        make_block(16, "lstm")
    with pytest.raises(RuntimeError, match="widedeepgp-m3"):
        make_block(16, "mamba3")
