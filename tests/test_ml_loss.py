"""The multi-disease loss and the metrics.  python -m pytest tests/test_ml_loss.py -q"""
import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from ml_helpers import random_labels

from widedeepgp import labels as L
from widedeepgp import metrics as M
from widedeepgp.losses import WideLoss
from widedeepgp.samplers import RotationBatchSampler


def test_one_label_shuffled_gamma0_is_the_published_loss():
    rng = np.random.default_rng(0)
    y = (rng.random((500, 1)) < 0.1).astype(np.float32)
    loss = WideLoss(y, gamma=0.0, sampler="shuffled")
    z = torch.randn(64, 1)
    yb = torch.from_numpy(y[:64])
    got, per = loss(z, yb, target=-1)
    want = nn.BCELoss()(torch.sigmoid(z), yb)
    assert abs(float(got) - float(want)) < 1e-6 and per.shape == (1,)
    assert torch.allclose(loss.logit_shift, torch.zeros(1))


def test_the_multi_disease_defaults_give_plain_cross_entropy():
    """main_multilabel.py's defaults (18 Nov 2026): uniform batches and no class weight, so every disease's term
    is its mean cross-entropy over the batch and the loss is their mean, whatever the prevalences."""
    import main_multilabel
    cfg = main_multilabel.parse_args(["--label", "x"])
    y = random_labels(n=2000)
    loss = WideLoss(y, gamma=cfg.gamma, w_max=cfg.w_max, sampler=cfg.sampler, correction=not cfg.no_sampling_correction,
                    outputs=cfg.loss_outputs)
    z = torch.randn(64, y.shape[1])
    yb = torch.from_numpy(y[:64])
    got, per = loss(z, yb, target=-1)
    want = nn.BCEWithLogitsLoss(reduction="none")(z, yb).mean(dim=0)
    assert torch.allclose(per, want, atol=1e-6) and abs(float(got) - float(want.mean())) < 1e-6
    assert torch.allclose(loss.c1, torch.ones(y.shape[1])) and torch.allclose(loss.logit_shift, torch.zeros(y.shape[1]))


def test_masked_entries_do_not_learn():
    y = random_labels(2000, missing=2)
    loss = WideLoss(y, sampler="rotation")
    yb = torch.from_numpy(y[:40])
    z = torch.randn(40, 4, requires_grad=True)
    out, _ = loss(z, yb, target=0)
    out.backward()
    nan = torch.isnan(yb)
    assert nan.any() and (z.grad[nan] == 0).all() and (z.grad[~nan] != 0).all()


def test_omega_in_the_loss_matches_the_label_module():
    y = random_labels(3000, missing=1)
    yb = torch.from_numpy(y[:200])
    for share in (0.5, 10 / 21):                                 # an even batch; a batch of 21 = 10 cases + 11 others
        loss = WideLoss(y, case_share=share)
        om = loss.omega(torch.nan_to_num(yb), ~torch.isnan(yb)).numpy()
        assert np.allclose(om, L.sampling_rate(y, *L.rotation_rates(y, share))[:200], rtol=1e-5)
        assert np.isclose(L.sampling_rate(y, case_share=share).mean(), 1.0)     # on average, drawn as often as uniformly
    assert np.allclose(L.rotation_rates(y)[0], L.rotation_rates(y, 0.5)[0])     # the default is half and half
    assert np.allclose(loss.logit_shift.numpy(), np.log(L.class_weights(L.prevalence(y), 0.5, 20.0)), atol=1e-6)


def test_every_label_is_on_the_same_scale():
    """With all logits at zero, each label's loss over the whole population is exactly ln 2."""
    y = random_labels(5000, missing=3)
    for gamma in (0.0, 0.5, 1.0):
        loss = WideLoss(y, gamma=gamma, sampler="shuffled")
        total, per = loss(torch.zeros(y.shape), torch.from_numpy(y), target=-1)
        assert torch.allclose(per, torch.full((4,), math.log(2.0)), atol=1e-5)
        assert abs(float(total) - math.log(2.0)) < 1e-5


@pytest.fixture
def one_thread():
    """Thousands of losses on a few people each: one thread is fastest, by far on a busy machine."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.mark.parametrize("batch_size", [40, 5])
@pytest.mark.parametrize("outputs", ["all", "target"])
def test_the_sampler_does_not_change_what_is_fitted(outputs, batch_size, one_thread):
    """Averaged over rotations, the batch loss equals the population loss for the same logits,
    also for an odd batch (5 = 2 cases + 3 non-cases), where rates for half and half do not."""
    y = random_labels(1200, prevalences=(0.3, 0.1, 0.05), shared=0.6, seed=2, missing=1)
    rng = np.random.default_rng(1)
    z_all = torch.from_numpy(rng.standard_normal(y.shape).astype(np.float32))   # any fixed prediction per person
    yt = torch.from_numpy(y)
    _, pop = WideLoss(y, sampler="shuffled")(z_all, yt, target=-1)              # the population objective
    s = RotationBatchSampler(y, batch_size=batch_size, steps_per_epoch=3, seed=0)
    loss = WideLoss(y, sampler="rotation", outputs=outputs, case_share=s.n_case / batch_size)
    half = WideLoss(y, sampler="rotation", outputs=outputs)                     # as if every batch were half cases
    acc, acc_half = torch.zeros(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64)
    n_rot = 2500
    for g in range(3 * n_rot):
        t, idx = s.batch(g)
        _, per = loss(z_all[idx], yt[idx], target=t)
        acc += per.double()
        if batch_size % 2:
            acc_half += half(z_all[idx], yt[idx], target=t)[1].double()
    n = 3 * n_rot if outputs == "all" else n_rot                                # each label is the target once per rotation
    assert torch.allclose((acc / n).float(), pop, rtol=0.04), (acc / n, pop)
    if batch_size % 2:
        assert not torch.allclose((acc_half / n).float(), pop, rtol=0.04)       # measured 4.5-12% off

    if outputs == "all":                                                        # without the correction it is not the same objective
        raw = WideLoss(y, sampler="rotation", correction=False)
        acc = torch.zeros(3, dtype=torch.float64)
        for g in range(3 * 400):
            t, idx = s.batch(g)
            acc += raw(z_all[idx], yt[idx], target=t)[1].double()
        assert not torch.allclose((acc / (3 * 400)).float(), pop, rtol=0.04)


def test_weighted_label_shares_match_the_population():
    """Over many rotations, people weighted by 1/omega reproduce every label's prevalence."""
    y = random_labels(1500, prevalences=(0.3, 0.1, 0.05), shared=0.6, seed=6)
    om = L.sampling_rate(y)
    s = RotationBatchSampler(y, batch_size=30, steps_per_epoch=3, seed=3)
    num, den = np.zeros(3), 0.0
    for g in range(3 * 3000):
        idx = s.batch(g)[1]
        w = 1.0 / om[idx]
        num += (w[:, None] * y[idx]).sum(0)
        den += w.sum()
    assert np.allclose(num / den, L.prevalence(y), rtol=0.03)


def test_target_only_needs_a_target():
    y = random_labels(500)
    with pytest.raises(ValueError, match="rotation sampler"):
        WideLoss(y, sampler="shuffled", outputs="target")
    with pytest.raises(ValueError, match="balanced on a label"):
        WideLoss(y, outputs="target")(torch.zeros(8, 4), torch.from_numpy(y[:8]), target=-1)


# ---- metrics ------------------------------------------------------------------------------------
def test_metrics_match_sklearn_and_skip_masked_entries():
    rng = np.random.default_rng(3)
    y = random_labels(900, missing=2)
    z = rng.standard_normal(y.shape).round(1) + 0.8 * np.nan_to_num(y)          # rounded: ties
    shift = np.array([0.0, 1.0, -0.5, 2.0])
    m = M.per_label_metrics(z, y, shift, names=list("abcd"))
    for j in range(4):
        ok = ~np.isnan(y[:, j])
        assert abs(m["auroc"][j] - roc_auc_score(y[ok, j], z[ok, j])) < 1e-12
        assert abs(m["ap"][j] - average_precision_score(y[ok, j], z[ok, j])) < 1e-12
        p = 1 / (1 + np.exp(-(z[ok, j] - shift[j])))
        want = -np.mean(y[ok, j] * np.log(p) + (1 - y[ok, j]) * np.log(1 - p))
        assert abs(m["bce"][j] - want) < 1e-9
        assert m["n"][j] == ok.sum() and m["n_cases"][j] == y[ok, j].sum()
    assert abs(m["auroc_mean"] - np.mean(m["auroc"])) < 1e-12
    rows = M.to_rows(m, split="val", step=10)
    assert len(rows) == 4 * 3 + 3 and rows[0]["label"] == "a" and rows[-1]["label"] == "MEAN" and rows[0]["step"] == 10


def test_metrics_stay_finite_without_cases():
    y = np.zeros((20, 3), dtype=np.float32)
    y[:5, 0] = 1                                             # label 1 has no case, label 2 is never applicable
    y[:, 2] = np.nan
    m = M.per_label_metrics(np.random.default_rng(0).standard_normal((20, 3)), y)
    assert not np.isnan(m["auroc"][0]) and np.isnan(m["auroc"][1]) and np.isnan(m["auroc"][2])
    assert np.isfinite(m["auroc_mean"]) and np.isfinite(m["bce"][1]) and np.isnan(m["bce"][2])
    none = M.per_label_metrics(np.zeros((4, 2)), np.zeros((4, 2)))
    assert none["auroc_mean"] == 0.5                         # nothing can be scored: still a finite number
