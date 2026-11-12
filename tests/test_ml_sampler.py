"""Batch samplers.  python -m pytest tests/test_ml_sampler.py -q"""
import numpy as np
import pytest
import torch

from ml_helpers import make_cfg, random_labels, synthetic_dataset

from widedeepgp import labels as L
from widedeepgp.losses import WideLoss
from widedeepgp.samplers import NO_TARGET, RotationBatchSampler, ShuffledBatchSampler, make_sampler


def test_rotation_batches_are_balanced_on_their_target():
    y = random_labels(3000, missing=1)
    s = RotationBatchSampler(y, batch_size=20, steps_per_epoch=40, seed=5)
    for g in range(200):
        t, idx = s.batch(g)
        assert len(idx) == 20
        assert (y[idx[:10], t] == 1).all() and (y[idx[10:], t] == 0).all()      # never a "not applicable" person
    odd = RotationBatchSampler(y, batch_size=25, steps_per_epoch=4, seed=5)     # odd batch: 12 cases, 13 non-cases
    t, idx = odd.batch(0)
    assert np.nansum(y[idx, t]) == 12
    with pytest.raises(ValueError, match="at least one case"):
        RotationBatchSampler(y, batch_size=1, steps_per_epoch=4)
    empty = y.copy()
    empty[:, 3] = 0
    with pytest.raises(ValueError, match="0 cases"):
        RotationBatchSampler(empty, batch_size=8, steps_per_epoch=4)


def test_a_rotation_visits_every_label_once_in_a_changing_order():
    y = random_labels(2000, prevalences=(0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.02))
    s = RotationBatchSampler(y, batch_size=8, steps_per_epoch=7, seed=2)
    orders = [[s.batch(r * 7 + p)[0] for p in range(7)] for r in range(30)]
    assert all(sorted(o) == list(range(7)) for o in orders)
    assert len({tuple(o) for o in orders}) > 20                                 # reshuffled every rotation


def test_batches_depend_only_on_seed_and_step():
    y = random_labels(2500)
    a = RotationBatchSampler(y, batch_size=16, steps_per_epoch=10, seed=9)
    b = RotationBatchSampler(y, batch_size=16, steps_per_epoch=10, seed=9)
    for g in (57, 3, 200, 4, 57, 0):                                            # any order, repeated
        ta, ia = a.batch(g)
        tb, ib = b.batch(g) if g != 57 else RotationBatchSampler(y, 16, 10, seed=9).batch(57)
        assert ta == tb and np.array_equal(ia, ib)
    other = RotationBatchSampler(y, batch_size=16, steps_per_epoch=10, seed=10)
    assert any(not np.array_equal(a.batch(g)[1], other.batch(g)[1]) for g in range(5))

    # an epoch is a window of batch numbers; resuming at epoch 3 gives the same batches
    a.set_epoch(3)
    got = list(a)
    assert len(got) == len(a) == 10
    for i, (t, idx) in enumerate(got):
        t2, idx2 = b.batch(30 + i)
        assert t == t2 and np.array_equal(idx, idx2)


def test_cases_come_round_before_they_repeat():
    y = random_labels(4000, prevalences=(0.3, 0.02))
    n_cases = int(y[:, 1].sum())
    s = RotationBatchSampler(y, batch_size=10, steps_per_epoch=2, seed=1)
    drawn = []
    visits = n_cases // 5
    for rot in range(visits):
        for p in range(2):
            t, idx = s.batch(rot * 2 + p)
            if t == 1:
                drawn.extend(idx[:5])
    assert len(drawn) == visits * 5 and len(set(drawn)) == len(drawn)           # one pass: no case twice
    assert set(drawn) <= set(np.flatnonzero(y[:, 1] == 1))


@pytest.mark.parametrize("batch_size", [30, 5])
def test_draw_rate_matches_omega(batch_size):
    """The rate at which the sampler draws each person equals the omega the loss divides by,
    for an odd batch too (5 = 2 cases + 3 non-cases, not half and half)."""
    y = random_labels(1500, prevalences=(0.3, 0.1, 0.05), shared=0.6, seed=4, missing=2)
    s = RotationBatchSampler(y, batch_size=batch_size, steps_per_epoch=3, seed=0)
    counts = np.zeros(len(y))
    n_batches = 3 * 6000 * 30 // batch_size                                     # the same number of draws
    for g in range(n_batches):
        np.add.at(counts, s.batch(g)[1], 1)
    rate = counts / (n_batches * batch_size / len(y))
    yt = torch.from_numpy(y)
    loss = WideLoss(y, case_share=s.n_case / batch_size)
    om = loss.omega(torch.nan_to_num(yt), ~torch.isnan(yt)).double().numpy()  # what the loss divides by
    half = L.sampling_rate(y)                                                   # as if every batch were half cases
    # compare by label pattern (people with the same labels have the same omega)
    keys = [tuple(np.nan_to_num(r, nan=-1)) for r in y]
    worst_half = 0.0
    for k in set(keys):
        m = np.array([kk == k for kk in keys])
        if m.sum() >= 30:
            assert abs(rate[m].mean() / om[m].mean() - 1) < 0.06, (k, rate[m].mean(), om[m].mean())
            worst_half = max(worst_half, abs(rate[m].mean() / half[m].mean() - 1))
    assert np.corrcoef(rate, om)[0, 1] > 0.97
    if batch_size % 2:
        assert worst_half > 0.15                                                # half and half is a fifth off here


def test_the_model_records_the_share_of_cases_its_sampler_draws(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    from widedeepgp.train import model_hparams, train_labels
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=66)
    cfg = make_cfg(data_dir, "multi")
    dm = SNPStoreDataModule(cfg)
    for b in (16, 21, 19, 69):
        cfg.batch_size = b
        n_case = RotationBatchSampler(train_labels(dm), b, steps_per_epoch=4).n_case
        assert model_hparams(cfg, dm)["case_share"] == n_case / b
    assert n_case == 34                                                         # 69 = 34 cases + 35 non-cases
    cfg.sampler = "shuffled"                                                    # no target: nothing to correct
    assert model_hparams(cfg, dm)["case_share"] == 0.5


def test_shuffled_sampler():
    s = ShuffledBatchSampler(103, batch_size=10, steps_per_epoch=11, seed=3)
    t, idx = s.batch(0)
    assert t == NO_TARGET and len(idx) == 10
    one_pass = np.concatenate([s.batch(g)[1] for g in range(11)])[:103]
    assert sorted(one_pass) == list(range(103))                                 # everyone once per pass
    assert np.array_equal(s.batch(7)[1], ShuffledBatchSampler(103, 10, 11, seed=3).batch(7)[1])
    assert isinstance(make_sampler("shuffled", np.zeros((50, 2)), 5, 10), ShuffledBatchSampler)
    with pytest.raises(ValueError):
        make_sampler("other", np.zeros((50, 2)), 5, 10)
