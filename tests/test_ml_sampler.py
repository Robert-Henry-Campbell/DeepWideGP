"""Batch samplers.  python -m pytest tests/test_ml_sampler.py -q"""
import numpy as np
import pytest

from ml_helpers import random_labels

from widedeepgp import labels as L
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


def test_draw_rate_matches_omega():
    """The rate at which the sampler draws each person equals the omega the loss divides by."""
    y = random_labels(1500, prevalences=(0.3, 0.1, 0.05), shared=0.6, seed=4, missing=2)
    s = RotationBatchSampler(y, batch_size=30, steps_per_epoch=3, seed=0)
    counts = np.zeros(len(y))
    n_batches = 3 * 6000
    for g in range(n_batches):
        np.add.at(counts, s.batch(g)[1], 1)
    rate = counts / (n_batches * 30 / len(y))
    om = L.sampling_rate(y)
    # compare by label pattern (people with the same labels have the same omega)
    keys = [tuple(np.nan_to_num(r, nan=-1)) for r in y]
    for k in set(keys):
        m = np.array([kk == k for kk in keys])
        if m.sum() >= 30:
            assert abs(rate[m].mean() / om[m].mean() - 1) < 0.06, (k, rate[m].mean(), om[m].mean())
    assert np.corrcoef(rate, om)[0, 1] > 0.97


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
