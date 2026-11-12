# -*- coding: utf-8 -*-
"""Batch samplers for the multi-disease model.

Both samplers yield whole batches as ``(target, indices)``; ``StoreBatchDataset`` turns that
into one encoded batch and passes ``target`` through to the model.

    RotationBatchSampler   batch g is balanced on one "target" label: half its people are
                           cases of that label, half are not. One rotation visits every label
                           once, in an order reshuffled each rotation.
    ShuffledBatchSampler   plain shuffling; target is -1.

Every batch is a pure function of (seed, global batch number). Nothing depends on how many
batches were drawn before, so a run resumed at an epoch boundary continues the same sequence,
and several workers could draw disjoint batch numbers.

An "epoch" here is whatever number of batches the training code asks for per validation
interval (``steps_per_epoch``); Lightning calls ``set_epoch`` before each one.
"""
import numpy as np
from torch.utils.data import Sampler

NO_TARGET = -1


def cases_per_batch(batch_size, case_fraction=0.5):
    """Cases of the target in a rotation batch, round(batch_size * case_fraction); the rest are
    non-cases. Python rounds halves to even, so an odd batch is not split exactly in half (21:
    10 cases and 11 non-cases; 19: 10 and 9), and the loss uses the share this gives."""
    return int(round(int(batch_size) * float(case_fraction)))


class _Queue:
    """Endless shuffled passes over a fixed set of indices, addressable by position.

    Position p lies in pass p // n; each pass is its own permutation, derived from the seed
    and the pass number. Cases of a rare label therefore come round many times while the
    non-cases come round once: "use the cases multiple times and each control once".
    """

    def __init__(self, items, seed_key):
        self.items = np.asarray(items, dtype=np.int64)
        self.n = len(self.items)
        self.seed_key = tuple(int(s) for s in seed_key)
        self._pass, self._perm = None, None

    def _order(self, pass_no):
        if self._pass != pass_no:
            rng = np.random.default_rng(self.seed_key + (int(pass_no),))
            self._pass, self._perm = pass_no, self.items[rng.permutation(self.n)]
        return self._perm

    def take(self, start, count):
        out = []
        while count > 0:
            pass_no, off = divmod(int(start), self.n)
            k = min(count, self.n - off)
            out.append(self._order(pass_no)[off:off + k])
            start += k
            count -= k
        return out[0] if len(out) == 1 else np.concatenate(out)


class _StepSampler(Sampler):
    def __init__(self, batch_size, steps_per_epoch, seed):
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        g0 = self.epoch * self.steps_per_epoch
        for i in range(self.steps_per_epoch):
            yield self.batch(g0 + i)


class ShuffledBatchSampler(_StepSampler):
    """Shuffled passes over all training people, cut into batches."""

    kind = "shuffled"

    def __init__(self, n_items, batch_size, steps_per_epoch, seed=0):
        super().__init__(batch_size, steps_per_epoch, seed)
        self.queue = _Queue(np.arange(int(n_items)), (self.seed, 1))

    def batch(self, g):
        return NO_TARGET, self.queue.take(int(g) * self.batch_size, self.batch_size)


class RotationBatchSampler(_StepSampler):
    """One target label per batch; half the batch are its cases, half its non-cases (an odd
    batch cannot be halved: see cases_per_batch).

    labels: float [N, n_labels] with 1, 0 or NaN (not applicable; never drawn for that target).
    case_fraction: share of the batch taken from the target's cases (0.5 = balanced).
    """

    kind = "rotation"

    def __init__(self, labels, batch_size, steps_per_epoch, seed=0, case_fraction=0.5):
        super().__init__(batch_size, steps_per_epoch, seed)
        y = np.asarray(labels, dtype=np.float32)
        if y.ndim != 2:
            raise ValueError("labels must be [people, labels]")
        self.n_labels = y.shape[1]
        self.n_case = cases_per_batch(self.batch_size, case_fraction)
        self.n_ctrl = self.batch_size - self.n_case
        if self.n_case < 1 or self.n_ctrl < 1:
            raise ValueError("the batch must hold at least one case and one non-case of the target")
        self.cases, self.ctrls = [], []
        for j in range(self.n_labels):
            c1, c0 = np.flatnonzero(y[:, j] == 1), np.flatnonzero(y[:, j] == 0)
            if len(c1) == 0 or len(c0) == 0:
                raise ValueError(f"label {j} has {len(c1)} cases and {len(c0)} non-cases in the training set")
            self.cases.append(_Queue(c1, (self.seed, 2, j)))
            self.ctrls.append(_Queue(c0, (self.seed, 3, j)))
        self._rot, self._order = None, None

    def target_of(self, g):
        rot, pos = divmod(int(g), self.n_labels)
        if self._rot != rot:
            self._rot = rot
            self._order = np.random.default_rng((self.seed, 4, rot)).permutation(self.n_labels)
        return int(self._order[pos]), rot

    def batch(self, g):
        t, rot = self.target_of(g)             # label t is the target once per rotation: its rot-th visit
        idx = np.concatenate([self.cases[t].take(rot * self.n_case, self.n_case),
                              self.ctrls[t].take(rot * self.n_ctrl, self.n_ctrl)])
        return t, idx


def make_sampler(kind, labels, batch_size, steps_per_epoch, seed=0):
    if kind == "rotation":
        return RotationBatchSampler(labels, batch_size, steps_per_epoch, seed)
    if kind == "shuffled":
        return ShuffledBatchSampler(len(labels), batch_size, steps_per_epoch, seed)
    raise ValueError("sampler must be 'rotation' or 'shuffled'")
