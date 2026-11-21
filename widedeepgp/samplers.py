"""Batch samplers for the multi-disease model (methods.md 5.1). Both yield whole batches as
(target, indices), target being the label the batch is balanced on (NO_TARGET when shuffled),
which StoreBatchDataset passes through to the model.

Every batch is a pure function of (seed, global batch number), so a run resumed at an epoch
boundary continues the same sequence. An "epoch" is the number of batches the training code asks
for per validation interval (steps_per_epoch); Lightning calls set_epoch before each one.
"""
import numpy as np
from torch.utils.data import Sampler

NO_TARGET = -1


def cases_per_batch(batch_size, case_fraction=0.5):
    """Cases of the target in a rotation batch, round(batch_size * case_fraction), the rest non-cases.
    Python rounds halves to even, so an odd batch is not split in half (21: 10 cases and 11
    non-cases; 19: 10 and 9), and the loss uses the share this gives."""
    return int(round(int(batch_size) * float(case_fraction)))


class _Queue:
    """Endless shuffled passes over a fixed set of indices, addressable by position: position p lies
    in pass p // n, each pass its own permutation from the seed key and the pass number. The cases of
    a rare label so come round many times while its non-cases come round once."""

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

    def __init__(self, n_items, batch_size, steps_per_epoch, seed=0):
        super().__init__(batch_size, steps_per_epoch, seed)
        self.queue = _Queue(np.arange(int(n_items)), (self.seed, 1))

    def batch(self, g):
        return NO_TARGET, self.queue.take(int(g) * self.batch_size, self.batch_size)


class RotationBatchSampler(_StepSampler):
    """One target label per batch, which takes its cases and non-cases (cases_per_batch); a rotation
    makes every label the target once, in an order reshuffled every rotation. labels: float
    [N, n_labels] of 1, 0 or NaN (not applicable: never drawn for that target)."""

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

    def batch(self, g):
        rot, pos = divmod(int(g), self.n_labels)
        if self._rot != rot:
            self._rot = rot
            self._order = np.random.default_rng((self.seed, 4, rot)).permutation(self.n_labels)
        t = int(self._order[pos])                # the target, on its rot-th visit
        return t, np.concatenate([self.cases[t].take(rot * self.n_case, self.n_case),
                                  self.ctrls[t].take(rot * self.n_ctrl, self.n_ctrl)])


class IndexBatches(Sampler):
    """A batch sampler's index arrays without their target, for one-disease models that take (x, y)."""

    def __init__(self, sampler):
        self.sampler = sampler

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        return (idx for _, idx in self.sampler)


def balanced_one_disease(y, batch_size, seed=0):
    """Half cases in every batch for one 0/1 label; an epoch shows every non-case once, the cases as often as
    that takes (they are recycled in reshuffled passes)."""
    y = np.asarray(y, dtype=np.float32).reshape(len(y), -1)
    if y.shape[1] != 1:
        raise ValueError("one label column expected")
    rot = RotationBatchSampler(y, batch_size, 1, seed)
    rot.steps_per_epoch = -(-int((y[:, 0] == 0).sum()) // rot.n_ctrl)
    return IndexBatches(rot)


def make_sampler(kind, labels, batch_size, steps_per_epoch, seed=0):
    if kind == "rotation":
        return RotationBatchSampler(labels, batch_size, steps_per_epoch, seed)
    if kind == "shuffled":
        return ShuffledBatchSampler(len(labels), batch_size, steps_per_epoch, seed)
    raise ValueError("sampler must be 'rotation' or 'shuffled'")
