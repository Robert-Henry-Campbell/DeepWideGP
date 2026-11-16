"""Label matrices: float [N, n_labels] holding 1 (case), 0 (not a case) or NaN (the label does not
apply to that person, e.g. a sex-limited disease; left out of the loss and of every metric)."""
import numpy as np


def check_labels(y, names=None):
    """Raise unless every entry is 0, 1 or NaN and every label has both cases and non-cases."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError(f"a label matrix must be 2-D (people x labels), got shape {y.shape}")
    names = list(names) if names is not None else [str(j) for j in range(y.shape[1])]
    bad = ~(np.isnan(y) | (y == 0) | (y == 1))
    if bad.any():
        raise ValueError(f"labels must be 0, 1 or empty; label {names[int(np.argwhere(bad)[0, 1])]} has other values")
    n1, n0 = case_counts(y)
    for j, name in enumerate(names):
        if n1[j] == 0 or n0[j] == 0:
            raise ValueError(f"label {name} has {int(n1[j])} cases and {int(n0[j])} non-cases; it cannot be learned")


def case_counts(y):
    """(cases, non-cases) per label, among the people each label applies to."""
    y = np.asarray(y, dtype=np.float64)
    return np.nansum(y, axis=0), np.nansum(1.0 - y, axis=0)


def prevalence(y):
    n1, n0 = case_counts(y)
    return n1 / (n1 + n0)


def class_weights(pi, gamma=0.5, w_max=20.0):
    """Weight of a case relative to a non-case, min(((1 - pi) / pi) ** gamma, w_max) (methods.md 5.2)."""
    pi = np.asarray(pi, dtype=np.float64)
    return np.minimum(((1.0 - pi) / pi) ** float(gamma), float(w_max))


def rotation_rates(y, case_share=0.5):
    """How often the rotation sampler draws a case / a non-case of each label, relative to uniform
    drawing: (N s / cases, N (1 - s) / non-cases), s the share of a batch taken from its target's
    cases (samplers.cases_per_batch)."""
    y = np.asarray(y, dtype=np.float64)
    n1, n0 = case_counts(y)
    n = y.shape[0]
    s = float(case_share)
    return n * s / n1, n * (1.0 - s) / n0


def sampling_rate(y, rate_case=None, rate_ctrl=None, case_share=0.5):
    """omega: how often the rotation sampler draws each person, relative to uniform drawing: the
    mean over labels of their rate when that label is the target (0 where it does not apply). Its
    mean over the people ``rate_*`` were computed from is exactly 1."""
    y = np.asarray(y, dtype=np.float64)
    if rate_case is None:
        rate_case, rate_ctrl = rotation_rates(y, case_share)
    r = np.where(np.isnan(y), 0.0, y * rate_case + (1.0 - y) * rate_ctrl)
    return r.mean(axis=1)
