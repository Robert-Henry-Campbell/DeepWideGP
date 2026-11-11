# -*- coding: utf-8 -*-
"""Per-disease metrics over a whole evaluation set.

Everything is computed once over all evaluated people (never batch by batch: a small batch
often holds no case of a disease, and its AUROC is then undefined), leaving out the entries
where a label does not apply. A label with no case or no non-case among the people evaluated
gets NaN, and the means skip it; if nothing can be scored the mean AUROC is reported as 0.5 so
that callers watching it (early stopping, the tuner) always see a finite number.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def auroc(score, y):
    """Area under the ROC curve from ranks (ties get their average rank)."""
    y = np.asarray(y) > 0.5
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cross_entropy(logit, y):
    """Mean binary cross-entropy of sigmoid(logit) against y, in nats."""
    logit = np.asarray(logit, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, logit) - y * logit))


def per_label_metrics(logits, y, logit_shift=None, names=None):
    """logits, y: [N, n]; y holds 1, 0 or NaN. logit_shift [n] is subtracted before the
    cross-entropy, so that it is evaluated at population prevalence.

    Returns {"labels": names, "auroc": [n], "ap": [n], "bce": [n], "n": [n], "n_cases": [n],
             "auroc_mean", "ap_mean", "bce_mean"}.
    """
    logits = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_labels = y.shape[1]
    shift = np.zeros(n_labels) if logit_shift is None else np.asarray(logit_shift, dtype=np.float64)
    out = {k: np.full(n_labels, np.nan) for k in ("auroc", "ap", "bce")}
    out["n"] = np.zeros(n_labels, dtype=np.int64)
    out["n_cases"] = np.zeros(n_labels, dtype=np.int64)
    for j in range(n_labels):
        ok = ~np.isnan(y[:, j])
        yj, zj = y[ok, j], logits[ok, j]
        out["n"][j], out["n_cases"][j] = len(yj), int(yj.sum())
        if len(yj) == 0:
            continue
        out["bce"][j] = cross_entropy(zj - shift[j], yj)
        if 0 < out["n_cases"][j] < len(yj):
            out["auroc"][j] = auroc(zj, yj)
            out["ap"][j] = float(average_precision_score(yj, zj))
    for k in ("auroc", "ap", "bce"):
        v = out[k][~np.isnan(out[k])]
        out[k + "_mean"] = float(v.mean()) if len(v) else (0.5 if k == "auroc" else float("nan"))
    out["labels"] = list(names) if names is not None else [str(j) for j in range(n_labels)]
    return out


def to_rows(metrics, **extra):
    """Long format: one dict per (label, metric), plus any constant columns in ``extra``."""
    rows = []
    for j, name in enumerate(metrics["labels"]):
        for k in ("auroc", "ap", "bce"):
            rows.append(dict(extra, label=name, metric=k, value=float(metrics[k][j]),
                             n=int(metrics["n"][j]), n_cases=int(metrics["n_cases"][j])))
    for k in ("auroc", "ap", "bce"):
        rows.append(dict(extra, label="MEAN", metric=k, value=float(metrics[k + "_mean"]),
                         n=int(metrics["n"].max()) if len(metrics["n"]) else 0, n_cases=int(metrics["n_cases"].sum())))
    return rows
