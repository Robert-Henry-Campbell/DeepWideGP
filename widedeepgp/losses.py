# -*- coding: utf-8 -*-
"""The multi-disease training loss.

The loss is defined so that it does not depend on how the batches were drawn:

    loss = (1/n) sum_j [ sum_k m_kj * c_j(y_kj) * BCE(z_kj, y_kj) / omega_k ] / (B * Z_j)

    c_j(1) = min(((1 - pi_j) / pi_j) ** gamma, w_max),  c_j(0) = 1      class weight
    omega_k = how often the sampler draws person k, relative to uniform  sampling-rate correction
    Z_j     = f_j * (pi_j * c_j(1) + 1 - pi_j)                           expected weight per draw

k runs over the B people of the batch, j over the n labels; m is 1 where a label applies; pi_j
is the training prevalence of label j among the people it applies to and f_j the share of
people it applies to.

Dividing by omega makes the expected batch sum equal to the same population average whatever
the sampler: the disease-balanced batches change how often an example is seen, not what the
network is asked to fit. Z_j and the 1/n put every label on the same scale, so a frequent
disease cannot dominate the sum. A population-scale probability is recovered from a logit by
subtracting ``logit_shift`` = log c_j(1).

Variants, for comparison runs:
    correction=False   leave out omega (the non-target labels then learn from a sample that is
                       enriched for whatever co-occurs with the batch's target)
    outputs="target"   only the label the batch was balanced on learns from it; its cases and
                       non-cases are weighted back to the same per-label objective
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import labels as L


def label_counts(train_labels):
    """Cases, non-cases and number of people of a training label matrix, as plain lists."""
    y = np.asarray(train_labels, dtype=np.float64)
    n1, n0 = L.case_counts(y)
    return {"cases": [float(v) for v in n1], "non_cases": [float(v) for v in n0], "n": int(y.shape[0])}


class WideLoss(nn.Module):
    def __init__(self, train_labels=None, gamma=0.5, w_max=20.0, sampler="rotation", correction=True, outputs="all",
                 counts=None, case_share=0.5):
        """train_labels: the training label matrix, or pass ``counts`` = {"cases": [...],
        "non_cases": [...], "n": people} (what ``label_counts`` returns), which is all the loss
        needs and is small enough to keep with a saved model. case_share: the share of a
        rotation batch drawn from its target's cases (one half unless the batch size is odd;
        samplers.cases_per_batch)."""
        super().__init__()
        if outputs not in ("all", "target"):
            raise ValueError("outputs must be 'all' or 'target'")
        if outputs == "target" and sampler != "rotation":
            raise ValueError("outputs='target' needs the rotation sampler, which names a target per batch")
        if counts is None:
            counts = label_counts(train_labels)
        n1 = np.asarray(counts["cases"], dtype=np.float64)
        n0 = np.asarray(counts["non_cases"], dtype=np.float64)
        n_people = float(counts["n"])
        pi = n1 / (n1 + n0)
        c1 = L.class_weights(pi, gamma, w_max)
        f = (n1 + n0) / n_people
        s = float(case_share)
        rate_case, rate_ctrl = n_people * s / n1, n_people * (1.0 - s) / n0      # as labels.rotation_rates
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        self.register_buffer("pi", t(pi))
        self.register_buffer("c1", t(c1))
        self.register_buffer("z", t(f * (pi * c1 + 1.0 - pi)))
        self.register_buffer("rate_case", t(rate_case))
        self.register_buffer("rate_ctrl", t(rate_ctrl))
        self.register_buffer("logit_shift", t(np.log(c1)))
        self.n_labels = len(n1)
        self.gamma, self.w_max, self.case_share = float(gamma), float(w_max), s
        self.sampler, self.correction, self.outputs = sampler, bool(correction), outputs

    def omega(self, y, mask):
        """Sampling rate of each person in the batch under the rotation sampler."""
        r = torch.where(mask, y * self.rate_case + (1.0 - y) * self.rate_ctrl, torch.zeros_like(y))
        return r.mean(dim=1)

    def forward(self, logits, y, target=None):
        """logits, y: [B, n]; y holds 1, 0 or NaN. Returns (loss, per-label losses [n])."""
        logits = logits.float()                      # the loss is always computed in float32
        mask = ~torch.isnan(y)
        y0 = torch.nan_to_num(y, nan=0.0).float()
        bce = F.binary_cross_entropy_with_logits(logits, y0, reduction="none")
        w = torch.where(y0 > 0.5, self.c1.expand_as(y0), torch.ones_like(y0)) * mask
        batch = logits.shape[0]
        if self.outputs == "target":
            t = int(target)
            if t < 0:
                raise ValueError("outputs='target' needs a batch that was balanced on a label")
            # the batch is a case-control sample for label t only: weight it back to the population
            back = torch.where(y0[:, t] > 0.5, 1.0 / self.rate_case[t], 1.0 / self.rate_ctrl[t])
            per_label = torch.zeros(self.n_labels, device=logits.device)
            per_label[t] = (w[:, t] * back * bce[:, t]).sum() / (batch * self.z[t])
            return per_label[t], per_label.detach()
        if self.sampler == "rotation" and self.correction:
            w = w / self.omega(y0, mask).clamp_min(1e-12).unsqueeze(1)
        per_label = (w * bce).sum(dim=0) / (batch * self.z)
        return per_label.mean(), per_label.detach()

    def extra_repr(self):
        return (f"n_labels={self.n_labels}, gamma={self.gamma}, w_max={self.w_max}, sampler={self.sampler}, "
                f"correction={self.correction}, outputs={self.outputs}, case_share={self.case_share:.4g}")
