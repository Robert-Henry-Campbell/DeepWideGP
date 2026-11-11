# -*- coding: utf-8 -*-
"""Decode genotype-store batches with torch (on whatever device the tensors live on).

The DataLoader ships *encoded* bytes (~0.29 byte per genotype); this turns them into the
float32 [B, L, 1] tensors the model consumes. Doing it after the transfer to the GPU keeps
host RAM, worker CPU and host-to-device traffic small.
"""
import torch

MISSING = 255
_SHIFTS = (0, 2, 4, 6)


def unpack_codes(packed):
    """uint8 [..., nbytes] -> uint8 [..., nbytes * 4] of 2-bit codes."""
    sh = torch.tensor(_SHIFTS, dtype=torch.uint8, device=packed.device)
    codes = torch.bitwise_and(torch.bitwise_right_shift(packed.unsqueeze(-1), sh), 3)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * 4)


def decode_packed(packed, ovr, width=None):
    """Torch twin of genostore.format.decode_packed: -> uint8 q [..., width]."""
    codes = unpack_codes(packed)
    if width is not None and width != codes.shape[-1]:
        codes = codes[..., :width]
    exc = codes == 3
    q = codes * 100                      # codes 0,1,2 -> 0,100,200 (code 3 wraps; overwritten next)
    n_exc = int(exc.sum())
    if n_exc != ovr.numel():
        raise ValueError(f"exception count mismatch: {n_exc} code-3 slots but {ovr.numel()} bytes")
    if n_exc:
        q[exc] = ovr
    return q


def decode_segment(full, full_ovr, tail, tail_ovr, block_variants, tail_width, skip, n):
    """One RawRegion (as tensors; absent parts are empty tensors) -> uint8 q [B, n]."""
    parts = []
    if full.numel():
        nbf, bsz = full.shape[0], full.shape[1]
        q = decode_packed(full, full_ovr)                                   # [nbf, B, W]
        parts.append(q.permute(1, 0, 2).reshape(bsz, nbf * block_variants))
    if tail.numel():
        parts.append(decode_packed(tail, tail_ovr, tail_width))
    q = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    if skip or n != q.shape[1]:
        q = q[:, skip:skip + n]
    return q


def to_model_input(q, mean=None, inv_std=None):
    """uint8 q [B, L] -> float32 [B, L, 1].

    With mean/inv_std (float32 [L], dosage units): standardised dosage, missing -> 0 (= the mean).
    Without: raw dosage in [0, 2], missing -> 0.
    """
    x = q.to(torch.float32)
    x.mul_(0.01)
    if mean is not None:
        x.sub_(mean).mul_(inv_std)
    miss = q == MISSING
    if bool(miss.any()):
        x.masked_fill_(miss, 0.0)
    return x.unsqueeze(-1)
