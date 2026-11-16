"""Decode genotype-store batches with torch, on the device the tensors are on: the DataLoader ships
encoded bytes (~0.29 byte per genotype), which keeps host RAM, worker CPU and transfers small."""
import torch

from .format import MISSING


def unpack_codes(packed):
    """uint8 [..., nbytes] -> uint8 [..., nbytes * 4] of 2-bit codes."""
    sh = torch.arange(0, 8, 2, dtype=torch.uint8, device=packed.device)
    codes = torch.bitwise_and(torch.bitwise_right_shift(packed.unsqueeze(-1), sh), 3)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * 4)


def decode_packed(packed, ovr, width=None):
    """Torch twin of genostore.format.decode_packed: -> uint8 q [..., width]."""
    codes = unpack_codes(packed)
    codes = codes[..., :width] if width is not None and width != codes.shape[-1] else codes
    exc = codes == 3
    q = codes * 100                      # codes 0,1,2 -> 0,100,200 (code 3 wraps; overwritten next)
    n_exc = int(exc.sum())
    if n_exc != ovr.numel():
        raise ValueError(f"exception count mismatch: {n_exc} code-3 slots but {ovr.numel()} bytes")
    if n_exc:
        q[exc] = ovr
    return q


def decode_segment(full, full_ovr, tail, tail_ovr, block_variants, tail_width, skip, n):
    """One RawRegion as tensors (absent parts empty) -> uint8 q [B, n]; twin of GenoStore.decode."""
    parts = []
    if full.numel():
        nbf, bsz = full.shape[0], full.shape[1]
        q = decode_packed(full, full_ovr)                                   # [nbf, B, W]
        parts.append(q.permute(1, 0, 2).reshape(bsz, nbf * block_variants))
    if tail.numel():
        parts.append(decode_packed(tail, tail_ovr, tail_width))
    q = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    return q[:, skip:skip + n] if skip or n != q.shape[1] else q


def to_model_input(q, mean=None, inv_std=None):
    """uint8 q [B, L] -> float32 [B, L, 1]: dosage, standardised with mean/inv_std (float32 [L],
    dosage units) when given; missing -> 0 (the mean, once standardised)."""
    x = q.to(torch.float32)
    x.mul_(0.01)
    if mean is not None:
        x.sub_(mean).mul_(inv_std)
    x.masked_fill_(q == MISSING, 0.0)
    return x.unsqueeze(-1)
