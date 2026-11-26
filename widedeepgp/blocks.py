"""The sequence block (methods.md 4.3), with the interface of models.DeepGP.Block:
``hidden_states, residual = block(hidden_states, residual)``.

    bimamba1   the published block: bidirectional Mamba-1 of the Vision-Mamba fork of mamba_ssm
               ("v1": the two directions share convolution and projections; "v2", selected by
               --bi_direct, gives the backward direction weights of its own)
    mamba3     upstream Mamba-3, one direction only, as a pair reading forwards and backwards
               (BiDirectional); it needs its own environment (the fork and upstream share a name)
    transformer  a pre-norm transformer encoder layer (torch.nn), for the block tokens, or in SNP mode for the
               SNPs of each input sequence; it keeps no residual stream outside the layer, so it returns
               (hidden_states, None)
"""
from functools import partial

import torch
import torch.nn as nn

MIXERS = ("bimamba1", "mamba3", "transformer")


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, activation="gelu",
                                                batch_first=True, norm_first=True)

    def forward(self, hidden_states, residual=None):
        return self.layer(hidden_states), None


def make_block(d_model, mixer="bimamba1", bi_direct=False, m3_d_state=None, n_heads=1, dropout=0.0):
    if mixer == "transformer":
        return TransformerBlock(d_model, n_heads, dropout)
    from models.DeepGP import Block, create_block

    if mixer == "bimamba1":
        try:
            return create_block(d_model, bimamba_type="v2" if bi_direct else "v1")
        except TypeError as e:
            raise RuntimeError("the published block needs the Vision-Mamba fork of mamba_ssm (bimamba_type); "
                               "this environment has a different mamba_ssm") from e
    if mixer == "mamba3":
        return Block(d_model, partial(BiDirectional, kind="mamba3", d_state=m3_d_state),
                     norm_cls=partial(nn.LayerNorm, eps=1e-5), fused_add_norm=False)
    raise ValueError(f"mixer must be one of {MIXERS}")


def mamba3_settings(d_model, d_state=None):
    """Mamba-3 at DeepGP's size, in single-input mode: headdim the largest power of two up to 64 that
    divides the inner width (the kernels need powers of two); d_state 16 by default (the published
    block's; Mamba-3's 128 would outweigh the rest of a narrow block), which must be even (rotary
    embedding) and a power of two; chunk_size 64, upstream's advice for this kernel, which computes its
    recurrence in bf16 whatever the precision of the rest."""
    d_inner = 2 * d_model
    headdim = 64
    while d_inner % headdim:
        headdim //= 2
    d_state = int(d_state) if d_state else 16
    if d_state < 4 or d_state & (d_state - 1):
        raise ValueError(f"m3_d_state must be a power of two of at least 4, got {d_state}")
    return dict(d_state=d_state, expand=2, headdim=headdim, is_mimo=False, chunk_size=64)


class BiDirectional(nn.Module):
    """Two one-directional blocks with weights of their own, one reading the sequence forwards and
    one backwards; their outputs are summed."""

    def __init__(self, dim, kind="mamba3", d_state=None):
        super().__init__()
        if kind != "mamba3":
            raise ValueError("only Mamba-3 is wrapped")
        try:
            from mamba_ssm import Mamba3
        except ImportError as e:
            raise RuntimeError("Mamba-3 needs upstream mamba_ssm installed from source "
                               "(the widedeepgp-m3 environment)") from e
        kw = mamba3_settings(dim, d_state)
        self.fwd = Mamba3(d_model=dim, **kw)
        self.bwd = Mamba3(d_model=dim, **kw)

    def forward(self, hidden_states, inference_params=None):
        out = self.fwd(hidden_states)
        back = self.bwd(torch.flip(hidden_states, dims=[1]))
        return out + torch.flip(back, dims=[1])
