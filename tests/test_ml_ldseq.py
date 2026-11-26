"""Classic DeepGP's per-sequence encoders on one input sequence per LD block (--input_mode snps with --partition;
methods.md 4.13): the blocks, their dosages, the model's shape and size, the refusals, and that the other paths are
unchanged. On the CPU; forward passes and runs need the GPU (tests/test_ml_ldseq_gpu.py).
    CUDA_VISIBLE_DEVICES= python -m pytest tests/test_ml_ldseq.py -q
"""
import json
import os

import numpy as np
import pytest
import torch

from ml_helpers import HAVE_PRIVE, have_fork, ld_top_rung_model, make_cfg, prive_block_lengths, synthetic_dataset
from test_ml_blocks import _write_partition

MIXERS = ["transformer"] + (["bimamba1"] if have_fork() else [])
needs_prive = pytest.mark.skipif(not HAVE_PRIVE, reason="needs the project's HapMap3+ store and Privé partition")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return synthetic_dataset(tmp_path_factory.mktemp("d"))[0]           # 22 chromosomes of six SNPs


def _cfg(data_dir, **over):
    base = dict(input_mode="snps", partition="uniform:2", mixer="transformer", d_model=8, n_heads=2, input_mask=0.0)
    return make_cfg(data_dir, "multi", **{**base, **over})


def _data(data_dir, **over):
    from widedeepgp.train import build_data
    return build_data(_cfg(data_dir, **over))


def _model(data_dir, **over):
    from widedeepgp.train import build_data, build_model
    cfg = _cfg(data_dir, **over)
    return cfg, build_model(cfg, build_data(cfg))


def _batch(dm, n=24):
    ds = dm._dataset({k: v[:n] for k, v in dm.splits["val"].items()})
    return dm.decode_batch(ds[list(range(n))])


def _count(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


# ---- the blocks ---------------------------------------------------------------------------------
def test_each_ld_block_becomes_one_input_sequence(data_dir):
    dm = _data(data_dir)                                                # uniform:2: three blocks per chromosome
    assert type(dm).__name__ == "LDBlockData" and len(dm.regions) == 22  # read and decoded by chromosome
    assert dm.cuts == [[2, 2, 2]] * 22 and dm.seq_len == [2] * 66 and dm.partition is None
    assert all(np.array_equal(p, np.arange(2)) for p in dm.pos_chr)     # the index along the sequence: here the block
    dm4 = _data(data_dir, partition="uniform:4")                        # 4 + 2 SNPs per chromosome
    assert dm4.cuts[0] == [4, 2] and dm4.seq_len[:3] == [4, 2, 4]
    whole, plain = _data(data_dir, partition="uniform:6"), _data(data_dir, partition=None)
    assert type(plain).__name__ == "SNPStoreDataModule" and whole.seq_len == plain.seq_len == [6] * 22
    assert all(np.array_equal(a, b) for a, b in zip(whole.pos_chr, plain.pos_chr))   # one block per chromosome
    bp, chrom = _data(data_dir, pos_mode="pos"), _data(data_dir, partition=None, pos_mode="pos")
    assert len(bp.pos_chr) == 66 and np.array_equal(np.concatenate(bp.pos_chr), np.concatenate(chrom.pos_chr))


def test_blocks_follow_the_partition_file_and_the_datasets_own_sequences(data_dir, tmp_path):
    store = _data(data_dir, partition=None).store
    _write_partition(tmp_path / "partition.npz", store, [np.array([0, 0, 1, 1, 1, 2]) + 3 * i for i in range(22)])
    assert _data(data_dir, partition=str(tmp_path)).cuts == [[2, 3, 1]] * 22
    # sequences that keep part of a chromosome, one of them in two segments: the blocks are clipped to the sequences,
    # and the block of three SNPs spans the two segments of the first one
    c1, c2 = (str(c) for c in store.chroms[:2])
    regions = [[[c1, 1, 3], [c1, 3, 6]], [[c2, 0, 4]]]
    sub = _data(data_dir, partition=str(tmp_path), geno_regions=regions)
    assert sub.regions == regions and sub.cuts == [[1, 3, 1], [2, 2]] and sub.seq_len == [1, 3, 1, 2, 2]
    x_all, _ = _batch(sub)
    assert [tuple(x.shape[1:]) for x in x_all[:-1]] == [(1, 1), (3, 1), (1, 1), (2, 1), (2, 1)]
    _write_partition(tmp_path / "bad.npz", store, [np.array([0, 1, 0, 1, 2, 2]) + 3 * i for i in range(22)])
    with pytest.raises(ValueError, match="contiguous"):                 # the partition's own checks still apply
        _data(data_dir, partition=str(tmp_path / "bad.npz"))


@pytest.mark.parametrize("stats", ["store", "train"])
def test_the_block_sequences_carry_the_chromosomes_standardised_dosages(data_dir, stats):
    blocks, plain = _data(data_dir, partition="uniform:4", snp_stats=stats), _data(data_dir, partition=None, snp_stats=stats)
    (xb, yb), (xp, yp) = _batch(blocks), _batch(plain)
    pieces = [t for x in xp[:-1] for t in torch.split(x, [4, 2], dim=1)]
    assert len(xb) == len(pieces) + 1 and all(torch.equal(a, b) for a, b in zip(xb[:-1], pieces))
    assert torch.equal(xb[-1], xp[-1]) and np.array_equal(yb.numpy(), yp.numpy(), equal_nan=True)   # covariates, labels


# ---- the model ----------------------------------------------------------------------------------
@pytest.mark.parametrize("mixer", MIXERS)
def test_the_model_has_the_published_encoder_for_every_block_and_a_head_over_all_of_them(data_dir, mixer):
    from widedeepgp.model import WideDeepGP
    cfg, model = _model(data_dir, mixer=mixer)
    _, plain = _model(data_dir, mixer=mixer, partition=None)
    assert len(model.encoders) == len(model.dataEmbed) == len(model.norm_f) == model.n_seq == 66 and plain.n_seq == 22
    assert not model.blocks and model.hp["input_mode"] == "snps" and model.hp["seq_len"] == [2] * 66
    assert type(model.encoders[0][0]).__name__ == ("TransformerBlock" if mixer == "transformer" else "Block")
    d = cfg.d_model
    assert model.combine.layers[0].in_features == (66 + 1) * d          # the 66 block vectors and the covariates'
    per_seq = _count(plain.encoders[0], plain.dataEmbed[0], plain.norm_f[0])
    assert _count(model) - _count(plain) == (66 - 22) * (per_seq + d * cfg.head_hidden)
    with pytest.raises(ValueError, match="positions"):                  # its checkpoint refuses the chromosome sequences
        WideDeepGP(model.hp, [np.arange(6)] * 22)


@needs_prive
def test_the_prive_partition_gives_431_sequences_over_all_hapmap3plus_snps():
    lengths = [n for c in prive_block_lengths() for n in c]
    assert len(lengths) == 431 and sum(lengths) == 1_258_200
    assert (min(lengths), int(np.median(lengths)), max(lengths)) == (119, 1636, 15194)


@needs_prive
@pytest.mark.parametrize("mixer,total", [("bimamba1", 8_370_814), ("transformer", 9_101_790)])
def test_the_top_rung_models_size(data_dir, mixer, total):
    """methods.md 4.13: 431 encoders of the published size and a head over 432 vectors (30 outputs, 12 covariates)."""
    if mixer not in MIXERS:
        pytest.skip("needs the Vision-Mamba fork of mamba_ssm")
    model = ld_top_rung_model(data_dir, mixer)
    per_seq = {"bimamba1": 11_008, "transformer": 12_704}[mixer] + 3 * 32 + 32        # layer, embedding, final norm
    assert model.n_seq == 431 and _count(model.encoders, model.dataEmbed, model.norm_f) == 431 * per_seq
    assert _count(model.combine) == 432 * 32 * 256 + 256 + 30 * 257 and _count(model) == total


# ---- refusals, and the paths that do not change -------------------------------------------------
def test_a_snp_list_is_refused_and_the_other_paths_are_unchanged(data_dir):
    from genostore.datamodule import SNPStoreDataModule, dataset_dir
    from widedeepgp.train import build_data
    blocks = build_data(_cfg(data_dir, input_mode="blocks", final_pool="mean", embed_dim=4))
    assert type(blocks).__name__ == "SNPStoreDataModule" and blocks.partition["n_blocks"] == 66   # blocks mode as before
    assert len(SNPStoreDataModule(_cfg(data_dir)).seq_len) == 22         # what main_genome.py builds: chromosomes
    src = dataset_dir(_cfg(data_dir))
    with open(os.path.join(src, "store.json")) as f:
        spec = json.load(f)
    spec["store"] = spec["store"] if os.path.isabs(spec["store"]) else os.path.join(src, spec["store"])
    listed = os.path.join(os.path.dirname(src), "listed")
    os.makedirs(listed, exist_ok=True)
    with open(os.path.join(listed, "store.json"), "w") as f:
        json.dump({**spec, "snps": "snps.npz"}, f)
    with pytest.raises(ValueError, match='"snps" list'):                # refused before any table is read
        build_data(_cfg(data_dir, rd_mode="listed"))
