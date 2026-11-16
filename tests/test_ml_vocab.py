"""--snp_embed vocab: the published 201-entry table (layers/Embed.py DataEmbeddingVocab), looked up
at the genotype store's own dosage codes, on raw dosages (methods.md 11). Until 16 November 2026
the model passed the float dosages to the table, which fails at the first forward pass."""
import numpy as np
import pytest
import torch

from ml_helpers import have_fork, make_cfg, synthetic_dataset

fork = pytest.mark.skipif(not have_fork(), reason="the published block needs the Vision-Mamba fork of mamba_ssm")
gpu = pytest.mark.skipif(not (torch.cuda.is_available() and have_fork()),
                         reason="needs a GPU and the Vision-Mamba fork of mamba_ssm (the published block)")
OFF_BY_ONE = [53, 59, 105, 106, 117, 118]        # the levels the published conversion puts one code low


def test_the_codes_are_the_store_s_own():
    """Each of the store's 201 levels comes back as its own code; a missing dosage (decoded to 0) as 0."""
    from genostore import torch_decode as td
    from widedeepgp.model import vocab_codes
    q = torch.tensor(list(range(201)) + [td.MISSING], dtype=torch.uint8).view(1, -1)
    codes = vocab_codes(td.to_model_input(q))
    assert codes.dtype == torch.int64 and codes.shape == (1, 202)
    assert codes[0].tolist() == list(range(201)) + [0]


def test_the_published_conversion_puts_six_levels_one_code_low():
    """Why the model does not reuse DataEmbeddingVocab.forward's conversion (round to two decimals,
    times 100, then truncate): in float32 six levels land one code low."""
    from genostore import torch_decode as td
    x = td.to_model_input(torch.arange(201, dtype=torch.uint8).view(1, -1))
    published = (torch.round(x[:, :, 0], decimals=2) * 100).int()[0].tolist()
    assert [q for q in range(201) if published[q] != q] == OFF_BY_ONE
    assert all(published[q] == q - 1 for q in OFF_BY_ONE)


def _build(tmp_path, snp_embed):
    from genostore.datamodule import SNPStoreDataModule
    from widedeepgp.train import build_model, schedule, train_labels
    data_dir, _ = synthetic_dataset(tmp_path, n=300, snps=60)
    cfg = make_cfg(data_dir, "multi", snp_embed=snp_embed)
    dm = SNPStoreDataModule(cfg)
    torch.manual_seed(1)
    return dm, build_model(cfg, dm, schedule(cfg, len(train_labels(dm)))).eval()


def _inputs(model):
    """Store codes [B, L] for the first input sequence holding every one of the 201 levels and
    missing, and the model input they decode to without standardisation."""
    from genostore import torch_decode as td
    L = int(model.hp["seq_len"][0])
    levels = list(range(201)) + [td.MISSING]
    levels += levels[:(-len(levels)) % L]                              # whole rows of L
    q = torch.tensor(levels, dtype=torch.uint8).view(-1, L)
    return q, td.to_model_input(q)


@fork
def test_a_vocab_model_reads_raw_dosages_and_embeds_the_store_s_codes(tmp_path):
    """The embedding step is the published module's sum of its table and position parts, with the
    table looked up at the store's codes, and equals the published forward wherever the published
    conversion is right."""
    from genostore import torch_decode as td
    dm, model = _build(tmp_path, "vocab")
    assert dm.snp_mean is None and dm.snp_inv_std is None             # not standardised
    emb = model.dataEmbed[0]
    q, x = _inputs(model)
    codes = torch.where(q == td.MISSING, torch.zeros_like(q), q).long()
    with torch.no_grad():
        got = model._embed(0, x)
        assert torch.equal(got, emb.value_embedding(codes) + emb.position_embedding(x))
        right = ~torch.isin(codes, torch.tensor(OFF_BY_ONE))
        assert right.any() and (~right).any()
        assert torch.equal(got[right], emb(x)[right])
        assert not torch.equal(got[~right], emb(x)[~right])


@fork
def test_the_cov_embedding_step_is_still_the_published_module_s(tmp_path):
    """--snp_embed cov (the default): the embedding step is DataEmbedding.forward unchanged."""
    dm, model = _build(tmp_path, "cov")
    _, x = _inputs(model)
    with torch.no_grad():
        assert torch.equal(model._embed(0, x), model.dataEmbed[0](x))


def test_vocab_needs_the_linear_path_off():
    """The linear PRS path's prior and learning rate assume standardised SNPs; vocab reads raw dosages."""
    from types import SimpleNamespace
    from widedeepgp.train import wide_hparams
    for wide in ("on", "only"):
        with pytest.raises(ValueError, match="use --wide off"):
            wide_hparams(SimpleNamespace(wide=wide, snp_embed="vocab", prs_grid=None), None)
    assert wide_hparams(SimpleNamespace(wide="off", snp_embed="vocab", prs_grid=None), None) == {}


@gpu
def test_a_vocab_model_trains(tmp_path):
    """A short run with --snp_embed vocab (the network alone) trains and validates."""
    import widedeepgp.train as T
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    cfg = make_cfg(data_dir, "multi", snp_embed="vocab", wide="off", run_dir=str(tmp_path / "runs"), experiment="t",
                   exp_name="vocab", steps_per_epoch_equiv=12, val_per_epoch=2, stop_patience=0,
                   snapshots_per_decade=4, max_epoch_equiv=1.0)
    out = T.run_training(cfg)
    assert len(out["history"]) == 2 and np.isfinite(out["best_val_auroc_mean"])
    assert out["datamodule"].snp_mean is None
