"""Blocks mode on the GPU: both mixers in bf16, activation checkpointing, and the memory of the largest
planned models at 2,000 block tokens (the plan's S5) at the batch sizes the chains use.
    CUDA_VISIBLE_DEVICES=<gpu> python -m pytest tests/test_ml_blocks_gpu.py -q
"""
import json

import numpy as np
import pytest
import torch

from ml_helpers import have_fork, make_cfg, synthetic_dataset

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
MIXERS = ["transformer"] + (["bimamba1"] if have_fork() else [])
# the store's SNPs per chromosome (data/reference/hm3plus_in_ukb_pgen.json), for the memory probes
SEQ_LEN = [102457, 105260, 88030, 81218, 79470, 82981, 70720, 68454, 57151, 66178, 63751, 61302, 46885, 41528,
           37389, 39588, 34842, 37465, 25645, 31869, 17844, 18173]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return synthetic_dataset(tmp_path_factory.mktemp("d"))[0]


def _build(data_dir, **over):
    from widedeepgp.train import build_data, build_model
    base = dict(input_mode="blocks", partition="uniform:2", final_pool="mean", n_heads=2, d_model=8, embed_dim=4)
    cfg = make_cfg(data_dir, "multi", **{**base, **over})
    dm = build_data(cfg)
    return cfg, dm, build_model(cfg, dm)


def _batch(dm, n=16):
    ds = dm._dataset({k: v[:n] for k, v in dm.splits["train"].items()})
    x_all, y = dm.decode_batch(ds[list(range(n))])
    return [x.cuda() for x in x_all], y.cuda()


@gpu
@pytest.mark.parametrize("mixer", MIXERS)
def test_both_mixers_train_a_step_in_bf16(data_dir, mixer):
    _, dm, model = _build(data_dir, mixer=mixer, precision="bf16", e_layers=2)
    model.cuda().train()
    x_all, y = _batch(dm)
    loss, _ = model.loss_fn(model(x_all), y, 0)
    loss.backward()                                                      # no loss penalty in blocks mode
    assert torch.isfinite(loss) and all(torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0 for t in model.encoder.tables)
    assert model.encoder.tables[0].dtype == torch.float32                 # master weights stay in float32


@gpu
@pytest.mark.parametrize("mixer", MIXERS)
def test_activation_checkpointing_changes_nothing(data_dir, mixer):
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(0)
        _, dm, model = _build(data_dir, mixer=mixer, e_layers=2, dropout=0.0, grad_ckpt=ckpt)
        model.cuda().train()
        with torch.no_grad():
            for t in model.encoder.tables:
                t.normal_(0, 0.1, generator=None)
        x_all, y = _batch(dm)
        loss, _ = model.loss_fn(model(x_all), y, 0)
        loss.backward()
        grads.append((float(loss), torch.cat([t.grad.flatten() for t in model.encoder.tables]),
                      torch.cat([p.grad.flatten() for p in model.layers.parameters() if p.grad is not None])))
    assert grads[0][0] == pytest.approx(grads[1][0], rel=1e-4)
    assert torch.allclose(grads[0][1], grads[1][1], rtol=1e-3, atol=1e-6)
    assert torch.allclose(grads[0][2], grads[1][2], rtol=1e-3, atol=1e-6)


def _big_model(data_dir, n_blocks, d, layers, mixer, batch, embed_dim=16, grad_ckpt=False):
    """A model over the store's real SNP counts and ``n_blocks`` blocks of about equal size, with random inputs."""
    from genostore.datamodule import partition_digest
    from widedeepgp.model import WideDeepGP
    _, _, small = _build(data_dir)
    M = sum(SEQ_LEN)
    ids, start = [], 0
    for L in SEQ_LEN:
        k = max(1, round(n_blocks * L / M))
        ids.append(np.arange(L) * k // L + start)
        start += k
    part = {"ids": ids, "n_blocks": start, "digest": partition_digest(ids), "name": f"fake{n_blocks}"}
    pos = [np.arange(L) for L in SEQ_LEN]
    from widedeepgp.model import positions_digest
    counts = {"cases": [batch] * 30, "non_cases": [batch] * 30, "n": 200000}
    hp = dict(small.hp, d_model=d, e_layers=layers, n_heads=max(1, d // 16), mixer=mixer, embed_dim=embed_dim, head_hidden=4 * d,
              label_names=[f"d{j}" for j in range(30)], label_counts=counts, seq_len=SEQ_LEN, pos_digest=positions_digest(pos),
              partition=part["name"], partition_digest=part["digest"], n_blocks=start, precision="bf16", grad_ckpt=grad_ckpt,
              lr_table=1e-3, n_cov=12, dropout=0.1)
    model = WideDeepGP(hp, pos, blocks=part).cuda().train()
    x_all = [torch.randn(batch, L, 1, device="cuda") for L in SEQ_LEN] + [torch.randn(batch, 12, device="cuda")]
    y = (torch.rand(batch, 30, device="cuda") < 0.5).float()
    return model, x_all, y


@gpu
@pytest.mark.parametrize("mixer, layers, batch", [("transformer", 8, 128)] + ([("bimamba1", 14, 128)] if have_fork() else []))
def test_the_largest_planned_model_fits_at_2000_tokens(data_dir, mixer, layers, batch):
    torch.cuda.reset_peak_memory_stats()
    model, x_all, y = _big_model(data_dir, 2000, 256, layers, mixer, batch)
    loss, _ = model.loss_fn(model(x_all), y, 0)
    loss.backward()                                                      # no loss penalty in blocks mode
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    counts = model.parameter_counts()
    print(f"\n{mixer} S5 (d 256 x {layers}) at 2,001 tokens, batch {batch}: peak {peak:.1f} GiB; mixer parameters "
          f"{counts['mixer']:,}, table {counts['table']:,}", flush=True)
    assert peak < 40
    del model, x_all, y
    torch.cuda.empty_cache()


@gpu
def test_the_planned_sizes_and_their_mixer_parameter_counts(data_dir):
    """Prints the mixer parameter count of every planned size for both mixers (the chains pin the Mamba depths
    to match the transformer's N within a factor 1.3)."""
    rows = []
    for mixer in MIXERS:
        for d, layers in [(16, 1), (32, 2), (64, 4), (128, 6), (256, 8), (16, 2), (32, 3), (64, 6), (128, 9), (256, 12),
                          (256, 14)]:
            model, _, _ = _big_model(data_dir, 431, d, layers, mixer, 2)
            counts = model.parameter_counts()
            rows.append({"block": mixer, "d": d, "layers": layers, **counts})
            del model
            torch.cuda.empty_cache()
    print("\n" + json.dumps(rows, indent=0), flush=True)
    t = {(r["d"], r["layers"]): r["mixer"] for r in rows if r["block"] == "transformer"}
    assert t[(64, 4)] == pytest.approx(12 * 4 * 64 ** 2, rel=0.05)      # about 12 L d^2
