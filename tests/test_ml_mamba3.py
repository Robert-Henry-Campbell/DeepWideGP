"""Mamba-3 as the sequence block.

Runs only where upstream ``mamba_ssm`` (with Mamba3) is installed, i.e. in the second
environment (scripts/setup_mamba3_env.sh); everywhere else the file is skipped.

    CUDA_VISIBLE_DEVICES=<gpu> /mnt/sdg/robert/envs/widedeepgp-m3/bin/python -m pytest tests/test_ml_mamba3.py -q

The first call of each kernel compiles it, which takes a minute or two.
"""
import pytest
import torch

from ml_helpers import have_mamba3, make_cfg, synthetic_dataset

pytestmark = [pytest.mark.skipif(not have_mamba3(), reason="needs upstream mamba_ssm with Mamba3 (widedeepgp-m3)"),
              pytest.mark.skipif(not torch.cuda.is_available(), reason="the Mamba-3 kernels need a GPU")]

D, LENGTH, SPOT = 32, 300, 200            # LENGTH is not a multiple of the kernel's chunk of 64


def _x(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, LENGTH, D, generator=g).cuda()


def _changed(a, b):
    """Positions along the sequence at which two outputs differ."""
    return (a.float() - b.float()).abs().amax(dim=(0, 2)) > 1e-4


def test_one_mamba3_block_reads_in_one_direction():
    from mamba_ssm import Mamba3
    from widedeepgp.blocks import mamba3_settings
    torch.manual_seed(0)
    block = Mamba3(d_model=D, **mamba3_settings(D)).cuda().eval()
    x = _x()
    x2 = x.clone()
    x2[:, SPOT] += 1.0
    with torch.no_grad():
        changed = _changed(block(x), block(x2))
    assert not changed[:SPOT].any()                           # nothing before the changed SNP moves
    assert changed[SPOT:].any()


def test_the_wrapper_reads_both_ways_and_reverses_with_its_input():
    from widedeepgp.blocks import BiDirectional
    torch.manual_seed(0)
    a = BiDirectional(D).cuda().eval()
    b = BiDirectional(D).cuda().eval()
    b.fwd.load_state_dict(a.bwd.state_dict())                 # the same two blocks with their roles swapped
    b.bwd.load_state_dict(a.fwd.state_dict())
    x = _x()
    x2 = x.clone()
    x2[:, SPOT] += 1.0
    with torch.no_grad():
        y = a(x)
        changed = _changed(y, a(x2))
        y_rev = b(torch.flip(x, dims=[1]))
    assert changed[:SPOT].any() and changed[SPOT + 1:].any()  # SNPs on both sides are reached
    # reversing the sequence and swapping the directions' weights reverses the output
    assert torch.allclose(torch.flip(y_rev, dims=[1]).float(), y.float(), atol=1e-5)
    assert sum(p.numel() for p in a.fwd.parameters()) == sum(p.numel() for p in a.bwd.parameters())
    assert not any(p is q for p, q in zip(a.fwd.parameters(), a.bwd.parameters()))   # weights of their own


def _model(tmp_path, **over):
    from genostore.datamodule import SNPStoreDataModule
    from widedeepgp.train import build_model
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    cfg = make_cfg(data_dir, "multi", mixer="mamba3", **over)
    dm = SNPStoreDataModule(cfg)
    return cfg, dm, build_model(cfg, dm).cuda()


def _inputs(dm, batch=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return ([torch.randn(batch, n, 1, generator=g).cuda() for n in dm.seq_len]
            + [torch.randn(batch, dm.n_cov, generator=g).cuda()])


@pytest.mark.parametrize("precision", ["32", "bf16"])
def test_the_network_runs_and_learns_from_every_output(tmp_path, precision):
    cfg, dm, model = _model(tmp_path, precision=precision)
    model.train()
    x = _inputs(dm, batch=6)
    logits = model(x)
    y = (torch.rand(6, model.n_labels, device="cuda") < 0.3).float()
    loss, per_label = model.loss_fn(logits, y, 0)
    loss.backward()
    assert logits.shape == (6, 6) and logits.dtype == torch.float32 and torch.isfinite(logits).all()
    assert torch.isfinite(loss) and per_label.shape == (6,)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_checkpointing_changes_nothing_with_mamba3(tmp_path):
    from widedeepgp.train import build_model
    cfg, dm, plain = _model(tmp_path)
    plain.train()
    cfg.grad_ckpt = True
    ckpt = build_model(cfg, dm).cuda().train()
    ckpt.load_state_dict(plain.state_dict())
    x = _inputs(dm, batch=6)
    grads = []
    for m in (plain, ckpt):
        torch.manual_seed(7)                                  # same dropout masks
        out = m(x)
        out.square().sum().backward()
        grads.append((out.detach(), {k: p.grad.clone() for k, p in m.named_parameters()}))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-5)
    num = sum(float((grads[0][1][k] - grads[1][1][k]).pow(2).sum()) for k in grads[0][1])
    den = sum(float(grads[0][1][k].pow(2).sum()) for k in grads[0][1])
    assert (num / den) ** 0.5 < 1e-4


def test_a_run_trains_saves_and_reloads(tmp_path):
    import numpy as np
    from widedeepgp.train import load_model, predict_split, run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    cfg = make_cfg(data_dir, "multi", mixer="mamba3", run_dir=str(tmp_path / "runs"), experiment="t", exp_name="m3",
                   steps_per_epoch_equiv=8, val_per_epoch=2, max_epoch_equiv=1.0, snapshots_per_decade=4)
    out = run_training(cfg)
    assert 0.0 < out["best_val_auroc_mean"] < 1.0 and out["samples_seen"] == 8 * 16
    model, ckpt = load_model(out["best_checkpoint"], out["datamodule"], "cuda")
    assert ckpt["hyper_parameters"]["hp"]["mixer"] == "mamba3"
    a = predict_split(model, out["datamodule"], "val")[0]
    b = predict_split(out["model"].cuda(), out["datamodule"], "val")[0]
    assert np.allclose(a, b, atol=1e-5)
