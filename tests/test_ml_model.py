"""The WideDeepGP network against the published one, and its switches.

Needs a GPU (the Mamba kernels are CUDA only):  python -m pytest tests/test_ml_model.py -q
"""
import json

import numpy as np
import pytest
import torch

from ml_helpers import have_fork, make_cfg, synthetic_dataset

pytestmark = [pytest.mark.skipif(not torch.cuda.is_available(), reason="the Mamba kernels need a GPU"),
              pytest.mark.skipif(not have_fork(), reason="the published block needs the Vision-Mamba fork of mamba_ssm")]


def _data(tmp_path, **over):
    from genostore.datamodule import SNPStoreDataModule
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    cfg = make_cfg(data_dir, "multi", **over)
    return cfg, SNPStoreDataModule(cfg)


def _inputs(dm, batch=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, n, 1, generator=g).cuda() for n in dm.seq_len] + [torch.randn(batch, dm.n_cov, generator=g).cuda()]


def test_same_weights_same_output_as_the_published_model(tmp_path):
    from models import DeepGP
    from widedeepgp.train import build_model
    cfg, dm = _data(tmp_path, labels="shared_a")
    cfg.enc_len_chr, cfg.enc_len, cfg.pos = list(dm.seq_len), int(sum(dm.seq_len)), dm.pos_chr
    torch.manual_seed(0)
    old = DeepGP.SNPCOV_Model_mamba(cfg).cuda().eval()
    torch.manual_seed(0)
    new = build_model(cfg, dm).cuda().eval()
    old_sd, new_sd = old.state_dict(), new.state_dict()
    body = [k for k in new_sd if not k.startswith("loss_fn.")]
    # same seed, same construction order: the two classes start from identical weights
    assert set(body) == {k for k in old_sd if not k.endswith("position_embedding.pe")}
    assert all(torch.equal(new_sd[k], old_sd[k]) for k in body)

    torch.manual_seed(1)
    other = DeepGP.SNPCOV_Model_mamba(cfg).cuda().eval()                # "published weights"
    assert new.load_weights(other.state_dict()) == []
    x = _inputs(dm)
    with torch.no_grad():
        assert torch.allclose(torch.sigmoid(new(x)), other(x), atol=1e-6)
        assert new(x).shape == (5, 1)


def test_one_output_per_label_and_a_training_step(tmp_path):
    from widedeepgp.samplers import RotationBatchSampler
    from widedeepgp.train import build_model, train_labels
    cfg, dm = _data(tmp_path)
    model = build_model(cfg, dm).cuda()
    assert model.n_labels == 6 and model.combine.layers[-1].out_features == 6
    assert sum(p.numel() for p in model.combine.parameters()) - (23 * 16 * 256 + 256) == 6 * 257
    dm.train_sampler = RotationBatchSampler(train_labels(dm), 16, 6, seed=0)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    from lightning_fabric.utilities.apply_func import move_data_to_device
    x_all, y, target = dm.on_after_batch_transfer(move_data_to_device(batch, "cuda"), 0)
    logits = model(x_all)
    loss, per_label = model.loss_fn(logits, y, int(target))
    assert logits.shape == (16, 6) and logits.dtype == torch.float32 and torch.isfinite(loss) and per_label.shape == (6,)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("precision,tol", [("32", 1e-5), ("bf16", 2e-2)])
def test_checkpointing_changes_nothing(tmp_path, precision, tol):
    from widedeepgp.train import build_model
    cfg, dm = _data(tmp_path, precision=precision)
    plain = build_model(cfg, dm).cuda().train()
    cfg.grad_ckpt = True
    ckpt = build_model(cfg, dm).cuda().train()
    ckpt.load_state_dict(plain.state_dict())
    x = _inputs(dm, batch=6)
    grads = []
    for m in (plain, ckpt):
        torch.manual_seed(7)                                            # same dropout masks
        out = m(x)
        out.square().sum().backward()
        grads.append((out.detach(), {k: p.grad.clone() for k, p in m.named_parameters()}))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-6 if precision == "32" else 1e-3)
    num = sum(float((grads[0][1][k] - grads[1][1][k]).pow(2).sum()) for k in grads[0][1])
    den = sum(float(grads[0][1][k].pow(2).sum()) for k in grads[0][1])
    assert (num / den) ** 0.5 < tol


def test_bf16_keeps_pooling_head_and_loss_in_float32(tmp_path):
    from widedeepgp.train import build_model
    cfg, dm = _data(tmp_path)
    fp32 = build_model(cfg, dm).cuda().eval()
    cfg.precision = "bf16"
    bf16 = build_model(cfg, dm).cuda().eval()
    bf16.load_state_dict(fp32.state_dict())
    x = _inputs(dm, batch=4)
    with torch.no_grad():
        pooled = bf16._encode(0, x[0])
        a, b = fp32(x), bf16(x)
    assert pooled.dtype == torch.float32 and b.dtype == torch.float32
    assert float((a - b).abs().max()) < 0.05 and not torch.equal(a, b)  # the encoders really ran in bf16
    y = torch.zeros(4, 6, device="cuda")
    bf16.train()
    loss, _ = bf16.loss_fn(bf16(x), y, 0)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)


def test_annotations_start_as_the_plain_model(tmp_path):
    from widedeepgp.train import build_model
    cfg, dm = _data(tmp_path)
    plain = build_model(cfg, dm).cuda().eval()
    cfg.annot = True
    annotated = build_model(cfg, dm).cuda().eval()
    assert annotated.annot_dim == 4 and annotated.hp["annot_names"][0] == "causal"
    left = annotated.load_weights(plain.state_dict())                   # un-annotated weights load
    assert sorted(left) == ["annot_gain.bias", "annot_gain.weight", "annot_offset.bias", "annot_offset.weight"]
    x = _inputs(dm)
    with torch.no_grad():
        assert torch.allclose(annotated(x), plain(x), atol=1e-6)        # gain 0 and offset 0: nothing changes
        torch.nn.init.normal_(annotated.annot_gain.weight, std=0.5)
        assert not torch.allclose(annotated(x), plain(x), atol=1e-4)    # and the annotations do reach the network
    extra = sum(p.numel() for p in annotated.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 2 * (4 * 16 + 16)                                   # one gain map, one offset map, shared


def test_the_loss_corrects_for_the_share_of_cases_the_sampler_draws(tmp_path):
    from widedeepgp.model import WideDeepGP
    from widedeepgp.train import build_model
    cfg, dm = _data(tmp_path, batch_size=21)                            # 10 cases and 11 non-cases per batch
    model = build_model(cfg, dm)
    assert model.hp["case_share"] == 10 / 21 and model.loss_fn.case_share == 10 / 21
    saved_before = {k: v for k, v in model.hp.items() if k != "case_share"}     # saved before it was recorded
    assert WideDeepGP(saved_before, dm.pos_chr).loss_fn.case_share == 0.5


def test_saved_models_hold_no_tables_and_reload(tmp_path):
    from widedeepgp.model import annotations_digest
    from widedeepgp.train import load_annotations, load_model, predict_split, run_training
    data_dir, _ = synthetic_dataset(tmp_path, n=600, snps=154)
    cfg = make_cfg(data_dir, "multi", run_dir=str(tmp_path / "runs"), experiment="t", exp_name="a", annot=True,
                   steps_per_epoch_equiv=8, val_per_epoch=2, max_epoch_equiv=1.0, snapshots_per_decade=4)
    out = run_training(cfg)
    ck = torch.load(out["best_checkpoint"], map_location="cpu", weights_only=False)
    keys = list(ck["state_dict"])
    assert not any(k.endswith("position_embedding.pe") for k in keys)
    assert not any(k.startswith("annot_") and "gain" not in k and "offset" not in k for k in keys)
    hp = ck["hyper_parameters"]["hp"]
    json.dumps(hp)                                                      # plain data: no arrays, no positions
    assert hp["label_names"][0] == "shared_a" and hp["annot_dim"] == 4 and len(hp["seq_len"]) == 22
    assert hp["annot_digest"] == annotations_digest(load_annotations(out["datamodule"])[0])
    model, _ = load_model(out["best_checkpoint"], out["datamodule"], "cuda")
    a = predict_split(model, out["datamodule"], "val")[0]
    b = predict_split(out["model"].cuda(), out["datamodule"], "val")[0]
    assert np.allclose(a, b, atol=1e-6)

    import os
    run = out["run_dir"]
    snaps = sorted(os.listdir(os.path.join(run, "snapshots")))
    assert snaps[0] == "step_000000001.ckpt" and snaps[-1] == "step_000000008.ckpt"
    snap = torch.load(os.path.join(run, "snapshots", snaps[-1]), map_location="cpu", weights_only=False)
    assert "optimizer_states" not in snap and snap["samples_seen"] == 8 * 16
    for f in ("config.json", "metrics.csv", "snapshots.csv", "results.json", "checkpoints/last.ckpt"):
        assert os.path.exists(os.path.join(run, f)), f

    # an annotation file rewritten after training is refused; a model saved before the digest was
    # kept is rebuilt with whatever annotations it is given, with a warning
    dm = out["datamodule"]
    path = os.path.join(dm.dir, json.load(open(os.path.join(dm.dir, "store.json")))["annotations"]["file"])
    z = dict(np.load(path))
    z["chr1"][0, 0] += 1.0
    np.savez(path, **z)
    with pytest.raises(ValueError, match="annotation vectors given differ"):
        load_model(out["best_checkpoint"], dm, "cuda")
    del ck["hyper_parameters"]["hp"]["annot_digest"]
    torch.save(ck, str(tmp_path / "saved_before.ckpt"))
    with pytest.warns(UserWarning, match="without a digest"):
        load_model(str(tmp_path / "saved_before.ckpt"), dm, "cuda")


def test_weights_move_to_a_model_with_other_labels(tmp_path):
    from genostore.datamodule import SNPStoreDataModule
    from widedeepgp.train import build_model
    cfg, dm = _data(tmp_path)
    six = build_model(cfg, dm).cuda()
    cfg2 = make_cfg(cfg.data_dir, "multi", labels="solo_d,null_f")
    two = build_model(cfg2, SNPStoreDataModule(cfg2)).cuda()
    with pytest.raises(ValueError, match="saved shape"):
        two.load_weights(six.state_dict())
    left = two.load_weights(six.state_dict(), allow_new_head=True)
    assert sorted(left) == ["combine.layers.3.bias", "combine.layers.3.weight"]
    a, b = six.state_dict(), two.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if not k.startswith(("loss_fn.", "combine.layers.3")))
    # the same two diseases the other way round: the output layer has the same shape, but each
    # row belongs to one disease, so given the source's names it starts afresh
    cfg3 = make_cfg(cfg.data_dir, "multi", labels="null_f,solo_d")
    flipped = build_model(cfg3, SNPStoreDataModule(cfg3)).cuda()
    initial = {k: v.clone() for k, v in flipped.state_dict().items() if k.startswith("combine.layers.3")}
    left = flipped.load_weights(two.state_dict(), allow_new_head=True, source_labels=two.label_names)
    assert sorted(left) == ["combine.layers.3.bias", "combine.layers.3.weight"]
    assert all(torch.equal(flipped.state_dict()[k], v) for k, v in initial.items())
    a, b = two.state_dict(), flipped.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a if not k.startswith(("loss_fn.", "combine.layers.3")))
    assert flipped.load_weights(flipped.state_dict(), allow_new_head=True, source_labels=["null_f", "solo_d"]) == []
    assert flipped.load_weights(two.state_dict(), allow_new_head=True) == []    # a source without names: as before
