"""Classic DeepGP's encoders on LD-block sequences (methods.md 4.13) on the GPU: both mixers train in bf16, a run
re-scores to its record and resumes, the transformer runs the largest Privé block (15,194 SNPs) through a fast
attention kernel, and the top-rung model's memory and step time (slow; printed with -s).
    CUDA_VISIBLE_DEVICES=<gpu> python -m pytest tests/test_ml_ldseq_gpu.py -q -s
"""
import json
import os
import time

import numpy as np
import pytest
import torch

from ml_helpers import HAVE_PRIVE, have_fork, ld_top_rung_model, make_cfg, synthetic_dataset

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
MIXERS = ["transformer"] + (["bimamba1"] if have_fork() else [])


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    return synthetic_dataset(tmp_path_factory.mktemp("d"))[0]


def _run_args(tmp_path, data_dir, name, mixer, **over):
    """The published settings of the 30-disease classic run (runs/classic_deepgp/run_classic30_n215872.sh), small."""
    args = ["--data_dir", data_dir, "--label", "multi", "--rd_mode", "store", "--run_dir", str(tmp_path),
            "--experiment", "t", "--exp_name", name,
            "--input_mode", "snps", "--partition", "uniform:2", "--mixer", mixer, "--d_model", "8", "--n_heads", "2",
            "--e_layers", "1", "--final_pool", "atten", "--lr", "7e-4", "--l2", "1e-4", "--dropout", "0.2",
            "--lr_schedule", "constant", "--warmup_steps", "0", "--stop_burnin", "0", "--precision", "bf16",
            "--batch_size", "16", "--num_workers", "0", "--max_epoch_equiv", "1", "--val_per_epoch", "2",
            "--min_participants", "0", "--val_min_participants", "0", "--stop_patience", "0",
            "--snapshots_per_decade", "2", "--tune_trials", "0", "--eval_batch_size", "64"]
    for k, v in over.items():
        args += [f"--{k}", str(v)]
    return args


@gpu
@pytest.mark.parametrize("mixer", MIXERS)
def test_both_mixers_train_a_step_on_ld_block_sequences_in_bf16(data_dir, mixer):
    from widedeepgp.train import build_data, build_model
    cfg = make_cfg(data_dir, "multi", input_mode="snps", partition="uniform:2", mixer=mixer, d_model=8, n_heads=2,
                   precision="bf16", input_mask=0.0)
    dm = build_data(cfg)
    model = build_model(cfg, dm).cuda().train()
    ds = dm._dataset({k: v[:16] for k, v in dm.splits["train"].items()})
    x_all, y = dm.decode_batch(ds[list(range(16))])
    loss, _ = model.loss_fn(model([x.cuda() for x in x_all]), y.cuda(), 0)
    loss.backward()
    assert torch.isfinite(loss) and model.n_seq == 66
    assert all(any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters()) for enc in model.encoders)


@gpu
@pytest.mark.parametrize("mixer", MIXERS)
def test_a_run_on_ld_block_sequences_rescores_to_its_record_and_resumes(data_dir, tmp_path, mixer):
    import main_multilabel
    import score_snapshots
    out = main_multilabel.main(_run_args(tmp_path, data_dir, mixer, mixer))
    config = json.load(open(os.path.join(out["run_dir"], "config.json")))
    assert config["model"]["input_mode"] == "snps" and config["model"]["seq_len"] == [2] * 66 and "blocks" not in config
    assert config["args"]["partition"] == "uniform:2" and config["model"]["mixer"] == mixer
    rows = score_snapshots.main(["--run", out["run_dir"], "--split", "val", "--snapshots", "best", "--device", "cuda"])
    mean = [r["value"] for r in rows if r["snapshot"] == "best" and r["label"] == "MEAN" and r["metric"] == "auroc"]
    assert len(mean) == 1 and abs(float(mean[0]) - out["best_val_auroc_mean"]) < 1e-4    # the record, re-scored
    out2 = main_multilabel.main(_run_args(tmp_path, data_dir, mixer, mixer, max_epoch_equiv=2) + ["--resume"])
    assert out2["samples_seen"] > out["samples_seen"] and len(out2["history"]) > len(out["history"])
    with pytest.raises(ValueError):                                      # not on the chromosome sequences
        main_multilabel.main(_run_args(tmp_path, data_dir, mixer, mixer, max_epoch_equiv=3, partition="uniform:6")
                             + ["--resume"])


@gpu
def test_the_transformer_runs_the_largest_ld_block_through_a_fast_attention_kernel():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from widedeepgp.blocks import make_block
    layer = make_block(32, "transformer", n_heads=4, dropout=0.2).cuda().train()
    x = torch.randn(4, 15194, 32, device="cuda", requires_grad=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):      # no plain-kernel fallback
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _ = layer(x)
        h.float().square().mean().backward()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    # the plain kernel would hold 4 people x 4 heads x 15,194^2 scores: 7.4 GiB in bf16
    assert torch.isfinite(x.grad).all() and peak < 1.0, peak


@gpu
@pytest.mark.slow
@pytest.mark.skipif(not HAVE_PRIVE, reason="needs the project's HapMap3+ store and Privé partition")
@pytest.mark.parametrize("ckpt", [False, True])
@pytest.mark.parametrize("mixer", MIXERS)
def test_the_top_rung_model_trains_a_batch_and_reports_its_memory_and_step_time(data_dir, mixer, ckpt):
    """431 sequences of the real block lengths, 30 outputs, 12 covariates, width 32, one layer, bf16, batch 16;
    random standardised dosages (the step costs the same whatever the genotypes)."""
    torch.manual_seed(1)
    model = ld_top_rung_model(data_dir, mixer, precision="bf16", grad_ckpt=ckpt).cuda().train()
    b = 16
    x_all = [torch.randn(b, n, 1, device="cuda") for n in model.hp["seq_len"]] + [torch.randn(b, 12, device="cuda")]
    y = (torch.rand(b, 30, device="cuda") < 0.1).float()
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(4):
        t = time.perf_counter()
        loss, _ = model.loss_fn(model(x_all), y, 0)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"\n{mixer} checkpointing={ckpt} batch {b}: {np.median(times[1:]):.2f} s a step after the first "
          f"({times[0]:.1f} s), peak {peak:.1f} GiB, {peak / b:.2f} GiB a person")
    assert torch.isfinite(loss) and peak < 40
