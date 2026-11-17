"""Masked-SNP pretraining of the first layer of blocks mode (DeepWideGP-Scripts
scripts/widedeepgp-blocks/methods.md, section 2 step 6): hide a share of each participant's SNPs and
predict every hidden genotype class (0, 1 or 2 copies) from its block's token. No labels, no mixer.
Trains the per-SNP table, the identity vectors and the block MLP's hidden layer on the dataset's
training rows only, and writes an --init_encoder file for main_multilabel.py.

    CUDA_VISIBLE_DEVICES=<gpu> python pretrain_blocks.py --data_dir <data_dir> --label ard30 --rd_mode <rung> \
        --partition <partition.npz> --embed_dim 16 --snp_code dosage --out <dataset>/blocks/pretrained_<name>.pt
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_fabric.utilities.apply_func import move_data_to_device

from main_multilabel import parse_args
from genostore.datamodule import SNPStoreDataModule
from widedeepgp.blockencoder import BlockEncoder
from widedeepgp.train import code_version


class MaskedDecoder(nn.Module):
    """Logit of class g for hidden SNP s: u_s . W_g h, with h the hidden layer of the SNP's block token
    and u_s one vector per SNP (the output table)."""

    def __init__(self, encoder):
        super().__init__()
        de = int(encoder.tables[0].shape[1])
        self.u = nn.ParameterList([nn.Parameter(torch.randn(sum(rows) // encoder.codes, de) * de ** -0.5)
                                   for rows in encoder.lengths])
        self.w = nn.Linear(4 * de, 3 * de)

    def logits(self, i, hidden, block_of, b, s):
        """hidden [B, n_blocks, 4 de]; block_of [L_i] the global block of each SNP; (b, s) the hidden SNPs."""
        z = self.w(hidden[b, block_of[s]]).view(len(b), 3, -1)
        return (z * self.u[i][s][:, None]).sum(-1)


def genotype_classes(x, mean, std):
    """Model inputs [B, L, 1] -> the genotype class 0/1/2 of each SNP, undoing the standardisation when there is one."""
    raw = x[:, :, 0] if mean is None else x[:, :, 0] * std + mean
    return raw.round().clamp(0, 2).long()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the --init_encoder file to write")
    ap.add_argument("--mask_rate", type=float, default=0.15)
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the training rows")
    ap.add_argument("--log_every", type=int, default=50)
    own, rest = ap.parse_known_args(argv)
    cfg = parse_args(rest + ["--input_mode", "blocks"])
    torch.manual_seed(int(cfg.seed))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dm = SNPStoreDataModule(cfg)
    dm.setup("fit")
    enc = BlockEncoder(dm.partition, cfg.embed_dim, 1, cfg.snp_code, 0.0).to(dev)
    dec = MaskedDecoder(enc).to(dev)
    block_of = [torch.as_tensor(ids, device=dev) for ids in dm.partition["ids"]]
    scale = None
    if dm.snp_mean is not None:
        scale = [(torch.from_numpy(m).to(dev), torch.from_numpy(np.where(s > 0, 1.0 / np.where(s > 0, s, 1.0), 0.0)
                                                                .astype(np.float32)).to(dev))
                 for m, s in zip(dm.snp_mean, dm.snp_inv_std)]
    params = [p for p in list(enc.parameters()) + list(dec.parameters()) if p is not enc.mlp[2].weight and p is not enc.mlp[2].bias]
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=0.0)
    n_rows = len(dm.splits["train"]["rows"])
    total_steps = max(1, int(round(own.epochs * n_rows / cfg.batch_size)))
    t0, step, hist = time.time(), 0, []
    while step < total_steps:
        for batch in dm._loader(dm.snp_train, True):
            x_all = dm.decode_batch(move_data_to_device(batch, dev))[0]
            masks = [torch.rand(x.shape[:2], device=dev) < own.mask_rate for x in x_all[:-1]]
            hidden = enc.hidden(enc.pool(x_all[:-1], masks))
            loss, correct, count, majority = 0.0, 0, 0, 0
            for i, (x, mask) in enumerate(zip(x_all[:-1], masks)):
                b, s = mask.nonzero(as_tuple=True)
                if not len(b):
                    continue
                cls = genotype_classes(x, *scale[i]) if scale else genotype_classes(x, None, None)
                logits = dec.logits(i, hidden, block_of[i], b, s)
                target = cls[b, s]
                loss = loss + F.cross_entropy(logits, target, reduction="sum")
                correct += int((logits.argmax(1) == target).sum())
                majority += int(torch.bincount(target, minlength=3).max())
                count += len(b)
            loss = loss / max(1, count)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            hist.append((float(loss), correct / max(1, count), majority / max(1, count)))
            if step % own.log_every == 0 or step == total_steps:
                recent = np.array(hist[-own.log_every:])
                print(f"step {step}/{total_steps}: masked loss {recent[:, 0].mean():.4f}, accuracy {recent[:, 1].mean():.4f} "
                      f"(majority class {recent[:, 2].mean():.4f}), {time.time() - t0:.0f} s", flush=True)
            if step >= total_steps:
                break
    keep = ("tables.", "block_emb.", "chrom_emb.", "mlp.0.")
    state = {k: v.detach().cpu() for k, v in enc.state_dict().items() if k.startswith(keep)}
    rows = np.ascontiguousarray(dm.splits["train"]["rows"], dtype=np.int64)
    first, last = np.array(hist[:own.log_every]), np.array(hist[-own.log_every:])
    manifest = {"kind": "masked", "partition_digest": dm.partition["digest"], "partition": dm.partition["name"],
                "snp_code": cfg.snp_code, "embed_dim": int(cfg.embed_dim), "n_blocks": int(dm.partition["n_blocks"]),
                "seq_len": [int(n) for n in dm.seq_len],
                "source": {"dataset_dir": os.path.abspath(dm.dir), "n_train": int(n_rows),
                           "train_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest()[:16],
                           "mask_rate": own.mask_rate, "epochs": own.epochs, "steps": step, "batch_size": int(cfg.batch_size),
                           "lr": float(cfg.lr), "seed": int(cfg.seed), "loss_first": float(first[:, 0].mean()),
                           "loss_last": float(last[:, 0].mean()), "accuracy_last": float(last[:, 1].mean()),
                           "majority_accuracy_last": float(last[:, 2].mean()), "seconds": round(time.time() - t0, 1),
                           "written": time.strftime("%Y-%m-%d %H:%M:%S"), **code_version()}}
    os.makedirs(os.path.dirname(os.path.abspath(own.out)), mode=0o700, exist_ok=True)
    torch.save({"manifest": manifest, "state": state}, own.out)
    os.chmod(own.out, 0o600)
    print(json.dumps(manifest["source"], indent=1))
    return manifest


if __name__ == "__main__":
    main()
