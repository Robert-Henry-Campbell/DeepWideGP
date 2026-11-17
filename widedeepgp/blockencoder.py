"""The first layer of blocks mode (DeepWideGP-Scripts scripts/widedeepgp-blocks/methods.md, section 2):
a learned vector per SNP, scaled by the participant's dosage and summed over the SNPs of each LD block
(``pool``), plus a learned identity vector per block and per chromosome, through one small MLP shared
by all blocks (``tokens``): one token per block for the mixer. With embed_dim 1 the pooled values are the
per-block sums of a ridge PRS, so the first layer contains a ridge; everything the model predicts from them
goes through the mixer and the head."""
import numpy as np
import torch
import torch.nn as nn

SNP_CODES = ("dosage", "genotype")


def genotype_weights(x):
    """Raw dosages [B, L] -> [B, L, 3]: the weights of the 0-, 1- and 2-copy vectors (hat functions at
    0, 1 and 2, so a dosage between two classes interpolates between their vectors)."""
    return torch.stack([(1 - x).clamp_min(0), 1 - (x - 1).abs(), (x - 1).clamp_min(0)], -1)


def table_learning_rate(kappa, batch_eff, n_snps):
    """Learning rate of the table, kappa sqrt(B) / sqrt(M): the linear PRS path's rule (model.wide_learning_rate)
    without its prior-variance and class-weight factors, as each entry's gradient is its dosage times the block
    token's gradient, of order 1 / sqrt(M). The table's optimiser has a large epsilon and a per-step weight
    decay (model.WideDeepGP.configure_optimizers), so an entry with a tiny, inconsistent gradient barely moves."""
    return float(kappa) * float(batch_eff) ** 0.5 / float(n_snps) ** 0.5


class BlockEncoder(nn.Module):
    """partition: as genostore.datamodule.load_partition returns it. The tables hold ``codes`` rows per
    SNP (1 for the dosage code, 3 for the genotype code, SNP-major), so block k of sequence i is rows
    [codes * start_k, codes * stop_k) of tables[i]."""

    def __init__(self, partition, embed_dim, d_model, snp_code="dosage", dropout=0.0):
        super().__init__()
        if snp_code not in SNP_CODES:
            raise ValueError(f"snp_code must be one of {SNP_CODES}")
        self.codes = 3 if snp_code == "genotype" else 1
        de = int(embed_dim)
        self.lengths, chrom = [], []                      # table rows per block, and each block's sequence
        for i, ids in enumerate(partition["ids"]):
            counts = np.unique(ids, return_counts=True)[1]
            self.lengths.append([int(c) * self.codes for c in counts])
            chrom += [i] * len(counts)
        self.tables = nn.ParameterList([nn.Parameter(torch.zeros(sum(rows), de)) for rows in self.lengths])
        # each block's sum is divided by the square root of its SNP count, so blocks of 119 and 15,194 SNPs start on one scale
        snps = [rows // self.codes for seq in self.lengths for rows in seq]
        self.register_buffer("block_scale", torch.tensor(snps, dtype=torch.float32).rsqrt()[None, :, None], persistent=False)
        self.block_emb = nn.Embedding(len(chrom), de)
        self.chrom_emb = nn.Embedding(len(self.lengths), de)
        self.register_buffer("block_chrom", torch.tensor(chrom, dtype=torch.long), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(de, 4 * de), nn.GELU(), nn.Linear(4 * de, int(d_model)))
        self.drop = nn.Dropout(float(dropout))

    def pool(self, x_all, masks=None):
        """Dosages, one [B, L_i, 1] per sequence (standardised for the dosage code, raw for the genotype
        code) -> the pooled vector of every block, [B, n_blocks, embed_dim]: the sum of the block's dosage-scaled
        vectors over the square root of its SNP count. ``masks``: optional bool [B, L_i] per sequence; a masked
        SNP contributes nothing (pretraining)."""
        out = []
        for i, (x, table, rows) in enumerate(zip(x_all, self.tables, self.lengths)):
            w = x[:, :, 0]
            if masks is not None:
                w = w.masked_fill(masks[i], 0.0)
            if self.codes == 3:
                w = genotype_weights(w).reshape(w.shape[0], -1)
            if torch.is_autocast_enabled() and w.is_cuda:   # cast once, not once per block
                dt = torch.get_autocast_dtype("cuda")
                w, table = w.to(dt), table.to(dt)
            out += [a @ t for a, t in zip(w.split(rows, 1), table.split(rows, 0))]
        return torch.stack(out, 1) * self.block_scale

    def hidden(self, pooled):
        """The MLP's hidden layer [B, n_blocks, 4 embed_dim] of pooled vectors with the identities added."""
        return self.mlp[1](self.mlp[0](pooled + self.block_emb.weight + self.chrom_emb(self.block_chrom)))

    def tokens(self, pooled):
        """Pooled vectors [B, n_blocks, embed_dim] -> the block tokens [B, n_blocks, d_model]."""
        return self.drop(self.mlp[2](self.hidden(pooled)))

    def forward(self, x_all):
        return self.tokens(self.pool(x_all))

    def load_init(self, path, digest, snp_code, freeze=False):
        """Start from the tables (and whatever else is there) of an --init_encoder file, which must have
        been made for this partition, SNP code and width. Returns the names loaded."""
        f = torch.load(path, map_location="cpu", weights_only=False)
        m, own = f["manifest"], self.state_dict()
        want = {"partition_digest": digest, "snp_code": snp_code, "embed_dim": int(self.tables[0].shape[1])}
        if {k: m.get(k) for k in want} != want:
            raise ValueError(f"--init_encoder {path}: made for another partition, SNP code or width than this model's")
        state = f["state"]
        bad = [k for k, v in state.items() if k not in own or tuple(v.shape) != tuple(own[k].shape)]
        if bad or not any(k.startswith("tables.") for k in state):
            raise ValueError(f"--init_encoder {path}: unknown or misshaped entries {bad[:5]}, or no tables")
        self.load_state_dict({k: v.to(own[k].dtype) for k, v in state.items()}, strict=False)
        if freeze:
            for k in state:
                self.get_parameter(k).requires_grad_(False)
        return sorted(state)
