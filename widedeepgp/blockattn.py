"""Within-block interactions over the strongest SNPs of each block (Robert, 20 November 2026, items 2 and 6): for every
block the K SNPs whose rows of the frozen per-SNP table are largest form a short sequence, one token per SNP (the
dosage-scaled table row, the dosage itself and a learned identity for that SNP) and one token for the block's pooled
vector; a small mixer shared by all blocks runs over each block's sequence (a transformer layer: self-attention over
the pairs of SNPs; or the published bi-Mamba block, reading the SNPs in genomic order), and the mean of its outputs is
added to the block's token before the mixer over blocks. The table rows stay fixed; only the small mixer, the input
maps and the identities learn. The selection is made once the table is loaded (``select_from``) and saved with the model."""
import torch
import torch.nn as nn

from .blocks import make_block


class BlockAttention(nn.Module):
    def __init__(self, encoder, k, d_attn, d_model, mixer="transformer", n_layers=1, n_heads=4, dropout=0.0):
        super().__init__()
        self.copies = int(encoder.k)                        # fold-routed tables: one set of rows per copy
        if encoder.codes != 1:
            raise ValueError("--block_attn needs the dosage code")
        self.k, self.d_e = int(k), int(encoder.tables[0].shape[-1])
        self.n_seq = len(encoder.lengths)
        smallest = min(rows for seq in encoder.lengths for rows in seq)
        if not 1 <= self.k <= smallest:
            raise ValueError(f"--block_attn must be between 1 and the smallest block's SNP count ({smallest})")
        n_blocks = sum(len(seq) for seq in encoder.lengths)
        for i, seq in enumerate(encoder.lengths):            # filled by select_from; saved with the model
            self.register_buffer(f"idx_{i}", torch.zeros(len(seq), self.k, dtype=torch.long))
        self.register_buffer("rows", torch.zeros(self.copies, n_blocks, self.k, self.d_e))
        self.inp = nn.Linear(self.d_e + 1, d_attn)
        self.pool_inp = nn.Linear(self.d_e, d_attn)
        self.snp_emb = nn.Embedding(n_blocks * self.k, d_attn)
        self.layers = nn.ModuleList([make_block(d_attn, mixer, n_heads=n_heads, dropout=dropout) for _ in range(int(n_layers))])
        self.norm = nn.LayerNorm(d_attn)
        self.out = nn.Linear(d_attn, d_model)

    @torch.no_grad()
    def select_from(self, encoder):
        """The K SNPs of each block with the largest table rows (L2 norm; of the copies' mean for a fold-routed table), in
        genomic order, and those rows of every copy."""
        rows = []
        for i, (table, lengths) in enumerate(zip(encoder.tables, encoder.lengths)):
            idx = getattr(self, f"idx_{i}")
            copies = table.float() if table.dim() == 3 else table.float()[None]           # [copies, L_i, d_e]
            start = 0
            for b, n in enumerate(lengths):
                norm = copies[:, start:start + n].mean(0).norm(dim=1)
                top = torch.topk(norm, self.k).indices.sort().values + start
                idx[b] = top
                rows.append(copies[:, top])
                start += n
        self.rows.copy_(torch.stack(rows, 1))
        return self

    def select(self, x_all):
        """Standardised dosages, one [B, L_i, 1] per sequence -> the selected SNPs' dosages [B, n_blocks, K], rounded to
        bfloat16 (what the cache keeps) and returned in float32."""
        out = [x[:, :, 0].index_select(1, getattr(self, f"idx_{i}").reshape(-1)).reshape(x.shape[0], -1, self.k)
               for i, x in enumerate(x_all)]
        return torch.cat(out, 1).to(torch.bfloat16).float()

    def forward(self, sel, pooled, fold=None):
        """sel [B, n_blocks, K] (select), pooled [B, n_blocks, d_e], fold [B] (each participant's copy; None: the one copy)
        -> what each block's token gains, [B, n_blocks, d_model]."""
        b, nb, k = sel.shape
        sel = sel.float()[..., None]
        rows = self.rows[0] if fold is None else self.rows[fold.long()]
        h = self.inp(torch.cat([sel * rows, sel], -1)) + self.snp_emb.weight.view(1, nb, k, -1)
        h = torch.cat([self.pool_inp(pooled.float())[:, :, None], h], 2).reshape(b * nb, k + 1, -1)
        residual = None
        for layer in self.layers:
            h, residual = layer(h, residual)
        h = self.norm(h if residual is None else h + residual)
        return self.out(h.mean(1)).view(b, nb, -1)
