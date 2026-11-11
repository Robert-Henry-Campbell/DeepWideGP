# -*- coding: utf-8 -*-
"""Reader for the DeepGP genotype store (numpy only; safe to use from DataLoader workers).

Nothing is loaded up front except small indexes; genotypes are memory-mapped and only the
rows asked for are read.
"""
import glob
import os
import re
from collections import namedtuple

import numpy as np

from .format import MISSING, decode_packed, read_json, chrom_paths

# Raw (still encoded) genotypes of some rows for variants [v0, v0 + n) of one chromosome.
#   full     : uint8 [n_full_blocks, B, block_bytes] or None   (full-width blocks)
#   full_ovr : uint8 1-D, exception bytes for ``full`` in C order (block, sample, variant)
#   tail     : uint8 [B, tail_bytes] or None                   (the chromosome's last, narrower block)
#   tail_ovr : uint8 1-D
#   skip     : variants to drop from the front after decoding (v0 is rarely block-aligned)
#   n        : number of variants wanted
RawRegion = namedtuple("RawRegion", "full full_ovr tail tail_ovr tail_width skip n")

_EMPTY = np.empty(0, dtype=np.uint8)


def _natural_key(c):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(c))]


def _runs_to_index(starts, lens):
    """Concatenated index of the half-open runs [start, start + len)."""
    total = int(lens.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(lens)
    return np.repeat(starts - (ends - lens), lens) + np.arange(total, dtype=np.int64)


class GenoStore:
    def __init__(self, store_dir):
        self.dir = store_dir
        self.meta = read_json(os.path.join(store_dir, "meta.json"))
        self.n_samples = int(self.meta["n_samples"])
        self.w = int(self.meta["block_variants"])
        self.block_bytes = self.w // 4
        self.info = {}
        for jp in glob.glob(os.path.join(store_dir, "chr*.json")):
            if jp.endswith(".progress.json"):
                continue
            inf = read_json(jp)
            if inf.get("complete"):
                self.info[str(inf["chrom"])] = inf
        self.chroms = sorted(self.info, key=_natural_key)
        self._samples = None
        self._maps = {}
        self._pid = None

    # ---- small metadata ----------------------------------------------------------------
    @property
    def samples(self):
        if self._samples is None:
            with open(os.path.join(self.dir, "samples.txt")) as f:
                self._samples = np.array([l.rstrip("\n") for l in f if l.strip()], dtype=object)
            assert len(self._samples) == self.n_samples
        return self._samples

    def n_variants(self, chrom):
        return int(self.info[str(chrom)]["n_variants"])

    def stats(self, chrom):
        """(mean, std, n) per variant, in dosage units (0..2), over the samples in this store."""
        s = np.load(chrom_paths(self.dir, chrom)["stats"])
        n = np.maximum(s[2], 1.0)
        mean = s[0] / n
        var = np.maximum(s[1] / n - mean * mean, 0.0)
        return mean / self.meta["dosage_scale"], np.sqrt(var) / self.meta["dosage_scale"], s[2]

    def variants(self, chrom):
        """dict of arrays: pos (int64), id, ref, alt."""
        pos, ids, ref, alt = [], [], [], []
        with open(chrom_paths(self.dir, chrom)["variants"]) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.rstrip("\n").split("\t")
                pos.append(int(p[1])); ids.append(p[2]); ref.append(p[3]); alt.append(p[4])
        return {"pos": np.array(pos, dtype=np.int64), "id": np.array(ids, dtype=object),
                "ref": np.array(ref, dtype=object), "alt": np.array(alt, dtype=object)}

    def positions(self, chrom):
        """Variant positions (the POS column, or whatever array the converter was given)."""
        p = chrom_paths(self.dir, chrom)["pos"]
        return np.load(p) if os.path.exists(p) else self.variants(chrom)["pos"]

    def nbytes(self, chrom=None):
        cs = self.chroms if chrom is None else [str(chrom)]
        return sum(self.info[c]["base_bytes"] + self.info[c]["ovr_bytes"] for c in cs)

    # ---- memory maps (opened lazily, once per process) ----------------------------------
    def _open(self, chrom):
        if self._pid != os.getpid():        # forked DataLoader worker: do not share handles
            self._maps, self._pid = {}, os.getpid()
        chrom = str(chrom)
        mp = self._maps.get(chrom)
        if mp is None:
            inf, p = self.info[chrom], chrom_paths(self.dir, chrom)
            n, m, nb = self.n_samples, inf["n_variants"], inf["n_blocks"]
            n_full = m // self.w
            tail_width = m - n_full * self.w
            full = (np.memmap(p["base"], dtype=np.uint8, mode="r", shape=(n_full, n, self.block_bytes))
                    if n_full else None)
            tail = (np.memmap(p["base"], dtype=np.uint8, mode="r", offset=n_full * n * self.block_bytes,
                              shape=(n, (tail_width + 3) // 4)) if tail_width else None)
            ovr = np.memmap(p["ovr"], dtype=np.uint8, mode="r") if inf["ovr_bytes"] else _EMPTY
            ptr = np.memmap(p["ptr"], dtype="<u4", mode="r", shape=(nb, n + 1))
            mp = {"full": full, "tail": tail, "ovr": ovr, "ptr": ptr, "n_full": n_full,
                  "tail_width": tail_width, "bstart": np.asarray(inf["ovr_block_start"], dtype=np.int64)}
            self._maps[chrom] = mp
        return mp

    # ---- reading -----------------------------------------------------------------------
    def read_raw(self, chrom, rows, v0=0, v1=None):
        """Encoded genotypes of ``rows`` (int array) for variants [v0, v1) of ``chrom``."""
        mp = self._open(chrom)
        m = self.n_variants(chrom)
        v1 = m if v1 is None else v1
        if not (0 <= v0 < v1 <= m):
            raise IndexError(f"chr{chrom}: bad variant range [{v0}, {v1}) for {m} variants")
        rows = np.asarray(rows, dtype=np.int64)
        b0, b1 = v0 // self.w, (v1 - 1) // self.w            # first and last block, inclusive
        bf = min(b1 + 1, mp["n_full"])                       # full blocks are [b0, bf)
        full, full_ovr = None, _EMPTY
        if bf > b0:
            full = np.ascontiguousarray(mp["full"][b0:bf][:, rows, :])
            p0 = mp["ptr"][b0:bf][:, rows].astype(np.int64)
            p1 = mp["ptr"][b0:bf][:, rows + 1].astype(np.int64)
            idx = _runs_to_index((mp["bstart"][b0:bf, None] + p0).ravel(), (p1 - p0).ravel())
            full_ovr = np.asarray(mp["ovr"][idx]) if len(idx) else _EMPTY
        tail, tail_ovr = None, _EMPTY
        if b1 >= mp["n_full"]:                               # region reaches the narrow last block
            b = mp["n_full"]
            tail = np.ascontiguousarray(mp["tail"][rows])
            p0 = mp["ptr"][b][rows].astype(np.int64)
            p1 = mp["ptr"][b][rows + 1].astype(np.int64)
            idx = _runs_to_index(mp["bstart"][b] + p0, p1 - p0)
            tail_ovr = np.asarray(mp["ovr"][idx]) if len(idx) else _EMPTY
        return RawRegion(full, full_ovr, tail, tail_ovr, mp["tail_width"], v0 - b0 * self.w, v1 - v0)

    def decode(self, raw):
        """RawRegion -> uint8 q [B, n] (reference numpy decoder; the training path decodes on GPU)."""
        parts = []
        if raw.full is not None:
            nbf, bsz, _ = raw.full.shape
            q = decode_packed(raw.full, raw.full_ovr)                    # [nbf, B, W]
            parts.append(q.transpose(1, 0, 2).reshape(bsz, nbf * self.w))
        if raw.tail is not None:
            parts.append(decode_packed(raw.tail, raw.tail_ovr, raw.tail_width))
        q = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        return np.ascontiguousarray(q[:, raw.skip:raw.skip + raw.n])

    def read_q(self, chrom, rows, v0=0, v1=None):
        """uint8 q = round(dosage*100) [len(rows), v1 - v0]; 255 = missing."""
        return self.decode(self.read_raw(chrom, rows, v0, v1))

    def read_dosage(self, chrom, rows, v0=0, v1=None):
        """float32 dosage [len(rows), v1 - v0]; NaN = missing."""
        q = self.read_q(chrom, rows, v0, v1)
        d = q.astype(np.float32) / np.float32(self.meta["dosage_scale"])
        if (q == MISSING).any():
            d[q == MISSING] = np.nan
        return d
