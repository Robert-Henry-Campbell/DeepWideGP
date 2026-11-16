"""On-disk format of the DeepGP genotype store, version 1 ("base2+ovr8"). numpy only.

Layout of a store directory
---------------------------
meta.json                 store-level facts (format, block size, number of samples, provenance)
samples.txt               one sample ID per line; line i is row i of every chromosome
chr<C>.json               per-chromosome index; present and "complete": true once converted
chr<C>.base.bin           2-bit codes, 4 per byte
chr<C>.ovr.bin            exception bytes
chr<C>.ovr_ptr.bin        uint32 [n_blocks, n_samples + 1] run offsets into each block's exceptions
chr<C>.stats.npy          float64 [3, n_variants]: sum, sum of squares, count of non-missing
                          (in stored units, i.e. dosage * 100)
chr<C>.variants.tsv       CHROM POS ID REF ALT, one line per variant (column order)
chr<C>.pos.npy            the positions again as an array (fast to load)

Encoding
--------
A genotype is the ALT-allele dosage in [0, 2], stored as q = round(dosage * 100) in a byte
(0..200; 255 = missing). That is exactly the resolution DeepGP's "vocab" embedding uses.
In UKB imputed data ~96% of genotypes are exactly 0, 1 or 2, so each genotype gets a 2-bit
code: 0, 1, 2 mean q = 0, 100, 200 and code 3 means "exception": the true byte is the next
unread byte of that sample's exception run. Exceptions need no position index because the
code-3 slots mark where they go. Cost: 0.25 byte + 1 byte per exception, ~0.29 byte per
genotype, lossless with respect to q.

Variants are grouped in blocks of ``block_variants`` (16384). Files are block-major: block b
holds all samples for variants [b*16384, (b+1)*16384). Inside a block, data is sample-major:
    base : n_samples rows of ceil(width/4) bytes (4096 for a full block); variant j of the
           block sits in byte j // 4 at bit offset 2 * (j % 4)
    ovr  : sample 0's exception bytes in variant order, then sample 1's, ...
Block-major files can be written sequentially from a variant-major source (.pgen), which is
what makes conversion a single streaming pass; reading one sample costs one 4 KB base read
plus one short exception read per block.

A store holds participant-level data: its directory is created 0700 and its files 0600,
whatever the umask.
"""
import hashlib
import json
import os
import time

import numpy as np

FORMAT = "deepgp-genostore"
VERSION = 1
ENCODING = "base2+ovr8"
BLOCK_VARIANTS = 16384
DOSAGE_SCALE = 100
MISSING = 255
DIR_MODE, FILE_MODE = 0o700, 0o600          # owner only

CODE_OF = np.full(256, 3, dtype=np.uint8)
CODE_OF[0], CODE_OF[100], CODE_OF[200] = 0, 1, 2
VALUE_OF = np.array([0, 100, 200, 0], dtype=np.uint8)
SHIFTS = np.array([0, 2, 4, 6], dtype=np.uint8)


def quantize_dosage(d):
    """float dosage array (missing: negative sentinel or NaN) -> uint8 q, 255 = missing.

    q = floor(dosage * 100 + 0.5), i.e. round half up, clipped to [0, 200]. pgenlib returns
    multiples of 1/16384, for which this is exact in float32.
    """
    t = np.multiply(d, np.float32(DOSAGE_SCALE), dtype=np.float32)
    if t.size and t.min() >= 0.0 and t.max() <= 2.0 * DOSAGE_SCALE:   # usual case: none missing (NaN fails)
        t += np.float32(0.5)
        return t.astype(np.uint8)
    miss = ~(t >= -0.5)
    t[miss] = 0
    np.clip(t, 0, 2 * DOSAGE_SCALE, out=t)
    t += np.float32(0.5)
    q = t.astype(np.uint8)
    q[miss] = MISSING
    return q


def encode_block(Q, slab=8192, pool=None):
    """Q: uint8 [n_samples, width] of q values -> (packed [n, ceil(width/4)], ovr bytes, counts).

    ``pool``: optional executor; row slabs are independent and numpy releases the GIL in these
    loops, so threads speed up large blocks.
    """
    Q = np.asarray(Q)
    assert Q.dtype == np.uint8 and Q.ndim == 2
    n, width = Q.shape
    nbytes = (width + 3) // 4
    packed = np.empty((n, nbytes), dtype=np.uint8)
    counts = np.empty(n, dtype=np.int64)

    def work(r0):
        q = Q[r0:r0 + slab]
        code = np.take(CODE_OF, q)
        exc = code == 3
        counts[r0:r0 + slab] = np.count_nonzero(exc, axis=1)
        if nbytes * 4 != width:                   # padding is code 0, so it never claims an exception
            code = np.pad(code, ((0, 0), (0, nbytes * 4 - width)))
        x = np.ascontiguousarray(code).view("<u4")    # 4 codes per uint32: c0 | c1<<8 | c2<<16 | c3<<24
        x = x | (x >> 6) | (x >> 12) | (x >> 18)      # low byte = c0 | c1<<2 | c2<<4 | c3<<6
        packed[r0:r0 + slab] = x.astype(np.uint8)
        return q[exc]                             # sample-major, variants ascending

    starts = range(0, n, slab)
    parts = list(pool.map(work, starts)) if pool is not None else [work(r0) for r0 in starts]
    ovr = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint8)
    return packed, ovr, counts


def decode_packed(packed, ovr, width=None):
    """Inverse of encode_block for any leading shape.

    packed: uint8 [..., nbytes]; ovr: exception bytes in C order of the leading axes.
    Returns uint8 [..., width] (width defaults to nbytes * 4).
    """
    packed = np.asarray(packed)
    lead, nbytes = packed.shape[:-1], packed.shape[-1]
    codes = ((packed[..., None] >> SHIFTS) & 3).reshape(*lead, nbytes * 4)[..., :None if width == nbytes * 4 else width]
    q = VALUE_OF[codes]
    m = codes == 3
    n_exc = int(m.sum())
    if n_exc != len(ovr):
        raise ValueError(f"exception count mismatch: {n_exc} code-3 slots but {len(ovr)} bytes")
    if n_exc:
        q[m] = ovr
    return q


def new_file(path, mode="w"):
    """Open ``path`` for writing, created (or emptied) readable and writable by its owner only."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    os.fchmod(fd, FILE_MODE)                     # a file that already existed would keep its mode
    return os.fdopen(fd, mode)


def _write_json_atomic(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with new_file(tmp) as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def chrom_paths(store_dir, chrom):
    p = os.path.join(store_dir, f"chr{chrom}")
    return {"json": p + ".json", "progress": p + ".progress.json", "base": p + ".base.bin",
            "ovr": p + ".ovr.bin", "ptr": p + ".ovr_ptr.bin", "stats": p + ".stats.npy",
            "variants": p + ".variants.tsv", "pos": p + ".pos.npy"}


def samples_digest(sample_ids):
    return hashlib.sha256(b"".join(str(s).encode() + b"\n" for s in sample_ids)).hexdigest()[:16]


def create_store(store_dir, sample_ids, block_variants=BLOCK_VARIANTS, provenance=None):
    """Create the store directory (or validate that an existing one has the same samples)."""
    os.makedirs(store_dir, mode=DIR_MODE, exist_ok=True)
    meta_path = os.path.join(store_dir, "meta.json")
    sample_ids = [str(s) for s in sample_ids]
    meta = {"format": FORMAT, "version": VERSION, "encoding": ENCODING,
            "dosage_scale": DOSAGE_SCALE, "missing": MISSING, "block_variants": int(block_variants),
            "n_samples": len(sample_ids), "samples_sha256_16": samples_digest(sample_ids),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "provenance": provenance or {}}
    if os.path.exists(meta_path):
        old = read_json(meta_path)
        for k in ("format", "version", "encoding", "block_variants", "n_samples", "samples_sha256_16"):
            if old.get(k) != meta[k]:
                raise ValueError(f"{store_dir} already holds a store with different {k}: "
                                 f"{old.get(k)!r} vs {meta[k]!r}")
        return old
    with new_file(os.path.join(store_dir, "samples.txt")) as f:
        f.write("\n".join(sample_ids) + "\n")
    _write_json_atomic(meta_path, meta)
    return meta


class ChromWriter:
    """Append blocks of one chromosome in order. Resumable at block granularity."""

    def __init__(self, store_dir, chrom, n_samples, n_variants, block_variants=BLOCK_VARIANTS,
                 resume=True, drop_cache=True):
        self.chrom = str(chrom)
        self.n, self.m, self.w = int(n_samples), int(n_variants), int(block_variants)
        assert self.w % 4 == 0
        self.n_blocks = (self.m + self.w - 1) // self.w
        self.p = chrom_paths(store_dir, self.chrom)
        self.drop_cache = drop_cache
        if resume and os.path.exists(self.p["progress"]):
            st = read_json(self.p["progress"])
            if (st["n_samples"], st["n_variants"], st["block_variants"]) != (self.n, self.m, self.w):
                raise ValueError(f"progress file for chr{self.chrom} does not match this conversion")
        else:
            st = {"n_samples": self.n, "n_variants": self.m, "block_variants": self.w,
                  "blocks_done": 0, "base_bytes": 0, "ovr_bytes": 0, "ovr_block_start": [0],
                  "n_override": 0, "n_missing": 0, "elapsed_s": 0.0}
            for k in ("base", "ovr", "ptr"):
                new_file(self.p[k], "wb").close()
            with new_file(self.p["stats"], "wb") as f:
                np.save(f, np.zeros((3, self.m)))
        self.st = st
        self.f = {}
        for k, size in (("base", st["base_bytes"]), ("ovr", st["ovr_bytes"]),
                        ("ptr", st["blocks_done"] * (self.n + 1) * 4)):
            fh = open(self.p[k], "r+b")
            fh.truncate(size)        # drop any half-written block from an interrupted run
            fh.seek(size)
            self.f[k] = fh
        self.stats = np.load(self.p["stats"], mmap_mode="r+")
        self._t0 = time.time()

    @property
    def next_block(self):
        return self.st["blocks_done"]

    @property
    def complete(self):
        return self.st["blocks_done"] >= self.n_blocks

    def block_range(self, b):
        return b * self.w, min((b + 1) * self.w, self.m)

    def _append(self, key, arr):
        fh = self.f[key]
        off = fh.tell()
        arr = np.ascontiguousarray(arr)
        fh.write(memoryview(arr).cast("B"))
        fh.flush()
        os.fsync(fh.fileno())
        if self.drop_cache:                   # not read back soon: leave the page cache to other jobs
            os.posix_fadvise(fh.fileno(), off, arr.nbytes, os.POSIX_FADV_DONTNEED)
        return arr.nbytes

    def append_block(self, Q, stats=None, pool=None):
        """Q: uint8 [n_samples, width] for block ``next_block``.
        stats: optional float [3, width] (sum, sumsq, n non-missing) if already computed.
        pool: optional thread pool for the encoder."""
        b = self.next_block
        v0, v1 = self.block_range(b)
        if Q.shape != (self.n, v1 - v0):
            raise ValueError(f"block {b} must have shape {(self.n, v1 - v0)}, got {Q.shape}")
        packed, ovr, counts = encode_block(Q, pool=pool)
        total = int(counts.sum())
        if total >= 2 ** 32:
            raise OverflowError("more than 2^32 exceptions in one block; uint32 run offsets overflow")
        ptr = np.zeros(self.n + 1, dtype="<u4")
        np.cumsum(counts, out=ptr[1:])
        if stats is None:
            stats = np.zeros((3, v1 - v0))
            for r0 in range(0, self.n, 8192):
                q = Q[r0:r0 + 8192]
                ok = q != MISSING
                qf = np.where(ok, q, 0).astype(np.float64)
                stats[0] += qf.sum(axis=0)
                stats[1] += (qf * qf).sum(axis=0)
                stats[2] += ok.sum(axis=0)
        self.st["base_bytes"] += self._append("base", packed)
        self.st["ovr_bytes"] += self._append("ovr", ovr)
        self._append("ptr", ptr)
        self.stats[:, v0:v1] = stats
        self.stats.flush()
        self.st["ovr_block_start"].append(self.st["ovr_bytes"])
        self.st["n_override"] += total
        self.st["n_missing"] += int(self.n * (v1 - v0) - np.asarray(stats[2]).sum())
        self.st["blocks_done"] = b + 1
        now = time.time()
        self.st["elapsed_s"] += now - self._t0
        self._t0 = now
        _write_json_atomic(self.p["progress"], self.st)

    def finalize(self, variant_lines=None, provenance=None, pos=None):
        """variant_lines: iterable of 'CHROM\\tPOS\\tID\\tREF\\tALT' strings (no header).
        pos: optional explicit position array (else the POS column is used)."""
        if not self.complete:
            raise RuntimeError(f"chr{self.chrom}: only {self.next_block}/{self.n_blocks} blocks written")
        for fh in self.f.values():
            fh.close()
        if variant_lines is not None:
            tsv_pos = np.empty(self.m, dtype=np.int64)
            with new_file(self.p["variants"]) as f:
                f.write("#CHROM\tPOS\tID\tREF\tALT\n")
                k = 0
                for line in variant_lines:
                    line = line.rstrip("\n")
                    if k < self.m:
                        tsv_pos[k] = int(line.split("\t", 2)[1])
                    f.write(line + "\n")
                    k += 1
            if k != self.m:
                raise ValueError(f"chr{self.chrom}: {k} variant lines for {self.m} variants")
            if pos is None:
                pos = tsv_pos
        if pos is not None:
            pos = np.asarray(pos)
            if len(pos) != self.m:
                raise ValueError(f"chr{self.chrom}: {len(pos)} positions for {self.m} variants")
            with new_file(self.p["pos"], "wb") as f:
                np.save(f, pos)
        info = {"chrom": self.chrom, "n_samples": self.n, "n_variants": self.m,
                "block_variants": self.w, "n_blocks": self.n_blocks,
                "base_bytes": self.st["base_bytes"], "ovr_bytes": self.st["ovr_bytes"],
                "ovr_block_start": self.st["ovr_block_start"], "n_override": self.st["n_override"],
                "n_missing": self.st["n_missing"], "elapsed_s": round(self.st["elapsed_s"], 1),
                "bytes_per_genotype": (self.st["base_bytes"] + self.st["ovr_bytes"]) / max(1, self.n * self.m),
                "provenance": provenance or {}, "complete": True}
        _write_json_atomic(self.p["json"], info)
        return info


def write_chrom_from_array(store_dir, chrom, q, variant_lines=None, block_variants=BLOCK_VARIANTS,
                           provenance=None, pos=None):
    """Write one chromosome from an in-memory uint8 q matrix [n_samples, n_variants].
    (Use quantize_dosage() first for float dosages.) For small data / tests / legacy pickles."""
    q = np.asarray(q)
    assert q.dtype == np.uint8 and q.ndim == 2
    w = ChromWriter(store_dir, chrom, q.shape[0], q.shape[1], block_variants, resume=False, drop_cache=False)
    while not w.complete:
        v0, v1 = w.block_range(w.next_block)
        w.append_block(np.ascontiguousarray(q[:, v0:v1]))
    if variant_lines is None:
        variant_lines = (f"{chrom}\t{j + 1}\tchr{chrom}_snp{j}\tA\tG" for j in range(q.shape[1]))
    return w.finalize(variant_lines, provenance, pos)
