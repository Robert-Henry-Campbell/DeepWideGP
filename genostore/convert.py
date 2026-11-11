# -*- coding: utf-8 -*-
"""Convert PLINK2 .pgen files (variant-major) into a DeepGP genotype store (sample-major).

One streaming pass per chromosome, bounded memory, resumable per block. Needs ``pgenlib``
(only here; training reads the store with numpy alone).

    python -m genostore.convert --pgen-dir /path/Genomics_PGEN --pattern "UKB_chr{chrom}" \
        --chroms 1 --out /path/store_chr1_all
    python -m genostore.convert ... --chroms 1 2 3 --keep-n 4096 --seed 0 --out /path/store_wg_n4096
    python -m genostore.convert verify --out /path/store

Memory: one block of n_samples x 16384 bytes (5.5 GB for 333k samples) plus ~1 GB of buffers.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import format as gf
from .prefetch import CacheProbe, Prefetcher, REQ_BYTES, disk_busy
from .store import GenoStore


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def warm_file(path, start=0, length=None):
    """Queue readahead for a small file/range (so a seek-bound disk serves it in one sweep)."""
    size = os.path.getsize(path)
    end = size if length is None else min(size, start + length)
    fd = os.open(path, os.O_RDONLY)
    try:
        off = (start // REQ_BYTES) * REQ_BYTES
        while off < end:
            os.posix_fadvise(fd, off, min(REQ_BYTES, end - off), os.POSIX_FADV_WILLNEED)
            off += REQ_BYTES
    finally:
        os.close(fd)


def read_psam_ids(path):
    warm_file(path)
    ids, iid_col = [], None
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if line.startswith("#"):
                cols = [c.lstrip("#") for c in p]
                iid_col = cols.index("IID")
                continue
            if iid_col is None:                  # headerless .fam-style: FID IID ...
                iid_col = 1
            ids.append(p[iid_col])
    return ids


def read_pvar_lines(path, max_variants=None):
    """'CHROM\\tPOS\\tID\\tREF\\tALT' for every variant, in file order."""
    warm_file(path)
    out, cols = [], None
    with open(path) as f:
        for line in f:
            if line.startswith("##"):
                continue
            p = line.rstrip("\n").split("\t")
            if line.startswith("#"):
                names = [c.lstrip("#") for c in p]
                cols = [names.index(k) for k in ("CHROM", "POS", "ID", "REF", "ALT")]
                continue
            if cols is None:                     # headerless .bim-style is not handled
                raise ValueError(f"{path}: no #CHROM header line")
            out.append("\t".join(p[i] for i in cols))
            if max_variants is not None and len(out) >= max_variants:
                break
    return out


def pgen_header(path):
    with open(path, "rb") as f:
        h = f.read(12)
    if h[:2] != b"\x6c\x1b":
        raise ValueError(f"{path}: not a .pgen file")
    return {"mode": h[2], "n_variants": int.from_bytes(h[3:7], "little"),
            "n_samples": int.from_bytes(h[7:11], "little"), "fmt": h[11]}


def pgen_variant_offsets(path):
    """File offset of every variant record (uint64 [M + 1], last = file size).

    Parsed from the .pgen header index so the prefetcher knows exactly which bytes the
    decoder needs next. Returns (offsets, exact); if the header is not the layout we know,
    offsets are a linear interpolation, which is good enough for prefetching.
    """
    size = os.path.getsize(path)
    h = pgen_header(path)
    m = h["n_variants"]
    approx = np.linspace(12, size, m + 1).astype(np.uint64)
    low = h["fmt"] & 15
    if h["mode"] != 0x10 or low > 7:
        return approx, False
    type_bits = 4 if low < 4 else 8
    len_bytes = (low & 3) + 1
    allele_ct_bytes = (h["fmt"] >> 4) & 3
    nonref_per_variant = ((h["fmt"] >> 6) & 3) == 3
    n_vblocks = (m + 65535) // 65536
    header_guess = 12 + n_vblocks * 8 + m * (len_bytes + 1 + allele_ct_bytes) + m // 8 + 4096
    warm_file(path, 0, header_guess)
    offs = np.empty(m + 1, dtype=np.uint64)
    with open(path, "rb") as f:
        f.seek(12)
        vb_off = np.frombuffer(f.read(8 * n_vblocks), dtype="<u8")
        ends = []
        for vb in range(n_vblocks):
            cnt = min(65536, m - vb * 65536)
            f.seek(cnt if type_bits == 8 else (cnt + 1) // 2, 1)
            raw = np.frombuffer(f.read(cnt * len_bytes), dtype=np.uint8).reshape(cnt, len_bytes)
            lens = np.zeros(cnt, dtype=np.uint64)
            for k in range(len_bytes):
                lens |= raw[:, k].astype(np.uint64) << np.uint64(8 * k)
            if allele_ct_bytes:
                f.seek(cnt * allele_ct_bytes, 1)
            if nonref_per_variant:
                f.seek((cnt + 7) // 8, 1)
            cs = np.cumsum(lens, dtype=np.uint64)
            offs[vb * 65536] = vb_off[vb]
            offs[vb * 65536 + 1: vb * 65536 + cnt] = vb_off[vb] + cs[:-1]
            ends.append(int(vb_off[vb]) + int(cs[-1]))
        header_end = f.tell()
    offs[m] = size
    ok = header_end == int(vb_off[0]) and ends[-1] == size
    ok = ok and all(ends[i] == int(vb_off[i + 1]) for i in range(n_vblocks - 1))
    return (offs, True) if ok else (approx, False)


def choose_samples(all_ids, keep_n=None, seed=0, keep_file=None):
    """Sorted uint32 row indices into the .psam order, or None for all samples."""
    n = len(all_ids)
    if keep_file:
        with open(keep_file) as f:
            want = {l.split()[-1] for l in f if l.strip() and not l.startswith("#")}
        idx = np.array([i for i, s in enumerate(all_ids) if s in want], dtype=np.uint32)
        if len(idx) != len(want):
            log(f"WARNING: {len(want) - len(idx)} IDs in {keep_file} are not in the .psam")
        return idx
    if keep_n and keep_n < n:
        return np.sort(np.random.default_rng(seed).choice(n, size=keep_n, replace=False)).astype(np.uint32)
    return None


class _Stopped(Exception):
    pass


def readable_now(pgen, store_dir, chrom, block_variants, max_variants=None, _cache={}):
    """True if the next stretch this chromosome needs is already in the page cache
    (so converting it now costs the source disk nothing)."""
    pr = CacheProbe(pgen)
    try:
        if not pr.supported or pr.cached(0, 65536) < 65536:
            return False
        if pgen not in _cache:
            _cache[pgen] = pgen_variant_offsets(pgen)[0]
        offs = _cache[pgen]
        m = len(offs) - 1 if max_variants is None else min(max_variants, len(offs) - 1)
        prog = gf.chrom_paths(store_dir, chrom)["progress"]
        v = gf.read_json(prog)["blocks_done"] * block_variants if os.path.exists(prog) else 0
        if v >= m:
            return True
        need = min(4 * REQ_BYTES, int(offs[m]) - int(offs[v]))
        return pr.cached(int(offs[v]), need) >= need
    finally:
        pr.close()


def convert_chrom(pgen, pvar, store_dir, chrom, sample_idx, n_raw, block_variants=gf.BLOCK_VARIANTS,
                  chunk=256, policy="idle", duty=0.5, max_ahead=1536 << 20, prefetch=True,
                  max_variants=None, deadline=None, n_audit=8, threads=8, stall_timeout=None):
    """Returns "done", or "deadline" / "stalled" if it stopped early (rerun to resume).

    stall_timeout: give up (keeping finished blocks) when no new bytes became readable for
    this many seconds -- used by the opportunistic scheduler to move to another chromosome.
    """
    import pgenlib

    p = gf.chrom_paths(store_dir, chrom)
    if os.path.exists(p["json"]) and gf.read_json(p["json"]).get("complete"):
        log(f"chr{chrom}: already complete, skipping")
        return "done"
    hdr = pgen_header(pgen)
    if hdr["n_samples"] != n_raw:
        raise ValueError(f"{pgen}: {hdr['n_samples']} samples but the .psam has {n_raw}")
    m = hdr["n_variants"] if max_variants is None else min(max_variants, hdr["n_variants"])
    n = n_raw if sample_idx is None else len(sample_idx)
    offs, exact = pgen_variant_offsets(pgen)
    log(f"chr{chrom}: {m} variants x {n} samples (of {n_raw}); record offsets "
        f"{'exact' if exact else 'APPROXIMATE'}; {int(offs[m]) / 1e9:.2f} GB to read")
    w = gf.ChromWriter(store_dir, chrom, n, m, block_variants)
    if w.next_block:
        log(f"chr{chrom}: resuming at block {w.next_block}/{w.n_blocks}")
    audit_rows = np.unique(np.linspace(0, n - 1, min(n_audit, n)).astype(np.int64))
    audit_path = p["json"][:-5] + ".audit.npy"
    if w.next_block and os.path.exists(audit_path):
        audit = np.load(audit_path, mmap_mode="r+")
    else:
        audit = np.lib.format.open_memmap(audit_path, mode="w+", dtype=np.uint8, shape=(len(audit_rows), m))
    np.save(p["json"][:-5] + ".audit_rows.npy", audit_rows)

    v_start = w.block_range(w.next_block)[0] if not w.complete else m
    pf = None
    if prefetch and not w.complete:
        pf = Prefetcher(pgen, start=int(offs[v_start]), end=int(offs[m]), policy=policy, duty=duty,
                        max_ahead=max_ahead, log=log, log_every=60.0)
        pf.set_consumed(int(offs[v_start]))
        pf.start()
    reader = (pgenlib.PgenReader(pgen.encode(), raw_sample_ct=n_raw) if sample_idx is None else
              pgenlib.PgenReader(pgen.encode(), raw_sample_ct=n_raw,
                                 sample_subset=np.ascontiguousarray(sample_idx, dtype=np.uint32)))
    fbuf = np.empty((chunk, n), dtype=np.float32)
    threads = max(1, int(threads))
    slab = max(256, -(-n // (threads * 4)))            # ~4 slabs per thread, cache-sized
    slab_starts = list(range(0, n, slab))
    pool = ThreadPoolExecutor(max_workers=threads)
    slack = 0 if exact else 64 << 20
    t_start, v_done0 = time.time(), v_start
    status = "done"
    try:
        while not w.complete:
            if deadline is not None and time.time() > deadline:
                status = "deadline"
                break
            b = w.next_block
            v0, v1 = w.block_range(b)
            Q = np.empty((n, v1 - v0), dtype=np.uint8)
            st = np.zeros((3, v1 - v0))
            t_b, t_dec, t_wait, t_proc = time.time(), 0.0, 0.0, 0.0
            for k0 in range(v0, v1, chunk):
                k1 = min(k0 + chunk, v1)
                k = k1 - k0
                if pf is not None:
                    t = time.time()
                    while not pf.wait_until(int(offs[k1]) + slack, timeout=20.0):
                        if stall_timeout is not None and time.time() - pf.last_progress > stall_timeout:
                            raise _Stopped("stalled")
                        if deadline is not None and time.time() > deadline:
                            raise _Stopped("deadline")
                    t_wait += time.time() - t
                t = time.time()
                reader.read_dosages_range(k0, k1, fbuf[:k])
                t_dec += time.time() - t
                if pf is not None:
                    pf.set_consumed(int(offs[k1]))
                t = time.time()
                d = fbuf[:k]                                             # [k, n], variant-major
                c0, c1 = k0 - v0, k1 - v0

                def work(s0, d=d, c0=c0, c1=c1):
                    # one slab of samples: quantise, per-variant sums, transposed store
                    qv = gf.quantize_dosage(d[:, s0:s0 + slab])
                    miss = qv == gf.MISSING
                    nm = miss.sum(axis=1) if miss.any() else 0
                    qz = np.where(miss, 0, qv) if np.any(nm) else qv
                    sq = qz.astype(np.uint16)
                    np.multiply(sq, sq, out=sq)
                    Q[s0:s0 + slab, c0:c1] = qv.T
                    return (qz.sum(axis=1, dtype=np.int64), sq.sum(axis=1, dtype=np.int64), nm)

                for s_sum, s_sq, s_nm in pool.map(work, slab_starts):
                    st[0, c0:c1] += s_sum
                    st[1, c0:c1] += s_sq
                    st[2, c0:c1] -= s_nm
                st[2, c0:c1] += n
                audit[:, k0:k1] = Q[audit_rows, c0:c1]
                t_proc += time.time() - t
            t = time.time()
            w.append_block(Q, st, pool=pool)
            audit.flush()
            t_enc = time.time() - t
            el = time.time() - t_start
            rate = (v1 - v_done0) / max(el, 1e-9)
            log(f"chr{chrom} block {b + 1}/{w.n_blocks}: {v1 - v0} variants in {time.time() - t_b:.1f}s "
                f"(wait-for-disk {t_wait:.1f}s, decode {t_dec:.1f}s, quantise+transpose {t_proc:.1f}s, "
                f"encode+write {t_enc:.1f}s); "
                f"{rate:.0f} variants/s, ETA {(m - v1) / max(rate, 1e-9) / 60:.1f} min")
    except _Stopped as e:
        status = str(e)
    finally:
        if pf is not None:
            pf.stop()
        reader.close()
        pool.shutdown(wait=True)
    if status != "done":
        why = "time budget reached" if status == "deadline" else "source not readable without disturbing other jobs"
        log(f"chr{chrom}: paused after block {w.next_block}/{w.n_blocks} ({why}); resumable")
        return status
    lines = read_pvar_lines(pvar, m)
    info = w.finalize(lines, {"pgen": pgen, "pvar": pvar, "n_raw_samples": n_raw,
                              "pgen_bytes_read": int(offs[m]), "max_variants": max_variants})
    log(f"chr{chrom}: complete. {info['bytes_per_genotype']:.4f} bytes/genotype "
        f"({(info['base_bytes'] + info['ovr_bytes']) / 1e9:.2f} GB), exceptions "
        f"{100 * info['n_override'] / (n * m):.2f}%, missing {info['n_missing']}, "
        f"writer time {info['elapsed_s'] / 60:.1f} min")
    return "done"


def verify_store(store_dir, chroms=None):
    """Compare the rows captured straight from the decoder with what the reader returns."""
    gs = GenoStore(store_dir)
    ok = True
    for c in (chroms or gs.chroms):
        base = gf.chrom_paths(store_dir, c)["json"][:-5]
        if not os.path.exists(base + ".audit.npy"):
            log(f"chr{c}: no audit file, skipped")
            continue
        rows = np.load(base + ".audit_rows.npy")
        audit = np.load(base + ".audit.npy", mmap_mode="r")
        m = gs.n_variants(c)
        bad = 0
        for v0 in range(0, m, 1 << 18):
            v1 = min(m, v0 + (1 << 18))
            bad += int((gs.read_q(c, rows, v0, v1) != audit[:, v0:v1]).sum())
        mean, std, cnt = gs.stats(c)
        log(f"chr{c}: {len(rows)} audit rows x {m} variants -> {bad} mismatches; "
            f"mean ALT dosage {mean.mean():.4f}, monomorphic variants {(std == 0).sum()}")
        ok = ok and bad == 0
    return ok


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "verify":
        ap = argparse.ArgumentParser(prog="genostore.convert verify")
        ap.add_argument("--out", required=True)
        ap.add_argument("--chroms", nargs="*")
        a = ap.parse_args(argv[1:])
        return 0 if verify_store(a.out, a.chroms) else 1
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgen-dir", required=True)
    ap.add_argument("--pattern", default="UKB_chr{chrom}", help="file prefix; {chrom} is substituted")
    ap.add_argument("--chroms", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="store directory")
    ap.add_argument("--keep-n", type=int, default=None, help="random subset of this many samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-file", default=None, help="file of sample IDs to keep (last column)")
    ap.add_argument("--anonymize-ids", action="store_true",
                    help="with --keep-n: write anonymous row labels and do not record which samples were drawn "
                         "(pass the same --seed again to resume or add chromosomes)")
    ap.add_argument("--block-variants", type=int, default=gf.BLOCK_VARIANTS)
    ap.add_argument("--chunk", type=int, default=256, help="variants decoded per pgenlib call")
    ap.add_argument("--disk-policy", choices=["idle", "share"], default="idle",
                    help="idle (default): read the source disk only while nobody else is reading it; "
                         "share: read alongside other jobs at --duty")
    ap.add_argument("--duty", type=float, default=0.5, help="with --disk-policy share: max share of disk time")
    ap.add_argument("--max-ahead-gb", type=float, default=1.5)
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--opportunistic", action="store_true",
                    help="convert whichever chromosome is readable without disturbing other jobs: one already in "
                         "the page cache (e.g. being read by a GWAS), or any once the source disk is idle")
    ap.add_argument("--stall-s", type=float, default=120.0,
                    help="with --opportunistic: leave a chromosome after this long without new readable data")
    ap.add_argument("--threads", type=int, default=8, help="threads for quantise/transpose/encode")
    ap.add_argument("--max-variants", type=int, default=None, help="only the first N variants (testing)")
    ap.add_argument("--time-budget-s", type=float, default=None, help="stop cleanly after this long; rerun to resume")
    a = ap.parse_args(argv)

    deadline = None if a.time_budget_s is None else time.time() + a.time_budget_s
    prefix = lambda c: os.path.join(a.pgen_dir, a.pattern.format(chrom=c))
    is_done = lambda c: (os.path.exists(gf.chrom_paths(a.out, c)["json"])
                         and gf.read_json(gf.chrom_paths(a.out, c)["json"]).get("complete"))
    pending = [c for c in a.chroms if not is_done(c)]
    state = {}                                   # filled on first use: sample subset, store meta

    def open_store(c):
        """Create/validate the store from the .psam of chromosome ``c`` (all .psam must agree)."""
        all_ids = read_psam_ids(prefix(c) + ".psam")
        if "idx" not in state:
            idx = choose_samples(all_ids, a.keep_n, a.seed, a.keep_file)
            ids = all_ids if idx is None else [all_ids[i] for i in idx]
            subset = "all" if idx is None else ({"keep_file": a.keep_file} if a.keep_file
                                                else {"keep_n": a.keep_n, "seed": a.seed})
            if a.anonymize_ids:
                # Data minimisation for test/benchmark stores: rows carry no participant ID and
                # the store does not record which .psam rows (or which seed) they came from.
                if idx is None:
                    raise SystemExit("--anonymize-ids is meant for small subsets; combine it with --keep-n")
                ids = [f"anon{i:06d}" for i in range(len(idx))]
                subset = {"keep_n": len(idx), "anonymized": True}
            prov = {"psam": prefix(c) + ".psam", "n_raw_samples": len(all_ids),
                    "raw_samples_sha256_16": gf.samples_digest(all_ids), "subset": subset}
            os.makedirs(a.out, mode=0o700, exist_ok=True)
            meta = gf.create_store(a.out, ids, a.block_variants, prov)
            if idx is not None and not a.anonymize_ids:
                np.save(os.path.join(a.out, "sample_rows.npy"), idx)
            state.update(idx=idx, n_raw=len(all_ids), digest=meta["provenance"]["raw_samples_sha256_16"])
            log(f"store {a.out}: {len(ids)} samples (of {len(all_ids)}), block {meta['block_variants']}")
        if gf.samples_digest(all_ids) != state["digest"]:
            raise ValueError(f"chr{c}: .psam sample order differs from the one this store was built from")

    t_note = 0.0
    while pending:
        if deadline is not None and time.time() > deadline:
            break
        pick = pending[0]
        if a.opportunistic:
            # Only touch what costs the source disk nothing: data another job already pulled into
            # the page cache, or anything at all once nobody else is reading the disk.
            pick = next((c for c in pending
                         if readable_now(prefix(c) + ".pgen", a.out, c, a.block_variants, a.max_variants)), None)
            if pick is None and not disk_busy(prefix(pending[0]) + ".pgen"):
                pick = pending[0]
            if pick is None:
                if time.time() - t_note > 600:
                    log(f"source disk is busy and none of chr{','.join(pending)} is in the page cache; waiting")
                    t_note = time.time()
                time.sleep(15)
                continue
        open_store(pick)
        status = convert_chrom(prefix(pick) + ".pgen", prefix(pick) + ".pvar", a.out, pick, state["idx"],
                               state["n_raw"], a.block_variants, a.chunk, a.disk_policy, a.duty,
                               int(a.max_ahead_gb * (1 << 30)), not a.no_prefetch, a.max_variants, deadline,
                               threads=a.threads, stall_timeout=a.stall_s if a.opportunistic else None)
        if status == "done":
            pending.remove(pick)
        elif status == "deadline":
            break
    if pending:
        log(f"not finished: chr{','.join(pending)} (rerun the same command to continue)")
    return 0 if not pending else 3


if __name__ == "__main__":
    sys.exit(main())
