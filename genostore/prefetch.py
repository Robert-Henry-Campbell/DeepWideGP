# -*- coding: utf-8 -*-
"""Throttled sequential page-cache prefetcher (stdlib only).

Why this exists
---------------
The UKB .pgen files live on a rotational disk that other jobs seek on. Ordinary buffered
reads then degrade to read_ahead_kb-sized (128 KB) requests that each wait for a scheduler
slot: ~0.3 MB/s was measured for a plain sequential reader on 2026-11-10, i.e. days per
chromosome. ``posix_fadvise(WILLNEED)`` lets us queue a *burst* of max_sectors_kb-sized
requests at once, which the elevator serves as one sequential sweep (tens of MB/s), after
which the real reader (pgenlib) finds every page already in the page cache.

Politeness
----------
While our bursts are queued, other readers of the same disk are nearly starved, so their
demand cannot be measured during a burst. We therefore go quiet for ``probe_s`` at intervals
and read /proc/diskstats: bytes read from the disk during that window are all somebody
else's. Two policies:

``idle`` (default)  never read while anyone else is reading the disk. A burst is issued only
                    after a quiet probe saw no other activity; if others show up we stop
                    within ``probe_every`` seconds and wait until the disk is idle again.
                    The most this can cost another job is one ~0.3 s burst.
``share``           read alongside others, sleeping after each burst so that our share of
                    disk time is at most ``duty``. Faster for us, but it does slow the other
                    readers, so only use it when those jobs are yours to slow down.

Riding the cache
----------------
If another job is already reading the same file (e.g. a GWAS over the same .pgen), its
reads put our data in the page cache. ``ride_cache`` (default on) consumes whatever is
already cached with non-blocking reads (preadv2 + RWF_NOWAIT), which never queue a request
on the disk, so we can follow such a job at its own pace even while the disk is "busy".

Usable as a library (``Prefetcher``) or standalone to warm a whole file:
    python -m genostore.prefetch /path/to/file.pgen
"""
import argparse
import ctypes
import errno
import os
import sys
import threading
import time

REQ_BYTES = 1280 * 1024  # one WILLNEED call triggers at most max(io_pages, ra_pages) of readahead


RWF_NOWAIT = 8


class _IoVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


try:
    _preadv2 = ctypes.CDLL(None, use_errno=True).preadv2
    _preadv2.argtypes = [ctypes.c_int, ctypes.POINTER(_IoVec), ctypes.c_int, ctypes.c_longlong, ctypes.c_int]
    _preadv2.restype = ctypes.c_ssize_t
except (OSError, AttributeError):          # no glibc preadv2: cache riding is simply unavailable
    _preadv2 = None


class CacheProbe:
    """Read only what is already in the page cache; never blocks on (or queues) disk I/O."""

    def __init__(self, path, bufsize=REQ_BYTES):
        self.fd = os.open(path, os.O_RDONLY)
        # no readahead on this handle, so a miss costs at most one forced 4 KB read
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.bufsize = bufsize
        self.buf = ctypes.create_string_buffer(bufsize)
        self.supported = _preadv2 is not None

    def cached(self, off, n):
        """Number of bytes, starting at ``off`` and at most ``n``, that are cached right now."""
        if not self.supported:
            return 0
        done = 0
        while done < n:
            want = min(self.bufsize, n - done)
            iov = _IoVec(ctypes.addressof(self.buf), want)
            r = _preadv2(self.fd, ctypes.byref(iov), 1, off + done, RWF_NOWAIT)
            if r < 0:
                e = ctypes.get_errno()
                if e not in (errno.EAGAIN, errno.EINTR):
                    self.supported = False      # e.g. EOPNOTSUPP on this filesystem
                break
            done += r
            if r < want:
                break
        return done

    def close(self):
        os.close(self.fd)


def disk_busy(path, window_s=0.5, threshold_bps=150e3):
    """True if anybody is reading the disk that holds ``path`` right now."""
    dev = disk_of(path)
    d0 = disk_read_bytes(dev)
    time.sleep(window_s)
    d1 = disk_read_bytes(dev)
    return d0 is not None and d1 is not None and (d1 - d0) / window_s > threshold_bps


def disk_of(path):
    """Name of the whole-disk block device (e.g. 'sdh') holding ``path``, or None."""
    try:
        st = os.stat(path)
        major, minor = os.major(st.st_dev), os.minor(st.st_dev)
        real = os.path.realpath(f"/sys/dev/block/{major}:{minor}")
        name = os.path.basename(real)
        parent = os.path.basename(os.path.dirname(real))
        # for a partition, the parent directory is the disk
        return parent if os.path.exists(f"/sys/block/{parent}") else name
    except OSError:
        return None


def disk_read_bytes(dev):
    """Cumulative bytes read from block device ``dev`` (all processes), or None."""
    if dev is None:
        return None
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if p[2] == dev:
                    return int(p[5]) * 512
    except OSError:
        pass
    return None


class Prefetcher(threading.Thread):
    """Warm ``[start, end)`` of ``path`` into the page cache, sequentially, in bursts.

    consumer protocol (optional):
        pf.set_consumed(pos)   # the reader has finished with everything below ``pos``
        pf.wait_until(pos)     # block until everything below ``pos`` has been prefetched
    The prefetcher never runs more than ``max_ahead`` bytes ahead of the consumed position,
    so prefetched pages are used before the kernel has a reason to evict them.
    """

    def __init__(self, path, start=0, end=None, burst_reqs=16, policy="idle", duty=0.5,
                 max_ahead=2 << 30, contended_bps=150e3, probe_every=3.0, probe_s=0.25,
                 backoff_s=5.0, ride_cache=True, log=None, log_every=30.0):
        super().__init__(daemon=True)
        self.path = path
        self.size = os.path.getsize(path)
        self.start_pos = (start // REQ_BYTES) * REQ_BYTES
        self.end = self.size if end is None else min(end, self.size)
        self.burst_reqs = int(burst_reqs)
        if policy not in ("idle", "share"):
            raise ValueError("policy must be 'idle' or 'share'")
        self.policy = policy
        self.duty = float(duty)
        self.backoff_s = backoff_s
        self.t_wait_idle = 0.0
        self.ride_cache = ride_cache
        self.bytes_from_cache = 0
        self.last_progress = time.time()
        self.max_ahead = None if max_ahead is None else int(max_ahead)
        self.contended_bps = contended_bps
        self.probe_every = probe_every
        self.probe_s = probe_s
        self.others_bps = float('nan')
        self.log = log
        self.log_every = log_every
        self.dev = disk_of(path)
        self.pos = self.start_pos          # everything below pos is prefetched
        self.consumed = self.start_pos
        self.cv = threading.Condition()
        self.stop_flag = False
        self.error = None
        self.bytes_done = 0
        self.t_sleep = 0.0
        self.t_io = 0.0
        self.contended_cycles = 0
        self.cycles = 0

    # ---- consumer side -------------------------------------------------------------
    def set_consumed(self, pos):
        with self.cv:
            if pos > self.consumed:
                self.consumed = pos
                self.cv.notify_all()

    def wait_until(self, pos, timeout=None):
        pos = min(pos, self.end)
        t0 = time.time()
        with self.cv:
            while self.pos < pos and self.error is None and self.is_alive():
                self.cv.wait(1.0)
                if timeout is not None and time.time() - t0 > timeout:
                    return False
        if self.error is not None:
            raise self.error
        return True

    def stop(self):
        with self.cv:
            self.stop_flag = True
            self.cv.notify_all()

    # ---- worker --------------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except BaseException as e:  # surface to the consumer instead of dying silently
            self.error = e
            with self.cv:
                self.cv.notify_all()

    def _probe(self):
        """Quiet window: nothing of ours is in flight, so all disk reads seen are others'."""
        d0 = disk_read_bytes(self.dev)
        time.sleep(self.probe_s)
        d1 = disk_read_bytes(self.dev)
        if d0 is None or d1 is None:
            return False                      # cannot measure: assume the disk is ours
        self.others_bps = (d1 - d0) / self.probe_s
        return self.others_bps > self.contended_bps

    def _run(self):
        fd = os.open(self.path, os.O_RDONLY)
        probe = None
        try:
            contended = True                  # assume the disk is busy until a probe says otherwise
            t_probe = -1e18                   # force a probe before the very first burst
            probe = CacheProbe(self.path) if self.ride_cache else None
            if probe is not None and not probe.supported:
                probe.close()
                probe = None
            riding = False
            t_log = time.time()
            b_log = 0
            t_wait_log = 0.0
            while True:
                with self.cv:
                    while (not self.stop_flag and self.max_ahead is not None
                           and self.pos >= self.consumed + self.max_ahead and self.pos < self.end):
                        self.cv.wait(0.5)
                    if self.stop_flag or self.pos >= self.end:
                        break
                    pos = self.pos
                burst = min(self.burst_reqs * REQ_BYTES, self.end - pos)
                if probe is not None:
                    # Whatever is already cached (typically read by another job) costs the disk nothing.
                    got = probe.cached(pos, burst)
                    if got >= min(REQ_BYTES, self.end - pos):
                        got = self.end - pos if pos + got >= self.end else (got // 4096) * 4096
                        with self.cv:
                            self.pos = pos + got
                            self.bytes_done += got
                            self.bytes_from_cache += got
                            self.last_progress = time.time()
                            self.cv.notify_all()
                        b_log += got
                        riding = True
                        continue
                if self.dev is not None and (contended or time.perf_counter() - t_probe >= self.probe_every):
                    contended = self._probe()
                    self.t_sleep += self.probe_s
                    if self.policy == "idle" and contended:
                        # Somebody else is using the disk: do not queue anything. Come back later
                        # (the cached fast path above is retried first every time).
                        if self.log is not None and time.time() - t_wait_log >= 300:
                            self.log(f"prefetch {os.path.basename(self.path)}: disk busy (others reading "
                                     f"{self.others_bps / 1e6:.1f} MB/s); waiting, at {self.pos / 1e9:.2f}/"
                                     f"{self.end / 1e9:.2f} GB ({self.bytes_from_cache / 1e9:.2f} GB taken from cache)")
                            t_wait_log = time.time()
                        nap = 1.0 if riding else self.backoff_s
                        riding = False
                        time.sleep(nap)
                        self.t_wait_idle += nap
                        continue
                    t_probe = time.perf_counter()
                n = burst
                t0 = time.perf_counter()
                off = pos
                while off < pos + n:
                    os.posix_fadvise(fd, off, min(REQ_BYTES, pos + n - off), os.POSIX_FADV_WILLNEED)
                    off += REQ_BYTES
                # Wait for the burst: a blocking read of its last page returns once the
                # (in-order) readahead has reached it.
                os.pread(fd, 1, pos + n - 1)
                dt = time.perf_counter() - t0
                self.t_io += dt
                if self.policy == "share" and contended and self.duty < 1.0:
                    sleep_s = min(2.0, dt * (1.0 / self.duty - 1.0))
                    time.sleep(sleep_s)
                    self.t_sleep += sleep_s
                self.cycles += 1
                self.contended_cycles += int(contended)
                with self.cv:
                    self.pos = pos + n
                    self.bytes_done += n
                    self.last_progress = time.time()
                    self.cv.notify_all()
                b_log += n
                if self.log is not None and time.time() - t_log >= self.log_every:
                    el = time.time() - t_log
                    self.log(f"prefetch {os.path.basename(self.path)}: {self.pos / 1e9:.2f}/{self.end / 1e9:.2f} GB "
                             f"({b_log / el / 1e6:.1f} MB/s over the last {el:.0f}s; {self.bytes_from_cache / 1e9:.2f} GB "
                             f"taken from cache; waited for the disk {self.t_wait_idle / 60:.1f} min in total)")
                    t_log = time.time()
                    b_log = 0
        finally:
            os.close(fd)
            if probe is not None:
                probe.close()
            with self.cv:
                self.cv.notify_all()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Warm a file into the page cache with large queued reads.")
    ap.add_argument("path")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--policy", choices=["idle", "share"], default="idle",
                    help="idle: only read while nobody else reads the disk; share: read alongside others")
    ap.add_argument("--duty", type=float, default=0.5, help="with --policy share: max share of disk time")
    ap.add_argument("--burst", type=int, default=16, help="1.28 MB requests queued per burst")
    a = ap.parse_args(argv)

    def log(m):
        print(time.strftime("%H:%M:%S"), m, flush=True)

    pf = Prefetcher(a.path, a.start, a.end, burst_reqs=a.burst, policy=a.policy, duty=a.duty, max_ahead=None,
                    log=log, log_every=30.0)
    t0 = time.time()
    pf.start()
    pf.join()
    if pf.error is not None:
        raise pf.error
    el = time.time() - t0
    log(f"done {a.path}: {pf.bytes_done / 1e9:.2f} GB in {el:.0f}s ({pf.bytes_done / max(el, 1e-9) / 1e6:.1f} MB/s); "
        f"io {pf.t_io:.0f}s, waited for idle disk {pf.t_wait_idle:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
