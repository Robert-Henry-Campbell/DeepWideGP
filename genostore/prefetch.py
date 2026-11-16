"""Throttled sequential page-cache prefetcher (stdlib only).

The UKB .pgen files live on a rotational disk that other jobs seek on, where buffered reads
degrade to read_ahead_kb-sized (128 KB) requests that each wait for a scheduler slot (~0.3 MB/s
measured on 2026-11-10: days per chromosome). ``posix_fadvise(WILLNEED)`` queues a burst of
max_sectors_kb-sized requests at once, which the elevator serves as one sequential sweep (tens of
MB/s); the real reader (pgenlib) then finds every page in the page cache.

Politeness: our queued bursts starve other readers, so their demand is measured in quiet probes
(``probe_s`` with nothing of ours in flight; every disk read /proc/diskstats then counts is theirs).
  idle (default)  never read while anyone else reads the disk: a burst only follows a quiet probe;
                  when others start we stop within ``probe_every`` s and wait for the disk to be idle
                  again, so the most another job loses is one ~0.3 s burst.
  share           read alongside others, sleeping after each burst so our share of disk time is at
                  most ``duty``: faster, but it slows them, so only when those jobs are yours to slow.
Cache riding (``ride_cache``, default on): what another job reading the same file (e.g. a GWAS)
has already put in the page cache is taken with non-blocking reads (preadv2 + RWF_NOWAIT), which
never queue a request on the disk, so we follow that job at its pace even while the disk is busy.

    python -m genostore.prefetch /path/to/file.pgen      # warm a whole file
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
                if ctypes.get_errno() not in (errno.EAGAIN, errno.EINTR):
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
    bps = read_bps(disk_of(path), window_s)
    return bps is not None and bps > threshold_bps


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


def read_bps(dev, window_s):
    """Bytes per second read from block device ``dev`` (all processes) over the next ``window_s``
    seconds, or None if it cannot be measured."""
    d0 = disk_read_bytes(dev)
    time.sleep(window_s)
    d1 = disk_read_bytes(dev)
    return None if d0 is None or d1 is None else (d1 - d0) / window_s


def willneed(fd, start, end):
    """Queue readahead for bytes [start, end) as requests a seek-bound disk serves in one sweep."""
    off = start
    while off < end:
        os.posix_fadvise(fd, off, min(REQ_BYTES, end - off), os.POSIX_FADV_WILLNEED)
        off += REQ_BYTES


class Prefetcher(threading.Thread):
    """Warm ``[start, end)`` of ``path`` into the page cache, sequentially, in bursts.

    A reader may report pf.set_consumed(pos) (done with everything below ``pos``) and call
    pf.wait_until(pos) (block until everything below ``pos`` is prefetched); the prefetcher then
    stays at most ``max_ahead`` bytes ahead, so its pages are used before the kernel evicts them.
    """

    def __init__(self, path, start=0, end=None, burst_reqs=16, policy="idle", duty=0.5,
                 max_ahead=2 << 30, contended_bps=150e3, probe_every=3.0, probe_s=0.25,
                 backoff_s=5.0, ride_cache=True, log=None, log_every=30.0):
        super().__init__(daemon=True)
        self.path = path
        size = os.path.getsize(path)
        self.pos = self.consumed = (start // REQ_BYTES) * REQ_BYTES      # everything below pos is prefetched
        self.end = size if end is None else min(end, size)
        self.burst_reqs = int(burst_reqs)
        if policy not in ("idle", "share"):
            raise ValueError("policy must be 'idle' or 'share'")
        self.policy, self.duty = policy, float(duty)
        self.last_progress = time.time()
        self.max_ahead = None if max_ahead is None else int(max_ahead)
        self.contended_bps, self.probe_every, self.probe_s = contended_bps, probe_every, probe_s
        self.backoff_s, self.ride_cache = backoff_s, ride_cache
        self.log, self.log_every = log, log_every
        self.dev = disk_of(path)
        self.cv = threading.Condition()
        self.stop_flag = False
        self.error = None
        self.bytes_done = self.bytes_from_cache = 0
        self.t_io = self.t_wait_idle = 0.0
        self.others_bps = float("nan")

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

    def run(self):
        try:
            self._run()
        except BaseException as e:  # surface to the consumer instead of dying silently
            self.error = e
            with self.cv:
                self.cv.notify_all()

    def _probe(self):
        """Quiet window: nothing of ours is in flight, so all disk reads seen are others'."""
        bps = read_bps(self.dev, self.probe_s)
        if bps is None:
            return False                      # cannot measure: assume the disk is ours
        self.others_bps = bps
        return bps > self.contended_bps

    def _advance(self, pos, n, from_cache):
        with self.cv:
            self.pos = pos + n
            self.bytes_done += n
            if from_cache:
                self.bytes_from_cache += n
            self.last_progress = time.time()
            self.cv.notify_all()

    def _run(self):
        fd = os.open(self.path, os.O_RDONLY)
        probe = None
        try:
            contended, t_probe = True, -1e18      # the disk counts as busy until a probe says otherwise
            probe = CacheProbe(self.path) if self.ride_cache else None
            if probe is not None and not probe.supported:
                probe.close()
                probe = None
            riding = False
            t_log, b_log, t_wait_log = time.time(), 0, 0.0
            while True:
                with self.cv:
                    while (not self.stop_flag and self.max_ahead is not None
                           and self.pos >= self.consumed + self.max_ahead and self.pos < self.end):
                        self.cv.wait(0.5)
                    if self.stop_flag or self.pos >= self.end:
                        break
                    pos = self.pos
                n = min(self.burst_reqs * REQ_BYTES, self.end - pos)
                if probe is not None:
                    # whatever is already cached (typically read by another job) costs the disk nothing
                    got = probe.cached(pos, n)
                    if got >= min(REQ_BYTES, self.end - pos):
                        got = self.end - pos if pos + got >= self.end else (got // 4096) * 4096
                        self._advance(pos, got, True)
                        b_log += got
                        riding = True
                        continue
                if self.dev is not None and (contended or time.perf_counter() - t_probe >= self.probe_every):
                    contended = self._probe()
                    if self.policy == "idle" and contended:
                        # queue nothing while others read; the cached path above is retried first
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
                t0 = time.perf_counter()
                willneed(fd, pos, pos + n)
                os.pread(fd, 1, pos + n - 1)      # returns once the (in-order) readahead reaches the last page
                dt = time.perf_counter() - t0
                self.t_io += dt
                if self.policy == "share" and contended and self.duty < 1.0:
                    time.sleep(min(2.0, dt * (1.0 / self.duty - 1.0)))
                self._advance(pos, n, False)
                b_log += n
                if self.log is not None and time.time() - t_log >= self.log_every:
                    el = time.time() - t_log
                    self.log(f"prefetch {os.path.basename(self.path)}: {self.pos / 1e9:.2f}/{self.end / 1e9:.2f} GB "
                             f"({b_log / el / 1e6:.1f} MB/s over the last {el:.0f}s; {self.bytes_from_cache / 1e9:.2f} GB "
                             f"taken from cache; waited for the disk {self.t_wait_idle / 60:.1f} min in total)")
                    t_log, b_log = time.time(), 0
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
    ap.add_argument("--end", type=int)
    ap.add_argument("--policy", choices=["idle", "share"], default="idle",
                    help="idle: only read while nobody else reads the disk; share: read alongside others")
    ap.add_argument("--duty", type=float, default=0.5, help="with --policy share: max share of disk time")
    ap.add_argument("--burst", type=int, default=16, help="1.28 MB requests queued per burst")
    a = ap.parse_args(argv)

    def log(m):
        print(time.strftime("%H:%M:%S"), m, flush=True)

    pf = Prefetcher(a.path, a.start, a.end, burst_reqs=a.burst, policy=a.policy, duty=a.duty, max_ahead=None, log=log)
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
