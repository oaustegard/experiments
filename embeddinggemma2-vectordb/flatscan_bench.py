#!/usr/bin/env python3
"""Flat (exhaustive) inner-product scan cost per vector, per code format.

Synthetic random unit Gaussian vectors (scan cost depends on bytes and code
format, not on what the vectors mean). One index per format is grown
incrementally through the size points, so each point is a prefix of the same
index. Quantizers are trained on a 20k sample. Search: 32 queries, k=10, best
of --reps wall-clock runs, at each thread count.

Vectors are drawn from a fixed pool of POOL_N unit Gaussians per dimension,
cycled to reach N. Duplicates do not change scan cost; this avoids spending
hours generating tens of millions of Gaussians for the 4 GB points.

Usage:
  python flatscan_bench.py --smoke --out SCRATCH/results_flat_smoke.json
  python flatscan_bench.py                       # full run, writes results_flat.json
  python flatscan_bench.py --sizes 16 64 --threads 1 4
"""
import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

HERE = Path(__file__).resolve().parent
TRAIN_N = 20_000
POOL_N = 100_000
NQ = 32
K = 10
ADD_CHUNK = 50_000
RSS_CAP_MB = 10 * 1024
OVERHEAD_MB = 1536          # estimate of non-code RSS (pool, query, train, process)
DEFAULT_SIZES = [16, 64, 256, 1024, 4096]
DEFAULT_THREADS = [1, 4]

# name, dimension, kind, nb_bits (bits for SQ/RaBitQ)
FORMATS = [
    ("fp32_d768", 768, "flat", None),
    ("fp32_d128", 128, "flat", None),
    ("sq8_d768", 768, "sq", 8),
    ("sq4_d768", 768, "sq", 4),
    ("rabitq1_d768", 768, "rabitq", 1),
    ("rabitq2_d768", 768, "rabitq", 2),
    ("rabitq2fs_d768", 768, "rabitq_fs", 2),
    ("rabitq2_d512", 512, "rabitq", 2),
]


def gen_unit(rng, n, d, chunk=10_000):
    out = np.empty((n, d), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        x = rng.standard_normal((hi - lo, d), dtype=np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        out[lo:hi] = x
    return out


def build_trained(kind, d, nb, train):
    ip = faiss.METRIC_INNER_PRODUCT
    if kind == "flat":
        return faiss.IndexFlatIP(d)
    if kind == "sq":
        qt = {8: faiss.ScalarQuantizer.QT_8bit, 4: faiss.ScalarQuantizer.QT_4bit}[nb]
        idx = faiss.IndexScalarQuantizer(d, qt, ip)
        idx.train(train)
        return idx
    if kind == "rabitq":
        idx = faiss.IndexRaBitQ(d, ip, nb)   # nb_bits must be given to the constructor
        idx.train(train)
        return idx
    if kind == "rabitq_fs":
        rq = faiss.IndexRaBitQ(d, ip, nb)
        rq.train(train)
        return faiss.IndexRaBitQFastScan(rq)  # empty, trained copy; then add()
    raise ValueError(kind)


def add_cycled(idx, pool, start, stop):
    """Add pool rows (start..stop-1) cyclically, in chunks."""
    for lo in range(start, stop, ADD_CHUNK):
        m = min(ADD_CHUNK, stop - lo)
        rows = np.arange(lo, lo + m) % len(pool)
        idx.add(pool[rows])


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=float, nargs="+", default=None, help="code storage sizes in MB")
    ap.add_argument("--threads", type=int, nargs="+", default=None, help="thread counts to time")
    ap.add_argument("--reps", type=int, default=3, help="best-of reps per timing point")
    ap.add_argument("--smoke", action="store_true", help="16 MB only, 1 thread")
    ap.add_argument("--out", type=Path, default=HERE / "results_flat.json")
    a = ap.parse_args()
    sizes = a.sizes or ([16] if a.smoke else DEFAULT_SIZES)
    threads = a.threads or ([1] if a.smoke else DEFAULT_THREADS)
    sizes = sorted(sizes)

    print(f"faiss {faiss.__version__}  sizes={sizes} MB  threads={threads}  reps={a.reps}", flush=True)
    pools, queries_by_d, train_by_d = {}, {}, {}

    def data_for(d):
        if d not in pools:
            t = time.perf_counter()
            pools[d] = gen_unit(np.random.default_rng(1000 + d), POOL_N, d)
            queries_by_d[d] = gen_unit(np.random.default_rng(2000 + d), NQ, d)
            train_by_d[d] = np.ascontiguousarray(pools[d][:TRAIN_N])
            print(f"  generated pool d={d} ({POOL_N} vectors) in {time.perf_counter()-t:.1f}s", flush=True)
        return pools[d], queries_by_d[d], train_by_d[d]

    points, errors = [], []
    meta = {
        "script": Path(__file__).name,
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "faiss_version": faiss.__version__,
        "cpu": cpu_model(),
        "nproc": int(np.array(faiss.omp_get_max_threads())),
        "python": sys.version.split()[0],
        "sizes_mb": sizes, "threads": threads, "reps": a.reps,
        "k": K, "nq": NQ, "train_n": TRAIN_N, "pool_n": POOL_N,
        "vectors": "synthetic unit Gaussian, pool cycled to N",
        "bytes_per_vector": "IndexX.sa_code_size() (includes RaBitQ per-vector factors)",
        "ns_per_vector": "best search wall time / (nq * ntotal) * 1e9",
        "maxrss_note": "maxrss_mb is the process peak so far (ru_maxrss)",
    }

    def dump():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"meta": meta, "points": points, "errors": errors}, indent=2))

    for name, d, kind, nb in FORMATS:
        pool, Q, train = data_for(d)
        print(f"[{name}]", flush=True)
        try:
            idx = build_trained(kind, d, nb, train)
            bpv = idx.sa_code_size()
        except Exception as e:  # record, keep going
            errors.append({"format": name, "stage": "build", "error": repr(e)})
            print(f"  BUILD FAILED: {e!r}", flush=True)
            dump()
            continue
        added = 0
        for target_mb in sizes:
            n = int(target_mb * 2**20 // bpv)
            if target_mb + OVERHEAD_MB > RSS_CAP_MB:
                points.append({"format": name, "target_mb": target_mb, "skipped": "would exceed RSS cap"})
                continue
            faiss.omp_set_num_threads(max(threads))
            t = time.perf_counter()
            add_cycled(idx, pool, added, n)
            add_s = time.perf_counter() - t
            added = n
            for th in threads:
                faiss.omp_set_num_threads(th)
                best = None
                times = []
                for _ in range(a.reps):
                    t0 = time.perf_counter()
                    idx.search(Q, K)
                    dt = time.perf_counter() - t0
                    times.append(dt)
                    best = dt if best is None else min(best, dt)
                pt = {
                    "format": name, "d": d, "kind": kind, "nb_bits": nb,
                    "N": idx.ntotal, "bytes_per_vector": bpv,
                    "code_mb": round(idx.ntotal * bpv / 2**20, 2),
                    "threads": th,
                    "ns_per_vector": round(best / (NQ * idx.ntotal) * 1e9, 3),
                    "best_search_s": round(best, 5),
                    "search_s_all": [round(x, 5) for x in times],
                    "add_s": round(add_s, 2),
                    "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                }
                points.append(pt)
                print(f"  {target_mb:>6} MB N={idx.ntotal:>9} {bpv:>5} B/v  t={th}  "
                      f"{pt['ns_per_vector']:>9.3f} ns/vec  add {add_s:.1f}s  rss {pt['maxrss_mb']} MB", flush=True)
            dump()
        del idx
    dump()
    print(f"wrote {a.out} ({len(points)} points, {len(errors)} errors)", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
