#!/usr/bin/env python3
"""Flat (exhaustive, no index) recall of remex codes vs FAISS IndexRaBitQ on the Quora base.

Same data as hnsw_bench.py: the first --shards shards of data/quora/shards (fp32, 768d, normalized),
queries data/quora/q_dense.npy (first --nq). Ground truth: exact inner-product top-10 and top-100 on
the raw fp32 768d vectors, so truncation loss (Matryoshka d=128, d=512) counts against those arms.

Arms (inner product everywhere; every quantized arm scores float queries against codes):
  fp32 d=768 (sanity, recall 1.0) and fp32 d=128 (truncate + renormalize)
  remex d in {768, 512} (truncate + renormalize), bits in {1, 2, 4}, uncentered and centered
    (centered = Quantizer(mean=corpus mean), as bench.py), rotation "rht", 3 rotation seeds
  remex nested: encode at 8 bits, read at 4/2/1 (precision=), d=768, centered and uncentered, 3 seeds
  FAISS IndexRaBitQ(d, METRIC_INNER_PRODUCT, nb_bits) flat, nb_bits in {1, 2, 4}, d in {768, 512};
    nb_bits passed to the constructor (setting it afterwards crashes faiss 1.15.1); default qb=4 kept;
    trained on a seeded sample of min(N, 20000) base vectors.

Bytes per vector: remex = d*bits/8 (excludes the 4 B norm, as bench.py); FAISS = sa_code_size().
Encode seconds: remex = Quantizer.encode(); FAISS = train + add.

Output: results_remex_flat.json (or --out) and a table sorted by bytes per vector.

Usage:
  python remex_flat.py                  # full: 10 shards, 1000 queries
  python remex_flat.py --smoke          # 1 shard, 50 queries, 1 thread; JSON goes to the scratchpad
"""
import argparse
import os
import sys
import time
from pathlib import Path


def _pre_threads():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--threads", type=int, default=1)
    a, _ = p.parse_known_args()
    return a.threads


# Thread caps must be set before numpy / faiss load their thread pools.
_T = str(_pre_threads())
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _T

import json  # noqa: E402
import numpy as np  # noqa: E402
import faiss  # noqa: E402
import warnings  # noqa: E402
from remex import Quantizer, corpus_mean, AnisotropyWarning  # noqa: E402

# Uncentered arms on this corpus always trigger the anisotropy hint; the centered arms are the comparison.
warnings.filterwarnings("ignore", category=AnisotropyWarning)

HERE = Path(__file__).resolve().parent
SHARDS = HERE / "data" / "quora" / "shards"
QUERY_FILE = HERE / "data" / "quora" / "q_dense.npy"
SMOKE_OUT = Path("/tmp/claude-0/-home-user-claude-workspace/cb6fdb49-c401-521e-8359-7842aa851e70/scratchpad/results_remex_flat_smoke.json")
K1, K2 = 10, 100
SEEDS = (0, 1, 2)
FAISS_TRAIN_MAX = 20000


def l2n(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def trunc(X, d):
    return X if d >= X.shape[1] else l2n(X[:, :d])


def exact_topk(Q, X, k, chunk=256):
    """Exact inner-product top-k, ids sorted by score (descending)."""
    out = np.empty((len(Q), k), np.int64)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ X.T
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ss = np.take_along_axis(S, idx, 1)
        out[i:i + chunk] = np.take_along_axis(idx, np.argsort(-ss, 1), 1)
    return out


def recall(ret, gt, k):
    return float(np.mean([len(set(r[:k].tolist()) & set(g[:k].tolist())) / k for r, g in zip(ret, gt)]))


def load_base(n_shards):
    parts = [np.load(SHARDS / f"docs_{s:05d}.npy") for s in range(n_shards)]
    return np.ascontiguousarray(np.concatenate(parts).astype(np.float32))


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.min()), float(a.max())


def make_row(arm, family, d, bits, centered, nested, bytes_pv, r10, r100, enc_s, seeds):
    m10, lo10, hi10 = summarize(r10)
    m100, lo100, hi100 = summarize(r100)
    return dict(
        arm=arm, family=family, dim=d, bits=bits, centered=centered, nested=nested,
        bytes_per_vector=round(float(bytes_pv), 1),
        recall10=round(m10, 4), recall10_min=round(lo10, 4), recall10_max=round(hi10, 4),
        recall100=round(m100, 4), recall100_min=round(lo100, 4), recall100_max=round(hi100, 4),
        encode_s=round(float(np.mean(enc_s)), 3),
        seeds=seeds, recall10_by_seed=[round(x, 4) for x in r10], recall100_by_seed=[round(x, 4) for x in r100],
        encode_s_by_seed=[round(x, 3) for x in enc_s],
    )


def run_remex(X, Q, gt10, gt100, d, bits, centered, seed, mu, nested=False):
    """X, Q already truncated to d. Returns (recall10, recall100, encode_s)."""
    if nested:
        enc = Quantizer(d, bits=8, seed=seed, rotation="rht", mean=mu)
        t = time.perf_counter(); cv = enc.encode(X); te = time.perf_counter() - t
        Xhat = enc.decode(cv, precision=bits)
    else:
        enc = Quantizer(d, bits=bits, seed=seed, rotation="rht", mean=mu)
        t = time.perf_counter(); cv = enc.encode(X); te = time.perf_counter() - t
        Xhat = enc.decode(cv)
    # Asymmetric: float query against decoded (dequantized) codes, as bench.py does.
    top = exact_topk(Q, Xhat, K2)
    return recall(top, gt10, K1), recall(top, gt100, K2), te


def run_faiss(X, Q, gt10, gt100, d, nb_bits, train_seed=0):
    idx = faiss.IndexRaBitQ(d, faiss.METRIC_INNER_PRODUCT, nb_bits)
    rng = np.random.default_rng(train_seed)
    ntr = min(len(X), FAISS_TRAIN_MAX)
    sample = X[rng.permutation(len(X))[:ntr]]
    t = time.perf_counter()
    idx.train(sample)
    idx.add(X)
    te = time.perf_counter() - t
    _, I = idx.search(Q, K2)
    I = np.where(I < 0, 0, I)
    return recall(I, gt10, K1), recall(I, gt100, K2), te, idx.sa_code_size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="1 shard, 50 queries, JSON to scratchpad")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    try:
        import torch
        torch.set_num_threads(a.threads)
    except ImportError:
        pass
    if a.smoke:
        a.shards, a.nq = 1, 50
    out = Path(a.out) if a.out else (SMOKE_OUT if a.smoke else HERE / "results_remex_flat.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time(); c0 = time.process_time()
    X = load_base(a.shards)
    Q = np.load(QUERY_FILE).astype(np.float32)[:a.nq]
    print(f"base {X.shape} queries {Q.shape} threads {a.threads} shards {a.shards}", flush=True)

    gt10 = exact_topk(Q, X, K1)
    gt100 = exact_topk(Q, X, K2)
    rows = []

    def log(r):
        rows.append(r)
        print(f"  {r['arm']:40s} {r['bytes_per_vector']:>7.1f} B  R@10 {r['recall10']:.4f}  R@100 {r['recall100']:.4f}  enc {r['encode_s']:.2f}s", flush=True)

    # fp32 sanity and Matryoshka truncation (exact search on truncated vectors)
    for d in (768, 512, 256, 128):
        Xd, Qd = trunc(X, d), trunc(Q, d)
        top = exact_topk(Qd, Xd, K2)
        log(make_row(f"fp32 d={d}", "fp32", d, 32, False, False, d * 4,
                     [recall(top, gt10, K1)], [recall(top, gt100, K2)], [0.0], None))

    for d in (768, 512, 256):
        Xd, Qd = trunc(X, d), trunc(Q, d)
        for centered in (False, True):
            mu = corpus_mean(Xd) if centered else None
            for bits in (1, 2, 4):
                r10, r100, es = [], [], []
                for s in SEEDS:
                    x10, x100, te = run_remex(Xd, Qd, gt10, gt100, d, bits, centered, s, mu)
                    r10.append(x10); r100.append(x100); es.append(te)
                name = f"remex{' centered' if centered else ''} {bits}-bit d={d}"
                log(make_row(name, "remex-c" if centered else "remex", d, bits, centered, False,
                             d * bits / 8, r10, r100, es, list(SEEDS)))

    for centered in (False, True):
        d = 768
        Xd, Qd = trunc(X, d), trunc(Q, d)
        mu = corpus_mean(Xd) if centered else None
        for bits in (4, 2, 1):
            r10, r100, es = [], [], []
            for s in SEEDS:
                x10, x100, te = run_remex(Xd, Qd, gt10, gt100, d, bits, centered, s, mu, nested=True)
                r10.append(x10); r100.append(x100); es.append(te)
            name = f"remex{' centered' if centered else ''} 8->{bits}-bit nested d={d}"
            log(make_row(name, "remex-nested", d, bits, centered, True,
                         d * bits / 8, r10, r100, es, list(SEEDS)))

    for d in (768, 512, 256):
        Xd, Qd = trunc(X, d), trunc(Q, d)
        for nb in (1, 2, 4):
            x10, x100, te, cs = run_faiss(Xd, Qd, gt10, gt100, d, nb)
            log(make_row(f"faiss RaBitQ {nb}-bit d={d}", "faiss-rabitq", d, nb, False, False,
                         cs, [x10], [x100], [te], None))

    rows.sort(key=lambda r: (r["bytes_per_vector"], r["arm"]))
    meta = dict(script="remex_flat.py", shards=a.shards, n_base=int(len(X)), n_queries=int(len(Q)),
                threads=a.threads, seeds=list(SEEDS), faiss_train_max=FAISS_TRAIN_MAX,
                faiss_version=faiss.__version__, wall_s=round(time.time() - t0, 1),
                cpu_s=round(time.process_time() - c0, 1))
    json.dump(dict(meta=meta, rows=rows), open(out, "w"), indent=1)

    print()
    print(f"{'arm':40s} {'B/vec':>8s} {'R@10 mean [min,max]':>26s} {'R@100 mean [min,max]':>26s} {'enc s':>7s}")
    for r in rows:
        if r["seeds"]:
            s10 = f"{r['recall10']:.4f} [{r['recall10_min']:.4f},{r['recall10_max']:.4f}]"
            s100 = f"{r['recall100']:.4f} [{r['recall100_min']:.4f},{r['recall100_max']:.4f}]"
        else:
            s10, s100 = f"{r['recall10']:.4f}", f"{r['recall100']:.4f}"
        print(f"{r['arm']:40s} {r['bytes_per_vector']:>8.1f} {s10:>26s} {s100:>26s} {r['encode_s']:>7.2f}")
    print(f"\nwrote {out}  wall {meta['wall_s']}s  cpu {meta['cpu_s']}s", flush=True)


if __name__ == "__main__":
    main()
