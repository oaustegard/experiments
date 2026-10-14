#!/usr/bin/env python3
"""Part 1 of the EmbeddingGemma 2 vector-DB test: recall@10 vs QPS for HNSW, IVF-RaBitQ and HNSW-RaBitQ.

Base: the first --shards shards of data/quora/shards (fp32, 768d, normalized).
Ground truth: exact inner-product top-10 on the raw fp32 768d vectors, so the
quality loss of truncation (Matryoshka d=128) counts against those configs.

Configs (inner product unless noted):
  HNSW (M=32, efConstruction=200), swept over efSearch:
    fp32_d768, fp32_d128 (Matryoshka: truncate + renormalize), sq8_d768,
    sq4_d768, sq4_d768_remex (centre + rotate)
  IVF-RaBitQ (IndexIVFRaBitQ, IP), swept over nprobe:
    rq2_d768, rq2_d768_remex, rq2_d512, rq2_d512_remex, rq1_d768, rq1_d768_remex
    (remex = centre + rotate)
  HNSW-RaBitQ (IndexHNSWRaBitQ, METRIC_L2 on unit-norm vectors, nb_bits in the
  constructor, M=32, efConstruction=200), swept over efSearch:
    hnswrq2_d768, hnswrq2_d768_rot, hnswrq2_d512, hnswrq2_d512_rot,
    hnswrq1_d768, hnswrq1_d768_rot
    (_rot = remex rotation ONLY, no centring: centring breaks unit norm, and
    L2 ranking equals IP ranking only for unit-norm vectors)
  Quantized configs also run with rescore (IndexRefineFlat, k_factor 4).

L2 == IP on unit vectors: ||q-x||^2 = 2 - 2 q.x, so for unit q and x the L2
top-k is the IP top-k. The smoke test asserts this on every HNSW-RaBitQ arm
(exact L2 top-10 vs exact IP top-10 on the same prepared vectors; for d=768
also vs the raw ground truth).

remex preprocessing: the base is centred on its own mean, then rotated with
remex's randomized Hadamard rotation (seed 42). Queries are rotated but not
centred. Then <Rq, R(x - m)> = <q, x> - <q, m>, so every query's ranking is
unchanged. The smoke test checks this against the uncentred vectors.

Usage:
  python hnsw_bench.py                       # full: 10 shards, q_dense.npy queries
  python hnsw_bench.py --smoke --out X.json  # 2 shards, 100 held-out queries, ef/nprobe 32/128
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import faiss
from remex import Quantizer, corpus_mean

HERE = Path(__file__).resolve().parent
SHARDS = HERE / "data" / "quora" / "shards"
QUERY_FILE = HERE / "data" / "quora" / "q_dense.npy"
K = 10
M = 32
EF_CONSTRUCTION = 200
REFINE_K_FACTOR = 4.0
SWEEP = [16, 32, 64, 128, 256, 512]
SEED = 42

CONFIGS = [
    dict(name="fp32_d768", dim=768, kind="hnsw", quant="fp32", remex=False),
    dict(name="fp32_d128", dim=128, kind="hnsw", quant="fp32", remex=False),
    dict(name="sq8_d768", dim=768, kind="hnsw", quant="sq8", remex=False),
    dict(name="sq4_d768", dim=768, kind="hnsw", quant="sq4", remex=False),
    dict(name="sq4_d768_remex", dim=768, kind="hnsw", quant="sq4", remex=True),
    dict(name="rq2_d768", dim=768, kind="ivf", quant="rq", nb_bits=2, remex=False),
    dict(name="rq2_d768_remex", dim=768, kind="ivf", quant="rq", nb_bits=2, remex=True),
    dict(name="rq2_d512", dim=512, kind="ivf", quant="rq", nb_bits=2, remex=False),
    dict(name="rq2_d512_remex", dim=512, kind="ivf", quant="rq", nb_bits=2, remex=True),
    dict(name="rq1_d768", dim=768, kind="ivf", quant="rq", nb_bits=1, remex=False),
    dict(name="rq1_d768_remex", dim=768, kind="ivf", quant="rq", nb_bits=1, remex=True),
    dict(name="hnswrq2_d768", dim=768, kind="hnsw_rq", quant="rq", nb_bits=2, remex=False),
    dict(name="hnswrq2_d768_rot", dim=768, kind="hnsw_rq", quant="rq", nb_bits=2, remex=True),
    dict(name="hnswrq2_d512", dim=512, kind="hnsw_rq", quant="rq", nb_bits=2, remex=False),
    dict(name="hnswrq2_d512_rot", dim=512, kind="hnsw_rq", quant="rq", nb_bits=2, remex=True),
    dict(name="hnswrq1_d768", dim=768, kind="hnsw_rq", quant="rq", nb_bits=1, remex=False),
    dict(name="hnswrq1_d768_rot", dim=768, kind="hnsw_rq", quant="rq", nb_bits=1, remex=True),
]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def load_base(n_shards):
    parts = [np.load(SHARDS / f"docs_{s:05d}.npy") for s in range(n_shards)]
    return np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)


def unit(x):
    return np.ascontiguousarray(x / np.linalg.norm(x, axis=1, keepdims=True), dtype=np.float32)


def exact_topk(Q, X, k=K, chunk=200):
    """Exact inner-product top-k, row-chunked to bound memory."""
    out = np.empty((len(Q), k), dtype=np.int64)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ X.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, part, axis=1), axis=1)
        out[i:i + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def exact_topk_l2(Q, X, k=K, chunk=200):
    """Exact squared-L2 top-k (smallest first), row-chunked."""
    out = np.empty((len(Q), k), dtype=np.int64)
    xn = (X * X).sum(axis=1)[None, :]
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk]
        D = (q * q).sum(axis=1)[:, None] - 2.0 * (q @ X.T) + xn
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(D, part, axis=1), axis=1)
        out[i:i + chunk] = np.take_along_axis(part, order, axis=1)
    return out


def recall_at_k(I, gt):
    hits = sum(len(set(I[i].tolist()) & set(gt[i].tolist())) for i in range(len(gt)))
    return hits / (len(gt) * K)


_ROT = {}


def remex_rotation(dim):
    if dim not in _ROT:
        R = np.asarray(Quantizer(dim, bits=4, seed=SEED, rotation="rht").R, dtype=np.float32)
        err = float(np.abs(R @ R.T - np.eye(dim, dtype=np.float32)).max())
        if err > 1e-4:
            raise RuntimeError(f"remex rotation d={dim} is not orthogonal (max err {err})")
        _ROT[dim] = R
    return _ROT[dim]


def prepare(cfg, base, Q):
    """Return (database, queries) for one config, in the space FAISS will see."""
    d = cfg["dim"]
    X, q = base[:, :d], Q[:, :d]
    if d < base.shape[1] or cfg["kind"] == "hnsw_rq":  # truncation or L2: unit norm
        X, q = unit(X), unit(q)
    if cfg["remex"]:
        R = remex_rotation(d)
        if cfg["kind"] == "hnsw_rq":
            # Rotation only. Centring would break unit norm, and with it L2 == IP.
            X = np.ascontiguousarray(X @ R.T, dtype=np.float32)
            q = np.ascontiguousarray(q @ R.T, dtype=np.float32)
        else:
            # Centring on the base mean, then rotation. Hypothesis for the IVF
            # result (rq2_d768 recall drops ~0.10 at nprobe 32 when this runs):
            # IVF assigns coarse cells by inner product on centred vectors that
            # are no longer unit norm, so the cell choice may get worse.
            mean = corpus_mean(X)
            X = np.ascontiguousarray((X - mean) @ R.T, dtype=np.float32)
            q = np.ascontiguousarray(q @ R.T, dtype=np.float32)  # rotate, do not centre
    return np.ascontiguousarray(X, dtype=np.float32), np.ascontiguousarray(q, dtype=np.float32)


def build(cfg, X):
    d = X.shape[1]
    faiss.omp_set_num_threads(4)
    t0 = time.perf_counter()
    if cfg["kind"] == "hnsw":
        if cfg["quant"] == "fp32":
            idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        else:
            qt = {"sq8": faiss.ScalarQuantizer.QT_8bit,
                  "sq4": faiss.ScalarQuantizer.QT_4bit}[cfg["quant"]]
            idx = faiss.IndexHNSWSQ(d, qt, M, faiss.METRIC_INNER_PRODUCT)
            idx.train(X)
        idx.hnsw.efConstruction = EF_CONSTRUCTION
        idx.add(X)
        code_bytes = {"fp32": 4 * d, "sq8": d, "sq4": (d + 1) // 2}[cfg["quant"]]
    elif cfg["kind"] == "hnsw_rq":
        # Constructor order is (d, M, nb_bits, metric); nb_bits must be passed here.
        idx = faiss.IndexHNSWRaBitQ(d, M, cfg["nb_bits"], faiss.METRIC_L2)
        if idx.metric_type != faiss.METRIC_L2:
            raise RuntimeError(f"{cfg['name']}: HNSW-RaBitQ metric is {idx.metric_type}")
        idx.hnsw.efConstruction = EF_CONSTRUCTION
        if not idx.is_trained:
            idx.train(X)
        idx.add(X)
        # Storage is a flat RaBitQ code per vector; size it with the same codec.
        code_bytes = int(faiss.IndexRaBitQ(d, faiss.METRIC_L2, cfg["nb_bits"]).sa_code_size())
        if code_bytes <= 0:
            raise RuntimeError(f"{cfg['name']}: HNSW-RaBitQ code size {code_bytes}")
    else:
        nlist = max(1, int(4 * np.sqrt(len(X))))
        quant = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
        idx = faiss.IndexIVFRaBitQ(quant, d, nlist, faiss.METRIC_INNER_PRODUCT, True,
                                   cfg["nb_bits"])
        idx.own_fields = True  # the index owns the quantizer from here on
        quant.this.disown()
        idx.train(X)
        idx.add(X)
        code_bytes = int(idx.code_size)
        if code_bytes <= 0:
            raise RuntimeError(f"{cfg['name']}: IVF-RaBitQ reports code_size {code_bytes}")
    return idx, code_bytes, time.perf_counter() - t0


def uses_ef(cfg):
    return cfg["kind"] in ("hnsw", "hnsw_rq")


def set_param(cfg, idx, value):
    if uses_ef(cfg):
        idx.hnsw.efSearch = value
    else:
        idx.nprobe = value


def sweep_values(cfg, idx):
    if uses_ef(cfg):
        return list(SWEEP)
    nlist = idx.nlist
    return sorted({min(v, nlist) for v in SWEEP})


def time_search(idx, Q):
    faiss.omp_set_num_threads(1)
    best = float("inf")
    I = None
    for _ in range(3):
        t0 = time.perf_counter()
        _, I_run = idx.search(Q, K)
        best = min(best, time.perf_counter() - t0)
        if I is None:
            I = I_run
    return I, len(Q) / best


def serialized_bytes(idx):
    return int(faiss.serialize_index(idx).nbytes)


KIND_LABEL = {"hnsw": "HNSW", "ivf": "IVF-RaBitQ", "hnsw_rq": "HNSW-RaBitQ"}


def run_config(cfg, base, Q, gt, n_base, n_q):
    X, q = prepare(cfg, base, Q)
    idx, code_bytes, build_s = build(cfg, X)
    log(f"[{cfg['name']}] built n={len(X)} d={X.shape[1]} code={code_bytes} B "
        f"build={build_s:.1f}s")
    variants = [("", idx, serialized_bytes(idx), False)]
    if cfg["quant"] != "fp32":
        refine = faiss.IndexRefineFlat(idx, faiss.swig_ptr(X))  # X kept alive below
        refine.k_factor = REFINE_K_FACTOR
        variants.append(("+rescore", refine, serialized_bytes(refine), True))
    rows = []
    for suffix, searcher, index_bytes, rescore in variants:
        for value in sweep_values(cfg, idx):
            set_param(cfg, idx, value)
            I, qps = time_search(searcher, q)
            rows.append({
                "config": cfg["name"] + suffix,
                "base_config": cfg["name"],
                "kind": KIND_LABEL[cfg["kind"]],
                "param": "efSearch" if uses_ef(cfg) else "nprobe",
                "value": int(value),
                "rescore": rescore,
                "dim": cfg["dim"],
                "remex": cfg["remex"],
                "remex_mode": ("rotate_only" if cfg["kind"] == "hnsw_rq" else "centre_rotate")
                if cfg["remex"] else None,
                "metric": "L2 (unit norm)" if cfg["kind"] == "hnsw_rq" else "IP",
                "quant": cfg["quant"],
                "nb_bits": cfg.get("nb_bits"),
                "recall@10": round(recall_at_k(I, gt), 4),
                "qps": round(qps, 1),
                "bytes_per_vector": code_bytes,
                "index_bytes": index_bytes,
                "build_s": round(build_s, 2),
                "n_base": n_base,
                "n_queries": n_q,
            })
            log(f"  {rows[-1]['config']:22s} {rows[-1]['param']}={value:<4d} "
                f"recall@10={rows[-1]['recall@10']:.4f} qps={qps:9.1f}")
    return rows


def check_remex_ranking(cfg, base, Q):
    """remex centring + rotation must not change the IP top-10 ranking."""
    d = cfg["dim"]
    X0, q0 = base[:, :d], Q[:, :d]
    if d < base.shape[1]:
        X0, q0 = unit(X0), unit(q0)
    X, q = prepare(cfg, base, Q)
    ref = exact_topk(q0, X0)
    got = exact_topk(q, X)
    same = np.mean([set(ref[i].tolist()) == set(got[i].tolist()) for i in range(len(Q))])
    return float(same)


def check_l2_ranking(cfg, base, Q, gt):
    """On the prepared unit-norm vectors, exact L2 top-10 must equal exact IP top-10."""
    X, q = prepare(cfg, base, Q)
    ip = exact_topk(q, X)
    l2 = exact_topk_l2(q, X)
    same = np.mean([set(ip[i].tolist()) == set(l2[i].tolist()) for i in range(len(Q))])
    if cfg["dim"] == 768:  # also against the raw 768d ground truth
        same_gt = np.mean([set(gt[i].tolist()) == set(l2[i].tolist()) for i in range(len(Q))])
    else:
        same_gt = None  # truncated space: the 768d ground truth is not the reference
    return float(same), (None if same_gt is None else float(same_gt))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=10, help="corpus shards (10 = 100k docs)")
    ap.add_argument("--nq", type=int, default=1000, help="number of queries")
    ap.add_argument("--smoke", action="store_true",
                    help="2 shards, 100 held-out base vectors as queries, ef/nprobe 32/128")
    ap.add_argument("--heldout", action="store_true",
                    help="take queries from the base (removed from it) instead of q_dense.npy")
    ap.add_argument("--out", default=str(HERE / "results_hnsw.json"))
    args = ap.parse_args()

    global SWEEP
    if args.smoke:
        args.shards, args.nq, args.heldout = 2, 100, True
        SWEEP = [32, 128]

    t_start = time.perf_counter()
    base = load_base(args.shards)
    if args.heldout:
        pick = np.random.default_rng(SEED).choice(len(base), args.nq, replace=False)
        Q = base[pick].copy()
        base = np.ascontiguousarray(np.delete(base, pick, axis=0))
    else:
        if not QUERY_FILE.exists():
            raise SystemExit(f"{QUERY_FILE} not found; the query encode has not finished")
        Q = np.ascontiguousarray(np.load(QUERY_FILE)[:args.nq], dtype=np.float32)
    n_base, n_q = len(base), len(Q)
    log(f"base={n_base} x {base.shape[1]}  queries={n_q}  out={args.out}")

    gt = exact_topk(Q, base)  # exact IP top-10 on raw fp32 768d
    log(f"ground truth done in {time.perf_counter() - t_start:.1f}s")

    results = []
    for cfg in CONFIGS:
        if cfg["remex"]:
            same = check_remex_ranking(cfg, base, Q)
            log(f"[{cfg['name']}] remex ranking check: top-10 sets identical to uncentred "
                f"vectors for {same * 100:.1f}% of queries")
            if same < 0.99:
                raise RuntimeError(f"{cfg['name']}: centring changed rankings ({same})")
        if cfg["kind"] == "hnsw_rq":
            same_ip, same_gt = check_l2_ranking(cfg, base, Q, gt)
            log(f"[{cfg['name']}] L2 vs IP exact top-10 identical for {same_ip * 100:.2f}% "
                f"of queries" + ("" if same_gt is None else
                                 f"; vs raw 768d ground truth {same_gt * 100:.2f}%"))
            if same_ip < 0.999 or (same_gt is not None and same_gt < 0.999):
                raise RuntimeError(f"{cfg['name']}: L2 top-10 does not match IP top-10")
        results.extend(run_config(cfg, base, Q, gt, n_base, n_q))
        Path(args.out).write_text(json.dumps(results, indent=1))  # checkpoint per config

    Path(args.out).write_text(json.dumps(results, indent=1))
    log(f"wrote {len(results)} points to {args.out} in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
