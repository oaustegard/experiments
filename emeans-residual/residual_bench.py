#!/usr/bin/env python3
"""Residual remex codes and IVF probing over k-means centroids from several sources.

Corpus: the 100,000 Quora EmbeddingGemma 2 vectors (unit norm, d=768); queries: the
1,000 test queries. Ground truth is the exact inner-product top 10 on fp32.

Centroids come from cluster_bench.py (trained on the first 90,000 rows): Lloyd after
2, 5 and 25 iterations, e-means with budgets of 2 and 5 epochs, and the shared random
initial points untrained. Every row of the corpus is assigned to its nearest centroid.

Residual arm. Row x in cell j is stored as remex codes of the direction of
r = x - c_j, plus ||r|| and c_j . r in fp32, and scored with remex 2's centred
estimator applied per cell:

    q . c_j  +  c_j . r  +  ||r|| * (q - c_j) . u_hat / kappa

K = 1 with c = corpus mean is remex's own centred mode; the script checks that its
scores match remex.Quantizer(mean=...) before running anything else.

IVF arm. Cells ranked by L2 distance from the query to the centroid; for each nprobe
the share of true top-10 neighbours inside the probed cells and the share of the
corpus scanned. Baseline: remex.IVFCoarseIndex (lsh and rotated_prefix) at the same
number of cells.

Writes results/residual.json and results/perquery.npz.
"""
import argparse
import json
import os
import time
import warnings
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--cent", default=None, help="directory of centroid .npy files")
p.add_argument("--ks", default="16,64,256,1024,4096")
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--bits", default="1,2,4")
p.add_argument("--threads", type=int, default=4)
p.add_argument("--nq", type=int, default=1000)
p.add_argument("--ndocs", type=int, default=100000)
p.add_argument("--out", default=None)
args = p.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(args.threads)

import numpy as np  # noqa: E402
import faiss  # noqa: E402
import remex  # noqa: E402
from remex import Quantizer, IVFCoarseIndex  # noqa: E402
from remex.codebook import assign_codes  # noqa: E402

warnings.filterwarnings("ignore", category=remex.AnisotropyWarning)
faiss.omp_set_num_threads(args.threads)

HERE = Path(__file__).resolve().parent
CENT = Path(args.cent) if args.cent else HERE / "cent"
OUT = Path(args.out) if args.out else HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
NTRAIN = 90000
TOPK = 10
NPROBES = (1, 2, 4, 8, 16, 32, 64, 128)
SOURCES = (("init", 0), ("lloyd", 2), ("lloyd", 5), ("lloyd", 25),
           ("emeans", 2), ("emeans", 5))

X = np.load(HERE / "data" / "docs.npy")[: args.ndocs]
Q = np.load(HERE / "data" / "quora" / "q_dense.npy")[: args.nq]
n, d = X.shape
nq = len(Q)
bits_list = [int(b) for b in args.bits.split(",")]
ks = [int(k) for k in args.ks.split(",")]
seeds = [int(s) for s in args.seeds.split(",")]


def topk(S, k=TOPK):
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return idx


S_exact = Q @ X.T
GT = topk(S_exact)
del S_exact
QUANT = {b: Quantizer(d, bits=b, seed=0) for b in bits_list}
Q_ROT = QUANT[bits_list[0]]._rotate_rows(Q)


def per_query_recall(S):
    got = topk(S)
    return np.array([len(np.intersect1d(got[i], GT[i])) for i in range(nq)],
                    dtype=np.float32) / TOPK


def residual_scores(C, assign):
    """Yield (bits, score matrix) for the corpus coded against centroids C."""
    Ca = C[assign]
    r = X.astype(np.float64) - Ca.astype(np.float64)
    lengths = np.sqrt((r ** 2).sum(axis=1))
    dots = (r * Ca).sum(axis=1)
    u = (r / np.maximum(lengths, 1e-8)[:, None]).astype(np.float32)
    del r
    q0 = QUANT[bits_list[0]]
    u_rot = q0._rotate_rows(u)
    del u
    ca_rot = q0._rotate_rows(np.ascontiguousarray(C, dtype=np.float32))[assign]
    offset = (Q @ C.T)[:, assign]
    for b in bits_list:
        qz = QUANT[b]
        codes = assign_codes(u_rot, qz.boundaries, b)
        v = qz.centroids[codes].astype(np.float32)
        vlen = np.sqrt((v.astype(np.float64) ** 2).sum(axis=1))
        scale = lengths / (np.maximum(vlen, 1e-12) * qz._cosine_factor())
        bias = dots - scale * (v * ca_rot).sum(axis=1, dtype=np.float64)
        y = v * scale[:, None].astype(np.float32)
        yield b, Q_ROT @ y.T + bias.astype(np.float32)[None, :] + offset


def ivf_kmeans(C, assign):
    K = len(C)
    depth = min(K, max(NPROBES))
    _, order = faiss.knn(Q, np.ascontiguousarray(C, dtype=np.float32), depth)
    rank = np.full((nq, K), K, dtype=np.int32)
    rank[np.arange(nq)[:, None], order] = np.arange(depth, dtype=np.int32)
    sizes = np.bincount(assign, minlength=K)
    gt_rank = rank[np.arange(nq)[:, None], assign[GT]]
    out = []
    for pr in NPROBES:
        if pr > K:
            break
        out.append({"nprobe": pr,
                    "recall": float((gt_rank < pr).mean()),
                    "scanned": float(sizes[order[:, :pr]].sum(axis=1).mean() / n)})
    return out


def ivf_remex(n_bits, mode, centred):
    mu = X[:NTRAIN].mean(axis=0).astype(np.float32) if centred else None
    qz = Quantizer(d, bits=1, seed=0, mean=mu)
    ivf = IVFCoarseIndex(qz, qz.encode(X), n_bits=n_bits, mode=mode, seed=0)
    sizes = np.diff(ivf.cell_offsets)
    cells_of_gt = ivf.cell_ids[GT].astype(np.int64)
    Qh = Q - mu if centred else Q
    qcells = [ivf._query_cell_from_rot(qz._rotate_query(q)) for q in Qh]
    out = []
    for pr in NPROBES:
        if pr > ivf.n_cells:
            break
        rec = scan = 0.0
        for i, qc in enumerate(qcells):
            cells = ivf.probe_cells(qc, pr)
            rec += np.isin(cells_of_gt[i], cells).mean()
            scan += sizes[cells].sum() / n
        out.append({"nprobe": pr, "recall": rec / nq, "scanned": scan / nq})
    return out


def bytes_per_vector(bits, K, table_dtype_bytes=4):
    cell_id = int(np.ceil(np.log2(K))) / 8 if K > 1 else 0.0
    row = d * bits / 8 + 8 + cell_id          # codes, fp32 ||r|| and c.r, cell id
    table = K * d * table_dtype_bytes / n     # centroid table spread over the corpus
    return {"row_bytes": row, "table_bytes": table, "total_bytes": row + table}


records, perquery = [], {}


def run_residual(tag, K, seed, C, assign, table_bytes=4, extra=None):
    mse = float(((X - C[assign]).astype(np.float64) ** 2).sum(axis=1).mean())
    for b, S in residual_scores(C, assign):
        pq = per_query_recall(S)
        key = f"{tag}|K{K}|s{seed}|b{b}"
        perquery[key] = pq
        rec = {"kind": "residual", "source": tag, "K": K, "seed": seed, "bits": b,
               "recall10": float(pq.mean()), "corpus_mse": mse,
               **bytes_per_vector(b, K, table_bytes), **(extra or {})}
        records.append(rec)
        print(json.dumps(rec), flush=True)


# --- control: K = 1 must reproduce remex's centred search ---------------------------
mu = X[:NTRAIN].mean(axis=0).astype(np.float32)
zero = np.zeros(n, dtype=np.int64)
control = {}
for b, S in residual_scores(mu[None, :], zero):
    qz = Quantizer(d, bits=b, seed=0, mean=mu)
    Y, bias = qz._score_vectors(qz.encode(X))
    S_ref = Q @ Y.T + bias[None, :]
    # The max is not the test: this path rotates a float64-derived direction and remex
    # a float32 one, so about 7 of 76.8M coordinates land on the other side of a cell
    # boundary, and at 1 bit one flipped sign moves that row's score by ~0.005.
    absdiff = np.abs(S - S_ref)
    diff = float(absdiff.max())
    # A flipped code shifts one corpus row for every query, so the test is the median
    # difference and the share of rows that differ at all, not a high percentile.
    median = float(np.median(absdiff[:100]))
    rows_off = float((absdiff[:100].max(axis=0) > 1e-4).mean())
    del absdiff
    r_mine, r_ref = float(per_query_recall(S).mean()), float(per_query_recall(S_ref).mean())
    control[b] = {"max_abs_score_diff": diff, "median_abs_score_diff": median,
                  "share_of_rows_differing": rows_off,
                  "recall_mine": r_mine, "recall_remex": r_ref}
    print("control", b, control[b], flush=True)
    assert median < 1e-5 and rows_off < 1e-3 and abs(r_mine - r_ref) < 2e-3, control[b]
run_residual("mean", 1, 0, mu[None, :], zero)

# --- centroid sources ---------------------------------------------------------------
ivf_records = []
t_start = time.time()
for K in ks:
    for seed in seeds:
        for arm, budget in SOURCES:
            f = CENT / f"{arm}_k{K}_s{seed}_t{budget:g}.npy"
            if not f.exists():
                print("missing", f.name, flush=True)
                continue
            C = np.load(f)
            _, lab = faiss.knn(X, C, 1)
            assign = lab.ravel().astype(np.int64)
            tag = f"{arm}{budget}" if arm != "init" else "random"
            run_residual(tag, K, seed, C, assign)
            rec = {"kind": "ivf", "source": tag, "K": K, "seed": seed,
                   "curve": ivf_kmeans(C, assign)}
            ivf_records.append(rec)
            # fp16 centroid table: encoder and scorer both see the rounded centroids
            if arm == "emeans" and budget == 5 and seed == seeds[0]:
                C16 = C.astype(np.float16).astype(np.float32)
                _, lab16 = faiss.knn(X, C16, 1)
                run_residual("emeans5-fp16", K, seed, C16,
                             lab16.ravel().astype(np.int64), table_bytes=2)
    print(f"K={K} done at {time.time() - t_start:.0f}s", flush=True)

for K in ks:
    nb = int(np.log2(K))
    for mode in ("lsh", "rotated_prefix"):
        for centred in (False, True):
            rec = {"kind": "ivf", "source": f"remex-{mode}" + ("-centred" if centred else ""),
                   "K": K, "seed": 0, "curve": ivf_remex(nb, mode, centred)}
            ivf_records.append(rec)
            print(json.dumps(rec), flush=True)

json.dump({"n": n, "d": d, "nq": nq, "remex": remex.__version__,
           "faiss": faiss.__version__, "control": control,
           "residual": records, "ivf": ivf_records},
          open(OUT / "residual.json", "w"), indent=1)
np.savez_compressed(OUT / "perquery.npz", **perquery)
print("DONE", flush=True)
