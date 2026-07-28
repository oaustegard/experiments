#!/usr/bin/env python3
"""k-sweep on the Mac-search (muninn) corpus — recall@10 vs stack count k.

Answers the open follow-up from remax `bench/results/CROSSOVER.md`: that sweep
was on SPECTER2; the shipped `.kb`/`.kbi` artifacts use a different embedder.
The Mac search corpus is embedded with **Gemini `gemini-embedding-001`** (768-d,
RETRIEVAL_DOCUMENT, L2-normalized) via the Cloudflare AI Gateway — so this is
the sweep on the embedder that production actually uses.

Protocol (mirrors CROSSOVER's self-retrieval fidelity test):
  1. Walk the corpus, chunk it (remax_kb default_chunker, ~500 chars) — the
     exact chunk set the production builder produces.
  2. Embed every chunk once as a document. Cache to embeddings.npz so the
     expensive step runs once; every k after that is pure re-binarization.
  3. Ground truth = float cosine top-10 in the FULL 768-d space (the real
     semantic neighbours), self-excluded.
  4. For each k: replicate the production transform (center by corpus mean →
     truncate to dim=256 → StackedSignBitQuantizer(d=256,k,seed=0).encode),
     then Hamming top-10. Report mean recall@10 vs the float ground truth.
  5. A "float dim=256" row is the truncation-only ceiling (k -> infinity).

Isolates the binarizer: both sides use document embeddings, so the only moving
part is k. Real queries use the RETRIEVAL_QUERY adapter — that asymmetry is held
out of scope here on purpose.
"""
from __future__ import annotations
import importlib.util, os, sys, time, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITE = ROOT / ".spokes" / "muninn.austegard.com"
sys.path[:0] = [str(ROOT / ".spokes" / "remax_kb"), str(ROOT / ".spokes" / "remax" / "src")]

DIM, SEED = 256, 0
KS = [1, 2, 3, 4, 6, 8, 12, 16]
TOPK = 10
EMB_CACHE = HERE / "embeddings.npz"


def load_build_module():
    spec = importlib.util.spec_from_file_location(
        "bm", SITE / "scripts" / "build_muninn_kb.py")
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    return bm


def get_embeddings(bm):
    if EMB_CACHE.exists():
        d = np.load(EMB_CACHE, allow_pickle=True)
        print(f"[cache] {d['vecs'].shape} from {EMB_CACHE.name}")
        return d["vecs"], list(d["ids"])
    chunks = list(bm.walk_corpus(SITE))
    print(f"[corpus] {len(chunks)} chunks")
    emb = bm.GeminiGatewayEmbedder(
        account_id=os.environ["CF_ACCOUNT_ID"],
        gateway_id=os.environ["CF_GATEWAY_ID"],
        gateway_token=os.environ["CF_API_TOKEN"])
    t0 = time.time()
    vecs = emb.encode([c.text for c in chunks], prompt="document")
    print(f"[embed] {vecs.shape} in {time.time()-t0:.1f}s")
    ids = [c.id for c in chunks]
    np.savez(EMB_CACHE, vecs=vecs.astype(np.float32), ids=np.array(ids, dtype=object))
    return vecs.astype(np.float32), ids


def topk_float(X, k, exclude_self=True):
    """Top-k by cosine (X assumed L2-normalized) — returns (n, k) indices."""
    sims = X @ X.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    return np.argpartition(-sims, k, axis=1)[:, :k]


def recall_at_k(pred_idx, gt_sets):
    hits = sum(len(set(pred_idx[i]) & gt_sets[i]) for i in range(len(gt_sets)))
    return hits / (len(gt_sets) * TOPK)


def main():
    bm = load_build_module()
    vecs, ids = get_embeddings(bm)
    N = len(ids)

    # already L2-normalized client-side, but enforce for the cosine math
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    # ground truth: float top-10 in full 768-d
    gt = topk_float(vecs, TOPK)
    gt_sets = [set(row) for row in gt]

    # production transform pieces
    mean_full = vecs.mean(axis=0).astype(np.float32)
    centered = vecs - mean_full
    trunc = np.ascontiguousarray(centered[:, :DIM])
    trunc_n = trunc / (np.linalg.norm(trunc, axis=1, keepdims=True) + 1e-12)

    from remax import StackedSignBitQuantizer, hamming_distances

    rows = []
    # ceiling: float cosine in the truncated dim=256 space (vs full-768 GT)
    ceil_idx = topk_float(trunc_n, TOPK)
    ceil_r = recall_at_k(ceil_idx, gt_sets)
    gt256_sets = [set(row) for row in ceil_idx]  # binarizer-only target
    print(f"[ceiling] float dim={DIM}: R@10(vs768)={ceil_r:.4f}")

    for k in KS:
        q = StackedSignBitQuantizer(d=DIM, k=k, seed=SEED)
        codes = q.encode(trunc)  # (N, DIM*k//8) packed uint8
        # per-query Hamming top-10 over the corpus, self-excluded
        pred = np.empty((N, TOPK), dtype=np.int64)
        for i in range(N):
            dists = hamming_distances(codes, codes[i]).astype(np.int64)
            dists[i] = 1 << 30  # exclude self
            idx = np.argpartition(dists, TOPK)[:TOPK]
            pred[i] = idx
        r768 = recall_at_k(pred, gt_sets)      # absolute (vs full-768 float)
        r256 = recall_at_k(pred, gt256_sets)   # binarizer-only (vs dim-256 float)
        bytes_per = DIM * k // 8
        rows.append((k, bytes_per, r768, r256))
        print(f"[k={k:2d}] bytes/chunk={bytes_per:3d}  R@10(vs768)={r768:.4f}  R@10(vs256)={r256:.4f}")

    out = {
        "corpus": "muninn.austegard.com (Mac search corpus)",
        "embedder": "gemini-embedding-001 @ output_dim=768, RETRIEVAL_DOCUMENT",
        "n_chunks": N, "dim": DIM, "seed": SEED, "topk": TOPK,
        "ceiling_float_dim256_R10_vs768": ceil_r,
        "sweep": [{"k": k, "bytes_per_chunk": b, "R_at_10_vs768": r768,
                   "R_at_10_vs256": r256} for k, b, r768, r256 in rows],
    }
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print("wrote results.json")


if __name__ == "__main__":
    main()
