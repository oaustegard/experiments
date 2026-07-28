#!/usr/bin/env python3
"""int8 rotations: how much recall do we trade to quarter the rotation sidecar?

The .kbi ships binarizer/rotations.f32 — k Haar matrices (dim x dim) at 4 B/elem,
corpus-independent, and the fat part of a small .kbi. They feed only a SIGN test
(x @ Q >= 0), so full f32 precision is likely overkill. This quantizes them to
int8 (k*dim^2 bytes + a tiny per-column scale -> 4x smaller) and measures the
cost three ways at dim=768/k=2:

  baseline  f32 rotations both sides              (what ships today)
  int8 A    int8 rotations both sides (re-pack)   (consistent; real format change)
  int8 B    f32 doc codes, int8 query rotations   (migration w/o re-encoding docs)

Reports: doc-code bit-flip rate vs f32, self-retrieval R@10, and query R@10 vs
float-768 gold (the number that tracks live search). Reuses cached doc + query
embeddings; embeds the 40 verify queries once and caches them.
"""
from __future__ import annotations
import importlib.util, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SITE = ROOT / ".spokes" / "muninn.austegard.com"
sys.path[:0] = [str(ROOT / ".spokes" / "remax_kb"), str(ROOT / ".spokes" / "remax" / "src")]

DIM, K, SEED, TOPK = 768, 2, 0, 10


def load_query_vecs():
    qcache = HERE / "queries.npz"
    vspec = importlib.util.spec_from_file_location("vf", HERE / "verify.py")
    vf = importlib.util.module_from_spec(vspec); vspec.loader.exec_module(vf)
    queries = vf.QUERIES
    if qcache.exists():
        return np.load(qcache)["qv"].astype(np.float32), queries
    spec = importlib.util.spec_from_file_location("bm", SITE / "scripts" / "build_muninn_kb.py")
    bm = importlib.util.module_from_spec(spec); spec.loader.exec_module(bm)
    real = bm.GeminiGatewayEmbedder(account_id=os.environ["CF_ACCOUNT_ID"],
        gateway_id=os.environ["CF_GATEWAY_ID"], gateway_token=os.environ["CF_API_TOKEN"])
    qv = real.encode(queries, prompt="query").astype(np.float32)
    np.savez(qcache, qv=qv)
    return qv, queries


def quant_int8_percol(R):
    """R: (k, d_in, d_out). Per-output-column symmetric int8. Returns dequantized."""
    scale = np.abs(R).max(axis=1, keepdims=True) / 127.0  # (k,1,d_out)
    scale = np.where(scale == 0, 1.0, scale)
    Ri = np.round(R / scale).astype(np.int8)
    return (Ri.astype(np.float32) * scale).astype(np.float32), Ri, scale


def main():
    d = np.load(HERE / "embeddings.npz", allow_pickle=True)
    docs = d["vecs"].astype(np.float32)
    docs = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    qv, _ = load_query_vecs()
    qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)
    N = docs.shape[0]

    from remax import StackedSignBitQuantizer, hamming_distances

    mean = docs.mean(axis=0).astype(np.float32)
    doc_t = np.ascontiguousarray((docs - mean)[:, :DIM])
    q_t = np.ascontiguousarray((qv - mean)[:, :DIM])

    # gold: float-768 query.doc cosine top-10
    sims = qv @ docs.T
    gold = [set(np.argpartition(-sims[i], TOPK)[:TOPK]) for i in range(qv.shape[0])]

    def enc(rot, X):
        q = StackedSignBitQuantizer(d=DIM, k=K, seed=SEED)
        q.rotations_ = rot.astype(q.dtype)
        return q.encode(X)

    R_f32 = StackedSignBitQuantizer(d=DIM, k=K, seed=SEED).rotations_.astype(np.float32)
    R_i8_deq, Ri, scale = quant_int8_percol(R_f32)

    doc_f32 = enc(R_f32, doc_t)
    doc_i8 = enc(R_i8_deq, doc_t)
    q_f32 = enc(R_f32, q_t)
    q_i8 = enc(R_i8_deq, q_t)

    flips = np.unpackbits(np.bitwise_xor(doc_f32, doc_i8), axis=1).sum()
    total_bits = N * DIM * K
    print(f"doc-code bit-flip rate (f32->int8 rotations): {flips/total_bits*100:.3f}%")

    def qrecall(doc_codes, q_codes):
        rec = []
        for i in range(q_codes.shape[0]):
            dist = hamming_distances(doc_codes, q_codes[i])
            top = np.argpartition(dist, TOPK)[:TOPK]
            rec.append(len(set(top) & gold[i]) / TOPK)
        return float(np.mean(rec))

    rows = [
        ("baseline  f32 / f32", qrecall(doc_f32, q_f32)),
        ("int8 A    i8  / i8 ", qrecall(doc_i8, q_i8)),
        ("int8 B    f32 / i8 ", qrecall(doc_f32, q_i8)),
    ]
    print("\nquery R@10 vs float-768 gold:")
    for name, r in rows:
        print(f"  {name}  {r:.4f}")

    f32_kb = K * DIM * DIM * 4 / 1024
    i8_kb = (K * DIM * DIM + scale.size * 4) / 1024
    print(f"\nrotation sidecar: f32 {f32_kb:.0f} KB  ->  int8 {i8_kb:.0f} KB "
          f"({f32_kb/i8_kb:.1f}x smaller)")

    import json
    (HERE / "int8_rotations.json").write_text(json.dumps({
        "dim": DIM, "k": K, "n_chunks": N, "topk": TOPK,
        "doc_bitflip_pct": flips / total_bits * 100,
        "query_R10": {name.split()[0] + "_" + name.split()[1]: r for name, r in rows},
        "rotation_kb_f32": f32_kb, "rotation_kb_int8": i8_kb,
    }, indent=2))
    print("wrote int8_rotations.json")


if __name__ == "__main__":
    main()
