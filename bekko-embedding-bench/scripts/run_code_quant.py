"""Quantization ladder on CODE, scored on the real file-discovery task.

The byte-budget work elsewhere in this experiment ran on the muninn blog corpus
(179 chunks, self-retrieval). Bit-depth behaviour is known to be
distribution-specific — the one-bit-beats-two inversion is exactly that — so
none of it should be assumed to transfer to code.

This runs the same ladder over the **11,380-chunk scikit-learn AST corpus** and
scores it on the **n=59 mini-CTXBench task** (NL bug report -> gold file set,
file-level recall@5/@10), not self-retrieval. That is the number a code sidecar
would actually be judged on.

Also sizes a hypothetical `.kb` sidecar for the repo, including the observation
that a code sidecar need not carry the chunk text at all: the corpus *is* the
working tree, so chunks can be (path, line-range) pointers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("REMEX_ROOT", "/home/user/remex"))
sys.path.insert(0, os.environ.get("REMAX_ROOT", "/home/user/remax/src"))
from bekko import BekkoEncoder, matryoshka  # noqa: E402
from eval_search import extract_identifiers, arm_rg, recall_at, rrf  # noqa: E402

import remex  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("BEKKO_BENCH_REPO", "/home/user/sklearn-bench"))


def file_rank(scores: np.ndarray, chunks: list[dict], topn: int = 400) -> list[str]:
    """Best-chunk-per-file ranking from a (n_chunks,) score vector."""
    order = np.argsort(-scores)[:topn]
    best: dict[str, float] = {}
    for i in order:
        f = chunks[i]["file"]
        if f not in best:
            best[f] = float(scores[i])
    return sorted(best, key=lambda f: -best[f])


def hamming_rank(qcode: np.ndarray, dcodes: np.ndarray, chunks: list[dict]) -> list[str]:
    d = np.bitwise_count(np.bitwise_xor(qcode, dcodes)).sum(axis=1)
    return file_rank(-d.astype(np.float64), chunks)


def main() -> None:
    inst = json.load(open(HERE / "instances.json"))
    chunks = json.load(open(HERE / "chunks_ast.json"))
    n = len(chunks)
    mat = np.asarray(np.memmap(HERE / "vecs_ast_a25m.f32", dtype=np.float32,
                               mode="r", shape=(n, 384)))
    enc = BekkoEncoder("a25m", threads=4)
    queries = [it["title"] + "\n" + it["body"] for it in inst]
    qv_full = enc.encode(queries, batch_size=8)

    # grep baseline + gold, shared
    rg_ranked = {}
    for it in inst:
        r, _, _ = arm_rg(extract_identifiers(it["title"] + "\n" + it["body"]))
        rg_ranked[it["issue"]] = r

    rows = []

    def score(label, codec, param, dim, ranker, bytes_per_vec):
        r5 = np.zeros(len(inst)); r10 = np.zeros(len(inst)); f10 = np.zeros(len(inst))
        for i, it in enumerate(inst):
            ranked = ranker(i)
            r5[i] = recall_at(ranked, it["gold"], 5)
            r10[i] = recall_at(ranked, it["gold"], 10)
            f10[i] = recall_at(rrf(rg_ranked[it["issue"]], ranked), it["gold"], 10)
        rows.append({"codec": codec, "param": param, "dim": dim,
                     "bytes": bytes_per_vec, "r@5": r5.mean(), "r@10": r10.mean(),
                     "rrf_r@10": f10.mean()})
        print(f"  {label:28s} {bytes_per_vec:5d} B   r@5 {r5.mean():.3f}  "
              f"r@10 {r10.mean():.3f}   +rg(RRF) r@10 {f10.mean():.3f}", flush=True)

    print(f"CODE corpus: {n} chunks, {len({c['file'] for c in chunks})} files; "
          f"task = n={len(inst)} file discovery\n")

    for dim in (384, 256, 128, 64):
        d = matryoshka(mat, dim if dim < 384 else None)
        q = matryoshka(qv_full, dim if dim < 384 else None)
        score(f"fp32 d={dim}", "fp32", 32, dim,
              lambda i, q=q, d=d: file_rank(d @ q[i], chunks), dim * 4)

    d384 = matryoshka(mat, None); q384 = matryoshka(qv_full, None)
    for bits in (1, 2, 4):
        qz = remex.Quantizer(d=384, bits=bits, seed=0)
        xh = qz.decode(qz.encode(d384))
        xh = xh / np.clip(np.linalg.norm(xh, axis=1, keepdims=True), 1e-9, None)
        score(f"remex {bits}-bit d=384", "remex", bits, 384,
              lambda i, xh=xh: file_rank(xh @ q384[i], chunks), 384 * bits // 8)
    for k in (1, 2, 4, 8):
        sq = StackedSignBitQuantizer(d=384, k=k, seed=0).fit(d384)
        dc, qc = sq.encode(d384), sq.encode(q384)
        score(f"remax k={k} d=384", "remax", k, 384,
              lambda i, qc=qc, dc=dc: hamming_rank(qc[i], dc, chunks), 384 * k // 8)

    json.dump(rows, open(HERE / "results_code_quant.json", "w"), indent=1, default=float)

    # ── sidecar sizing ──────────────────────────────────────────────────────
    print("\n=== .kb sidecar sizing for scikit-learn ===")
    src = sum(f.stat().st_size for f in (REPO / "sklearn").rglob("*")
              if f.is_file() and f.name.endswith((".py", ".pyx", ".pxd", ".tp")))
    text = sum(len(c["text"]) for c in chunks)
    print(f"  source tree (.py/.pyx/.pxd/.tp)     {src / 2**20:8.1f} MB")
    print(f"  chunk text if carried in the .kb    {text / 2**20:8.1f} MB (raw)")
    for label, b in (("remex 1-bit d=384", 48), ("remex 2-bit d=384", 96),
                     ("remax k=8 d=384", 384), ("fp32 d=384", 1536)):
        print(f"  vectors only, {label:22s} {n * b / 2**20:8.2f} MB "
              f"({100 * n * b / src:.1f}% of source)")


if __name__ == "__main__":
    main()
