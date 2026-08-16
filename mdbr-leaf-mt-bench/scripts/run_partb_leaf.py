"""MongoDB/mdbr-leaf-mt as a remax_kb embedder — Part B of the bench family.

Same harness as ``bekko-embedding-bench``: self-retrieval over the 179-chunk
muninn blog subset (chunks read from the committed ``bench_bekko-a8m.kb``, so
no external fetch) plus the 179-chunk sklearn AST slice as the second
distribution. Head/body split identical to ``run_partb.py``; gold is the
diagonal. The jina v5 nano q4 incumbent is re-encoded here on the same splits
so leaf-vs-jina comparisons are paired per query, not read off old aggregates.

Measured:
  1. R@1/5/10 for leaf fp32 / int8 (model_quantized) / q4, MRL curve
     1024/512/384/256/128/64, both distributions.
  2. Fidelity of the quantized exports vs the fp32 export (per-doc cosine,
     Spearman of the score matrix).
  3. Paired significance (exact McNemar + paired bootstrap CI) for the
     decision-relevant comparisons against the incumbent.
"""
from __future__ import annotations

import json
import sys
import zipfile
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
BEKKO_BENCH = HERE.parent / "bekko-embedding-bench"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BEKKO_BENCH / "scripts"))

from jina import JinaQ4Encoder  # noqa: E402
from leaf import LeafMTEncoder, matryoshka  # noqa: E402

HEAD_CHARS = 180
RNG = np.random.default_rng(0)


def load_blog_chunks() -> list[str]:
    z = zipfile.ZipFile(BEKKO_BENCH / "bench_bekko-a8m.kb")
    return [json.loads(l)["text"] for l in z.read("chunks.jsonl").decode().splitlines()]


def load_code_chunks(n: int) -> list[str]:
    code = json.load(open(BEKKO_BENCH / "chunks_ast.json"))
    step = max(1, len(code) // n)
    return [c["text"] for c in code[::step]][:n]


def split_chunks(texts: list[str]) -> tuple[list[str], list[str]]:
    qs, ds = [], []
    for t in texts:
        t = " ".join(t.split())
        head, body = t[:HEAD_CHARS], t[HEAD_CHARS:]
        if len(body) < 120:
            body = t
        qs.append(head)
        ds.append(body)
    return qs, ds


def recall_at_k(sims: np.ndarray, k: int) -> float:
    order = np.argsort(-sims, axis=1)[:, :k]
    return float(np.mean([i in order[i] for i in range(sims.shape[0])]))


def hits_at(sims: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-sims, axis=1)[:, :k]
    return np.array([i in order[i] for i in range(sims.shape[0])], dtype=np.int8)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        o = np.argsort(x)
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(len(x))
        return r
    ra, rb = rank(a.ravel()), rank(b.ravel())
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return n01, n10, p


def boot_ci(a: np.ndarray, b: np.ndarray, reps: int = 20000) -> tuple[float, float]:
    n = len(a)
    idx = RNG.integers(0, n, size=(reps, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


LEAF_DIMS = [1024, 512, 384, 256, 128, 64]
JINA_DIMS = [768, 384, 256, 128, 64]


def main() -> None:
    blog = load_blog_chunks()
    qs, ds = split_chunks(blog)
    code = load_code_chunks(len(blog))
    cq, cd = split_chunks(code)
    dists = {"blog": (qs, ds), "code": (cq, cd)}
    print(f"blog {len(qs)} chunks, code {len(cq)} chunks", flush=True)

    out = {"retrieval": [], "fidelity": [], "claims": [], "n": len(qs)}
    vecs: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    leaf_variants = {
        "leaf-mt-fp32": "onnx/model.onnx",
        "leaf-mt-int8": "onnx/model_quantized.onnx",
        "leaf-mt-q4": "onnx/model_q4.onnx",
    }
    for name, onnx_name in leaf_variants.items():
        enc = LeafMTEncoder(onnx_name, threads=4)
        for dist, (Q, D) in dists.items():
            qv = enc.encode(Q, prompt="query", batch_size=8)
            dv = enc.encode(D, prompt="document", batch_size=8)
            vecs[(name, dist)] = (qv, dv)
            for d in LEAF_DIMS:
                q = matryoshka(qv, d if d < enc.full_dim else None)
                v = matryoshka(dv, d if d < enc.full_dim else None)
                sims = q @ v.T
                out["retrieval"].append({
                    "model": name, "dist": dist, "dim": d,
                    "r@1": recall_at_k(sims, 1), "r@5": recall_at_k(sims, 5),
                    "r@10": recall_at_k(sims, 10), "bytes_per_vec_fp32": d * 4,
                })
            print(f"{name}/{dist}: r@10 {out['retrieval'][-len(LEAF_DIMS)]['r@10']:.3f} "
                  f"@1024 ({enc.last_wall:.1f}s docs)", flush=True)
        mb = enc.model_bytes() / 2**20
        out.setdefault("model_mb", {})[name] = round(mb, 1)
        del enc

    jina = JinaQ4Encoder(threads=4)
    for dist, (Q, D) in dists.items():
        qv = jina.encode(Q, prompt="query", batch_size=8)
        dv = jina.encode(D, prompt="document", batch_size=8)
        vecs[("jina-q4", dist)] = (qv, dv)
        for d in JINA_DIMS:
            q = matryoshka(qv, d if d < 768 else None)
            v = matryoshka(dv, d if d < 768 else None)
            sims = q @ v.T
            out["retrieval"].append({
                "model": "jina-v5-nano-q4", "dist": dist, "dim": d,
                "r@1": recall_at_k(sims, 1), "r@5": recall_at_k(sims, 5),
                "r@10": recall_at_k(sims, 10), "bytes_per_vec_fp32": d * 4,
            })
        print(f"jina-q4/{dist} done", flush=True)
    out["model_mb"]["jina-v5-nano-q4"] = round(jina.model_bytes() / 2**20, 1)
    del jina

    # fidelity of the quantized leaf exports vs the fp32 export
    for name in ("leaf-mt-int8", "leaf-mt-q4"):
        for dist in dists:
            qv, dv = vecs[(name, dist)]
            fq, fd = vecs[("leaf-mt-fp32", dist)]
            out["fidelity"].append({
                "model": name, "dist": dist,
                "per_doc_cosine_vs_fp32": float(np.mean(np.sum(dv * fd, axis=1))),
                "spearman_scores_vs_fp32": spearman(qv @ dv.T, fq @ fd.T),
            })

    # paired significance — decision-relevant comparisons
    def H(model: str, dist: str, dim: int | None, k: int) -> np.ndarray:
        qv, dv = vecs[(model, dist)]
        full = qv.shape[1]
        q = matryoshka(qv, dim if dim and dim < full else None)
        v = matryoshka(dv, dim if dim and dim < full else None)
        return hits_at(q @ v.T, k)

    CLAIMS = [
        ("leaf fp32 @1024 vs jina @768, blog R@10",
         H("leaf-mt-fp32", "blog", None, 10), H("jina-q4", "blog", None, 10)),
        ("leaf fp32 @1024 vs jina @768, code R@10",
         H("leaf-mt-fp32", "code", None, 10), H("jina-q4", "code", None, 10)),
        ("leaf fp32 @1024 vs jina @768, blog R@1",
         H("leaf-mt-fp32", "blog", None, 1), H("jina-q4", "blog", None, 1)),
        ("leaf fp32 @1024 vs jina @768, code R@1",
         H("leaf-mt-fp32", "code", None, 1), H("jina-q4", "code", None, 1)),
        ("iso-byte 1024B: leaf @256 vs jina @256, blog R@10",
         H("leaf-mt-fp32", "blog", 256, 10), H("jina-q4", "blog", 256, 10)),
        ("iso-byte 1024B: leaf @256 vs jina @256, code R@10",
         H("leaf-mt-fp32", "code", 256, 10), H("jina-q4", "code", 256, 10)),
        ("iso-byte 256B: leaf @64 vs jina @64, blog R@10",
         H("leaf-mt-fp32", "blog", 64, 10), H("jina-q4", "blog", 64, 10)),
        ("iso-byte 256B: leaf @64 vs jina @64, code R@10",
         H("leaf-mt-fp32", "code", 64, 10), H("jina-q4", "code", 64, 10)),
        ("leaf int8 export vs leaf fp32 export, blog R@10",
         H("leaf-mt-int8", "blog", None, 10), H("leaf-mt-fp32", "blog", None, 10)),
        ("leaf q4 export vs leaf fp32 export, blog R@10",
         H("leaf-mt-q4", "blog", None, 10), H("leaf-mt-fp32", "blog", None, 10)),
        ("leaf q4 export vs leaf fp32 export, code R@10",
         H("leaf-mt-q4", "code", None, 10), H("leaf-mt-fp32", "code", None, 10)),
    ]
    print(f"\n{'claim':<52}{'Δ':>8}{'95% CI':>18}{'w/l':>9}{'p':>8}")
    for label, a, b in CLAIMS:
        diff = float(a.mean() - b.mean())
        lo, hi = boot_ci(a, b)
        n01, n10, p = mcnemar_exact(a, b)
        print(f"{label:<52}{diff:+8.3f}  [{lo:+.3f},{hi:+.3f}]  {n01:>3}/{n10:<4}{p:>8.4f}")
        out["claims"].append({"claim": label, "delta": diff, "ci_lo": lo, "ci_hi": hi,
                              "wins": n01, "losses": n10, "p": p,
                              "significant": bool(p < 0.05)})

    json.dump(out, open(HERE / "results_partb_leaf.json", "w"), indent=1)
    print("\nwrote results_partb_leaf.json", flush=True)


if __name__ == "__main__":
    main()
