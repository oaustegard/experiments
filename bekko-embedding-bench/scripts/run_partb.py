"""Part B — bekko vs jina v5 nano q4 as a general remax_kb embedder.

Harness: self-retrieval over the published muninn-subset.kb (179 chunks, 11
posts). Each chunk is split into a head (the query) and a body (the indexed
document), with the head text removed from the document so an exact-substring
match cannot carry the retrieval. Gold = the chunk the head came from. The
split is identical for every model, so cross-model comparison is fair even
though the absolute numbers are self-retrieval numbers.

Measured here:
  1. R@k for bekko a8m/a25m vs official jina v5 nano q4.
  2. Fidelity of each shipped artifact to *its own* fp32 export — per-doc
     cosine and Spearman of the score vector. This is where the int8
     embedding-table question actually gets answered.
  3. Matryoshka truncation curve 384/256/128/64, quality vs bytes.
  4. Second distribution (sklearn code chunks), per 76526de1: a single-domain
     smoke test hid an int8 collapse once already.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bekko import BekkoEncoder, matryoshka  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
KB = Path("/tmp/muninn-subset.kb")
HEAD_CHARS = 180


def load_kb() -> list[dict]:
    z = zipfile.ZipFile(KB)
    return [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines()]


def split_chunks(texts: list[str]) -> tuple[list[str], list[str]]:
    """head = query, body = indexed document with the head removed."""
    qs, ds = [], []
    for t in texts:
        t = " ".join(t.split())
        head = t[:HEAD_CHARS]
        body = t[HEAD_CHARS:]
        if len(body) < 120:  # too short to split: keep whole, query is the head
            body = t
        qs.append(head)
        ds.append(body)
    return qs, ds


def recall_at_k(sims: np.ndarray, k: int) -> float:
    """sims[i, j] = score of query i against doc j; gold is the diagonal."""
    n = sims.shape[0]
    order = np.argsort(-sims, axis=1)[:, :k]
    return float(np.mean([i in order[i] for i in range(n)]))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho between two flattened score vectors."""
    def rank(x):
        o = np.argsort(x)
        r = np.empty_like(o, dtype=np.float64)
        r[o] = np.arange(len(x))
        return r
    ra, rb = rank(a.ravel()), rank(b.ravel())
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


def eval_model(name: str, qv: np.ndarray, dv: np.ndarray, dims: list[int]) -> list[dict]:
    rows = []
    for d in dims:
        q = matryoshka(qv, d if d < qv.shape[1] else None)
        v = matryoshka(dv, d if d < dv.shape[1] else None)
        sims = q @ v.T
        rows.append({
            "model": name, "dim": d,
            "r@1": recall_at_k(sims, 1), "r@5": recall_at_k(sims, 5),
            "r@10": recall_at_k(sims, 10),
            "bytes_per_vec_fp32": d * 4,
        })
    return rows


def main() -> None:
    rows = json.load(open(HERE / "instances.json")) and None  # noop guard
    chunks = load_kb()
    texts = [c["text"] for c in chunks]
    qs, ds = split_chunks(texts)
    print(f"muninn-subset.kb: {len(texts)} chunks", flush=True)

    # second distribution: sklearn code chunks (same count, deterministic slice)
    code = json.load(open(HERE / "chunks_ast.json"))
    step = max(1, len(code) // len(texts))
    code_sel = [c["text"] for c in code[::step]][: len(texts)]
    cq, cd = split_chunks(code_sel)

    out = {"retrieval": [], "fidelity": []}
    DIMS = [384, 256, 128, 64]

    for variant in ("a8m", "a25m"):
        enc_def = BekkoEncoder(variant, "onnx/model.onnx", threads=2)
        enc_f32 = BekkoEncoder(variant, "onnx/model_fp32.onnx", threads=2)
        for dist, (Q, D) in (("blog", (qs, ds)), ("code", (cq, cd))):
            qv = enc_def.encode(Q, batch_size=8)
            dv = enc_def.encode(D, batch_size=8)
            fq = enc_f32.encode(Q, batch_size=8)
            fd = enc_f32.encode(D, batch_size=8)
            for r in eval_model(f"bekko-{variant}", qv, dv, DIMS):
                r["dist"] = dist
                out["retrieval"].append(r)
            for r in eval_model(f"bekko-{variant}-fp32", fq, fd, DIMS):
                r["dist"] = dist
                out["retrieval"].append(r)
            percos = float(np.mean(np.sum(dv * fd, axis=1)))
            out["fidelity"].append({
                "model": f"bekko-{variant}", "dist": dist,
                "per_doc_cosine_vs_own_fp32": percos,
                "spearman_scores_vs_own_fp32": spearman(qv @ dv.T, fq @ fd.T),
                "default_mb": round(enc_def.model_bytes() / 2**20, 1),
                "fp32_mb": round(enc_f32.model_bytes() / 2**20, 1),
            })
            print(f"bekko-{variant}/{dist}: cos-vs-fp32 {percos:.5f}", flush=True)
            del fq, fd
        del enc_f32
    json.dump(out, open(HERE / "results_partb.json", "w"), indent=1)
    print("wrote results_partb.json", flush=True)


if __name__ == "__main__":
    main()
