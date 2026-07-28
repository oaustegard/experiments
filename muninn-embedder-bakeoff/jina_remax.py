#!/usr/bin/env python3
"""Our special case: Jina-q4 -> remax 1-bit, as a Gemini replacement for muninn.

Embeds the muninn corpus + queries with the q4 Jina ONNX (the deployable
runtime), runs the DEPLOYED remax pipeline (center -> truncate dim -> StackedSign
k -> Hamming), and scores R@5/@10 vs the phase-0 gold. Reports the 1-bit number
at a few (dim,k) byte budgets plus the full-float q4 ceiling.

Baselines (same gold/corpus): lexical 1.00/1.00, Jina-fp32 full-float 0.90/1.00.
Production muninn = Gemini->remax (no key this session to measure live).
"""
from __future__ import annotations

import json, sys, time, zipfile
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/claude-workspace")
HERE = Path(__file__).resolve().parent
KB = ROOT / ".spokes/muninn.austegard.com/knowledge/muninn.kb"
JINA = ROOT / "experiments/jina-int8-remax_kb"
sys.path.insert(0, str(ROOT / ".spokes/remax_kb"))
sys.path.insert(0, str(ROOT / ".spokes/remax/src"))
sys.path.insert(0, str(ROOT / "experiments/lexical-kb-phase0"))
from sweep import QUERIES, stem            # noqa: E402
from remax_kb._hamming import hamming_scan, top_k   # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

KS = (5, 10)
CONFIGS = [(256, 8), (512, 4), (768, 2)]   # (dim, k) -> bytes = dim*k/8 (all 256B)


def load_chunks():
    z = zipfile.ZipFile(KB)
    c = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    return [x["text"] for x in c], [stem(x["meta"]["source_path"]) for x in c]


def embed_resumable(emb, texts, tag, dim=768, batch=16):
    mm = HERE / f".jr_{tag}.mm.npy"; prog = HERE / f".jr_{tag}.prog"; n = len(texts)
    if mm.exists() and prog.exists():
        mat = np.lib.format.open_memmap(mm, mode="r+"); done = int(prog.read_text())
        if mat.shape != (n, dim): mat = np.lib.format.open_memmap(mm, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    else:
        mat = np.lib.format.open_memmap(mm, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    for i in range(done, n, batch):
        v = emb.encode(texts[i:i+batch], prompt="document" if tag == "doc" else "query")
        mat[i:i+len(v)] = v; mat.flush(); prog.write_text(str(min(i+batch, n)))
        if (i//batch) % 25 == 0: print(f"  {tag} {min(i+batch,n)}/{n}", flush=True)
    out = np.array(mat); mm.unlink(missing_ok=True); prog.unlink(missing_ok=True)
    return out


def recall_floatcos(D, posts, Q, ks):
    agg = {k: 0.0 for k in ks}
    for j, q in enumerate(QUERIES):
        order = np.argsort(-(D @ Q[j])); seen, ranked = set(), []
        for i in order:
            if posts[i] not in seen: seen.add(posts[i]); ranked.append(posts[i])
            if len(ranked) >= max(ks): break
        pos = {p: i+1 for i, p in enumerate(ranked)}
        for k in ks: agg[k] += sum(1 for g in set(q["gold"]) if pos.get(g, 999) <= k)/len(q["gold"])
    return {k: agg[k]/len(QUERIES) for k in ks}


def recall_remax(D, posts, Q, dim, k, ks):
    mean = D.mean(0).astype(np.float32)
    qz = StackedSignBitQuantizer(d=dim, k=k, seed=0)
    dcodes = qz.encode(np.ascontiguousarray((D - mean)[:, :dim]))
    qcodes = qz.encode(np.ascontiguousarray((Q - mean)[:, :dim]))
    agg = {kk: 0.0 for kk in ks}
    for j, q in enumerate(QUERIES):
        idx = top_k(hamming_scan(dcodes, qcodes[j]), len(posts))
        seen, ranked = set(), []
        for i in idx:
            if posts[i] not in seen: seen.add(posts[i]); ranked.append(posts[i])
            if len(ranked) >= max(ks): break
        pos = {p: i+1 for i, p in enumerate(ranked)}
        for kk in ks: agg[kk] += sum(1 for g in set(q["gold"]) if pos.get(g, 999) <= kk)/len(q["gold"])
    return {kk: agg[kk]/len(QUERIES) for kk in ks}


def main():
    from remax_kb.embedders import JinaQ4ONNXEmbedder
    emb = JinaQ4ONNXEmbedder(model_path=JINA / "model.q4.onnx", tokenizer_path=JINA / "tokenizer.json")
    docs, posts = load_chunks()
    print(f"corpus {len(docs)} chunks / {len(set(posts))} posts; embedding with Jina-q4", flush=True)
    t0 = time.time()
    D = embed_resumable(emb, docs, "doc")
    Q = embed_resumable(emb, [q["query"] for q in QUERIES], "qry")
    print(f"embedded in {time.time()-t0:.0f}s\n", flush=True)
    fc = recall_floatcos(D, posts, Q, KS)
    print(f"Jina-q4 full-float cosine : R@5={fc[5]:.3f} R@10={fc[10]:.3f}", flush=True)
    for dim, k in CONFIGS:
        r = recall_remax(D, posts, Q, dim, k, KS)
        print(f"Jina-q4 -> remax d={dim:<3} k={k} ({dim*k//8}B): R@5={r[5]:.3f} R@10={r[10]:.3f}", flush=True)
    print("\nbaselines (same gold): lexical 1.00/1.00 | Jina-fp32 float 0.90/1.00 | Gemini->remax = incumbent (unmeasured, no key)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
