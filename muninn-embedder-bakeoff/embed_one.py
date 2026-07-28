#!/usr/bin/env python3
"""Embed the muninn corpus with ONE model, score R@5/@10 vs phase-0 gold.

    python embed_one.py BAAI/bge-base-en-v1.5 bge-base

Resumable: doc vectors checkpoint to a memmap every batch, so a SIGKILL resumes
instead of restarting. Run one model per call (foreground) to dodge the
background-reaping / arena-SIGKILL issues seen with the heavy combined run.
"""
from __future__ import annotations

import json, sys, time, zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment, spoke

HERE = Path(__file__).resolve().parent
KB = spoke("muninn.austegard.com") / "knowledge/muninn.kb"
sys.path.insert(0, str(experiment("lexical-kb-phase0")))
from sweep import QUERIES, stem  # noqa: E402

KS = (5, 10)
QINSTR = "Represent this sentence for searching relevant passages: "


def load_chunks():
    z = zipfile.ZipFile(KB)
    c = [json.loads(l) for l in z.read("chunks.jsonl").decode().splitlines() if l.strip()]
    return [x["text"] for x in c], [stem(x["meta"]["source_path"]) for x in c]


def embed_resumable(model, texts, tag, dim, batch=16):
    mm = HERE / f".doc_{tag}.mm.npy"; prog = HERE / f".doc_{tag}.prog"
    n = len(texts)
    if mm.exists() and prog.exists():
        mat = np.lib.format.open_memmap(mm, mode="r+"); done = int(prog.read_text())
        if mat.shape != (n, dim): mat = np.lib.format.open_memmap(mm, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    else:
        mat = np.lib.format.open_memmap(mm, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    for i in range(done, n, batch):
        v = model.encode(texts[i:i+batch], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        mat[i:i+len(v)] = v; mat.flush(); prog.write_text(str(min(i+batch, n)))
        if (i//batch) % 20 == 0: print(f"  {min(i+batch,n)}/{n}", flush=True)
    return np.array(mat)


def score(D, posts, qv, gold):
    order = np.argsort(-(D @ qv)); seen, ranked = set(), []
    for i in order:
        if posts[i] not in seen: seen.add(posts[i]); ranked.append(posts[i])
        if len(ranked) >= max(KS): break
    pos = {p: i+1 for i, p in enumerate(ranked)}
    return {k: sum(1 for g in gold if pos.get(g, 999) <= k)/len(gold) for k in KS}


def main():
    import torch; torch.set_num_threads(4)
    from sentence_transformers import SentenceTransformer
    model_id, tag = sys.argv[1], sys.argv[2]
    docs, posts = load_chunks()
    m = SentenceTransformer(model_id)
    dim = m.get_sentence_embedding_dimension()
    print(f"{tag} ({dim}-d): embedding {len(docs)} chunks", flush=True)
    t0 = time.time()
    D = embed_resumable(m, docs, tag, dim)
    Q = m.encode([QINSTR + q["query"] for q in QUERIES], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    a5 = a10 = 0.0
    for j, q in enumerate(QUERIES):
        r = score(D, posts, Q[j], set(q["gold"])); a5 += r[5]; a10 += r[10]
    n = len(QUERIES)
    print(f"RESULT {tag} ({dim}-d): R@5={a5/n:.3f} R@10={a10/n:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    (HERE / f".doc_{tag}.mm.npy").unlink(missing_ok=True); (HERE / f".doc_{tag}.prog").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
