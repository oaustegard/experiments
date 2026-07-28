#!/usr/bin/env python3
"""Bigger-corpus test + the real question: given remax 1-bits the vectors, does
embedder weight precision below int8 still matter?

Dataset: BEIR NFCorpus (3633 docs, 323 test queries, real qrels) — two orders of
magnitude more queries than the 5-query muninn probe.

For each Jina ONNX variant (fp32 / int8 / int4) we measure recall@{10,100} two
ways:
  * full-float cosine over the emitted fp32 vectors (the embedding ceiling)
  * 1-bit remax codes: center on corpus mean -> truncate dim -> StackedSignBit
    (d=256,k=8,seed=0) -> Hamming search (the DEPLOYED remax_kb path)

If the 1-bit recall barely moves across fp32/int8/int4, then precision above the
1-bit floor is largely wasted — int8 is not the limit.
"""
from __future__ import annotations

import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "nfcorpus"
TOKENIZER = HERE / "tokenizer.json"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke
sys.path.insert(0, str(spoke("remax_kb")))
sys.path.insert(0, str(spoke("remax") / "src"))

from remax_kb.embedders import JinaONNXEmbedder  # noqa: E402
from remax_kb._hamming import hamming_scan, top_k  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

VARIANTS = [("fp32", HERE / "model.onnx"),          # 847 MB
            ("int8", HERE / "model.int8.onnx"),     # 212 MB (quantize_dynamic, whole graph)
            ("q4",   HERE / "model.q4.onnx"),        # 170 MB (4-bit matmul + int8 embed mop-up)
            ("q2",   HERE / "model.q2.onnx")]        # 141 MB (2-bit matmul + int8 embed mop-up)
DIM, K, SEED = 256, 8, 0
KS = (10, 100)
# Subsample so the run finishes in one drivable window (CPU embed is slow on
# NFCorpus's long abstracts). All gold docs for the kept queries are retained, so
# recall stays well-defined; fillers add distractors. 24x the old 5-query probe.
CAP_QUERIES, CAP_DOCS, MAXLEN, SUB_SEED = 120, 600, 256, 0


def load_nfcorpus():
    docs, doc_ids = [], []
    for line in (DATA / "corpus.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        docs.append((d.get("title", "") + " " + d.get("text", "")).strip())
        doc_ids.append(d["_id"])
    qtext = {}
    for line in (DATA / "queries.jsonl").read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            qtext[q["_id"]] = q["text"]
    qrels = defaultdict(set)
    for line in (DATA / "qrels" / "test.tsv").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        qid, cid, score = line.split("\t")
        if int(score) > 0:
            qrels[qid].add(cid)
    # keep only test queries that have at least one relevant doc present
    doc_id_set = set(doc_ids)
    queries = [(qid, qtext[qid]) for qid in qrels
               if qid in qtext and (qrels[qid] & doc_id_set)]

    # Corpus-first subsample (NFCorpus is dense, ~38 rel/query, so query-first
    # blows past any doc cap): fix a random CAP_DOCS-doc index, then keep queries
    # whose gold survives (>=3). Recall is measured against surviving gold within
    # this fixed corpus — a smaller but well-defined retrieval task.
    rng = np.random.default_rng(SUB_SEED)
    if CAP_DOCS and len(doc_ids) > CAP_DOCS:
        sel = sorted(rng.choice(len(doc_ids), CAP_DOCS, replace=False))
        sub_ids = [doc_ids[i] for i in sel]
        sub_docs = [docs[i] for i in sel]
    else:
        sub_ids, sub_docs = doc_ids, docs
    sub_set = set(sub_ids)
    sub_qrels = {qid: (rel & sub_set) for qid, rel in qrels.items()}
    queries = [(qid, qt) for qid, qt in queries if len(sub_qrels[qid]) >= 3]
    if CAP_QUERIES and len(queries) > CAP_QUERIES:
        idx = sorted(rng.choice(len(queries), CAP_QUERIES, replace=False))
        queries = [queries[i] for i in idx]
    return sub_docs, sub_ids, queries, sub_qrels


def embed(emb, texts, prompt, cache: Path, batch=32, dim=768):
    """Resumable embed: checkpoints a (N, dim) memmap + a progress counter so a
    reaped background job resumes instead of restarting. Background jobs in this
    session die during user-idle, mid-variant — incremental checkpointing makes
    each re-run converge."""
    if cache.exists():
        d = np.load(cache)
        if d["m"].shape[0] == len(texts):
            return d["m"], 0.0
    mm_path = cache.with_suffix(".memmap.npy")
    prog_path = cache.with_suffix(".prog")
    n = len(texts)
    if mm_path.exists() and prog_path.exists():
        mat = np.lib.format.open_memmap(mm_path, mode="r+")
        done = int(prog_path.read_text().strip())
        if mat.shape != (n, dim):
            mat = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    else:
        mat = np.lib.format.open_memmap(mm_path, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    t0 = time.time()
    for i in range(done, n, batch):
        v = emb.encode(texts[i:i + batch], prompt=prompt)
        mat[i:i + v.shape[0]] = v
        mat.flush()
        prog_path.write_text(str(min(i + batch, n)))
        if (i // batch) % 20 == 0:
            print(f"    {prompt} {min(i+batch,n)}/{n}", flush=True)
    dt = time.time() - t0
    out = np.array(mat)
    np.savez(cache, m=out)
    mm_path.unlink(missing_ok=True); prog_path.unlink(missing_ok=True)
    return out, dt


def recall_at(ranked_ids, rel, ks):
    return {k: len(set(ranked_ids[:k]) & rel) / len(rel) for k in ks}


def eval_cosine(dvecs, doc_ids, qvecs, queries, qrels, ks):
    agg = {k: 0.0 for k in ks}
    for j, (qid, _) in enumerate(queries):
        order = np.argsort(-(dvecs @ qvecs[j]))
        ranked = [doc_ids[i] for i in order[:max(ks)]]
        r = recall_at(ranked, qrels[qid], ks)
        for k in ks:
            agg[k] += r[k]
    return {k: agg[k] / len(queries) for k in ks}


def eval_onebit(dvecs, doc_ids, qvecs, queries, qrels, ks):
    mean = dvecs.mean(axis=0).astype(np.float32)
    q = StackedSignBitQuantizer(d=DIM, k=K, seed=SEED)
    dcodes = q.encode(np.ascontiguousarray((dvecs - mean)[:, :DIM]))
    qcodes = q.encode(np.ascontiguousarray((qvecs - mean)[:, :DIM]))
    agg = {k: 0.0 for k in ks}
    for j, (qid, _) in enumerate(queries):
        dist = hamming_scan(dcodes, qcodes[j])
        idx = top_k(dist, max(ks))
        ranked = [doc_ids[i] for i in idx]
        r = recall_at(ranked, qrels[qid], ks)
        for k in ks:
            agg[k] += r[k]
    return {k: agg[k] / len(queries) for k in ks}


def main():
    docs, doc_ids, queries, qrels = load_nfcorpus()
    print(f"NFCorpus: {len(docs)} docs, {len(queries)} test queries w/ qrels\n", flush=True)
    rows = []
    for tag, mp in VARIANTS:
        if not mp.exists():
            print(f"[{tag}] missing {mp.name}, skip"); continue
        size = mp.stat().st_size / 1e6
        print(f"=== {tag} ({size:.0f} MB) ===", flush=True)
        emb = JinaONNXEmbedder(model_path=mp, tokenizer_path=TOKENIZER, max_length=MAXLEN)
        dvecs, dt = embed(emb, docs, "document", HERE / f".nf_doc_{tag}.npz")
        qvecs, _ = embed(emb, [t for _, t in queries], "query", HERE / f".nf_qry_{tag}.npz")
        cos = eval_cosine(dvecs, doc_ids, qvecs, queries, qrels, KS)
        bit = eval_onebit(dvecs, doc_ids, qvecs, queries, qrels, KS)
        rate = f"{len(docs)/dt:.1f}/s" if dt else "cached"
        rows.append((tag, size, rate, cos, bit))
        print(f"  cosine  R@10={cos[10]:.3f} R@100={cos[100]:.3f}", flush=True)
        print(f"  1-bit   R@10={bit[10]:.3f} R@100={bit[100]:.3f}  (d={DIM},k={K})\n", flush=True)

    print("=" * 70)
    print(f"{'variant':<8}{'MB':>6}{'rate':>9}{'cos@10':>8}{'cos@100':>9}{'1b@10':>8}{'1b@100':>9}")
    for tag, size, rate, cos, bit in rows:
        print(f"{tag:<8}{size:>6.0f}{rate:>9}{cos[10]:>8.3f}{cos[100]:>9.3f}{bit[10]:>8.3f}{bit[100]:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
