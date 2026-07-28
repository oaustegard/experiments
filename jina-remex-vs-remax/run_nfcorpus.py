#!/usr/bin/env python3
"""remex vs remax on BEIR NFCorpus — the credible curve (120 queries, real qrels).

Same operating points as run_muninn.py, but on NFCorpus (medical abstracts —
a domain far from muninn's tech prose, and the corpus where per-tensor int8
Jina was shown domain-fragile). Recall@{10,100} against test qrels.

Corpus-first subsample (CAP_DOCS docs, then queries whose surviving gold >=3),
matching experiments/jina-int8-remax_kb/bench_nfcorpus.py so numbers are
comparable to the shipped 1-bit d256/k8 reference there.

Embedder: q4 Jina ONNX (fp32-parity output). Vectors are L2-normalized, so
remex per-row norms are redundant and excluded from bytes/row.
"""
from __future__ import annotations

import json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/home/user/claude-workspace")
HERE = Path(__file__).resolve().parent
A = HERE / "assets"
DATA = HERE / "data" / "nfcorpus"
sys.path.insert(0, str(ROOT / ".spokes/remax_kb"))
sys.path.insert(0, str(ROOT / ".spokes/remax/src"))
from remax import StackedSignBitQuantizer             # noqa: E402
from remax_kb._hamming import hamming_scan, top_k     # noqa: E402
from remex import Quantizer                           # noqa: E402

KS = (10, 100)
SEED = 0
CAP_QUERIES, CAP_DOCS, MAXLEN, SUB_SEED = 120, 600, 256, 0


def load_nfcorpus():
    docs, doc_ids = [], []
    for line in (DATA / "corpus.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        docs.append((d.get("title", "") + " " + d.get("text", "")).strip()); doc_ids.append(d["_id"])
    qtext = {}
    for line in (DATA / "queries.jsonl").read_text().splitlines():
        if line.strip():
            q = json.loads(line); qtext[q["_id"]] = q["text"]
    qrels = defaultdict(set)
    for line in (DATA / "qrels" / "test.tsv").read_text().splitlines()[1:]:
        if not line.strip():
            continue
        qid, cid, score = line.split("\t")
        if int(score) > 0:
            qrels[qid].add(cid)
    doc_id_set = set(doc_ids)
    queries = [(qid, qtext[qid]) for qid in qrels if qid in qtext and (qrels[qid] & doc_id_set)]
    rng = np.random.default_rng(SUB_SEED)
    if CAP_DOCS and len(doc_ids) > CAP_DOCS:
        sel = sorted(rng.choice(len(doc_ids), CAP_DOCS, replace=False))
        sub_ids = [doc_ids[i] for i in sel]; sub_docs = [docs[i] for i in sel]
    else:
        sub_ids, sub_docs = doc_ids, docs
    sub_set = set(sub_ids)
    sub_qrels = {qid: (rel & sub_set) for qid, rel in qrels.items()}
    queries = [(qid, qt) for qid, qt in queries if len(sub_qrels[qid]) >= 3]
    if CAP_QUERIES and len(queries) > CAP_QUERIES:
        idx = sorted(rng.choice(len(queries), CAP_QUERIES, replace=False))
        queries = [queries[i] for i in idx]
    return sub_docs, sub_ids, queries, sub_qrels


def embed(emb, texts, tag, dim=768, batch=8):
    cache = A / f".nf_{tag}.npz"
    if cache.exists():
        m = np.load(cache)["m"]
        if m.shape[0] == len(texts):
            return m
    mm = A / f".nf_{tag}.mm.npy"; prog = A / f".nf_{tag}.prog"; n = len(texts)
    prompt = "document" if tag == "doc" else "query"
    if mm.exists() and prog.exists():
        mat = np.lib.format.open_memmap(mm, mode="r+"); done = int(prog.read_text().strip())
        if mat.shape != (n, dim):
            mat = np.lib.format.open_memmap(mm, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    else:
        mat = np.lib.format.open_memmap(mm, mode="w+", dtype=np.float32, shape=(n, dim)); done = 0
    t0 = time.time()
    for i in range(done, n, batch):
        v = np.asarray(emb.encode(texts[i:i + batch], prompt=prompt), dtype=np.float32)
        mat[i:i + v.shape[0]] = v; mat.flush(); prog.write_text(str(min(i + batch, n)))
        if (i // batch) % 20 == 0:
            print(f"  {tag} {min(i+batch,n)}/{n}", flush=True)
    out = np.array(mat); np.savez(cache, m=out)
    mm.unlink(missing_ok=True); prog.unlink(missing_ok=True)
    print(f"  embedded {tag} {out.shape} in {time.time()-t0:.0f}s", flush=True)
    return out


def recall_at(ranked_ids, rel, ks):
    return {k: len(set(ranked_ids[:k]) & rel) / len(rel) for k in ks}


def agg_recall(order_fn, doc_ids, queries, qrels, ks):
    agg = {k: 0.0 for k in ks}
    for j, (qid, _) in enumerate(queries):
        idx = order_fn(j)[: max(ks)]
        ranked = [doc_ids[i] for i in idx]
        r = recall_at(ranked, qrels[qid], ks)
        for k in ks:
            agg[k] += r[k]
    return {k: agg[k] / len(queries) for k in ks}


def main():
    from remax_kb.embedders import JinaQ4ONNXEmbedder
    emb = JinaQ4ONNXEmbedder(model_path=A / "model.q4.onnx", tokenizer_path=A / "tokenizer.json", max_length=MAXLEN)
    docs, doc_ids, queries, qrels = load_nfcorpus()
    print(f"NFCorpus: {len(docs)} docs, {len(queries)} queries w/ qrels", flush=True)
    D = embed(emb, docs, "doc")
    Q = embed(emb, [t for _, t in queries], "qry")
    print(f"unit-norm: doc norms mean={np.linalg.norm(D,axis=1).mean():.4f}\n", flush=True)

    rows = []

    def add(label, bits, dim, by, order_fn):
        r = agg_recall(order_fn, doc_ids, queries, qrels, KS)
        rows.append((label, bits, dim, by, r[10], r[100]))
        print(f"  {label:<22} R@10={r[10]:.3f} R@100={r[100]:.3f}", flush=True)

    add("fp32 cosine", "-", 768, 768 * 4, lambda j: np.argsort(-(D @ Q[j])))

    for bits, dim in [(8, 768), (4, 768), (2, 768), (4, 512)]:
        qz = Quantizer(d=dim, bits=bits, seed=SEED)
        comp = qz.encode(np.ascontiguousarray(D[:, :dim]))
        add(f"remex {bits}-bit", bits, dim, dim * bits // 8,
            lambda j, qz=qz, comp=comp, dim=dim: qz.search(comp, Q[j, :dim], k=len(D))[0])

    qz8 = Quantizer(d=768, bits=8, seed=SEED); comp8 = qz8.encode(np.ascontiguousarray(D))
    add("remex 8b 2stage(c4)", 8, 768, 768,
        lambda j: qz8.search_twostage(comp8, Q[j], k=len(D), candidates=200, coarse_precision=4)[0])

    for dim, k in [(256, 8), (512, 4), (768, 2)]:
        mean = D.mean(0).astype(np.float32)
        qz = StackedSignBitQuantizer(d=dim, k=k, seed=SEED)
        dcodes = qz.encode(np.ascontiguousarray((D - mean)[:, :dim]))
        qcodes = qz.encode(np.ascontiguousarray((Q - mean)[:, :dim]))
        add(f"remax 1-bit d{dim}/k{k}", 1, dim, dim * k // 8,
            lambda j, dcodes=dcodes, qcodes=qcodes: top_k(hamming_scan(dcodes, qcodes[j]), len(D)))

    print("\n" + "=" * 70)
    print(f"{'method':<22}{'bits':>5}{'dim':>5}{'B/row':>7}{'R@10':>8}{'R@100':>8}")
    print("-" * 70)
    for lab, b, dim, by, r10, r100 in rows:
        print(f"{lab:<22}{str(b):>5}{dim:>5}{by:>7}{r10:>8.3f}{r100:>8.3f}")
    (A / ".rows_nfcorpus.json").write_text(json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
