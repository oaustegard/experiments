#!/usr/bin/env python3
"""Bench: remex / remax / native levers on NeoMME-260M-Retriever, BEIR SciFact.

Reads data/scifact_enc (encode.py) or any --data dir carrying its own qrels.tsv (encode_vidore.py). Writes results.json (one record per arm,
resumable: arms already present are skipped) and perquery.npz (per-query
nDCG@10 for paired tests).

Arms
  dense:  fp32 x MRL dims; remex {4,2,1}-bit x MRL dims; remax k=1 (asym|sym),
          k=2 asym; Sentence Transformers' own quantize_embeddings int8 / binary.
  late:   fp32 x pool{1,2,4}; remex {4,2,1}-bit; remax k=1 (asym|sym), k=2;
          compositions pool2 + {remex 2-bit, remex 1-bit, remax k=1}.
  pipeline: dense top-100 candidates -> late-interaction rerank, each side
          fp32 or quantized.

Metrics vs qrels: nDCG@10, R@10, R@100. Fidelity: overlap@10 with the fp32
full-width ranking of the same head. Bytes per document as stored.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from neomme_quant import DenseIndex, MultiVectorIndex, MRL_DIMS, ndcg_at_k, recall_at_k, overlap_at_k, l2n  # noqa: E402

K_CAND = 100


def load(data: Path):
    dense = np.load(data / "dense.npy"); toks = np.load(data / "mv_tokens.npy"); off = np.load(data / "mv_offsets.npy")
    qd = np.load(data / "q_dense.npy"); qt = np.load(data / "q_mv_tokens.npy"); qoff = np.load(data / "q_mv_offsets.npy")
    doc_ids = json.loads((data / "doc_ids.json").read_text()); qids = json.loads((data / "query_ids.json").read_text())
    qrels = defaultdict(set)
    qrels_path = data / "qrels.tsv" if (data / "qrels.tsv").exists() else HERE / "data" / "scifact" / "qrels" / "test.tsv"
    for _, r in pd.read_csv(qrels_path, sep="\t").iterrows():
        if int(r["score"]) > 0:
            qrels[str(r["query-id"])].add(str(r["corpus-id"]))
    keep = [i for i, q in enumerate(qids) if qrels[q] & set(doc_ids)]
    return dense, toks, off, qd, qt, qoff, doc_ids, [qids[i] for i in keep], keep, qrels


def evaluate(rank_fn, qids, keep, qrels, doc_ids, ref_top=None):
    """rank_fn(i_query) -> array of doc indices, best first (>= K_CAND)."""
    nd, r10, r100, fid = [], [], [], []
    tops = []
    for qi, qid in zip(keep, qids):
        order = rank_fn(qi)
        ranked = [doc_ids[j] for j in order[:K_CAND]]
        nd.append(ndcg_at_k(ranked, qrels[qid], 10)); r10.append(recall_at_k(ranked, qrels[qid], 10)); r100.append(recall_at_k(ranked, qrels[qid], 100))
        tops.append(order[:10])
        if ref_top is not None:
            fid.append(overlap_at_k(order, ref_top[len(tops) - 1], 10))
    return dict(ndcg10=float(np.mean(nd)), r10=float(np.mean(r10)), r100=float(np.mean(r100)),
                fid10=float(np.mean(fid)) if fid else None), np.array(nd), tops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data" / "scifact_enc"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--only", default="", help="comma list of groups: dense,late,pipeline")
    args = ap.parse_args()
    data = Path(args.data); out = Path(args.out); pq_path = out.with_name(out.stem + "_perquery.npz")
    groups = set(args.only.split(",")) if args.only else {"dense", "late", "pipeline"}
    results = json.loads(out.read_text()) if out.exists() else {}
    perquery = dict(np.load(pq_path)) if pq_path.exists() else {}
    dense, toks, off, qd, qt, qoff, doc_ids, qids, keep, qrels = load(data)
    n_docs = len(doc_ids); tok_per_doc = off[-1] / n_docs
    print(f"{n_docs} docs, {off[-1]} tokens ({tok_per_doc:.0f}/doc), {len(qids)} judged queries with a relevant doc in corpus", flush=True)
    qtok = lambda i: qt[qoff[i]:qoff[i + 1]].astype(np.float32)

    def record(name, group, metrics, nd, extra):
        results[name] = dict(group=group, **metrics, **extra); perquery[name] = nd
        out.write_text(json.dumps(results, indent=1)); np.savez(pq_path, **perquery)
        print(f"  {name:38s} nDCG@10 {metrics['ndcg10']:.4f}  R@10 {metrics['r10']:.3f}  R@100 {metrics['r100']:.3f}  fid@10 {metrics['fid10'] if metrics['fid10'] is None else round(metrics['fid10'],3)}  {extra.get('bytes_per_doc', 0):>10.1f} B/doc  {extra.get('sec', 0):.0f}s", flush=True)

    # ---------------- dense
    ref_dense = None
    if "dense" in groups:
        arms = [DenseIndex("fp32", dim=d) for d in reversed(MRL_DIMS)]
        arms += [DenseIndex("remex", dim=d, bits=b) for b in (4, 2, 1) for d in reversed(MRL_DIMS)]
        arms += [DenseIndex("remax", dim=d, k=1, query_mode=m) for m in ("asym", "sym") for d in reversed(MRL_DIMS)]
        arms += [DenseIndex("remax", dim=d, k=2) for d in (1024, 512)]
        for a in arms:
            t = time.time(); a.build(dense); S = a.scores(qd)
            m, nd, tops = evaluate(lambda i: np.argsort(-S[i], kind="stable"), qids, keep, qrels, doc_ids, ref_dense)
            if ref_dense is None:
                ref_dense = tops
            record("dense " + a.name, "dense", m, nd, dict(bytes_per_doc=a.bytes_per_vec, sec=time.time() - t))
        # Sentence Transformers' own quantizers on the full-width dense head
        from sentence_transformers.util.quantization import quantize_embeddings
        t = time.time(); i8 = quantize_embeddings(dense, precision="int8", calibration_embeddings=dense)
        # ST's int8 rule: per-dim min/max of the calibration set, 255 steps, offset -128. Dequantise with the same ranges.
        ranges = np.vstack((np.min(dense, axis=0), np.max(dense, axis=0))); steps = (ranges[1] - ranges[0]) / 255
        deq = (i8.astype(np.float32) + 128) * steps + ranges[0]
        if True:
            S = qd @ deq.T
            m, nd, tops = evaluate(lambda i: np.argsort(-S[i], kind="stable"), qids, keep, qrels, doc_ids, ref_dense)
            record("dense ST int8 d=1024", "dense", m, nd, dict(bytes_per_doc=1024.0, sec=time.time() - t))
        for mode in ("asym", "sym"):
            t = time.time(); sb = np.where(dense > 0, 1.0, -1.0).astype(np.float32)        # ST binary = sign(x), no rotation, no centering
            Q = qd if mode == "asym" else np.where(qd > 0, 1.0, -1.0).astype(np.float32)
            S = Q @ sb.T
            m, nd, tops = evaluate(lambda i: np.argsort(-S[i], kind="stable"), qids, keep, qrels, doc_ids, ref_dense)
            record(f"dense ST binary d=1024 ({mode})", "dense", m, nd, dict(bytes_per_doc=128.0, sec=time.time() - t))

    # ---------------- late interaction
    ref_late = None
    pooled_cache = {}
    def pooled(pf):
        if pf not in pooled_cache:
            cp = data / f"pooled{pf}.npz"
            if cp.exists():
                z = np.load(cp); pooled_cache[pf] = (z["toks"], z["off"])
            else:
                from neomme_quant import pool_tokens
                t = time.time(); pt, po = pool_tokens(toks.astype(np.float32), off, pf); np.savez(cp, toks=pt.astype(np.float16), off=po)
                print(f"  pooled pool_factor={pf}: {off[-1]} -> {po[-1]} tokens in {time.time()-t:.0f}s", flush=True)
                pooled_cache[pf] = (pt, po)
        return pooled_cache[pf]

    def run_late(a: MultiVectorIndex, pf: int):
        nonlocal ref_late
        name = a.name
        if name in results:
            return
        t = time.time()
        src_t, src_o = (toks, off) if pf == 1 else pooled(pf)
        a.pool_factor = 1                      # pooling already applied to the cached tokens
        a.build(src_t, src_o)
        S = np.stack([a.scores(qtok(i)) for i in keep])
        m, nd, tops = evaluate(lambda i: np.argsort(-S[keep.index(i)], kind="stable"), qids, keep, qrels, doc_ids, ref_late)
        if ref_late is None:
            ref_late = tops
        record(name, "late", m, nd, dict(bytes_per_doc=a.bytes_per_doc, tokens_per_doc=a.n_tokens / a.n_docs, sec=time.time() - t))
        del a.T

    if "late" in groups:
        late_arms = [(MultiVectorIndex("fp32", pool_factor=pf), pf) for pf in (1, 2, 4)]
        late_arms += [(MultiVectorIndex("remex", bits=b), 1) for b in (4, 2, 1)]
        late_arms += [(MultiVectorIndex("remax", k=1, query_mode=m), 1) for m in ("asym", "sym")]
        late_arms += [(MultiVectorIndex("remax", k=2), 1)]
        late_arms += [(MultiVectorIndex("remex", bits=2, pool_factor=2), 2), (MultiVectorIndex("remex", bits=1, pool_factor=2), 2), (MultiVectorIndex("remax", k=1, pool_factor=2), 2)]
        for a, pf in late_arms:
            run_late(a, pf)

    # ---------------- pipeline: dense candidates -> late rerank
    if "pipeline" in groups:
        dense_sides = [DenseIndex("fp32", dim=1024), DenseIndex("remex", dim=1024, bits=2), DenseIndex("remex", dim=256, bits=2), DenseIndex("remax", dim=1024, k=1)]
        late_sides = [MultiVectorIndex("fp32"), MultiVectorIndex("remex", bits=2), MultiVectorIndex("remex", bits=1), MultiVectorIndex("remax", k=1)]
        cands = {}
        for ds in dense_sides:
            ds.build(dense); Sd = ds.scores(qd)
            cands[ds.name] = {i: np.argsort(-Sd[i], kind="stable")[:K_CAND] for i in keep}
        for ls in late_sides:                      # one late index resident at a time
            todo = [ds for ds in dense_sides if f"pipeline {ds.name} -> {ls.name} @{K_CAND}" not in results]
            if not todo:
                continue
            L = ls.build(toks, off)
            for ds in todo:
                t = time.time(); c_ = cands[ds.name]
                def rank(i):
                    c = c_[i]; s = L.scores(qtok(i), c); return c[np.argsort(-s, kind="stable")]
                m, nd, _ = evaluate(rank, qids, keep, qrels, doc_ids, None)
                record(f"pipeline {ds.name} -> {ls.name} @{K_CAND}", "pipeline", m, nd, dict(bytes_per_doc=ds.bytes_per_vec + L.bytes_per_doc, dense_bytes=ds.bytes_per_vec, late_bytes=L.bytes_per_doc, sec=time.time() - t))
            del L.T
    print("done ->", out)


if __name__ == "__main__":
    main()
