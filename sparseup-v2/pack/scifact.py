"""BEIR SciFact nDCG@10 of the quantized encoder variants against the fp32 reference.

    python3 scifact.py encode --variant fp32        # also bf16, int8, q4 (dequantized pack weights)
    python3 scifact.py score                        # results/scifact.json, paired bootstrap vs fp32

Documents are title + " " + text, kind "document"; the 300 test queries, kind "query"; max_length 512.
Encodings checkpoint to cache/scifact_<variant>.npz every 256 docs and resume.
Written by gemini-4-hydrogen from a spec (DELEGATION.md #2).
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pack import dequant_state_dict
from ref import Reference

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
DATA_DIR = CACHE_DIR / "scifact"
RESULTS_DIR = HERE / "results"


def load_data():
    corpus_df = pd.read_parquet(DATA_DIR / "corpus" / "corpus-00000-of-00001.parquet")
    queries_df = pd.read_parquet(DATA_DIR / "queries" / "queries-00000-of-00001.parquet")
    qrels_df = pd.read_csv(DATA_DIR / "test.tsv", sep="\t", dtype={"query-id": str, "corpus-id": str})

    qrels: dict[str, dict[str, float]] = {}
    for row in qrels_df.itertuples(index=False):
        qid, did, score = str(row[0]), str(row[1]), float(row[2])
        if score > 0:
            qrels.setdefault(qid, {})[did] = score

    queries_df["_id"] = queries_df["_id"].astype(str)
    queries_df = queries_df[queries_df["_id"].isin(qrels)].reset_index(drop=True)
    query_ids = queries_df["_id"].tolist()
    query_texts = queries_df["text"].fillna("").astype(str).tolist()

    corpus_df["_id"] = corpus_df["_id"].astype(str)
    doc_ids = corpus_df["_id"].tolist()
    titles = corpus_df["title"].fillna("").astype(str)
    texts = corpus_df["text"].fillna("").astype(str)
    doc_texts = (titles + " " + texts).str.strip().tolist()

    return query_ids, query_texts, doc_ids, doc_texts, qrels


def _csr_from_vecs(vecs: list[tuple[np.ndarray, np.ndarray]], vocab_size: int):
    indptr = np.zeros(len(vecs) + 1, dtype=np.int64)
    if vecs:
        lengths = np.fromiter((len(idx) for idx, _ in vecs), dtype=np.int64, count=len(vecs))
        np.cumsum(lengths, out=indptr[1:])
        indices = np.concatenate([idx for idx, _ in vecs]) if indptr[-1] > 0 else np.empty(0, dtype=np.int32)
        data = np.concatenate([val for _, val in vecs]) if indptr[-1] > 0 else np.empty(0, dtype=np.float32)
    else:
        indices = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=np.float32)
    shape = np.array([len(vecs), vocab_size], dtype=np.int64)
    return data, indices, indptr, shape


def _vecs_from_csr(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    vecs = []
    for i in range(len(indptr) - 1):
        s, e = int(indptr[i]), int(indptr[i + 1])
        vecs.append((indices[s:e], data[s:e]))
    return vecs


def _save_checkpoint(out_path: Path, query_ids: list[str], q_vecs: list[tuple[np.ndarray, np.ndarray]],
                     doc_ids: list[str], d_vecs: list[tuple[np.ndarray, np.ndarray]], vocab_size: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q_data, q_indices, q_indptr, q_shape = _csr_from_vecs(q_vecs, vocab_size)
    d_data, d_indices, d_indptr, d_shape = _csr_from_vecs(d_vecs, vocab_size)
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez(tmp_path, query_ids=np.array(query_ids, dtype=str), doc_ids=np.array(doc_ids, dtype=str),
             q_data=q_data, q_indices=q_indices, q_indptr=q_indptr, q_shape=q_shape,
             d_data=d_data, d_indices=d_indices, d_indptr=d_indptr, d_shape=d_shape)
    os.replace(tmp_path, out_path)


def _encode_one(ref: Reference, text: str, kind: str) -> tuple[np.ndarray, np.ndarray]:
    ids, pool = ref.tokenize(text, kind, max_length=512)
    vec = ref.encode_ids(ids, pool).cpu().numpy()
    nz = np.flatnonzero(vec).astype(np.int32)
    vals = vec[nz].astype(np.float32)
    return nz, vals


def encode_variant(variant: str, threads: int, limit: int | None, device: str = "cpu"):
    query_ids, query_texts, doc_ids, doc_texts, _ = load_data()
    if limit is not None:
        doc_ids = doc_ids[:limit]
        doc_texts = doc_texts[:limit]
        out_path = CACHE_DIR / f"scifact_{variant}_limit{limit}.npz"
    else:
        out_path = CACHE_DIR / f"scifact_{variant}.npz"

    q_vecs: list[tuple[np.ndarray, np.ndarray]] = []
    d_vecs: list[tuple[np.ndarray, np.ndarray]] = []
    vocab_size = 50370

    if out_path.exists():
        with np.load(out_path, allow_pickle=False) as ckpt:
            q_vecs = _vecs_from_csr(ckpt["q_data"], ckpt["q_indices"], ckpt["q_indptr"])
            d_vecs = _vecs_from_csr(ckpt["d_data"], ckpt["d_indices"], ckpt["d_indptr"])
            vocab_size = int(ckpt["q_shape"][1])
        print(f"[{variant}] Resuming from {len(d_vecs)}/{len(doc_ids)} docs")

    if len(q_vecs) == len(query_ids) and len(d_vecs) == len(doc_ids):
        print(f"[{variant}] Already complete ({len(doc_ids)} docs)")
        return

    ref = Reference(HERE / "hf", threads=threads, device=device)
    if variant in ("int8", "q4"):
        ref.load_weights(dequant_state_dict(HERE / "packs" / variant))
    elif variant == "bf16":
        ref.model.to(torch.bfloat16)

    with torch.inference_mode():
        if len(q_vecs) < len(query_ids):
            t0 = time.perf_counter()
            for text in query_texts[len(q_vecs):]:
                q_vecs.append(_encode_one(ref, text, "query"))
            print(f"[{variant}] Encoded {len(q_vecs)} queries in {time.perf_counter() - t0:.1f}s", flush=True)
            _save_checkpoint(out_path, query_ids, q_vecs, doc_ids[: len(d_vecs)], d_vecs, vocab_size)

        start_idx = len(d_vecs)
        total_docs = len(doc_ids)
        t0 = time.perf_counter()
        for i in range(start_idx, total_docs):
            d_vecs.append(_encode_one(ref, doc_texts[i], "document"))
            done = i + 1
            if done % 256 == 0 or done == total_docs:
                _save_checkpoint(out_path, query_ids, q_vecs, doc_ids[:done], d_vecs, vocab_size)
                elapsed = time.perf_counter() - t0
                rate = (done - start_idx) / max(elapsed, 1e-6)
                eta = (total_docs - done) / max(rate, 1e-6)
                print(f"[{variant}] {done}/{total_docs} docs ({rate:.2f} docs/s, ETA {eta:.0f}s)", flush=True)


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        query_ids = z["query_ids"].tolist()
        doc_ids = z["doc_ids"].tolist()
        q_mat = sp.csr_matrix((z["q_data"], z["q_indices"], z["q_indptr"]), shape=tuple(z["q_shape"]))
        d_mat = sp.csr_matrix((z["d_data"], z["d_indices"], z["d_indptr"]), shape=tuple(z["d_shape"]))
    return query_ids, doc_ids, q_mat, d_mat


def ndcg_at_10(ranked_doc_ids: list[str], rels: dict[str, float]) -> float:
    if not rels:
        return 0.0
    dcg = 0.0
    for rank, did in enumerate(ranked_doc_ids[:10], start=1):
        rel = rels.get(did, 0.0)
        if rel > 0.0:
            dcg += rel / np.log2(rank + 1)
    ideal_rels = sorted(rels.values(), reverse=True)[:10]
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    return dcg / idcg if idcg > 0.0 else 0.0


def bootstrap(diff: np.ndarray, n_boot: int) -> tuple[float, list[float], float]:
    rng = np.random.default_rng(0)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    frac_changed = float(np.mean(diff != 0.0))
    return float(diff.mean()), [float(lo), float(hi)], frac_changed


def score_variants(variants: list[str], n_boot: int):
    _, _, corpus_doc_ids, _, qrels = load_data()
    n_expected_docs = len(corpus_doc_ids)

    variant_results = {}
    per_query_ndcg: dict[str, np.ndarray] = {}
    per_query_top10: dict[str, list[set[str]]] = {}
    n_queries = 0

    for variant in variants:
        npz_path = CACHE_DIR / f"scifact_{variant}.npz"
        if not npz_path.exists():
            continue
        query_ids, doc_ids, q_mat, d_mat = load_npz(npz_path)
        if len(doc_ids) != n_expected_docs:
            continue

        n_queries = len(query_ids)
        scores = (q_mat @ d_mat.T).toarray()
        doc_id_arr = np.array(doc_ids)
        doc_to_idx = {did: idx for idx, did in enumerate(doc_ids)}

        ndcgs = np.zeros(n_queries, dtype=np.float64)
        recalls = np.zeros(n_queries, dtype=np.float64)
        top10_sets: list[set[str]] = []

        for qi, qid in enumerate(query_ids):
            row = scores[qi].copy()
            if qid in doc_to_idx:
                row[doc_to_idx[qid]] = -np.inf
            order = np.argsort(-row, kind="mergesort")
            top100_ids = doc_id_arr[order[:100]].tolist()
            top10_ids = top100_ids[:10]
            top10_sets.append(set(top10_ids))

            rels = qrels[qid]
            ndcgs[qi] = ndcg_at_10(top10_ids, rels)
            hits = sum(1 for did in top100_ids if did in rels)
            recalls[qi] = hits / len(rels) if rels else 0.0

        variant_results[variant] = {
            "ndcg10": round(float(ndcgs.mean()), 5),
            "recall100": round(float(recalls.mean()), 5),
            "nnz_query": round(float(np.mean(np.diff(q_mat.indptr))), 5),
            "nnz_doc": round(float(np.mean(np.diff(d_mat.indptr))), 5),
        }
        per_query_ndcg[variant] = ndcgs
        per_query_top10[variant] = top10_sets

    vs_fp32 = {}
    if "fp32" in per_query_ndcg:
        fp32_ndcg = per_query_ndcg["fp32"]
        fp32_top10 = per_query_top10["fp32"]
        for variant, ndcgs in per_query_ndcg.items():
            if variant == "fp32":
                continue
            mean_delta, ci95, frac_changed = bootstrap(ndcgs - fp32_ndcg, n_boot)
            overlap = float(np.mean([len(a & b) / 10.0 for a, b in zip(per_query_top10[variant], fp32_top10)]))
            vs_fp32[variant] = {
                "delta_ndcg10": round(mean_delta, 5),
                "ci95": [round(ci95[0], 5), round(ci95[1], 5)],
                "frac_queries_changed": round(frac_changed, 5),
                "top10_overlap": round(overlap, 5),
            }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"dataset": "BeIR/scifact test", "n_queries": n_queries, "n_docs": n_expected_docs, "max_length": 512,
           "variants": variant_results, "vs_fp32": vs_fp32}
    (RESULTS_DIR / "scifact.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"{'variant':<8} {'nDCG@10':>8} {'R@100':>8} {'nnz_q':>8} {'nnz_d':>8} {'delta':>9} {'95% CI':>19} {'ovlp@10':>8}")
    for variant, m in variant_results.items():
        if variant in vs_fp32:
            v = vs_fp32[variant]
            ci_str = f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"
            delta_str = f"{v['delta_ndcg10']:+.5f}"
            ovlp_str = f"{v['top10_overlap']:.5f}"
        else:
            ci_str, delta_str, ovlp_str = "-", "-", "-"
        print(f"{variant:<8} {m['ndcg10']:>8.5f} {m['recall100']:>8.5f} "
              f"{m['nnz_query']:>8.1f} {m['nnz_doc']:>8.1f} {delta_str:>9} {ci_str:>19} {ovlp_str:>8}")


def main():
    parser = argparse.ArgumentParser(description="BEIR SciFact SPLADE retrieval check")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser("encode", help="Encode queries and corpus docs")
    p_enc.add_argument("--variant", required=True, choices=["fp32", "bf16", "int8", "q4"])
    p_enc.add_argument("--threads", type=int, default=4)
    p_enc.add_argument("--limit", type=int, default=None)
    p_enc.add_argument("--device", default="cpu")

    p_score = sub.add_parser("score", help="Score encoded variants")
    p_score.add_argument("--variants", default="fp32,bf16,int8,q4")
    p_score.add_argument("--boot", type=int, default=10000)

    args = parser.parse_args()
    if args.cmd == "encode":
        encode_variant(args.variant, args.threads, args.limit, args.device)
    elif args.cmd == "score":
        score_variants([v.strip() for v in args.variants.split(",") if v.strip()], args.boot)


if __name__ == "__main__":
    main()
