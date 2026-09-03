"""Pass 3: can a token-embedding table FIT BY REGRESSION against the teacher's
own sentence vectors (rather than distilled token-by-token from the teacher's
weights, as in pass 2) produce a student compatible with the teacher's space?

X = per-chunk bag-of-bekko-tokens, row-normalized by total token count (so
X @ W reproduces exactly the mean-pool a StaticModel-style encoder would do:
sum of token rows for present tokens, divided by the FULL token count,
i.e. missing/dropped tokens contribute zero and are still counted in the
denominator). Y = teacher (bekko-a8m fp32 ONNX) sentence vectors.
W = argmin_W ||XW - Y||^2 + alpha||W||^2, fit per output column via
sklearn.linear_model.Ridge(solver="sparse_cg", fit_intercept=False) so W IS
literally a (n_fitted_tokens, 384) token-embedding table — no separate bias
to smuggle in at encode time.

Split: files (not chunks) 80/20, seed 0, so no chunk from a held-out file
leaks into train. Vocab restricted to tokens with train-split frequency >= 2
(the full 256k bekko vocab makes an un-restricted ridge fit too slow for a
single pass).

Two variants:
  A. fit on the 80% train chunks only.
  B. fit on train chunks + up to 2000 synthetic NL-ish texts extracted as
     `ast.get_docstring` docstrings from TRAIN-split ast chunks only (their
     teacher vectors are freshly computed by a real teacher forward pass —
     nothing about the 59-query test set is touched).
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from tokenizers import Tokenizer

BEKKO_DIR = Path("/home/user/experiments/bekko-embedding-bench")
sys.path.insert(0, str(BEKKO_DIR / "scripts"))
sys.path.insert(0, os.environ.get("REMEX_ROOT", "/usr/local/lib/python3.11/dist-packages"))
sys.path.insert(0, os.environ.get("REMAX_ROOT", "/home/user/oaustegard/remax/src"))
from bekko import BekkoEncoder  # noqa: E402
from eval_search import extract_identifiers, arm_rg, recall_at, rrf  # noqa: E402
from run_code_quant import file_rank, hamming_rank  # noqa: E402

import remex  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

HERE = Path(__file__).resolve().parent
TOK_PATH = "/home/user/models/bekko-a8m/tokenizer.json"
MAX_LEN = 512
ALPHAS = (0.1, 1.0, 10.0)
MIN_FREQ = 2
RNG = np.random.default_rng(0)

failed: list[str] = []


def log(m):
    print(m, flush=True)


def tokenize_ids(tok: Tokenizer, text: str) -> list[int]:
    ids = tok.encode(text, add_special_tokens=False).ids
    return ids[:MAX_LEN]


def build_rows(token_lists: list[list[int]], tok2col: dict[int, int]) -> sp.csr_matrix:
    """Row i = count(token)/L_i for kept tokens; L_i = full token count (incl. dropped)."""
    indptr = [0]
    indices = []
    data = []
    for ids in token_lists:
        L = max(1, len(ids))
        counts: dict[int, int] = {}
        for t in ids:
            j = tok2col.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in sorted(counts.items()):
            indices.append(j)
            data.append(c / L)
        indptr.append(len(indices))
    n_cols = len(tok2col)
    return sp.csr_matrix((data, indices, indptr), shape=(len(token_lists), n_cols), dtype=np.float32)


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def fit_and_pick_alpha(X_train, Y_train, X_val, Y_val, alphas=ALPHAS):
    best = None
    for alpha in alphas:
        t0 = time.time()
        ridge = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg")
        ridge.fit(X_train, Y_train)
        W = ridge.coef_.T.astype(np.float32)  # (n_features, 384)
        pred = np.asarray(X_val @ W)
        pred = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-9, None)
        c = cos_rows(pred, Y_val / np.clip(np.linalg.norm(Y_val, axis=1, keepdims=True), 1e-9, None))
        log(f"    alpha={alpha:<5} fit {time.time() - t0:.1f}s  held-out cosine mean={c.mean():.4f}")
        if best is None or c.mean() > best[1]:
            best = (alpha, c.mean(), W)
    return best  # (alpha, mean_cosine, W)


def student_encode(texts: list[str], tok: Tokenizer, tok2col: dict[int, int], W: np.ndarray) -> np.ndarray:
    n_cols, dim = W.shape
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        ids = tokenize_ids(tok, t)
        L = max(1, len(ids))
        acc = np.zeros(dim, dtype=np.float32)
        for tid in ids:
            j = tok2col.get(tid)
            if j is not None:
                acc += W[j]
        out[i] = acc / L
    norm = np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)
    return out / norm


def extract_docstrings(chunks: list[dict], idxs: list[int], cap: int) -> list[str]:
    out = []
    for i in idxs:
        if len(out) >= cap:
            break
        text = chunks[i]["text"]
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ds = ast.get_docstring(node)
                if ds and len(ds.strip()) >= 20:
                    out.append(ds.strip())
                    break
    return out[:cap]


def run_variant(name, X_train, Y_train, X_val, Y_val, tok2col, tok, chunks, all_idx,
                 queries, teacher_mat, teacher_qv, rg_ranked, inst):
    log(f"\n=== variant {name}: fitting ridge (train n={X_train.shape[0]}, "
        f"vocab={X_train.shape[1]}) ===")
    alpha, val_cos, W = fit_and_pick_alpha(X_train, Y_train, X_val, Y_val)
    log(f"  chosen alpha={alpha}  held-out cosine mean={val_cos:.4f}")

    # held-out chunk cosines (full stats, not just mean)
    pred_val = np.asarray(X_val @ W)
    pred_val_n = pred_val / np.clip(np.linalg.norm(pred_val, axis=1, keepdims=True), 1e-9, None)
    Y_val_n = Y_val / np.clip(np.linalg.norm(Y_val, axis=1, keepdims=True), 1e-9, None)
    c_chunks = cos_rows(pred_val_n, Y_val_n)
    log(f"  held-out chunk cosine: mean {c_chunks.mean():.4f} median {np.median(c_chunks):.4f} "
        f"min {c_chunks.min():.4f}")

    # query cosines (different distribution: NL bug reports vs code chunks)
    student_qv = student_encode(queries, tok, tok2col, W)
    c_query = cos_rows(student_qv, teacher_qv / np.clip(np.linalg.norm(teacher_qv, axis=1, keepdims=True), 1e-9, None))
    log(f"  query cosine (NL bug reports, different distribution from code chunks): "
        f"mean {c_query.mean():.4f} median {np.median(c_query):.4f} min {c_query.min():.4f}")

    # full-corpus student encoding for retrieval cells
    student_mat = student_encode([chunks[i]["text"] for i in all_idx], tok, tok2col, W)

    def score(ranker):
        r5 = np.zeros(len(inst)); r10 = np.zeros(len(inst)); f10 = np.zeros(len(inst))
        for i, it in enumerate(inst):
            ranked = ranker(i)
            r5[i] = recall_at(ranked, it["gold"], 5)
            r10[i] = recall_at(ranked, it["gold"], 10)
            f10[i] = recall_at(rrf(rg_ranked[it["issue"]], ranked), it["gold"], 10)
        return r5, r10, f10

    cells = {}
    r5, r10, f10 = score(lambda i: file_rank(student_mat @ student_qv[i], chunks))
    cells["student/student"] = {"r@5": float(r5.mean()), "r@10": float(r10.mean()), "rrf_r@10": float(f10.mean())}
    log(f"  student/student            r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}  RRF r@10 {f10.mean():.3f}")

    r5, r10, f10 = score(lambda i: file_rank(teacher_mat @ student_qv[i], chunks))
    cells["student-query/teacher-index"] = {"r@5": float(r5.mean()), "r@10": float(r10.mean()), "rrf_r@10": float(f10.mean())}
    log(f"  student-query/TEACHER-index r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}  RRF r@10 {f10.mean():.3f}")

    # same cell with the teacher index quantized (remex 2-bit / remax k=1), as in pass 1
    qz = remex.Quantizer(d=384, bits=2, seed=0)
    xh = qz.decode(qz.encode(teacher_mat))
    xh = xh / np.clip(np.linalg.norm(xh, axis=1, keepdims=True), 1e-9, None)
    r5, r10, f10 = score(lambda i: file_rank(xh @ student_qv[i], chunks))
    cells["student-query/teacher-index-remex2bit"] = {"r@5": float(r5.mean()), "r@10": float(r10.mean()), "rrf_r@10": float(f10.mean())}
    log(f"  student-query/teacher-remex2bit r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}  RRF r@10 {f10.mean():.3f}")

    sq = StackedSignBitQuantizer(d=384, k=1, seed=0).fit(teacher_mat)
    dc = sq.encode(teacher_mat)
    qc = sq.encode(student_qv)
    r5, r10, f10 = score(lambda i, qc=qc, dc=dc: hamming_rank(qc[i], dc, chunks))
    cells["student-query/teacher-index-remax_k1"] = {"r@5": float(r5.mean()), "r@10": float(r10.mean()), "rrf_r@10": float(f10.mean())}
    log(f"  student-query/teacher-remax_k1  r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}  RRF r@10 {f10.mean():.3f}")

    table_bytes = int(W.shape[0] * W.shape[1] * 2)  # fp16
    log(f"  fitted table: {W.shape[0]} tokens x {W.shape[1]} dim, fp16 = {table_bytes / 2**20:.2f} MB")

    return {
        "variant": name, "alpha": alpha, "n_fitted_tokens": int(W.shape[0]),
        "table_bytes_fp16": table_bytes,
        "held_out_chunk_cosine": {"mean": float(c_chunks.mean()), "median": float(np.median(c_chunks)),
                                   "min": float(c_chunks.min())},
        "query_cosine": {"mean": float(c_query.mean()), "median": float(np.median(c_query)),
                          "min": float(c_query.min())},
        "cells": cells,
    }


def main() -> None:
    inst = json.load(open(BEKKO_DIR / "instances.json"))
    chunks = json.load(open(BEKKO_DIR / "chunks_ast.json"))
    n_chunks = len(chunks)
    teacher_mat = np.asarray(np.memmap(BEKKO_DIR / "vecs_ast_a8m.f32", dtype=np.float32,
                                        mode="r", shape=(n_chunks, 384)))
    queries = [it["title"] + "\n" + it["body"] for it in inst]

    tok = Tokenizer.from_file(TOK_PATH)
    log(f"tokenizer vocab size: {tok.get_vocab_size()}")

    # ---- file-level 80/20 split, seed 0 ---------------------------------------
    files = sorted({c["file"] for c in chunks})
    perm = RNG.permutation(len(files))
    n_train_files = int(round(0.8 * len(files)))
    train_files = {files[i] for i in perm[:n_train_files]}
    train_idx = [i for i, c in enumerate(chunks) if c["file"] in train_files]
    test_idx = [i for i, c in enumerate(chunks) if c["file"] not in train_files]
    log(f"file split: {len(train_files)}/{len(files)} files train, "
        f"{len(train_idx)} train chunks / {len(test_idx)} held-out chunks")

    # ---- tokenize all train+test chunks once ----------------------------------
    t0 = time.time()
    train_ids = [tokenize_ids(tok, chunks[i]["text"]) for i in train_idx]
    test_ids = [tokenize_ids(tok, chunks[i]["text"]) for i in test_idx]
    log(f"tokenized {len(train_ids) + len(test_ids)} chunks in {time.time() - t0:.1f}s")

    # ---- vocab: train-frequency >= MIN_FREQ ------------------------------------
    freq: dict[int, int] = {}
    for ids in train_ids:
        for t in ids:
            freq[t] = freq.get(t, 0) + 1
    kept = sorted(t for t, c in freq.items() if c >= MIN_FREQ)
    tok2col = {t: j for j, t in enumerate(kept)}
    log(f"vocab: {len(freq)} distinct train tokens, {len(kept)} kept at freq>={MIN_FREQ}")

    X_train = build_rows(train_ids, tok2col)
    X_test = build_rows(test_ids, tok2col)
    Y_train = teacher_mat[train_idx]
    Y_test = teacher_mat[test_idx]

    # ---- rg baseline + teacher query vectors (shared across variants) ---------
    rg_ranked = {}
    for it in inst:
        r, _, _ = arm_rg(extract_identifiers(it["title"] + "\n" + it["body"]))
        rg_ranked[it["issue"]] = r
    teacher_enc = BekkoEncoder("a8m", threads=4)
    teacher_qv = teacher_enc.encode(queries, batch_size=8)

    def score_all(ranker):
        r5 = np.zeros(len(inst)); r10 = np.zeros(len(inst)); f10 = np.zeros(len(inst))
        for i, it in enumerate(inst):
            ranked = ranker(i)
            r5[i] = recall_at(ranked, it["gold"], 5)
            r10[i] = recall_at(ranked, it["gold"], 10)
            f10[i] = recall_at(rrf(rg_ranked[it["issue"]], ranked), it["gold"], 10)
        return r5, r10, f10

    log("\n=== reference: teacher/teacher ===")
    r5, r10, f10 = score_all(lambda i: file_rank(teacher_mat @ teacher_qv[i], chunks))
    log(f"  teacher/teacher  r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}  RRF r@10 {f10.mean():.3f}")
    teacher_ref = {"r@5": float(r5.mean()), "r@10": float(r10.mean()), "rrf_r@10": float(f10.mean())}

    all_idx = list(range(n_chunks))

    results = {"teacher_reference": teacher_ref, "vocab": {"train_distinct": len(freq),
               "kept_at_min_freq": len(kept), "min_freq": MIN_FREQ}, "variants": {}, "failed": failed}

    try:
        results["variants"]["A_chunks_only"] = run_variant(
            "A_chunks_only", X_train, Y_train, X_test, Y_test, tok2col, tok, chunks, all_idx,
            queries, teacher_mat, teacher_qv, rg_ranked, inst)
    except Exception as e:
        failed.append(f"variant A: {e!r}")
        log(f"FAILED variant A: {e!r}")

    # ---- variant B: augment with docstring texts from TRAIN chunks only -------
    log("\n=== extracting docstrings from TRAIN-split ast chunks ===")
    docstrings = extract_docstrings(chunks, train_idx, cap=2000)
    log(f"  extracted {len(docstrings)} docstrings (cap 2000)")

    if docstrings:
        try:
            t0 = time.time()
            y_aug = teacher_enc.encode(docstrings, batch_size=8)
            log(f"  encoded {len(docstrings)} docstrings with the teacher in {time.time() - t0:.1f}s")

            aug_ids = [tokenize_ids(tok, d) for d in docstrings]
            # extend vocab with docstring-only tokens meeting the same freq bar
            # over the combined (train chunks + docstrings) token pool
            freq_b = dict(freq)
            for ids in aug_ids:
                for t in ids:
                    freq_b[t] = freq_b.get(t, 0) + 1
            kept_b = sorted(t for t, c in freq_b.items() if c >= MIN_FREQ)
            tok2col_b = {t: j for j, t in enumerate(kept_b)}
            log(f"  variant B vocab: {len(kept_b)} kept at freq>={MIN_FREQ} "
                f"(vs {len(kept)} in variant A)")

            X_train_b = build_rows(train_ids + aug_ids, tok2col_b)
            Y_train_b = np.vstack([Y_train, y_aug])
            X_test_b = build_rows(test_ids, tok2col_b)

            results["variants"]["B_plus_docstrings"] = run_variant(
                "B_plus_docstrings", X_train_b, Y_train_b, X_test_b, Y_test, tok2col_b, tok,
                chunks, all_idx, queries, teacher_mat, teacher_qv, rg_ranked, inst)
            results["variants"]["B_plus_docstrings"]["n_docstrings_added"] = len(docstrings)
        except Exception as e:
            failed.append(f"variant B: {e!r}")
            log(f"FAILED variant B: {e!r}")
    else:
        failed.append("variant B: no docstrings extracted from train split")

    json.dump(results, open(HERE / "results_fit_table.json", "w"), indent=1, default=float)
    log(f"\nwrote {HERE / 'results_fit_table.json'}")
    if failed:
        log("\nFAILURES:")
        for f in failed:
            log(f"  - {f}")


if __name__ == "__main__":
    main()
