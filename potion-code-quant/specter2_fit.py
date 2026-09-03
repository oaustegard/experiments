"""Pass 4 — a static table compatible with a remax/remex-quantized SPECTER2 index.

Reuses pass 3's ridge-regression-table method (fit_table.py) but this time
against RAW, unnormalized SPECTER2 vectors (norm mean ~21.7), so the table
learns the teacher's mean and scale directly rather than fitting into an
already-unit-normalized target space.

Protocol mirrors remax/bench/sketch_matryoshka.py: rng(99) picks 100 query
papers out of 10,000; the remaining 9,900 are the corpus. Ground truth is
top-10/top-100 by raw float32 inner product with the teacher query vector
(exactly as the bench computes it) — no L2 normalization anywhere in the
teacher pipeline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from tokenizers import Tokenizer

sys.path.insert(0, "/home/user/oaustegard/remax/src")
import remex  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = Path("/home/user/oaustegard/remax/bench/.cache/SPECTER2")
TOK_PATH = "/home/user/models/specter2_base/tokenizer.json"
MAX_LEN = 512
ALPHAS = (0.1, 1.0, 10.0)
MIN_FREQ = 2
SEED = 99
N_QUERIES = 100
D = 768
RNG = np.random.default_rng(SEED)

failed: list[str] = []


def log(m):
    print(m, flush=True)


def tokenize_ids(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids[:MAX_LEN]


def build_rows(token_lists, tok2col) -> sp.csr_matrix:
    indptr = [0]; indices = []; data = []
    for ids in token_lists:
        L = max(1, len(ids))
        counts = {}
        for t in ids:
            j = tok2col.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in sorted(counts.items()):
            indices.append(j); data.append(c / L)
        indptr.append(len(indices))
    return sp.csr_matrix((data, indices, indptr), shape=(len(token_lists), len(tok2col)), dtype=np.float32)


def cos_rows(a, b):
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def encode_from_tokens(token_lists, tok2col_size_and_map, W):
    """Mean-pool W rows for each row's tokens (raw, no normalization)."""
    tok2col = tok2col_size_and_map
    dim = W.shape[1]
    out = np.zeros((len(token_lists), dim), dtype=np.float32)
    for i, ids in enumerate(token_lists):
        L = max(1, len(ids))
        acc = np.zeros(dim, dtype=np.float32)
        for t in ids:
            j = tok2col.get(t)
            if j is not None:
                acc += W[j]
        out[i] = acc / L
    return out


def recall_at_set(truth10: np.ndarray, pred: np.ndarray) -> float:
    k = truth10.shape[1]
    hits = sum(len(set(truth10[i].tolist()) & set(pred[i].tolist())) for i in range(truth10.shape[0]))
    return hits / (truth10.shape[0] * k)


def topN_by_score(scores: np.ndarray, N: int) -> np.ndarray:
    """scores: (n_queries, n_corpus). Returns (n_queries, N) top indices, sorted desc."""
    nq = scores.shape[0]
    N = min(N, scores.shape[1])
    idx = np.argpartition(-scores, N - 1, axis=1)[:, :N]
    out = np.empty((nq, N), dtype=np.intp)
    for i in range(nq):
        out[i] = idx[i][np.argsort(-scores[i, idx[i]])]
    return out


def hamming_topN(q_codes: np.ndarray, c_codes: np.ndarray, N: int) -> np.ndarray:
    nq = q_codes.shape[0]
    N = min(N, c_codes.shape[0])
    out = np.empty((nq, N), dtype=np.intp)
    for i in range(nq):
        d = np.bitwise_count(np.bitwise_xor(c_codes, q_codes[i])).sum(axis=1)
        idx = np.argpartition(d, N - 1)[:N]
        out[i] = idx[np.argsort(d[idx])]
    return out


def score_cell(truth10, pred):
    r10 = recall_at_set(truth10, pred[:, :10])
    r100 = recall_at_set(truth10, pred[:, :100])
    top1 = float(np.mean(pred[:, 0] == truth10[:, 0]))
    return {"r@10": float(r10), "r@100": float(r100), "top1": top1}


def main() -> None:
    X_emb = np.load(CACHE / "embeddings.npy")
    texts = json.load(open(CACHE / "texts.json"))
    n_total, d = X_emb.shape
    assert d == D
    log(f"SPECTER2: {n_total} vectors, d={d}, norm mean={np.linalg.norm(X_emb, axis=1).mean():.3f}")

    perm = np.random.default_rng(SEED).permutation(n_total)
    q_idx, c_idx = perm[:N_QUERIES], perm[N_QUERIES:]
    teacher_qv = X_emb[q_idx]          # (100, 768) raw
    teacher_corpus = X_emb[c_idx]      # (9900, 768) raw
    n_corpus = teacher_corpus.shape[0]
    log(f"split: {N_QUERIES} query papers, {n_corpus} corpus papers (seed {SEED})")

    truth10 = np.argsort(-(teacher_qv @ teacher_corpus.T), axis=1)[:, :10]

    # ---- tokenize ---------------------------------------------------------
    tok = Tokenizer.from_file(TOK_PATH)
    log(f"tokenizer vocab: {tok.get_vocab_size()}")
    t0 = time.time()
    corpus_texts = [texts[i] for i in c_idx]
    query_texts = [texts[i] for i in q_idx]
    corpus_ids = [tokenize_ids(tok, t) for t in corpus_texts]
    query_ids = [tokenize_ids(tok, t) for t in query_texts]
    log(f"tokenized {len(corpus_ids) + len(query_ids)} texts in {time.time() - t0:.1f}s")

    freq = {}
    for ids in corpus_ids:
        for t in ids:
            freq[t] = freq.get(t, 0) + 1
    kept = sorted(t for t, c in freq.items() if c >= MIN_FREQ)
    tok2col = {t: j for j, t in enumerate(kept)}
    log(f"vocab: {len(freq)} distinct corpus tokens, {len(kept)} kept at freq>={MIN_FREQ}")

    X_corpus = build_rows(corpus_ids, tok2col)
    Y_corpus = teacher_corpus  # RAW, unnormalized

    # ---- 5-fold CV to pick alpha (cosine, on RAW predictions) --------------
    log("\n=== 5-fold CV alpha selection (raw targets) ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {}
    oof_preds = {}  # alpha -> (n_corpus, 768) out-of-fold predictions
    for alpha in ALPHAS:
        t0 = time.time()
        oof = np.zeros_like(Y_corpus)
        fold_cos = []
        for train_i, val_i in kf.split(X_corpus):
            ridge = Ridge(alpha=alpha, fit_intercept=False, solver="sparse_cg")
            ridge.fit(X_corpus[train_i], Y_corpus[train_i])
            Wf = ridge.coef_.T.astype(np.float32)
            pred = np.asarray(X_corpus[val_i] @ Wf)
            oof[val_i] = pred
            fold_cos.append(cos_rows(pred, Y_corpus[val_i]).mean())
        cv_scores[alpha] = float(np.mean(fold_cos))
        oof_preds[alpha] = oof
        log(f"  alpha={alpha:<5} CV fit+score {time.time() - t0:.1f}s  mean fold cosine={cv_scores[alpha]:.4f}")

    best_alpha = max(cv_scores, key=cv_scores.get)
    log(f"chosen alpha={best_alpha}  (CV mean cosine={cv_scores[best_alpha]:.4f})")

    # held-out cosine on a 1000-row subset, from the CHOSEN alpha's out-of-fold predictions
    oof_best = oof_preds[best_alpha]
    sub = RNG.choice(n_corpus, size=min(1000, n_corpus), replace=False)
    c_holdout = cos_rows(oof_best[sub], Y_corpus[sub])
    log(f"held-out (1000-row, out-of-fold) cosine: mean {c_holdout.mean():.4f} "
        f"median {np.median(c_holdout):.4f} min {c_holdout.min():.4f}")

    # ---- refit on the FULL 9900-row corpus with the chosen alpha -----------
    log(f"\nrefitting on full corpus (n={n_corpus}) at alpha={best_alpha}")
    ridge = Ridge(alpha=best_alpha, fit_intercept=False, solver="sparse_cg")
    ridge.fit(X_corpus, Y_corpus)
    W = ridge.coef_.T.astype(np.float32)  # (n_fitted_tokens, 768)
    table_bytes = int(W.shape[0] * W.shape[1] * 2)
    log(f"fitted table: {W.shape[0]} tokens x {W.shape[1]} dim, fp16 = {table_bytes / 2**20:.2f} MB")

    student_qv_raw = encode_from_tokens(query_ids, tok2col, W)          # (100, 768) raw
    student_corpus_raw = np.asarray(X_corpus @ W)                       # (9900, 768) raw, self-index

    c_query = cos_rows(student_qv_raw, teacher_qv)
    log(f"cosine(student, teacher) on the 100 queries: mean {c_query.mean():.4f} "
        f"median {np.median(c_query):.4f} min {c_query.min():.4f}")

    # raw vs L2-normalized query variant — check whether ranks differ
    student_qv_norm = student_qv_raw / np.clip(np.linalg.norm(student_qv_raw, axis=1, keepdims=True), 1e-9, None)
    rank_raw = topN_by_score(student_qv_raw @ teacher_corpus.T, 10)
    rank_norm = topN_by_score(student_qv_norm @ teacher_corpus.T, 10)
    ranks_identical = bool(np.array_equal(rank_raw, rank_norm))
    log(f"raw vs L2-normalized student query: top-10 rankings against teacher float index "
        f"{'IDENTICAL' if ranks_identical else 'DIFFER'} (expected identical — inner-product "
        f"ranking against a fixed corpus is invariant to positive query scaling)")

    # ---- teacher remax 1-bit index (centered SimHash) ----------------------
    log("\n=== quantized teacher indexes ===")
    mu = teacher_corpus.mean(0)
    corpus_c = (teacher_corpus - mu).astype(np.float32)
    teacher_qv_c = (teacher_qv - mu).astype(np.float32)
    student_qv_c = (student_qv_raw - mu).astype(np.float32)

    sq = StackedSignBitQuantizer(d=D, k=1, seed=SEED).fit(corpus_c)
    corpus_codes = sq.encode(corpus_c)
    teacher_q_codes = sq.encode(teacher_qv_c)
    student_q_codes = sq.encode(student_qv_c)

    # sign-agreement rate: fraction of the 768 bits that match between
    # student and teacher query codes, post centering+rotation
    hamming = np.bitwise_count(np.bitwise_xor(teacher_q_codes, student_q_codes)).sum(axis=1)
    sign_agree = 1.0 - hamming / D
    log(f"sign-agreement rate (student vs teacher query, post center+rotate, n=100): "
        f"mean {sign_agree.mean():.4f} median {np.median(sign_agree):.4f} "
        f"min {sign_agree.min():.4f} max {sign_agree.max():.4f}  "
        f"(0.5 = chance level for a random bit)")

    # ---- remex 2-bit / 4-bit teacher index ----------------------------------
    remex_xh = {}
    for bits in (2, 4):
        qz = remex.Quantizer(d=D, bits=bits, seed=SEED)
        xh = qz.decode(qz.encode(teacher_corpus))
        remex_xh[bits] = xh

    # ---- cells ---------------------------------------------------------------
    rows = []

    def add(cell, index_codec, bits, metrics):
        row = {"cell": cell, "index_codec": index_codec, "bits": bits, **metrics}
        rows.append(row)
        log(f"  {cell:42s} {index_codec:>8} bits={str(bits):>4}  "
            f"r@10={metrics['r@10']:.3f}  r@100={metrics['r@100']:.3f}  top1={metrics['top1']:.3f}")

    log("\n=== cells (n=100 queries, 9900-paper corpus) ===")

    pred = topN_by_score(teacher_qv @ teacher_corpus.T, 100)
    add("teacher-query/teacher-float-index", "float32", 32, score_cell(truth10, pred))

    pred = hamming_topN(teacher_q_codes, corpus_codes, 100)
    add("teacher-query/teacher-remax-1bit-index", "remax", 1, score_cell(truth10, pred))

    for bits in (2, 4):
        pred = topN_by_score(teacher_qv @ remex_xh[bits].T, 100)
        add(f"teacher-query/teacher-remex-{bits}bit-index", "remex", bits, score_cell(truth10, pred))

    pred = topN_by_score(student_qv_raw @ teacher_corpus.T, 100)
    add("student-query/teacher-float-index", "float32", 32, score_cell(truth10, pred))

    pred = hamming_topN(student_q_codes, corpus_codes, 100)
    add("student-query/teacher-remax-1bit-index", "remax", 1, score_cell(truth10, pred))

    for bits in (2, 4):
        pred = topN_by_score(student_qv_raw @ remex_xh[bits].T, 100)
        add(f"student-query/teacher-remex-{bits}bit-index", "remex", bits, score_cell(truth10, pred))

    pred = topN_by_score(student_qv_raw @ student_corpus_raw.T, 100)
    add("student-query/student-index (reference only)", "float32", 32, score_cell(truth10, pred))

    out = {
        "seed": SEED, "n_queries": N_QUERIES, "n_corpus": n_corpus, "d": D,
        "vocab": {"corpus_distinct_tokens": len(freq), "kept_at_min_freq": len(kept), "min_freq": MIN_FREQ},
        "cv_scores": cv_scores, "chosen_alpha": best_alpha,
        "held_out_1000_cosine": {"mean": float(c_holdout.mean()), "median": float(np.median(c_holdout)),
                                  "min": float(c_holdout.min())},
        "query_cosine": {"mean": float(c_query.mean()), "median": float(np.median(c_query)),
                          "min": float(c_query.min())},
        "raw_vs_normalized_ranks_identical": ranks_identical,
        "sign_agreement": {"mean": float(sign_agree.mean()), "median": float(np.median(sign_agree)),
                            "min": float(sign_agree.min()), "max": float(sign_agree.max())},
        "table": {"n_fitted_tokens": int(W.shape[0]), "dim": int(W.shape[1]), "bytes_fp16": table_bytes},
        "rows": rows,
        "model2vec_comparison_skipped": "requires a SPECTER2 model download for token-level distillation; "
                                         "pass 2 already established that token-level distillation from a "
                                         "teacher's own weights is space-incompatible, so this was skipped "
                                         "per the brief's 'only if cheap' instruction.",
        "failed": failed,
    }
    json.dump(out, open(HERE / "specter2_results.json", "w"), indent=1, default=float)
    log(f"\nwrote {HERE / 'specter2_results.json'}")
    if failed:
        log("\nFAILURES:")
        for f in failed:
            log(f"  - {f}")


if __name__ == "__main__":
    main()
