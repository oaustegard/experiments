"""Build N=50 reranking slates from MSLR-WEB10K and fit the teacher.

Simulates a reranking stage: a first-stage retriever (BM25 over the whole
document, MSLR feature 110, 0-indexed 109) returns 50 candidates per query;
the reranker must order them.

Teacher = pointwise gradient-boosted trees on graded relevance, fit on train.
Its argsort is the permutation the students distill (the paper's Phase 1).
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
N_SLATE = 50
BM25_WHOLE_DOC = 109  # 0-indexed; MSLR feature 110
SEED = 20260908


def load_slates(path, n_slate=N_SLATE):
    t = pq.read_table(path).to_pydict()
    feats, labs, qids = [], [], []
    for q, lab, fl, n in zip(t["query"], t["labels"], t["features"], t["n"]):
        if n < n_slate:
            continue
        F = np.asarray(fl, dtype=np.float32).reshape(n, -1)
        L = np.asarray(lab, dtype=np.float32)
        top = np.argsort(-F[:, BM25_WHOLE_DOC], kind="stable")[:n_slate]
        feats.append(F[top])
        labs.append(L[top])
        qids.append(q)
    return np.stack(feats), np.stack(labs), np.asarray(qids)


def main():
    out = DATA / "slates.npz"
    Xtr, ytr, qtr = load_slates(DATA / "train.parquet")
    Xte, yte, qte = load_slates(DATA / "test.parquet")
    print(f"train slates {Xtr.shape}  test slates {Xte.shape}")
    print(f"train label mix {np.bincount(ytr.astype(int).ravel(), minlength=5)}")
    print(f"test  label mix {np.bincount(yte.astype(int).ravel(), minlength=5)}")

    # feature standardisation from train only
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mu, sd = flat.mean(0), flat.std(0)
    sd[sd < 1e-6] = 1.0

    teacher = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.1, max_depth=None,
        early_stopping=True, validation_fraction=0.1, random_state=SEED,
    )
    teacher.fit(flat, ytr.ravel())
    print(f"teacher fitted, {teacher.n_iter_} iters")

    str_ = teacher.predict(flat).reshape(ytr.shape).astype(np.float32)
    ste = teacher.predict(Xte.reshape(-1, Xte.shape[-1])).reshape(yte.shape).astype(np.float32)
    # teacher permutation: item indices, best first
    ptr = np.argsort(-str_, axis=1, kind="stable").astype(np.int16)
    pte = np.argsort(-ste, axis=1, kind="stable").astype(np.int16)

    np.savez_compressed(
        out,
        Xtr=Xtr, ytr=ytr, qtr=qtr, teacher_perm_tr=ptr, teacher_score_tr=str_,
        Xte=Xte, yte=yte, qte=qte, teacher_perm_te=pte, teacher_score_te=ste,
        mu=mu.astype(np.float32), sd=sd.astype(np.float32),
    )
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
