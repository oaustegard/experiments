"""Arms 2, 3, 4 and controls: can a linear map from each representation predict
Muninn's human tags on the labelled memories, and does SPARSEUP need its
learned weights (vs binarized) or its learned expansion (vs lexical TF-IDF)?

One stratified 5-fold split (stratify on each memory's most frequent label —
"most frequent" meaning globally most-used among its labels — seed 0), shared
by every arm and saved to results/folds.json. One-vs-rest LogisticRegression
per label; a label with no positives in a training fold gets all-zero
predictions for that fold.

Representations: sparseup, sparseup_binary, tfidf_word, tfidf_char,
tfidf_word+char, gte_small, plus controls prior (train-fold label frequency)
and shuffled (sparseup features, label rows permuted, seed 0).

Metrics on pooled out-of-fold predictions: micro-F1 and macro-F1 at threshold
0.5, P@5, per-label F1 and AP; the binarization gap (sparseup - sparseup_binary);
a per-label sparseup-vs-tfidf_word+char split by arm1's literal-mention rate;
and bootstrap 95% CIs (1000 resamples, seed 0) over documents for the paired
differences sparseup - tfidf_word+char and sparseup - sparseup_binary on
micro-F1 and P@5.

Usage:
    python3 probe.py                          # full run; requires data/sparse.npz, data/dense.npy
    python3 probe.py --limit 256 --folds 3     # smoke test against whatever data/parts exist
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, load_fixture

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEED = 0
C = 1.0
MAX_ITER = 2000
SOLVER = "liblinear"
N_JOBS = 4

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------------------------------------------------------------- data loading

def load_rows_sparse(n_limit):
    """Full data/sparse.npz, or (smoke test) vstack of whatever data/parts exist,
    truncated to the first n_limit rows."""
    if n_limit is None:
        return sp.load_npz(DATA / "sparse.npz").tocsr()
    part_files = sorted((DATA / "parts").glob("sparse_*.npz"))
    if not part_files:
        print(f"No data/parts/sparse_*.npz found; cannot smoke-test with --limit {n_limit}.",
              file=sys.stderr)
        sys.exit(1)
    S = sp.vstack([sp.load_npz(p) for p in part_files]).tocsr()
    return S[:min(n_limit, S.shape[0])]


def load_rows_dense(n_limit):
    if n_limit is None:
        return np.load(DATA / "dense.npy")
    part_files = sorted((DATA / "parts").glob("dense_*.npy"))
    if not part_files:
        print(f"No data/parts/dense_*.npy found; cannot smoke-test with --limit {n_limit}.",
              file=sys.stderr)
        sys.exit(1)
    D = np.vstack([np.load(p) for p in part_files])
    return D[:min(n_limit, D.shape[0])]


def labelled_subset(fixture, n_limit):
    """Row positions (into the full/limited sparse & dense matrices, and texts.json,
    which all share fixture order) of memories carrying >= 1 label."""
    memories = fixture["memories"]
    if n_limit is not None:
        memories = memories[:n_limit]
    idx = [i for i, m in enumerate(memories) if m["labels"]]
    labelled = [memories[i] for i in idx]
    return labelled, idx


# ---------------------------------------------------------------- folds

def make_folds(labelled, n_splits, seed=SEED):
    label_counts = load_fixture()["label_counts"]

    def primary(m):
        return max(m["labels"], key=lambda l: label_counts.get(l, 0))

    primary_labels = [primary(m) for m in labelled]
    from collections import Counter
    counts = Counter(primary_labels)
    # StratifiedKFold needs >= n_splits members per stratum; merge rare primaries
    # into one pseudo-class for the split only. True primary_label is kept in
    # folds.json regardless.
    strat_key = [pl if counts[pl] >= n_splits else "__rare__" for pl in primary_labels]
    rare_merged = sorted({pl for pl in primary_labels if counts[pl] < n_splits})

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = np.full(len(labelled), -1, dtype=int)
    for k, (_, te) in enumerate(skf.split(np.zeros(len(labelled)), strat_key)):
        fold_of[te] = k
    assert (fold_of >= 0).all()
    return fold_of, primary_labels, rare_merged


# ---------------------------------------------------------------- CV predictors


def _fit_one(Xtr, ytr, Xte):
    """One one-vs-rest fit; returns out-of-fold probabilities for the test rows."""
    pos = ytr.sum()
    if pos == 0:
        return None  # stays 0.0
    if pos == len(ytr):
        return np.ones(Xte.shape[0])
    clf = LogisticRegression(C=C, max_iter=MAX_ITER, solver=SOLVER)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _fit_all_labels(Xtr, Ytr, Xte):
    from joblib import Parallel, delayed
    # joblib memmaps large arrays read-only; scipy's max() sorts indices in place,
    # so canonicalise once here instead of copying inside every worker.
    for M in (Xtr, Xte):
        if hasattr(M, "sort_indices"):
            M.sum_duplicates(); M.sort_indices()
    return Parallel(n_jobs=N_JOBS, prefer="processes", batch_size=8)(
        delayed(_fit_one)(Xtr, Ytr[:, j], Xte) for j in range(Ytr.shape[1]))


def cv_predict_fixed(X, Y, fold_of, n_splits):
    n, L = Y.shape
    proba = np.zeros((n, L), dtype=np.float64)
    for k in range(n_splits):
        tr = fold_of != k
        te = fold_of == k
        Xtr, Xte = X[tr], X[te]
        for j, p in enumerate(_fit_all_labels(Xtr, Y[tr], Xte)):
            if p is not None:
                proba[te, j] = p
    return proba


def cv_predict_tfidf(texts, Y, fold_of, n_splits, kind):
    n, L = Y.shape
    proba = np.zeros((n, L), dtype=np.float64)
    for k in range(n_splits):
        tr_idx = np.where(fold_of != k)[0]
        te_idx = np.where(fold_of == k)[0]
        train_texts = [texts[i] for i in tr_idx]
        test_texts = [texts[i] for i in te_idx]
        if kind == "word":
            vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
            Xtr = vec.fit_transform(train_texts)
            Xte = vec.transform(test_texts)
        elif kind == "char":
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
            Xtr = vec.fit_transform(train_texts)
            Xte = vec.transform(test_texts)
        elif kind == "word+char":
            vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
            vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
            Xtr = hstack([vw.fit_transform(train_texts), vc.fit_transform(train_texts)]).tocsr()
            Xte = hstack([vw.transform(test_texts), vc.transform(test_texts)]).tocsr()
        else:
            raise ValueError(kind)
        for j, p in enumerate(_fit_all_labels(Xtr, Y[tr_idx], Xte)):
            if p is not None:
                proba[te_idx, j] = p
    return proba


def cv_predict_prior(Y, fold_of, n_splits):
    n, L = Y.shape
    proba = np.zeros((n, L), dtype=np.float64)
    for k in range(n_splits):
        tr = fold_of != k
        te = fold_of == k
        freq = np.asarray(Y[tr]).mean(axis=0)
        proba[te] = freq
    return proba


# ---------------------------------------------------------------- metrics

def micro_f1(Y, P, thresh=0.5):
    return f1_score(Y, (P >= thresh).astype(int), average="micro", zero_division=0)


def macro_f1(Y, P, thresh=0.5):
    return f1_score(Y, (P >= thresh).astype(int), average="macro", zero_division=0)


def per_label_f1(Y, P, thresh=0.5):
    return f1_score(Y, (P >= thresh).astype(int), average=None, zero_division=0)


def per_label_ap(Y, P):
    L = Y.shape[1]
    out = []
    for j in range(L):
        if Y[:, j].sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float(average_precision_score(Y[:, j], P[:, j])))
    return out


def p_at_5(Y, P):
    n = Y.shape[0]
    k = min(5, Y.shape[1])
    top = np.argsort(-P, axis=1)[:, :k]
    hits = np.take_along_axis(Y, top, axis=1).sum(axis=1)
    return float(np.mean(hits / k))


def bootstrap_ci_diff(Y, Pa, Pb, metric_fn, n_boot=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = Y.shape[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        diffs[b] = metric_fn(Y[idx], Pa[idx]) - metric_fn(Y[idx], Pb[idx])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(np.percentile(diffs, 2.5)),
        "ci_hi": float(np.percentile(diffs, 97.5)),
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="smoke-test on the first N fixture rows, using whatever data/parts exist")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0,
                     help="inverse regularisation strength, same for every arm; sweep it — "
                          "the arms differ in feature scale (TF-IDF and gte-small rows are "
                          "L2-normalised, SPARSEUP weights are not), so one C is not one prior")
    args = ap.parse_args()
    global C
    C = args.C
    ctag = f"C{args.C:g}"
    is_smoke = args.limit is not None

    t0 = time.time()
    fixture = load_fixture()
    labels = fixture["labels"]
    L = len(labels)
    label_idx = {lab: j for j, lab in enumerate(labels)}

    labelled, row_idx = labelled_subset(fixture, args.limit)
    n = len(labelled)
    print(f"{'[smoke test] ' if is_smoke else ''}{n} labelled memories "
          f"(of {args.limit or fixture['n_memories']} rows considered)", file=sys.stderr)
    if n < args.folds * 2:
        print(f"Too few labelled rows ({n}) for --folds {args.folds}; raise --limit.", file=sys.stderr)
        sys.exit(1)

    Y = np.zeros((n, L), dtype=np.int8)
    for i, m in enumerate(labelled):
        for lab in m["labels"]:
            Y[i, label_idx[lab]] = 1

    fold_of, primary_labels, rare_merged = make_folds(labelled, args.folds)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "folds.json").write_text(json.dumps({
        "smoke_test": is_smoke,
        "n": n,
        "n_splits": args.folds,
        "seed": SEED,
        "ids": [m["id"] for m in labelled],
        "fold_of": fold_of.tolist(),
        "primary_label": primary_labels,
        "rare_primary_labels_merged_for_stratification": rare_merged,
    }, indent=1))
    print(f"folds: {np.bincount(fold_of).tolist()}, "
          f"{len(rare_merged)} rare primary labels merged for stratification", file=sys.stderr)

    S_full = load_rows_sparse(args.limit)
    D_full = load_rows_dense(args.limit)
    S = S_full[row_idx]
    D = D_full[row_idx]
    S_bin = S.copy()
    S_bin.data = np.ones_like(S_bin.data)

    texts_j = json.loads((DATA / "texts.json").read_text())
    texts_by_id = dict(zip(texts_j["ids"], texts_j["texts"]))
    texts = [texts_by_id[m["id"]] for m in labelled]

    print("running CV per arm ...", file=sys.stderr)
    proba = {}
    arms = [
        ("sparseup", lambda: cv_predict_fixed(S, Y, fold_of, args.folds)),
        ("sparseup_binary", lambda: cv_predict_fixed(S_bin, Y, fold_of, args.folds)),
        ("tfidf_word", lambda: cv_predict_tfidf(texts, Y, fold_of, args.folds, "word")),
        ("tfidf_char", lambda: cv_predict_tfidf(texts, Y, fold_of, args.folds, "char")),
        ("tfidf_word+char", lambda: cv_predict_tfidf(texts, Y, fold_of, args.folds, "word+char")),
        ("gte_small", lambda: cv_predict_fixed(D, Y, fold_of, args.folds)),
        ("prior", lambda: cv_predict_prior(Y, fold_of, args.folds)),
    ]
    RESULTS.mkdir(exist_ok=True)
    for name, fn in arms:
        ta = time.time()
        cache = RESULTS / f"proba_{name}_{ctag}.npy"
        if cache.exists() and not is_smoke:
            proba[name] = np.load(cache)
            print(f"  arm {name}: cached", file=sys.stderr, flush=True)
            continue
        proba[name] = fn()
        if not is_smoke:
            np.save(cache, proba[name])
        print(f"  arm {name}: {time.time() - ta:.0f}s", file=sys.stderr, flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(n)
    Y_shuf = Y[perm]
    cache = RESULTS / f"proba_shuffled_{ctag}.npy"
    if cache.exists() and not is_smoke:
        proba["shuffled"] = np.load(cache)
    else:
        proba["shuffled"] = cv_predict_fixed(S, Y_shuf, fold_of, args.folds)
        if not is_smoke:
            np.save(cache, proba["shuffled"])
    print(f"CV done in {time.time() - t0:.0f}s", file=sys.stderr)

    # ---- per-arm metrics
    arm_metrics = {}
    for arm, P in proba.items():
        y_use = Y_shuf if arm == "shuffled" else Y
        arm_metrics[arm] = {
            "micro_f1": micro_f1(y_use, P),
            "macro_f1": macro_f1(y_use, P),
            "p_at_5": p_at_5(y_use, P),
            "per_label_f1": per_label_f1(y_use, P).tolist(),
            "per_label_ap": per_label_ap(y_use, P),
        }

    # ---- binarization gap: sparseup - sparseup_binary
    gap = {
        m: arm_metrics["sparseup"][m] - arm_metrics["sparseup_binary"][m]
        for m in ("micro_f1", "macro_f1", "p_at_5")
    }

    # ---- literal-mention split: sparseup vs tfidf_word+char, grouped by arm1 rate
    lit_rates = {}
    arm1_path = RESULTS / "arm1.json"
    if arm1_path.exists():
        arm1 = json.loads(arm1_path.read_text())
        for lab, entry in arm1.get("per_label", {}).items():
            r = entry.get("literal_mention_rate")
            if r is not None:
                lit_rates[lab] = r
    else:
        # fallback: compute inline exactly as arm1_recall.py does, on this subset
        id_to_labels = {m["id"]: m["labels"] for m in labelled}
        from collections import defaultdict as dd
        hits, tot = dd(int), dd(int)
        for m in labelled:
            t = texts_by_id[m["id"]].lower()
            for lab in m["labels"]:
                tot[lab] += 1
                lab_l = lab.lower()
                if lab_l in t or lab_l.replace("-", " ") in t:
                    hits[lab] += 1
        lit_rates = {lab: hits[lab] / tot[lab] for lab in tot}

    group_hi = [label_idx[l] for l in labels if lit_rates.get(l, 0.0) >= 0.5]
    group_lo = [label_idx[l] for l in labels if lit_rates.get(l, 0.0) < 0.5]

    def mean_f1(arm, idxs):
        f1s = np.array(arm_metrics[arm]["per_label_f1"])
        return float(np.nanmean(f1s[idxs])) if idxs else None

    literal_split = {
        "n_labels_high_literal_rate_ge_0.5": len(group_hi),
        "n_labels_low_literal_rate_lt_0.5": len(group_lo),
        "labels_missing_from_arm1": [l for l in labels if l not in lit_rates],
        "mean_f1": {
            "sparseup": {"high_literal": mean_f1("sparseup", group_hi),
                         "low_literal": mean_f1("sparseup", group_lo)},
            "tfidf_word+char": {"high_literal": mean_f1("tfidf_word+char", group_hi),
                                 "low_literal": mean_f1("tfidf_word+char", group_lo)},
        },
        "note": ("computed from results/arm1.json" if arm1_path.exists()
                 else "arm1.json not found; literal-mention rate computed inline on this subset"),
    }

    # ---- bootstrap CIs (paired, over documents)
    print("bootstrapping ...", file=sys.stderr)
    boot = {
        "sparseup_minus_tfidf_word+char": {
            "micro_f1": bootstrap_ci_diff(Y, proba["sparseup"], proba["tfidf_word+char"], micro_f1),
            "p_at_5": bootstrap_ci_diff(Y, proba["sparseup"], proba["tfidf_word+char"], p_at_5),
        },
        "sparseup_minus_sparseup_binary": {
            "micro_f1": bootstrap_ci_diff(Y, proba["sparseup"], proba["sparseup_binary"], micro_f1),
            "p_at_5": bootstrap_ci_diff(Y, proba["sparseup"], proba["sparseup_binary"], p_at_5),
        },
    }

    def scale(M):
        M = M.tocsr() if hasattr(M, "tocsr") else M
        norms = (np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel()) if hasattr(M, "multiply")
                 else np.linalg.norm(M, axis=1))
        mx = float(M.max()) if hasattr(M, "max") else float(np.abs(M).max())
        return {"max_value": mx, "mean_row_l2": float(norms.mean())}
    feature_scale = {"sparseup": scale(S), "sparseup_binary": scale(S_bin), "gte_small": scale(D)}

    out = {
        "smoke_test": is_smoke,
        "feature_scale": feature_scale,
        "n": n,
        "n_splits": args.folds,
        "seed": SEED,
        "classifier": {"kind": "one-vs-rest LogisticRegression", "C": C, "solver": SOLVER,
                        "max_iter": MAX_ITER, "class_weight": None},
        "arms": arm_metrics,
        "binarization_gap_sparseup_minus_binary": gap,
        "literal_mention_split": literal_split,
        "bootstrap_ci_diff": boot,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / f"probe_{ctag}.json").write_text(json.dumps(out, indent=1))

    print(f"\n{'SMOKE TEST — ' if is_smoke else ''}Probe: {n} labelled memories, "
          f"{args.folds}-fold CV, {L} labels")
    print(f"{'arm':<18}{'micro-F1':>10}{'macro-F1':>10}{'P@5':>8}")
    for arm in ("sparseup", "sparseup_binary", "tfidf_word", "tfidf_char",
                "tfidf_word+char", "gte_small", "prior", "shuffled"):
        m = arm_metrics[arm]
        print(f"{arm:<18}{m['micro_f1']:>10.4f}{m['macro_f1']:>10.4f}{m['p_at_5']:>8.4f}")
    print(f"\nbinarization gap (sparseup - sparseup_binary): micro-F1 {gap['micro_f1']:+.4f}, "
          f"macro-F1 {gap['macro_f1']:+.4f}, P@5 {gap['p_at_5']:+.4f}")
    b1 = boot["sparseup_minus_tfidf_word+char"]["micro_f1"]
    b2 = boot["sparseup_minus_sparseup_binary"]["micro_f1"]
    print(f"sparseup - tfidf_word+char micro-F1: {b1['mean_diff']:+.4f} "
          f"[{b1['ci_lo']:+.4f}, {b1['ci_hi']:+.4f}]")
    print(f"sparseup - sparseup_binary micro-F1: {b2['mean_diff']:+.4f} "
          f"[{b2['ci_lo']:+.4f}, {b2['ci_hi']:+.4f}]")
    print(f"\nWrote {RESULTS / f'probe_{ctag}.json'} and {RESULTS / 'folds.json'}")


if __name__ == "__main__":
    main()
