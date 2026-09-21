"""Round 2: the 2024 map step as the first layer, then the same LR probe.

Candidate phrases per memory (spaCy noun chunks + entities, 1-4 words, edge stop-words
stripped) -> K dims selected inside each training fold by document frequency, optionally
with a gte-small dissimilarity filter -> binary presence vector -> one-vs-rest LR on the
round-1 folds, C swept and selected by micro-AP.

Arms at each K: phrases+dissim, phrases (frequency only), unigrams (frequency only).
Writes results/phrase_vocab.json; caches candidates in data/phrases.json (gitignored).
"""
import argparse, json, re, sys, time
from collections import Counter
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from common import DATA, load_fixture

HERE = DATA.parent
RESULTS = HERE / "results"
KS = [128, 256, 512, 1024, 2048, None]
CS = [1, 10, 100, 1000, 10000]
SMOKE = False
STOP = set("a an the this that these those my our your his her its their some any all no each every "
           "of in on at to for from by with as into over under about after before between and or but "
           "is are was were be been being it he she they we you i me him them us what which who whom "
           "there here then than so such very more most less least much many few one two three".split())
WORD = re.compile(r"^[a-z0-9][a-z0-9._+/-]*$")


def p_at_5(Y, P):
    top = np.argsort(-P, axis=1)[:, :5]
    return float(np.mean([Y[i, top[i]].mean() for i in range(len(Y))]))


def candidates(texts, use_cache=True):
    cache = DATA / "phrases.json"
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp.max_length = 200000
    out = []
    t0 = time.time()
    for i, doc in enumerate(nlp.pipe([t[:20000] for t in texts], batch_size=32)):
        phr = set()
        spans = list(doc.noun_chunks) + list(doc.ents)
        for s in spans:
            toks = [t.text.lower() for t in s if not t.is_punct and not t.is_space]
            while toks and toks[0] in STOP: toks = toks[1:]
            while toks and toks[-1] in STOP: toks = toks[:-1]
            if 1 <= len(toks) <= 4 and all(WORD.match(t) for t in toks):
                phr.add(" ".join(toks))
        out.append(sorted(phr))
        if i % 500 == 0:
            print(f"  spacy {i}/{len(texts)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    if use_cache:
        cache.write_text(json.dumps(out))
    return out


def unigrams(texts):
    return [sorted({w for w in re.findall(r"[a-z0-9][a-z0-9._+/-]*", t.lower()) if w not in STOP and len(w) > 1})
            for t in texts]


def select_vocab(sets_tr, K, embed=None, min_df=3, thresh=0.9):
    df = Counter(p for s in sets_tr for p in s)
    ranked = [p for p, c in df.most_common() if c >= min_df]
    if K is None or embed is None:
        return ranked[:K] if K else ranked
    chosen, chosen_vecs = [], []
    for p in ranked:
        v = embed[p]
        if chosen_vecs and float(np.max(np.asarray(chosen_vecs) @ v)) > thresh:
            continue
        chosen.append(p); chosen_vecs.append(v)
        if len(chosen) >= K:
            break
    return chosen


def binarize(sets, vocab):
    idx = {p: j for j, p in enumerate(vocab)}
    rows, cols = [], []
    for i, s in enumerate(sets):
        for p in s:
            if p in idx:
                rows.append(i); cols.append(idx[p])
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(sets), len(vocab)))


def fit_one(Xtr, y, Xte, C):
    if y.sum() == 0: return None
    if y.sum() == len(y): return np.ones(Xte.shape[0])
    return LogisticRegression(C=C, max_iter=2000, solver="liblinear").fit(Xtr, y).predict_proba(Xte)[:, 1]


def cv(sets, Y, fold_of, K, C, embed=None):
    n, L = Y.shape
    P = np.zeros((n, L))
    sizes = []
    for k in range(5):
        tr, te = fold_of != k, fold_of == k
        vocab = select_vocab([s for s, m in zip(sets, tr) if m], K, embed)
        sizes.append(len(vocab))
        Xtr, Xte = binarize([s for s, m in zip(sets, tr) if m], vocab), binarize([s for s, m in zip(sets, te) if m], vocab)
        for M in (Xtr, Xte): M.sum_duplicates(); M.sort_indices()
        res = Parallel(n_jobs=4, prefer="processes", batch_size=8)(
            delayed(fit_one)(Xtr, Y[tr, j], Xte, C) for j in range(L))
        for j, p in enumerate(res):
            if p is not None: P[te, j] = p
    return P, sizes


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); args = ap.parse_args()
    t0 = time.time()
    fx = load_fixture(); labels = fx["labels"]; L = len(labels); lidx = {l: j for j, l in enumerate(labels)}
    folds = json.loads((RESULTS / "folds.json").read_text()); ids = folds["ids"]; fold_of = np.array(folds["fold_of"])
    if args.limit:
        ids, fold_of = ids[:args.limit], fold_of[:args.limit]
        global KS, CS
        KS, CS = [128, None], [100]
    lab_of = {m["id"]: m["labels"] for m in fx["memories"]}
    Y = np.zeros((len(ids), L), np.int8)
    for i, m in enumerate(ids):
        for l in lab_of[m]: Y[i, lidx[l]] = 1
    tj = json.loads((DATA / "texts.json").read_text()); tmap = dict(zip(tj["ids"], tj["texts"]))
    texts = [tmap[m] for m in ids]
    all_ids = tj["ids"]
    phr_all = candidates(tj["texts"]) if not args.limit else candidates(texts, use_cache=False)
    phr = [phr_all[all_ids.index(m)] for m in ids] if not args.limit else phr_all
    uni = unigrams(texts)
    print(f"{len(ids)} memories; phrases/memory median {np.median([len(s) for s in phr]):.0f}, "
          f"distinct {len({p for s in phr for p in s})}; unigrams distinct {len({p for s in uni for p in s})}",
          file=sys.stderr, flush=True)
    # phrase embeddings for the dissimilarity filter, over every candidate with df >= 3
    from sentence_transformers import SentenceTransformer
    df = Counter(p for s in phr for p in s)
    keep = [p for p, c in df.items() if c >= 3]
    st = SentenceTransformer("thenlper/gte-small", device="cpu")
    E = st.encode(keep, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    embed = dict(zip(keep, E))
    print(f"embedded {len(keep)} candidates in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)

    out = {"n": len(ids), "arms": {}, "predictions_file": "PLAN.md round 2"}
    RESULTS.mkdir(exist_ok=True)
    for K in KS:
        for arm, sets, emb in (("phrases+dissim", phr, embed), ("phrases", phr, None), ("unigrams", uni, None)):
            if K is None and arm == "phrases+dissim":
                continue
            best = None
            for C in CS:
                ta = time.time()
                P, sizes = cv(sets, Y, fold_of, K, C, emb)
                m = {"C": C, "micro_ap": float(average_precision_score(Y.ravel(), P.ravel())),
                     "macro_ap": float(np.nanmean([average_precision_score(Y[:, j], P[:, j]) if Y[:, j].sum() else np.nan for j in range(L)])),
                     "p_at_5": p_at_5(Y, P), "vocab_sizes": sizes, "seconds": round(time.time() - ta)}
                print(f"  K={K} {arm:16s} C={C:<6} micro-AP {m['micro_ap']:.4f} macro-AP {m['macro_ap']:.4f} "
                      f"P@5 {m['p_at_5']:.4f} vocab {sizes[0]} {m['seconds']}s", file=sys.stderr, flush=True)
                if best is None or m["micro_ap"] > best["micro_ap"]:
                    best = m
                    if not args.limit: np.save(RESULTS / f"proba_phrase_{arm}_K{K}.npy", P)
            out["arms"].setdefault(arm, {})[str(K)] = best
            if not args.limit: (RESULTS / "phrase_vocab.json").write_text(json.dumps(out, indent=1))
    out["wall_seconds"] = round(time.time() - t0)
    if not args.limit: (RESULTS / "phrase_vocab.json").write_text(json.dumps(out, indent=1))
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
