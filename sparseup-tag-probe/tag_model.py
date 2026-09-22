"""The tagger round 1 pointed at: TF-IDF word+char + one-vs-rest LR over the 325 tags.

  python3 tag_model.py train      # fits on the 3,315 labelled fixture memories, picks a
                                  # per-label threshold from the cached out-of-fold scores
  python3 tag_model.py predict <memory-id> [...]   # tags for memories fetched by id
  python3 tag_model.py predict --text "..."        # tags for arbitrary text

The fitted model holds n-grams of memory text, so it lives in data/ (gitignored).
"""
import argparse, json, pickle, sys, time
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from common import DATA, load_fixture, memory_module

MODEL = DATA / "tag_model.pkl"
RESULTS = DATA.parent / "results"
C = 10000  # the C selected by micro-AP in round 1 for this arm


def features(train_texts):
    vw = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    X = hstack([vw.fit_transform(train_texts), vc.fit_transform(train_texts)]).tocsr()
    return (vw, vc), X


def transform(vecs, texts):
    vw, vc = vecs
    return hstack([vw.transform(texts), vc.transform(texts)]).tocsr()


def train():
    t0 = time.time()
    fx = load_fixture(); labels = fx["labels"]; L = len(labels)
    folds = json.loads((RESULTS / "folds.json").read_text()); ids = folds["ids"]
    lab_of = {m["id"]: m["labels"] for m in fx["memories"]}
    tj = json.loads((DATA / "texts.json").read_text()); tmap = dict(zip(tj["ids"], tj["texts"]))
    texts = [tmap[m] for m in ids]
    Y = np.zeros((len(ids), L), np.int8)
    for i, m in enumerate(ids):
        for l in lab_of[m]: Y[i, labels.index(l)] = 1

    # per-label threshold from the round-1 out-of-fold scores of this exact arm and C
    P = np.load(RESULTS / f"proba_tfidf_word+char_C{C}.npy")
    grid = np.concatenate([np.arange(0.001, 0.05, 0.001), np.arange(0.05, 0.95, 0.01)])
    thr = np.full(L, 0.5)
    for j in range(L):
        if Y[:, j].sum() == 0: continue
        f = [f1_score(Y[:, j], P[:, j] >= t, zero_division=0) for t in grid]
        thr[j] = float(grid[int(np.argmax(f))])
    pred = P >= thr
    tp = (pred & (Y == 1)).sum(); fp = (pred & (Y == 0)).sum(); fn = (~pred & (Y == 1)).sum()
    oof = {"micro_precision": float(tp / (tp + fp)), "micro_recall": float(tp / (tp + fn)),
           "micro_f1": float(2 * tp / (2 * tp + fp + fn)), "tags_per_memory": float(pred.sum(1).mean()),
           "gold_tags_per_memory": float(Y.sum(1).mean())}
    # thresholds are chosen on the same OOF scores they are scored on: optimistic by construction

    vecs, X = features(texts)
    clfs = []
    for j in range(L):
        y = Y[:, j]
        clfs.append(LogisticRegression(C=C, max_iter=2000, solver="liblinear").fit(X, y))
    MODEL.write_bytes(pickle.dumps({"labels": labels, "vecs": vecs, "clfs": clfs, "thr": thr,
                                    "trained_on": len(ids), "C": C, "oof": oof}))
    print(json.dumps({"trained_on": len(ids), "labels": L, "features": int(X.shape[1]),
                      "seconds": round(time.time() - t0, 1), "oof_at_per_label_threshold": oof}, indent=1))


def predict(texts, top=8):
    m = pickle.loads(MODEL.read_bytes())
    X = transform(m["vecs"], texts)
    P = np.column_stack([c.predict_proba(X)[:, 1] for c in m["clfs"]])
    out = []
    for i in range(len(texts)):
        order = np.argsort(-P[i])
        above = [(m["labels"][j], round(float(P[i, j]), 3)) for j in order if P[i, j] >= m["thr"][j]]
        ranked = [(m["labels"][j], round(float(P[i, j]), 3)) for j in order[:top]]
        out.append({"tags": above, "top": ranked})
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    pp = sub.add_parser("predict"); pp.add_argument("ids", nargs="*"); pp.add_argument("--text")
    a = ap.parse_args()
    if a.cmd == "train":
        return train()
    if a.text:
        print(json.dumps(predict([a.text])[0], indent=1)); return
    memory = memory_module()
    q = ",".join("?" * len(a.ids))
    rows = {r["id"]: r for r in memory._exec(f"SELECT id, summary, tags FROM memories WHERE id IN ({q})", a.ids)}
    texts = [rows[i]["summary"] or "" for i in a.ids]
    for mid, res in zip(a.ids, predict(texts)):
        t = rows[mid].get("tags") or []
        if isinstance(t, str): t = json.loads(t)
        print(f"\n{mid[:8]}  human tags: {t}")
        print(f"  predicted: {res['tags']}")
        print(f"  top-8:     {res['top']}")


if __name__ == "__main__":
    main()
