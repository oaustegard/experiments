"""Per-member share of posts about AI, from 600 LLM-labelled posts and a classifier on Ettin-17m vectors.

Inputs (gitignored, data/): sample.json + labels0-5.json (label.sh), the 100,000-post snapshot and its fp32
Ettin vectors from ../answer-search-1bit/data/. Writes results/model.json (aggregate checks, committed) and
data/members.json (the per-member table: kept out of git, since it ranks named people).
"""
import gzip, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "answer-search-1bit" / "data"
buf = (SRC / "snapshot.bin").read_bytes(); hl = int.from_bytes(buf[4:8], "little"); h = json.loads(buf[8:8 + hl])
sec = lambda k: buf[8 + hl + h["sections"][k][0]: 8 + hl + h["sections"][k][0] + h["sections"][k][1]]
c = json.loads(gzip.decompress(sec("posts")))
X = np.load(SRC / "X.npy")
ids = json.loads((HERE / "data" / "sample.json").read_text())["ids"]
y = np.array(sum((json.loads((HERE / "data" / f"labels{b}.json").read_text()) for b in range(6)), []))
assert len(y) == len(ids) == 600

out = {"labelled": 600, "labelled_ai_share": float(y.mean())}
cv = StratifiedKFold(5, shuffle=True, random_state=0)
best = None
for C in (0.3, 1, 3, 10, 30):
    p = cross_val_predict(LogisticRegression(C=C, max_iter=2000), X[ids], y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    out[f"cv_auc_C{C}"] = round(auc, 4)
    if best is None or auc > best[1]:
        best = (C, auc, p)
C, auc, p = best
out.update(C=C, cv_auc=round(auc, 4), cv_mean_pred=round(float(p.mean()), 4), cv_acc_at_0_5=round(float(((p > .5) == y).mean()), 4))
m = LogisticRegression(C=C, max_iter=2000).fit(X[ids], y)
P = m.predict_proba(X)[:, 1]
out["corpus_mean_pred"] = round(float(P.mean()), 4)

members = {did: hd for did, hd in json.loads(Path(sys.argv[1]).read_text())}
words = np.array([len(t.split()) for t in c["texts"]])
rng = np.random.default_rng(0)
by = {}
for i, (a, _, _) in enumerate(c["posts"]):
    by.setdefault(a, []).append(i)
table = []
for a, ix in by.items():
    handle, did = c["authors"][a]
    if did not in members:
        continue                                   # already off the list
    ix = np.array(ix); pa = P[ix]
    long = ix[words[ix] >= 15]
    boot = [pa[rng.integers(0, len(pa), len(pa))].mean() for _ in range(1000)]
    table.append({"handle": members[did], "posts": len(ix), "ai_share": round(float(pa.mean()), 3),
                  "ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                  "long_posts": int(len(long)), "ai_share_long": round(float(P[long].mean()), 3) if len(long) else None})
table.sort(key=lambda r: r["ai_share"])
out["members_with_posts"] = len(table)
out["members_without_posts"] = len(members) - len(table)
(HERE / "results" / "model.json").write_text(json.dumps(out, indent=1))
(HERE / "data" / "members.json").write_text(json.dumps(table, indent=1))
silent = sorted(hd for did, hd in members.items() if did not in {c["authors"][a][1] for a in by})
(HERE / "data" / "silent.json").write_text(json.dumps(silent, indent=1))
print(json.dumps(out, indent=1))
