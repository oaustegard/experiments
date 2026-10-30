"""Negative-margin direction drop (RRC support selection, arXiv 2610.37707) on
BEIR Quora / EmbeddingGemma 2 (first 100k docs, 885 queries with in-corpus qrels).

Five-fold cross-fitting over queries: margins are estimated on four folds and
the fixed map is applied to the held-out fold and the corpus. Methods at each k:
MRL prefix truncation, PCA, PCA+drop (support selection), PCA+RRC, and the same
for whitening (gamma=1) and a mid prior (gamma=0.5). nDCG@10, self-id excluded.
"""
import glob, json, sys
import numpy as np

D = sys.argv[1]
SEEDS = [1999, 5, 2026]
KS = [768, 512, 256, 128]
GAMMAS = {"PCA": 0.0, "g0.5": 0.5, "Whiten": 1.0}

ids, X = [], []
for f in sorted(glob.glob(f"{D}/shards/docs_*.npy")):
    X.append(np.load(f).astype(np.float32))
    ids += json.load(open(f.replace(".npy", ".ids.json")))
X = np.concatenate(X)
Q = np.load(f"{D}/q_dense.npy").astype(np.float32)
qids = json.load(open(f"{D}/query_ids.json"))
qrels = json.load(open(f"{D}/qrels.json"))
pos_of = {d: i for i, d in enumerate(ids)}
rel = {}
for qi, q in enumerate(qids):
    r = [pos_of[d] for d in qrels.get(q, {}) if d in pos_of and d != q]
    if r:
        rel[qi] = r
keep = np.array(sorted(rel))
Q = Q[keep]
rel = [rel[i] for i in keep]
self_idx = [pos_of.get(qids[i], -1) for i in keep]
nq = len(keep)
print(f"docs {len(X)}  queries {nq}  positives {sum(map(len, rel))}", flush=True)

# Corpus spectral basis (label-free, shared by all methods)
mu = X.mean(0)
Xc = X - mu
Xc64 = Xc.astype(np.float64)
lam, V = np.linalg.eigh(Xc64.T @ Xc64 / len(X))
V = V.astype(np.float32)
order = np.argsort(lam)[::-1]
lam, V = lam[order], V[:, order]
print('eig min/max', lam.min(), lam.max(), flush=True)
lam = np.maximum(lam, 1e-8 * lam[0])
Zd = Xc @ V                      # all 768 spectral coords, centred
Zq = (Q - mu) @ V

DISC = 1 / np.log2(np.arange(2, 12))


def ndcg10(S, qrows):
    out = []
    for r, qi in enumerate(qrows):
        s = S[r]
        if self_idx[qi] >= 0:
            s[self_idx[qi]] = -np.inf
        top = np.argpartition(-s, 10)[:10]
        top = top[np.argsort(-s[top])]
        R = set(rel[qi])
        dcg = sum(DISC[j] for j, d in enumerate(top) if d in R)
        idcg = DISC[: min(len(R), 10)].sum()
        out.append(dcg / idcg)
    return np.array(out)


def scores(Aq, Ad):
    Aq = Aq / np.linalg.norm(Aq, axis=1, keepdims=True)
    Ad = Ad / np.linalg.norm(Ad, axis=1, keepdims=True)
    return Aq @ Ad.T


# Label-free top-200 pools from full-dim embeddings (for hard negatives)
S_full = Q @ X.T
for r in range(nq):
    if self_idx[r] >= 0:
        S_full[r, self_idx[r]] = -np.inf
pools = np.argpartition(-S_full, 200, axis=1)[:, :200]
base_full = ndcg10(Q @ X.T, range(nq))
print(f"Full dim 768: {100*base_full.mean():.2f}", flush=True)


def margins(train, rng, k):
    """Query-balanced margin mean/var per direction over the leading k."""
    M1 = np.zeros(k); M2 = np.zeros(k)
    for qi in train:
        R = set(rel[qi])
        pool = [d for d in pools[qi] if d not in R and d != self_idx[qi]]
        zs = []
        for p in rel[qi]:
            if rng.random() < 0.8 and pool:          # majority from pool
                n = pool[rng.integers(len(pool))]
            else:
                while True:
                    n = int(rng.integers(len(X)))
                    if n not in R and n != self_idx[qi]:
                        break
            zs.append(Zq[qi, :k] * (Zd[p, :k] - Zd[n, :k]))
        zs = np.array(zs)
        M1 += zs.mean(0); M2 += (zs ** 2).mean(0)
    m = M1 / len(train); v = M2 / len(train) - m ** 2
    return m, v


def weights(prior, m, v, lam_k, mode):
    if mode == "prior":
        return prior
    e = np.maximum(m, 0) / v
    sup = e > 0
    if mode == "drop":
        return np.where(sup, prior, 0.0)
    if mode == "direct":
        return e
    # RRC: residual of log e against a fitted power law in log lam, on support
    L = np.log(lam_k[sup]); y = np.log(e[sup])
    A = np.c_[np.ones_like(L), -L]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    u = y - A @ coef
    w = np.zeros_like(prior)
    w[sup] = prior[sup] * np.exp(u)
    return w


results = {}  # (method, k) -> list of per-seed means
counts = {}
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(nq), 5)
    per = {}
    for f, test in enumerate(folds):
        train = np.concatenate([folds[g] for g in range(5) if g != f])
        for k in KS:
            m, v = margins(train, rng, k)
            lam_k = lam[:k]
            # MRL prefix truncation (raw coords, no centring)
            if f == 0 or True:
                per.setdefault(("MRL prefix", k), []).append(
                    ndcg10(scores(Q[test][:, :k], X[:, :k]), test))
            for gname, g in GAMMAS.items():
                prior = lam_k ** -g
                for mode in ["prior", "drop", "rrc"]:
                    w = weights(prior, m, v, lam_k, mode)
                    s = np.sqrt(w)
                    name = gname if mode == "prior" else f"{gname}+{mode}"
                    per.setdefault((name, k), []).append(
                        ndcg10(scores(Zq[test, :k] * s, Zd[:, :k] * s), test))
                    if mode == "drop" and gname == "PCA":
                        counts.setdefault(k, []).append(int((w > 0).sum()))
            # Direct e (label-fitted trend), and full-dim uncentred-free reference
            w = weights(None, m, v, lam_k, "direct"); s = np.sqrt(w)
            per.setdefault(("Direct e", k), []).append(
                ndcg10(scores(Zq[test, :k] * s, Zd[:, :k] * s), test))
        print(f"seed {seed} fold {f} done", flush=True)
    for key, arrs in per.items():
        results.setdefault(key, []).append(np.concatenate(arrs))

# Report: mean over seeds of pooled per-query nDCG; paired t vs prior
from scipy.stats import ttest_rel
names = ["MRL prefix", "PCA", "PCA+drop", "PCA+rrc", "g0.5", "g0.5+drop",
         "g0.5+rrc", "Whiten", "Whiten+drop", "Whiten+rrc", "Direct e"]
print(f"\nnDCG@10 x100, {nq} queries, mean of {len(SEEDS)} seeds; full dim = {100*base_full.mean():.2f}")
print("method".ljust(14) + "".join(f"k={k}".rjust(16) for k in KS))
for n in names:
    row = n.ljust(14)
    for k in KS:
        a = np.mean(results[(n, k)], axis=0)
        cell = f"{100*a.mean():.2f}"
        base = n.split("+")[0]
        if "+" in n:
            b = np.mean(results[(base, k)], axis=0)
            p = ttest_rel(a, b).pvalue
            cell += f" ({100*(a.mean()-b.mean()):+.2f}{'*' if p < .05 else ''})"
        row += cell.rjust(16)
    print(row)
print("\ndirections kept by drop (PCA), mean over folds/seeds:",
      {k: f"{np.mean(c):.0f}/{k}" for k, c in counts.items()})
