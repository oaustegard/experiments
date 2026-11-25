"""Simulated listeners against the taste loop in web/taste.js.

Each listener has a hidden utility over candidate phrases. For T decisions it is shown a random
harvested candidate set; the arm scores the eight, samples one by softmax(score / tau) with an
exploration floor, and the listener reacts: like (y=1, w=1) above its 75th utility percentile, skip
(y=0, w=1) below its 35th, otherwise it lets the phrase play (y=1, w=W_PASSIVE). 10% of reactions are
random. The arm refits a weighted L2 logistic model after every reaction, as the page does.

Metric: the percentile of the played candidate's utility within its own set (0.5 = random choice,
1.0 = always the listener's favourite of the eight; tied candidates share a mid-rank).

Arms: random; feats (the 31 counted features); desc (Ettin vector of the description); code (Ettin
vector of the first bar's Strudel code); desc32 (desc projected on its top 32 principal components).
Listener families: linear (random weights on the counted features, the feature arm's own model class),
categorical (a liked subset of modes and sections), roots (likes chords on three random roots, a
property visible in the text but in no counted feature), interaction (busy drums only under a sparse
lead). Tau and lambda are chosen per arm on dev listeners (seed 1) and reported on test listeners (seed 2).

    python3 run/sim.py results/harvest.jsonl results/vectors.npz results/sim.json
"""
import json
import sys
from itertools import product

import numpy as np

W_PASSIVE, EPS = 0.3, 0.05          # web/taste.js
T, N_LIST = 60, 40                  # decisions per listener, listeners per family and split
TAUS, LAMBDAS = (0.25, 0.5, 1.0), (0.1, 1.0, 10.0)
ROOTS = ["C", "C♯", "D♭", "D", "D♯", "E♭", "E", "F", "F♯", "G♭", "G", "G♯", "A♭", "A", "A♯", "B♭", "B"]
PC = {"C": 0, "C♯": 1, "D♭": 1, "D": 2, "D♯": 3, "E♭": 3, "E": 4, "F": 5, "F♯": 6, "G♭": 6, "G": 7, "G♯": 8, "A♭": 8, "A": 9, "A♯": 10, "B♭": 10, "B": 11}


def root_pc(sym):
    for r in sorted(ROOTS, key=len, reverse=True):
        if sym.startswith(r):
            return PC[r]
    return None


def load(harvest, vectors):
    pts = [json.loads(l) for l in open(harvest) if l.strip()]
    z = np.load(vectors)
    F = z["feats"].astype(np.float64)                       # [S, 8, 31]
    reps = {"feats": F, "desc": z["desc"].astype(np.float64), "code": z["code"].astype(np.float64)}
    flat = reps["desc"].reshape(-1, reps["desc"].shape[-1])
    mu = flat.mean(0)
    _, _, vt = np.linalg.svd(flat - mu, full_matrices=False)
    reps["desc32"] = (reps["desc"] - mu) @ vt[:32].T
    for k, X in reps.items():                               # standardise over the pool (unlabelled statistics)
        f = X.reshape(-1, X.shape[-1])
        sd = f.std(0)
        reps[k] = (X - f.mean(0)) / np.where(sd > 1e-6, sd, 1)
    roots = np.array([[[root_pc(s) for s in c["meta"]["chords"]] for c in p["info"]] for p in pts], dtype=object)
    meta = [[c["meta"] for c in p["info"]] for p in pts]
    return pts, F, reps, roots, meta


MODES = ["ionian", "dorian", "phrygian", "lydian", "mixolydian", "aeolian"]
SECTIONS = ["intro", "groove", "lift", "build", "drop", "breakdown"]


def listener(family, rng, F, roots, meta):
    """Utility [S, 8] for one random listener of a family."""
    S = F.shape[0]
    num = F[..., :19]
    zf = (num - num.reshape(-1, 19).mean(0)) / (num.reshape(-1, 19).std(0) + 1e-9)
    if family == "linear":
        return zf @ rng.normal(size=19)
    if family == "categorical":
        ms, ss = set(rng.choice(MODES, 2, replace=False)), set(rng.choice(SECTIONS, 2, replace=False))
        tilt = rng.choice([-1, 1])
        return np.array([[(m["key"].split()[-1] in ms) + (m["section"] in ss) + 0.5 * (m["tilt"] == tilt) for m in row] for row in meta], float)
    if family == "roots":
        liked = set(rng.choice(12, 3, replace=False).tolist())
        return np.array([[sum(r in liked for r in rs) for rs in row] for row in roots], float)
    if family == "interaction":
        drums, lead = zf[..., 2] + zf[..., 4], zf[..., 9]
        return drums * (lead < -0.3) - drums * (lead > 0.3)
    raise ValueError(family)


def fit(w, b, Z, y, wt, lam, iters=150):
    W = wt.sum() or 1.0
    # step 1/L: the loss is (sum w_i CE_i + lam/2 |w|^2) / W, whose curvature is at most max|z|^2/4 + lam/W
    lr = 1.0 / (0.25 * float((Z ** 2).sum(1).max()) + lam / W + 0.25)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        e = (p - y) * wt
        w = w - lr * (Z.T @ e + lam * w) / W
        b = b - lr * e.sum() / W
    return w, b


def pctl(u):  # mid-rank percentile of each candidate within its set
    r = np.array([(u < x).sum() + 0.5 * ((u == x).sum() - 1) for x in u])
    return r / (len(u) - 1)


def run(arm, X, U, rng, tau, lam):
    S = U.shape[0]
    lo, hi = np.quantile(U, 0.35), np.quantile(U, 0.75)
    order = rng.permutation(S)[:T]
    d = X.shape[-1] if X is not None else 0
    w, b = np.zeros(d), 0.0
    Z, y, wt, got, liked = [], [], [], [], []
    for t, s in enumerate(order):
        if arm == "random":
            k = rng.integers(8)
        else:
            sc = X[s] @ w + b
            e = np.exp((sc - sc.max()) / tau)
            p = e / e.sum() * (1 - EPS) + EPS / 8
            k = rng.choice(8, p=p)
        u = U[s, k]
        if rng.random() < 0.1:
            r = rng.choice(["like", "skip", "pass"])
        else:
            r = "like" if u >= hi else "skip" if u <= lo else "pass"
        got.append(pctl(U[s])[k]); liked.append(r == "like")
        if arm != "random":
            Z.append(X[s, k]); y.append(0.0 if r == "skip" else 1.0); wt.append(W_PASSIVE if r == "pass" else 1.0)
            w, b = fit(w, b, np.array(Z), np.array(y), np.array(wt), lam)
    return np.array(got), np.array(liked)


def evaluate(arms, reps, F, roots, meta, seed, params):
    out = {}
    for fam in ("linear", "categorical", "roots", "interaction"):
        rng = np.random.default_rng(seed)
        users = [listener(fam, rng, F, roots, meta) for _ in range(N_LIST)]
        for arm in arms:
            tau, lam = params.get((arm, fam), params.get(arm, (0.5, 1.0)))
            g = np.stack([run(arm, reps.get(arm), U, np.random.default_rng(seed * 1000 + i), tau, lam)[0] for i, U in enumerate(users)])
            out[(arm, fam)] = g  # [listeners, T]
    return out


DATA = None


def _dev(job):
    arm, tau, lam = job
    pts, F, reps, roots, meta = DATA
    r = evaluate([arm], reps, F, roots, meta, 1, {arm: (tau, lam)})
    return job, float(np.mean([v.mean() for v in r.values()]))


def _test(arm_params):
    arm, params = arm_params
    pts, F, reps, roots, meta = DATA
    return evaluate([arm], reps, F, roots, meta, 2, params)


def main():
    global DATA
    from concurrent.futures import ProcessPoolExecutor
    harvest, vectors, dst = sys.argv[1:4]
    DATA = load(harvest, vectors)
    pts, F, reps, roots, meta = DATA
    arms = ["random", "feats", "desc", "code", "desc32"]
    # choose tau and lambda per arm on dev listeners, pooled over families
    params, dev = {}, {}
    with ProcessPoolExecutor(4) as ex:  # fork start method: workers inherit DATA
        for (arm, tau, lam), m in ex.map(_dev, [(a, t, l) for a in arms[1:] for t, l in product(TAUS, LAMBDAS)]):
            dev[f"{arm} tau={tau} lambda={lam}"] = round(m, 4)
            if arm not in params or m > params[arm][0]:
                params[arm] = (m, tau, lam)
        params = {a: v[1:] for a, v in params.items()}
        print("chosen on dev:", params, flush=True)
        res = {}
        for r in ex.map(_test, [(a, params) for a in arms]):
            res.update(r)
    table = {}
    for (arm, fam), g in res.items():
        se = g.mean(1).std(ddof=1) / np.sqrt(g.shape[0])
        table.setdefault(fam, {})[arm] = {"all": round(float(g.mean()), 4), "se": round(float(se), 4),
                                         "t1_20": round(float(g[:, :20].mean()), 4), "t41_60": round(float(g[:, 40:].mean()), 4)}
    ceiling = {}
    for fam in ("linear", "categorical", "roots", "interaction"):  # best possible: always the favourite of the eight
        rng = np.random.default_rng(2)
        users = [listener(fam, rng, F, roots, meta) for _ in range(N_LIST)]
        ceiling[fam] = round(float(np.mean([[pctl(U[s]).max() for s in range(U.shape[0])] for U in users])), 4)
    dup = int(sum(len({c["text"] for c in p["info"]}) < 8 for p in pts))
    json.dump({"T": T, "listeners": N_LIST, "sets": len(pts), "sets_with_duplicate_candidates": dup, "params": {k: list(v) for k, v in params.items()},
               "dev": dev, "test": table, "ceiling": ceiling}, open(dst, "w"), indent=1, ensure_ascii=False)
    for fam, row in table.items():
        print(fam, "ceiling", ceiling[fam], {a: (v["all"], v["t41_60"]) for a, v in row.items()})


if __name__ == "__main__":
    main()
