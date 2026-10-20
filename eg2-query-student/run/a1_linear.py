"""A1: a ridge map from an off-the-shelf small model's query vectors into EG2's 768 dimensions.

Usage: python3 -I run/a1_linear.py [--models all-MiniLM-L6-v2 ...] [--train msmarco_train synth_train]

Fits W, b on (student, EG2) pairs for the training files, ridge lambda picked on a 10%
validation slice by mean cosine, then scores each test file: top-10 of the mapped query on
the 2-bit codes against top-10 of the EG2 query on the same codes. Writes
results/a1_<train>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Index, boot_ci, overlap, unit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "out" / "queries"
TESTS = ["real_test", "synth_test"]


def fit(X, Y, lam):
    mx, my = X.mean(0), Y.mean(0)
    Xc = X - mx
    W = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(X.shape[1]), Xc.T @ (Y - my))
    return lambda Z: unit((Z - mx) @ W + my)


def kinds(name):
    return np.array([json.loads(l)["kind"] for l in (Q / f"{name}.jsonl").open()])


def score(ix, P, T, kind):
    ov = overlap(ix.topk(P), ix.topk(T))
    out = {"n": len(ov), "overlap@10": boot_ci(ov), "cos": float((P * T).sum(1).mean())}
    for k in sorted(set(kind)):
        if (kind == k).sum() < len(kind):
            out[f"overlap@10[{k}]"] = boot_ci(ov[kind == k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["paraphrase-MiniLM-L3-v2", "all-MiniLM-L6-v2",
                                                     "bge-small-en-v1.5", "mdbr-leaf-mt"])
    ap.add_argument("--train", nargs="+", default=["msmarco_train"])
    a = ap.parse_args()
    ix = Index()
    tests = [t for t in TESTS if (Q / f"{t}.eg2.npy").exists()]
    Yt = np.concatenate([np.load(Q / f"{t}.eg2.npy") for t in a.train])
    res = {"train": a.train, "n_train": len(Yt), "arms": {}}
    for m in a.models:
        S = ROOT / "out" / "student" / m
        X = np.concatenate([np.load(S / f"{t}.npy") for t in a.train])
        perm = np.random.default_rng(0).permutation(len(X))
        va, tr = perm[:len(X) // 10], perm[len(X) // 10:]
        best = max((float((fit(X[tr], Yt[tr], lam)(X[va]) * Yt[va]).sum(1).mean()), lam)
                   for lam in [1e-4, 1e-3, 1e-2, 1e-1])
        f = fit(X, Yt, best[1])
        r = {"dim": X.shape[1], "lambda": best[1], "val_cos": best[0]}
        for t in (t for t in tests if (S / f"{t}.npy").exists()):
            r[t] = score(ix, f(np.load(S / f"{t}.npy")), np.load(Q / f"{t}.eg2.npy"), kinds(t))
        res["arms"][m] = r
        print(m, json.dumps({k: (v["overlap@10"] if isinstance(v, dict) else v) for k, v in r.items()}),
              flush=True)
    out = ROOT / "results" / f"a1_{'+'.join(a.train)}.json"
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
