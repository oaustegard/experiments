#!/usr/bin/env python3
"""NanoBEIR nDCG@10 for every (query model, document model) pair, plus token alignment and a mapped arm.

Cross pairs score query tokens of one size against document tokens of another with plain MaxSim.
That is only meaningful if the two token heads share a space, which `align` measures directly:
all sizes share the tokenizer, the [D] marker and the skiplist, so row i of model A's document
tokens and row i of model B's embed the same token in the same context. Cosine near 1 means a
shared space; near 0 means unrelated ones.

The mapped arm asks whether a cheap fix closes the gap: an orthogonal Procrustes map from the
small model's 128-d token space into the big one's, fit on document tokens of the other 12 sets
(leave-one-set-out, 200k token pairs) and applied to the small model's query tokens.

Usage: evaluate.py [models...]   writes results/ndcg.json. MCB_DEVICE=cuda to run on a GPU.
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode import ENC, SETS

RES = Path(os.environ.get("MCB_RES", Path(__file__).resolve().parents[1] / "results"))
DEV = os.environ.get("MCB_DEVICE", "cpu")


def load(s, name, kind):
    z = np.load(ENC / s / f"{name}_{kind}.npz")
    return torch.from_numpy(z["tok"].astype(np.float32)).to(DEV), z["off"]


def maxsim_scores(qt, qoff, dt, doff):
    """(n_q, n_docs) MaxSim scores."""
    nq, nd = len(qoff) - 1, len(doff) - 1
    seg = torch.from_numpy(np.repeat(np.arange(nd), np.diff(doff))).to(DEV)
    out = torch.empty(nq, nd, device=DEV)
    for i in range(nq):
        s = qt[qoff[i]:qoff[i + 1]] @ dt.T
        m = torch.full((s.shape[0], nd), -1e9, device=DEV).scatter_reduce(1, seg.expand(s.shape[0], -1), s, "amax")
        out[i] = m.sum(0)
    return out


def ndcg10(scores, meta):
    dids, qids, rel = meta["dids"], meta["qids"], meta["rel"]
    top = torch.topk(scores, 10, dim=1).indices.cpu().numpy()
    vals = []
    for i, q in enumerate(qids):
        r = rel[q]
        dcg = sum(1 / np.log2(k + 2) for k, j in enumerate(top[i]) if dids[j] in r)
        idcg = sum(1 / np.log2(k + 2) for k in range(min(10, len(r))))
        vals.append(dcg / idcg)
    return vals


def procrustes(A, B):
    """Orthogonal W minimizing ||A W - B||."""
    u, _, vt = torch.linalg.svd(A.T @ B)
    return u @ vt


def main(models):
    torch.manual_seed(0)
    big = models[-1]
    out = {"per_set": {}, "per_query": {}, "align": {}}
    # token pairs per set for alignment and for fitting maps
    pairs = {}
    for s in SETS:
        D = {m: load(s, m, "d") for m in models}
        for m in models:
            assert (D[m][1] == D[big][1]).all(), f"{s}: token rows differ between {m} and {big}"
        idx = torch.randperm(D[big][0].shape[0], device=DEV)[:20000]
        pairs[s] = {m: D[m][0][idx] for m in models}
        out["align"][s] = {f"{a}~{b}": float((pairs[s][a] * pairs[s][b]).sum(1).mean())
                           for i, a in enumerate(models) for b in models[i + 1:]}
    out["align_mean"] = {k: float(np.mean([out["align"][s][k] for s in SETS])) for k in out["align"][SETS[0]]}
    print("align", out["align_mean"], flush=True)

    for s in SETS:
        meta = json.load(open(ENC / s / "meta.json"))
        out["per_set"][s] = {}
        D = {m: load(s, m, "d") for m in models}
        Q = {m: load(s, m, "q") for m in models}
        arms = {f"{qm}>{dm}": (Q[qm], D[dm]) for qm in models for dm in models}
        for small in models[:-1]:
            A = torch.cat([pairs[o][small] for o in SETS if o != s])
            B = torch.cat([pairs[o][big] for o in SETS if o != s])
            Wm = procrustes(A, B)
            qt, qoff = Q[small]
            arms[f"{small}*map>{big}"] = ((torch.nn.functional.normalize(qt @ Wm, dim=-1), qoff), D[big])
        for k, (q, d) in arms.items():
            v = ndcg10(maxsim_scores(*q, *d), meta)
            out["per_set"][s][k] = float(np.mean(v))
            out["per_query"].setdefault(k, []).extend(v)
        print(s, {k: round(v, 3) for k, v in out["per_set"][s].items()}, flush=True)
    keys = out["per_set"][SETS[0]].keys()
    out["mean"] = {k: float(np.mean([out["per_set"][s][k] for s in SETS])) for k in keys}
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / "ndcg.json", "w"), indent=1)
    print(json.dumps(out["mean"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ["1.5M", "27M", "115M"])
