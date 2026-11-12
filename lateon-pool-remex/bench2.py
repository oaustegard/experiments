#!/usr/bin/env python3
"""Round 2: rotation seeds on the frontier arms, side-information check across seeds,
and the integrated cut on the library's tree (Ward on the 1 - cos matrix, as
sentence-transformers' HierarchicalTokenPooling builds it) instead of true Ward.

Round 1 showed (SciFact) that one rotation seed moves a pooled 2-bit arm by up to 0.02,
that library pooling (cosine-distance Ward) beat true Ward at pool factor 4 by 0.015,
and that dropping the per-vector side scalar gained +0.017 at 1 bit on one seed.

Same row format, same results_<corpus>.jsonl; skips finished arms.
Prints 'BENCH2 DONE <corpus>'.
"""
from __future__ import annotations

import json, sys, time

import numpy as np

import bench
from bench import D, SIDE_BYTES, WORK, HERE, code, maxsim_top10, ndcg10, st_pool, WardForest

SEEDS = (1, 2, 3)  # haar seeds; seed 42 rht is the default-rotation run from round 1


class CosineWardForest(WardForest):
    """Ward linkage on the condensed 1 - cos matrix (the library's tree), same node
    errors and cuts as WardForest."""

    def __init__(self, T, off):
        import fastcluster
        from scipy.spatial.distance import squareform
        self.T, self.off = T, off
        self.Z, self.mc = [], []
        for i in range(len(off) - 1):
            X = T[off[i] + 1:off[i + 1]].astype(np.float64)
            n = len(X)
            if n < 2:
                self.Z.append(None); self.mc.append(None); continue
            Dm = np.clip(1.0 - X @ X.T, 0.0, 2.0)
            np.fill_diagonal(Dm, 0.0)
            Z = fastcluster.linkage(squareform(Dm, checks=False), method="ward")
            S = np.zeros((2 * n - 1, X.shape[1])); S[:n] = X
            cnt = np.ones(2 * n - 1); err = np.zeros(2 * n - 1)
            for j, (a, b) in enumerate(Z[:, :2].astype(np.int64)):
                S[n + j] = S[a] + S[b]; cnt[n + j] = cnt[a] + cnt[b]
                err[n + j] = max(2.0 * (1.0 - np.linalg.norm(S[n + j]) / cnt[n + j]), err[a], err[b])
            self.Z.append(Z); self.mc.append(err[n:])


def run(corpus):
    W = WORK / corpus
    res = HERE / f"results_{corpus}.jsonl"
    rows = {json.loads(l)["arm"]: json.loads(l) for l in open(res)}
    done = set(rows)
    T0 = np.load(W / "tokens.npy"); off0 = np.load(W / "offsets.npy")
    QT = np.load(W / "q_tokens.npy"); qoff = np.load(W / "q_offsets.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    Db = {int(k): v for k, v in json.load(open(W / "distortion.json")).items()}
    n_docs = len(off0) - 1
    print(f"[{corpus}] round 2, {len(done)} arms done", flush=True)

    def record(arm, T, off, bits, seed=42, rotation=None, side="fp16", **extra):
        if arm in done:
            return
        t0 = time.time()
        if bits is None:
            Ts, mu_norm = T, None
        else:
            Ts, mu_norm = code(T, bits, seed=seed, rotation=rotation, side=side)
        top = maxsim_top10(QT, qoff, Ts, off)
        nd = [ndcg10([dids[j] for j in r], qrels[q]) for r, q in zip(top, qids)]
        tpd = len(T) / n_docs
        side_b = 0 if side == "none" else SIDE_BYTES
        bpv = D * 4 if bits is None else D * bits / 8 + side_b
        row = dict(arm=arm, bits=bits, tokens_per_doc=round(tpd, 2), bytes_per_doc=round(tpd * bpv, 1),
                   bytes_per_doc_codes_only=round(tpd * (D * 4 if bits is None else D * bits / 8), 1),
                   ndcg10=round(float(np.mean(nd)), 4), norm_of_mean=mu_norm, seconds=round(time.time() - t0, 1),
                   perquery=[round(x, 5) for x in nd], round=2, side=side, **extra)
        if seed != 42:
            row.update(seed=seed, rotation=rotation, seed_arm=True)
        with open(res, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        done.add(arm)
        print(f"  {arm:30s} tok/doc {tpd:7.2f}  B/doc {row['bytes_per_doc']:9.1f}  nDCG@10 {row['ndcg10']:.4f}  "
              f"({row['seconds']}s)", flush=True)

    def seeded(base, T, off, bits, side="fp16", **extra):
        for s in SEEDS:
            record(f"{base} haar s{s}", T, off, bits, seed=s, rotation="haar", side=side, **extra)

    # A. unpooled: seeds, and the side-information check across seeds
    for b in (2, 1):
        seeded(f"none {b}-bit", T0, off0, b, family="none")
        seeded(f"none {b}-bit no-side", T0, off0, b, side="none", family="side")

    # B. library pooling at the iso-byte points
    for name, kw, b in (("st pf4", dict(pool_factor=4), 4), ("st pf2", dict(pool_factor=2), 2)):
        T, off = st_pool(T0, off0, kw)
        seeded(f"{name} {b}-bit", T, off, b, family="st")
        record(f"{name} {b}-bit no-side", T, off, b, side="none", family="side")
        del T

    # C. integrated (true Ward) and matched controls: seeds on the frontier points
    forest = WardForest(T0, off0)
    for c, b in ((1.0, 2), (4.0, 3), (2.0, 3), (1.0, 3)):
        tag = f"c{c:g}@{b}"
        T, off = forest.cut(tau=c * Db[b])
        f = forest.factor_for(len(T))
        Tm, offm = forest.cut(f=f)
        seeded(f"adapt {tag} {b}-bit", T, off, b, family="adapt", c=c, target_bits=b)
        seeded(f"match {tag} {b}-bit", Tm, offm, b, family="match", c=c, f=round(f, 4), target_bits=b)
        if (c, b) == (1.0, 2):
            record(f"adapt {tag} {b}-bit no-side", T, off, b, side="none", family="side")
        del T, Tm
    del forest

    # D. the integrated cut on the library's tree
    t0 = time.time()
    cforest = CosineWardForest(T0, off0)
    print(f"[{corpus}] cosine-Ward forest {time.time() - t0:.0f}s", flush=True)
    for f in (2, 4):
        T, off = cforest.cut(f=f)
        record(f"cu pf{f} fp32", T, off, None, family="cu", f=f)  # should track st pf{f}
    for c, b in ((1.0, 2), (2.0, 2), (0.5, 2), (1.0, 3), (2.0, 3), (4.0, 3)):
        tag = f"c{c:g}@{b}"
        T, off = cforest.cut(tau=c * Db[b])
        f = cforest.factor_for(len(T))
        Tm, offm = cforest.cut(f=f)
        record(f"cadapt {tag} {b}-bit", T, off, b, family="cadapt", c=c, target_bits=b, tau=round(c * Db[b], 5))
        record(f"cmatch {tag} {b}-bit", Tm, offm, b, family="cmatch", c=c, f=round(f, 4), target_bits=b,
               target_tokens=len(T))
        if (c, b) in ((1.0, 2), (4.0, 3)):
            record(f"cadapt {tag} fp32", T, off, None, family="cadapt", c=c, target_bits=b)
            seeded(f"cadapt {tag} {b}-bit", T, off, b, family="cadapt", c=c, target_bits=b)
            seeded(f"cmatch {tag} {b}-bit", Tm, offm, b, family="cmatch", c=c, f=round(f, 4), target_bits=b)
        del T, Tm
    print(f"BENCH2 DONE {corpus}", flush=True)


if __name__ == "__main__":
    import torch
    torch.set_num_threads(4)
    args = sys.argv[1:]
    for c in [a for a in args if a in bench.CORPORA] or bench.CORPORA:
        run(c)
