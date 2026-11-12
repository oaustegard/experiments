#!/usr/bin/env python3
"""Training-unaware hierarchical token pooling x centered remex on LateOn.

Three pooling families, every one scored after centered remex coding:
  st      sentence-transformers HierarchicalTokenPooling (the library default,
          the sequential baseline from token-pooling-remex): pool_factor 2/4,
          max_tokens 32. Ward on a cosine-distance matrix, first token protected.
  wu      true Ward (fastcluster, Euclidean on the unit tokens), first token
          protected, a uniform factor: k = max(1, round((n-1)/f)) clusters per doc.
  adapt   the integrated arm: the same Ward tree cut where pooling stops costing
          less than the codec. A cluster is kept only if it and all its
          sub-clusters satisfy  E_pool = 2(1 - ||mean of members||) <= c * D_b,
          where E_pool is the mean squared distance of the members to the
          renormalised mean (exact for unit vectors) and D_b is the measured
          mean squared error of centered remex at b bits on the unpooled tokens.
          The bit width sets the threshold, so tokens per document vary.
  match   wu with f chosen so the corpus has the same total tokens as an adapt
          arm: same bits, same bytes, uniform instead of adaptive allocation.

Codec: centered remex (Quantizer(128, b, mean=corpus_mean of the coded tokens)).
Each coded vector carries one fp16 side scalar s = mu.x; for a unit vector it gives
both stored fields exactly (||x-mu||^2 = 1 - 2s + ||mu||^2, mu.(x-mu) = s - ||mu||^2),
so the decode here rebuilds them from the fp16 s, and bytes count it:
  bytes/vector = 128*b/8 + 2   (fp32 arms: 512).
Scoring: brute-force sum-of-MaxSim over every document, queries fp32 unpooled.

Per-arm checkpoint: one JSON line per arm in results_<corpus>.jsonl (per-query
nDCG@10 inside), so a rerun skips finished arms. Prints 'BENCH DONE <corpus>'.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE / ".work"
CORPORA = ("scifact", "nfcorpus")
D = 128
SIDE_BYTES = 2
C_GRID = (0.5, 1.0, 2.0, 4.0)
ADAPT_BITS = (4, 3, 2, 1)


# ----------------------------------------------------------------------------- metrics
def ndcg10(ranked, rel: dict) -> float:
    dcg = sum(rel.get(d, 0) / np.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def maxsim_top10(QT, qoff, T, off):
    starts = off[:-1]
    out = np.empty((len(qoff) - 1, 10), dtype=np.int64)
    for i in range(len(qoff) - 1):
        S = QT[qoff[i]:qoff[i + 1]] @ T.T
        s = np.maximum.reduceat(S, starts, axis=1).sum(0)
        top = np.argpartition(-s, 10)[:10]
        out[i] = top[np.argsort(-s[top])]
    return out


# ----------------------------------------------------------------------------- codec
def code(T, bits, seed=42, rotation=None, side="fp16"):
    """Centered remex round trip. side: 'fp16' (rebuild both stored fields from an
    fp16 mu.x, the counted format), 'exact' (remex's own float32 fields), 'none'
    (replace mu.x by its corpus average: no per-vector side info)."""
    from remex import Quantizer, corpus_mean
    mu = corpus_mean(T)
    q = Quantizer(D, bits=bits, seed=seed, rotation=rotation, mean=mu)
    cv = q.encode(T)
    if side != "exact":
        mu64 = mu.astype(np.float64)
        mm = float(mu64 @ mu64)
        s = T.astype(np.float64) @ mu64
        if side == "fp16":
            s = s.astype(np.float16).astype(np.float64)
        elif side == "none":
            s = np.full_like(s, s.mean())
        cv.mean_dots = (s - mm).astype(np.float32)
        cv.norms = np.sqrt(np.maximum(1.0 - 2.0 * s + mm, 0.0)).astype(np.float32)
        cv._dir_len = {}
    return q.decode(cv).astype(np.float32), float(np.linalg.norm(mu))


def distortion(T, bits, n=200_000, seed=0):
    """Mean squared reconstruction error of the counted codec on a token sample."""
    rng = np.random.default_rng(seed)
    X = T[rng.choice(len(T), size=min(n, len(T)), replace=False)]
    Xh, _ = code(X, bits)
    return float(np.mean(np.sum((X - Xh) ** 2, axis=1)))


# ----------------------------------------------------------------------------- pooling
def st_pool(T, off, kwargs):
    import torch
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling
    p = HierarchicalTokenPooling(**kwargs)
    docs = [torch.from_numpy(T[off[i]:off[i + 1]]) for i in range(len(off) - 1)]
    out = []
    for s in range(0, len(docs), 256):
        out.extend(p.pool(docs[s:s + 256]))
    lens = [len(d) for d in out]
    return (np.concatenate([d.numpy() for d in out]).astype(np.float32),
            np.concatenate([[0], np.cumsum(lens)]).astype(np.int64))


class WardForest:
    """One Ward tree per document over tokens[1:] (the first token is kept as is),
    with each node's pooling error 2(1-||mean||) made monotone up the tree."""

    def __init__(self, T, off):
        import fastcluster
        self.T, self.off = T, off
        self.Z, self.mc = [], []
        for i in range(len(off) - 1):
            X = T[off[i] + 1:off[i + 1]].astype(np.float64)
            n = len(X)
            if n < 2:
                self.Z.append(None); self.mc.append(None); continue
            Z = fastcluster.linkage_vector(X, method="ward")
            S = np.zeros((2 * n - 1, X.shape[1])); S[:n] = X
            cnt = np.ones(2 * n - 1)
            err = np.zeros(2 * n - 1)
            for j, (a, b) in enumerate(Z[:, :2].astype(np.int64)):
                S[n + j] = S[a] + S[b]; cnt[n + j] = cnt[a] + cnt[b]
                err[n + j] = max(2.0 * (1.0 - np.linalg.norm(S[n + j]) / cnt[n + j]), err[a], err[b])
            self.Z.append(Z); self.mc.append(err[n:])  # scipy monocrit: one value per merge

    def _emit(self, i, labels):
        X = self.T[self.off[i] + 1:self.off[i + 1]]
        order = np.argsort(labels, kind="stable")
        _, starts = np.unique(labels[order], return_index=True)
        C = np.add.reduceat(X[order].astype(np.float64), starts, axis=0)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        return np.vstack([self.T[self.off[i]:self.off[i] + 1], C.astype(np.float32)])

    def cut(self, *, tau=None, f=None):
        from scipy.cluster.hierarchy import fcluster
        out = []
        for i in range(len(self.off) - 1):
            Z = self.Z[i]
            if Z is None:
                out.append(self.T[self.off[i]:self.off[i + 1]]); continue
            n = len(Z) + 1
            if tau is not None:
                lab = fcluster(Z, tau, criterion="monocrit", monocrit=self.mc[i])
            else:
                k = max(1, int(round(n / f)))
                lab = fcluster(Z, k, criterion="maxclust")
            out.append(self._emit(i, lab))
        lens = [len(d) for d in out]
        return np.concatenate(out).astype(np.float32), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)

    def factor_for(self, target_tokens):
        """Uniform factor f whose cut has the target total token count (bisection)."""
        ns = np.array([0 if z is None else len(z) + 1 for z in self.Z])
        fixed = len(self.Z) + sum(int(self.off[i + 1] - self.off[i] - 1) for i, z in enumerate(self.Z) if z is None)

        def total(f):
            return fixed + int(np.maximum(1, np.round(ns[ns > 0] / f)).sum())
        lo, hi = 1.0, 64.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if total(mid) > target_tokens:
                lo = mid
            else:
                hi = mid
        return min((lo, hi), key=lambda f: abs(total(f) - target_tokens))


# ----------------------------------------------------------------------------- driver
def run(corpus):
    W = WORK / corpus
    res = HERE / f"results_{corpus}.jsonl"
    done = {json.loads(l)["arm"] for l in open(res)} if res.exists() else set()
    T0 = np.load(W / "tokens.npy"); off0 = np.load(W / "offsets.npy")
    QT = np.load(W / "q_tokens.npy"); qoff = np.load(W / "q_offsets.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    n_docs = len(off0) - 1
    print(f"[{corpus}] {n_docs} docs, {len(T0)} tokens, {len(qids)} queries, {len(done)} arms done", flush=True)

    def record(arm, T, off, bits, **extra):
        if arm in done:
            return
        t0 = time.time()
        mu_norm = None
        if bits is None:
            Ts = T
        else:
            Ts, mu_norm = code(T, bits, seed=extra.pop("seed", 42), rotation=extra.pop("rotation", None),
                               side=extra.pop("side", "fp16"))
        top = maxsim_top10(QT, qoff, Ts, off)
        nd = [ndcg10([dids[j] for j in r], qrels[q]) for r, q in zip(top, qids)]
        tpd = len(T) / n_docs
        bpv = D * 4 if bits is None else D * bits / 8 + SIDE_BYTES
        row = dict(arm=arm, bits=bits, tokens_per_doc=round(tpd, 2), bytes_per_doc=round(tpd * bpv, 1),
                   bytes_per_doc_codes_only=round(tpd * (D * 4 if bits is None else D * bits / 8), 1),
                   ndcg10=round(float(np.mean(nd)), 4), norm_of_mean=mu_norm, seconds=round(time.time() - t0, 1),
                   perquery=[round(x, 5) for x in nd], **extra)
        with open(res, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        done.add(arm)
        print(f"  {arm:26s} tok/doc {tpd:7.2f}  B/doc {row['bytes_per_doc']:9.1f}  nDCG@10 {row['ndcg10']:.4f}  "
              f"({row['seconds']}s)", flush=True)

    # codec resolution, measured once on the unpooled tokens (sets every adapt threshold)
    dfile = W / "distortion.json"
    if dfile.exists():
        Db = {int(k): v for k, v in json.load(open(dfile)).items()}
    else:
        Db = {b: distortion(T0, b) for b in (1, 2, 3, 4)}
        json.dump(Db, open(dfile, "w"))
    print(f"[{corpus}] D_b (mean sq. error per token, centered remex + fp16 side) {Db}", flush=True)

    # A. unpooled
    record("none fp32", T0, off0, None, family="none")
    for b in (4, 3, 2, 1):
        record(f"none {b}-bit", T0, off0, b, family="none")
    for b in (2, 1):
        record(f"none {b}-bit haar s1", T0, off0, b, family="none", seed=1, rotation="haar", seed_arm=True)
        record(f"none {b}-bit exact-side", T0, off0, b, family="side", side="exact")
        record(f"none {b}-bit no-side", T0, off0, b, family="side", side="none")

    # B. library pooling (sequential baseline)
    for name, kw in (("st pf2", dict(pool_factor=2)), ("st pf4", dict(pool_factor=4)), ("st mt32", dict(max_tokens=32))):
        if all(f"{name} {s}" in done for s in ("fp32", "4-bit", "2-bit", "1-bit")):
            continue
        T, off = st_pool(T0, off0, kw)
        record(f"{name} fp32", T, off, None, family="st")
        for b in (4, 2, 1):
            record(f"{name} {b}-bit", T, off, b, family="st")
        del T

    # C-F. true Ward: uniform, adaptive (integrated), matched-uniform controls
    t0 = time.time()
    forest = WardForest(T0, off0)
    print(f"[{corpus}] Ward forest {time.time() - t0:.0f}s", flush=True)
    for f in (2, 4):
        T, off = forest.cut(f=f)
        record(f"wu pf{f} fp32", T, off, None, family="wu", f=f)
        record(f"wu pf{f} 4-bit", T, off, 4, family="wu", f=f)
    for b in ADAPT_BITS:
        for c in C_GRID:
            tau = c * Db[b]
            tag = f"c{c:g}@{b}"
            T, off = forest.cut(tau=tau)
            n_tok = len(T)
            record(f"adapt {tag} {b}-bit", T, off, b, family="adapt", c=c, tau=round(tau, 5), target_bits=b)
            if c in (1.0, 4.0):
                record(f"adapt {tag} fp32", T, off, None, family="adapt", c=c, tau=round(tau, 5), target_bits=b)
            seeds = b in (1, 2) and c in (1.0, 2.0)
            if seeds:
                record(f"adapt {tag} {b}-bit haar s1", T, off, b, family="adapt", c=c, tau=round(tau, 5),
                       target_bits=b, seed=1, rotation="haar", seed_arm=True)
            f = forest.factor_for(n_tok)
            Tm, offm = forest.cut(f=f)
            record(f"match {tag} {b}-bit", Tm, offm, b, family="match", c=c, f=round(f, 4), target_bits=b,
                   target_tokens=n_tok)
            if seeds:
                record(f"match {tag} {b}-bit haar s1", Tm, offm, b, family="match", c=c, f=round(f, 4),
                       target_bits=b, target_tokens=n_tok, seed=1, rotation="haar", seed_arm=True)
            del T, Tm
    print(f"BENCH DONE {corpus}", flush=True)


if __name__ == "__main__":
    import torch
    torch.set_num_threads(4)
    args = sys.argv[1:]
    for c in [a for a in args if a in CORPORA] or CORPORA:
        run(c)
