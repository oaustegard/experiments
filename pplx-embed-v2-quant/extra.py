#!/usr/bin/env python3
"""Follow-ups to bench.py on SciFact.

1. Centered remax with a float query, done right. bench.py's "remax ... (asym) centered"
   rotated the UNcentered query against sign codes of (x - mu). That would be harmless
   with exact residuals (q.mu is constant per query token), but a sign code carries no
   residual norm, so q.s = mu.s + (q - mu).s and the mu.s term (||mu|| ~ 0.94 against
   residual norms ~ 0.3) is noise that swamps the signal: nDCG@10 0.06. Here the query
   is centered as well: score (q - mu) R . s.
2. Centered remex nested: encode once at 8 bits centered, decode at 4/2/1 (does the
   Matryoshka bit nesting cost as much centered as uncentered?).
3. Seed floor for centered remex 1-bit and 2-bit (seed 1), and centered remax k=1 sym seed 1.
4. Paired bootstrap of the leading low-bit arms against PLAID at the same bit width.

Reads results_scifact_perquery.npz for the bench.py arms; writes results_extra_scifact.json.
"""
from __future__ import annotations

import json, sys, warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import importlib.util  # noqa: E402
# load this directory's bench.py under another name: it imports mxbai-edge-remex-quant's
# bench.py as "bench", which a plain `import bench` here would shadow
_spec = importlib.util.spec_from_file_location("pplx_bench", HERE / "bench.py")
pb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pb)
late_rank, D_TOK, paired_bootstrap, signs_pm1 = pb.late_rank, pb.D_TOK, pb.paired_bootstrap, pb.signs_pm1
ndcg10, recall100 = pb.ndcg10, pb.recall100
from remex import Quantizer, corpus_mean, AnisotropyWarning  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

warnings.filterwarnings("ignore", category=AnisotropyWarning)


def main(corpus="scifact"):
    W = HERE / ".work" / corpus
    T = np.load(W / "mv_tokens.npy"); off = np.load(W / "mv_offsets.npy")
    QT = np.load(W / "q_mv_tokens.npy"); qoff = np.load(W / "q_mv_offsets.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    tpd = len(T) / (len(off) - 1)
    mu = corpus_mean(T)
    pq = dict(np.load(HERE / f"results_{corpus}_perquery.npz"))
    ref = pq["fp32"]
    rows = []

    def score(top):
        ranked = [[dids[j] for j in r] for r in top]
        return (np.array([ndcg10(r, qrels[q]) for r, q in zip(ranked, qids)]),
                float(np.mean([recall100(r, qrels[q]) for r, q in zip(ranked, qids)])))

    def record(name, top, bytes_doc):
        nd, r100 = score(top)
        pq[name] = nd
        m, lo, hi, w, l = paired_bootstrap(nd - ref)
        row = dict(arm=name, bytes_per_doc=round(bytes_doc, 1), ndcg10=round(float(nd.mean()), 4),
                   recall100=round(r100, 4), delta=round(m, 4), ci=[round(lo, 4), round(hi, 4)])
        rows.append(row)
        print(f"  {name:40s} {bytes_doc:>9.1f} B/doc  nDCG@10 {row['ndcg10']:.4f} R@100 {r100:.4f}  "
              f"Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)

    for k, seed in [(1, 0), (2, 0), (4, 0), (1, 1)]:
        rq = StackedSignBitQuantizer(D_TOK, k, seed=seed, rotation="rht")
        S = signs_pm1(rq.encode(T - mu), k * D_TOK)
        R = rq._rotation_matrix
        record(f"remax k={k} (asym) centered q-centered" + (f" s{seed}" if seed else ""),
               late_rank(QT, qoff, S, off, lambda x, R=R: (x - mu) @ R), tpd * D_TOK * k / 8)
    rq = StackedSignBitQuantizer(D_TOK, 1, seed=1, rotation="rht")
    S = signs_pm1(rq.encode(T - mu), D_TOK); R = rq._rotation_matrix
    record("remax k=1 (sym) centered s1",
           late_rank(QT, qoff, S, off, lambda x, R=R: np.where((x - mu) @ R > 0, 1.0, -1.0).astype(np.float32)),
           tpd * D_TOK / 8)

    q8 = Quantizer(D_TOK, bits=8, seed=0, rotation="rht", mean=mu)
    c8 = q8.encode(T)
    for bits in (8, 4, 2, 1):
        Td = q8.decode(c8, precision=bits) if bits < 8 else q8.decode(c8)
        record(f"remex {bits}-bit centered nested-from-8", late_rank(QT, qoff, Td, off), tpd * D_TOK * bits / 8)
    del c8
    for bits in (2, 1):
        q = Quantizer(D_TOK, bits=bits, seed=1, rotation="rht", mean=mu)
        record(f"remex {bits}-bit centered s1", late_rank(QT, qoff, q.decode(q.encode(T)), off), tpd * D_TOK * bits / 8)

    pairs = [("remex 2-bit centered", "plaid 2-bit"), ("remex 1-bit centered", "plaid 1-bit"),
             ("remex 4-bit centered", "plaid 4-bit"), ("remex 1-bit centered", "remax k=1 (sym) centered"),
             ("remax k=1 (asym) centered q-centered", "remex 1-bit centered"),
             ("remex 2-bit centered", "trunc 64d + remex 4-bit"), ("remex 1-bit centered", "remex 1-bit"),
             ("remex 2-bit direct s0", "remex 2-bit")]
    comps = []
    print("  paired (a - b):", flush=True)
    for a, b in pairs:
        m, lo, hi, w, l = paired_bootstrap(pq[a] - pq[b])
        comps.append(dict(a=a, b=b, delta=round(m, 4), ci=[round(lo, 4), round(hi, 4)], wins=w, losses=l))
        print(f"    {a:38s} - {b:30s} {m:+.4f} [{lo:+.4f},{hi:+.4f}] w/l {w}/{l}", flush=True)
    json.dump(dict(rows=rows, paired=comps), open(HERE / f"results_extra_{corpus}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
