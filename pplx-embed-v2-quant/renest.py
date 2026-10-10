#!/usr/bin/env python3
"""Does re-solving the centred length per precision repair remex's nested centred decode?

remex centred mode (remex 1.1.0, core.py `_centred_lengths`) stores, per vector, the
length m that puts ||mu + m*u_hat|| == ||x|| for the FULL-precision direction u_hat.
`decode(c, precision=p)` reuses that m with the p-bit direction u_p, so ||mu + m*u_p||
!= ||x||; with ||mu|| ~ 0.94 the solve is sensitive to mu.u, and extra.py measured
1-bit nested-from-8 at nDCG@10 0.527 against 0.765 for a direct 1-bit centred encode.

The fix needs no stored bytes: ||x|| is recoverable from the full-precision decode, so
re-solve m for u_p:  m_p = -b + sqrt(b^2 - (||mu||^2 - ||x||^2)),  b = mu.u_p,
falling back to the stored m where no positive root exists (as remex does).

Writes results_renest_<corpus>.json.
"""
from __future__ import annotations

import importlib.util, json, sys, warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pplx_bench", HERE / "bench.py")
pb = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pb)
from remex import Quantizer, corpus_mean, AnisotropyWarning  # noqa: E402

warnings.filterwarnings("ignore", category=AnisotropyWarning)


def resolve_lengths(Xp, mu, xnorm):
    """Re-solve m for the p-bit direction so ||mu + m*u_p|| == ||x||."""
    d = (Xp - mu).astype(np.float64)
    stored = np.linalg.norm(d, axis=1)
    u = d / np.maximum(stored, 1e-12)[:, None]
    b = u @ mu.astype(np.float64)
    disc = b * b - (float(mu.astype(np.float64) @ mu) - xnorm.astype(np.float64) ** 2)
    m = np.where(disc > 0, -b + np.sqrt(np.maximum(disc, 0)), stored)
    return (mu + m[:, None] * u).astype(np.float32), m, stored


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

    q8 = Quantizer(pb.D_TOK, bits=8, seed=0, rotation="rht", mean=mu)
    c8 = q8.encode(T)
    X8 = q8.decode(c8)
    xnorm = np.linalg.norm(X8.astype(np.float64), axis=1)          # recovered from codes
    true_err = float(np.abs(xnorm - np.linalg.norm(T, axis=1)).max())
    print(f"  max | ||decode_8(x)|| - ||x|| | = {true_err:.2e}", flush=True)
    for p in (4, 3, 2, 1):
        Xp = q8.decode(c8, precision=p)
        Xfix, m_new, m_old = resolve_lengths(Xp, mu, xnorm)
        len_err_old = float(np.median(np.abs(np.linalg.norm(Xp, axis=1) - xnorm)))
        len_err_new = float(np.median(np.abs(np.linalg.norm(Xfix, axis=1) - xnorm)))
        for name, X in ((f"remex {p}-bit centered nested-from-8", Xp),
                        (f"remex {p}-bit centered nested-from-8 re-solved", Xfix)):
            top = pb.late_rank(QT, qoff, X, off)
            ranked = [[dids[j] for j in r] for r in top]
            nd = np.array([pb.ndcg10(r, qrels[q]) for r, q in zip(ranked, qids)])
            m, lo, hi, w, l = pb.paired_bootstrap(nd - ref)
            row = dict(arm=name, bytes_per_doc=round(tpd * pb.D_TOK * p / 8, 1), ndcg10=round(float(nd.mean()), 4),
                       delta=round(m, 4), ci=[round(lo, 4), round(hi, 4)],
                       median_length_error=round(len_err_new if "re-solved" in name else len_err_old, 4))
            if f"remex {p}-bit centered" in pq:
                dm, dlo, dhi, _, _ = pb.paired_bootstrap(nd - pq[f"remex {p}-bit centered"])
                row["vs_direct_centered"] = dict(delta=round(dm, 4), ci=[round(dlo, 4), round(dhi, 4)])
            rows.append(row)
            print(f"  {name:48s} nDCG@10 {row['ndcg10']:.4f} Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}] "
                  f"len-err {row['median_length_error']:.4f} "
                  f"{('vs direct ' + format(row['vs_direct_centered']['delta'], '+.4f')) if 'vs_direct_centered' in row else ''}",
                  flush=True)
    json.dump(dict(max_recovered_norm_error=true_err, rows=rows), open(HERE / f"results_renest_{corpus}.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
