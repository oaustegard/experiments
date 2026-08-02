#!/usr/bin/env python3
"""Quantization FIDELITY vs the fp32 ranking — the saturation-proof metric.

Recall-vs-qrels conflates embedder quality with quantization loss: muninn
ceilings (fp32=0.90/1.00) and NFCorpus floors (fp32=0.24) both hide damage.
The quantization question is "does the code reproduce what the fp32 vector
would retrieve?" — so we score each code against fp32's OWN ranking:

  * recall@{1,5,10,100} vs fp32-kNN  — agreement with the float top-k
    (this is remax's own bench metric; rotation-decorrelation used it too).
    Chunk-level, NO collapse-to-post — 1238 muninn targets, not 73.
  * Spearman rho  — rank correlation of code scores vs fp32 scores over the
    whole corpus, per query, averaged. Continuous, very sensitive.
  * recon cosine  — mean cosine(decode(code), fp32 vector). remex only
    (remax codes are binary, no dequantization).

METRIC CONSISTENCY (fixed 2026-08-01, see experiments#9): the reference
ranking is cosine, so every candidate ranking must be cosine too.  Scoring
`q @ xhat` without dividing by ||xhat|| silently mixes the angular question
with a reconstruction-norm question, and it does not do so evenly across
codecs: a 1-bit code has constant reconstruction norm *by construction* and
therefore pays no penalty at all, while a multi-bit code carries real norm
error.  In the sibling ablation that manufactured a false low-rate result
(0.663 -> 0.689 once renormalised).  Both sides are normalised below.  If you
want to score MIPS instead, normalise NEITHER side -- but then the reference
must be MIPS as well.

Reads cached fp32 vectors (no re-embedding). Each Quantizer is built once.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
A = HERE / "assets"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke
sys.path.insert(0, str(spoke("remax_kb")))
sys.path.insert(0, str(spoke("remax") / "src"))
from remax import StackedSignBitQuantizer             # noqa: E402
from remax_kb._hamming import hamming_scan            # noqa: E402
from remex import Quantizer                           # noqa: E402

KS = (1, 5, 10, 100)
SEED = 0
CORPORA = [("muninn", ".vec_doc.npz", ".vec_qry.npz"),
           ("nfcorpus", ".nf_doc.npz", ".nf_qry.npz")]


def recall_vs_gt(gt_topk, cand_order, ks):
    out = {}
    for k in ks:
        gt = set(gt_topk[:k].tolist())
        out[k] = len(gt & set(cand_order[:k].tolist())) / k
    return out


def eval_scores(scores_per_q, gt_top, ks):
    """scores_per_q: (m, n) aligned code scores (higher=better). Returns
    mean recall@k vs fp32 top-k and mean Spearman rho."""
    m = scores_per_q.shape[0]
    agg = {k: 0.0 for k in ks}; rho = 0.0
    for j in range(m):
        order = np.argsort(-scores_per_q[j])
        r = recall_vs_gt(gt_top[j], order, ks)
        for k in ks:
            agg[k] += r[k]
        rho += spearmanr(scores_per_q[j], gt_scores_cache[j]).statistic
    return {k: agg[k] / m for k in ks}, rho / m


def main():
    all_rows = {}
    for corpus, dn, qn in CORPORA:
        dp, qp = A / dn, A / qn
        if not dp.exists() or not qp.exists():
            print(f"[{corpus}] no cache, skip"); continue
        D = np.load(dp)["m"].astype(np.float32); Q = np.load(qp)["m"].astype(np.float32)
        n, m = D.shape[0], Q.shape[0]
        print(f"\n### {corpus}: {n} docs (chunk-level), {m} queries", flush=True)

        # fp32 ground truth, under the same metric the candidates are scored
        # with.  Row-normalising D is a no-op for an already-unit-norm encoder
        # output and a correction otherwise; the CV is printed so the choice
        # is visible rather than assumed.
        dn = np.linalg.norm(D, axis=1)
        print(f"  doc-norm CV {dn.std() / dn.mean():.4f} "
              f"({'unit-norm' if dn.std() / dn.mean() < 1e-4 else 'unnormalised'})")
        global gt_scores_cache
        Dn = D / (dn[:, None] + 1e-12)
        gt_scores_cache = (Q @ Dn.T).astype(np.float32)             # (m, n)
        gt_top = np.argsort(-gt_scores_cache, axis=1)               # (m, n)

        rows = []  # (label, bits, dim, B/row, rec{1,5,10,100}, rho, recon)

        def add(label, bits, dim, by, scores, recon):
            rec, rho = eval_scores(scores, gt_top, KS)
            rows.append((label, bits, dim, by, rec, rho, recon))
            print(f"  {label:<20} R@1={rec[1]:.3f} R@10={rec[10]:.3f} "
                  f"R@100={rec[100]:.3f} rho={rho:.3f} recon={recon}", flush=True)

        # remex family — build each Quantizer once, decode for aligned scores.
        # Note the d512 row is scored against the FULL-width fp32 ranking, so
        # it reports truncation + quantization together.  That is the intended
        # reading for a bytes-per-row comparison; it is not a pure codec number.
        for bits, dim in [(8, 768), (4, 768), (2, 768), (1, 768), (4, 512)]:
            t = time.time()
            qz = Quantizer(d=dim, bits=bits, seed=SEED)
            comp = qz.encode(np.ascontiguousarray(D[:, :dim]))
            Xhat = qz.decode(comp)                                   # (n, dim) approx fp32
            # Cosine, to match the reference ranking -- see METRIC CONSISTENCY.
            Xn = Xhat / (np.linalg.norm(Xhat, axis=1, keepdims=True) + 1e-12)
            scores = Q[:, :dim] @ Xn.T                               # (m, n) aligned
            recon = float(np.mean(np.sum(Xhat * D[:, :dim], axis=1) /
                          (np.linalg.norm(Xhat, axis=1) * np.linalg.norm(D[:, :dim], axis=1) + 1e-9)))
            add(f"remex {bits}b d{dim}", bits, dim, dim * bits // 8, scores, f"{recon:.4f}")
            print(f"    (built in {time.time()-t:.0f}s)", flush=True)

        # remax family — binary codes, score = -hamming (aligned), no recon
        for dim, k in [(256, 8), (512, 4), (768, 2)]:
            mean = D.mean(0).astype(np.float32)
            qz = StackedSignBitQuantizer(d=dim, k=k, seed=SEED)
            dcodes = qz.encode(np.ascontiguousarray((D - mean)[:, :dim]))
            qcodes = qz.encode(np.ascontiguousarray((Q - mean)[:, :dim]))
            scores = np.vstack([-hamming_scan(dcodes, qcodes[j]).astype(np.float32) for j in range(m)])
            add(f"remax d{dim}/k{k}", 1, dim, dim * k // 8, scores, "n/a")

        all_rows[corpus] = rows

    # serialize for the writeup / plot
    ser = {c: [[lab, b, dim, by, rec, rho, recon] for lab, b, dim, by, rec, rho, recon in rows]
           for c, rows in all_rows.items()}
    (A / ".fidelity.json").write_text(json.dumps(ser))

    # summary tables
    for corpus, rows in all_rows.items():
        print(f"\n{'='*78}\n{corpus} — fidelity vs fp32 ranking (chunk-level)")
        print(f"{'method':<16}{'B/row':>6}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'R@100':>7}{'rho':>7}{'recon':>8}")
        print("-" * 78)
        for lab, b, dim, by, rec, rho, recon in rows:
            print(f"{lab:<16}{by:>6}{rec[1]:>7.3f}{rec[5]:>7.3f}{rec[10]:>7.3f}{rec[100]:>7.3f}{rho:>7.3f}{recon:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
