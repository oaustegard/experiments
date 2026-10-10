#!/usr/bin/env python3
"""Compression grid for EmbeddingGemma 2 (text-only) single vectors, BEIR SciFact + NFCorpus.

The vendor's one size lever is Matryoshka truncation (768/512/256/128, re-normalised);
the card ships no quantized vectors. So the grid crosses the two axes at matched bytes:

  fp32 x MRL dims                         the vendor lever
  fp16, int8 (global scale) at 768        trivial codecs
  ST int8 (per-dim calibrated), ST binary asym/sym at 768   sentence-transformers' own quantize_embeddings
  remex {8,4,3,2,1}-bit x MRL dims      plain (RHT, renorm on)
  remex centered {4,3,2,1}-bit x MRL dims Quantizer(mean=corpus mean)
  remex 8-bit encoded once, read at 4/2/1 (nested precision), plain and centered
  remex 2/1-bit haar seeds 0,1 at 768     the single-seed noise floor
  remax k=1 asym/sym x MRL dims, k=2 asym at 768/512

Bytes per document exclude remex's 4 B norm (every vector is unit-norm, before and
after MRL renormalisation) and the one corpus mean (3 KB total) in centered mode.
Reference for deltas: fp32 d=768. Paired bootstrap, 5,000 resamples, 95% CI.

Writes results_<corpus>.json and results_<corpus>_perquery.npz.
"""
from __future__ import annotations

import json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "neomme-remex-quant"))
from neomme_quant import paired_bootstrap, overlap_at_k, signs_pm1, l2n  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("mxbai_bench", HERE.parent / "mxbai-edge-remex-quant" / "bench.py")
_mx = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mx)
ndcg10, recall100, dense_rank = _mx.ndcg10, _mx.recall100, _mx.dense_rank

from remex import Quantizer, corpus_mean  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402

WORK = HERE / ".work"
DIMS = (768, 512, 256, 128)
REF = "fp32 d=768"


def trunc(X, d):
    return l2n(X[:, :d])


def main(corpus):
    W = WORK / corpus
    D = np.load(W / "dense.npy"); Q = np.load(W / "q_dense.npy")
    dids = json.load(open(W / "doc_ids.json")); qids = json.load(open(W / "query_ids.json"))
    qrels = json.load(open(W / "qrels.json"))
    n = len(dids)
    print(f"[{corpus}] docs={n} queries={len(qids)}", flush=True)
    pred = {d: float(np.linalg.norm(corpus_mean(trunc(D, d)))) for d in DIMS}
    cos = trunc(D, 768)[np.random.default_rng(0).integers(0, n, (2, 20000))]
    rp = (cos[0] * cos[1]).sum(1)
    print(f"  centering predictor ||mean|| (unit vectors) {pred}; random-pair cos mean {rp.mean():.3f} p05 {np.percentile(rp, 5):.3f}", flush=True)

    rows, perq, ranks = [], {}, {}

    def record(name, family, top, bytes_doc, extra=None):
        ranked = [[dids[j] for j in r] for r in top]
        nd = np.array([ndcg10(r, qrels[q]) for r, q in zip(ranked, qids)])
        r100 = float(np.mean([recall100(r, qrels[q]) for r, q in zip(ranked, qids)]))
        perq[name] = nd; ranks[name] = top
        row = dict(arm=name, family=family, bytes_per_doc=round(bytes_doc, 1), ndcg10=round(float(nd.mean()), 4), recall100=round(r100, 4))
        if REF in perq and name != REF:
            m, lo, hi, w, l = paired_bootstrap(nd - perq[REF])
            row.update(delta=round(m, 4), ci=[round(lo, 4), round(hi, 4)], wins=w, losses=l,
                       overlap10=round(float(np.mean([overlap_at_k(a, b) for a, b in zip(top, ranks[REF])])), 3))
        if extra:
            row.update(extra)
        rows.append(row)
        tail = f"  Δ {row['delta']:+.4f} [{row['ci'][0]:+.4f},{row['ci'][1]:+.4f}] ov {row['overlap10']:.2f}" if "delta" in row else ""
        print(f"  {name:36s} {row['bytes_per_doc']:>7.1f} B  nDCG@10 {row['ndcg10']:.4f}  R@100 {row['recall100']:.3f}{tail}", flush=True)

    t0 = time.time()
    for d in DIMS:
        record(f"fp32 d={d}", "mrl", dense_rank(trunc(Q, d), trunc(D, d)), d * 4, dict(dim=d, bits=32))
    record("fp16 d=768", "scalar", dense_rank(Q, D.astype(np.float16).astype(np.float32)), 768 * 2, dict(dim=768, bits=16))
    s8 = 127.0 / np.abs(D).max()
    record("int8 global d=768", "scalar", dense_rank(Q, np.round(D * s8).astype(np.int8).astype(np.float32) / s8), 768, dict(dim=768, bits=8))

    from sentence_transformers.util.quantization import quantize_embeddings
    i8 = quantize_embeddings(D, precision="int8", calibration_embeddings=D)
    lo_, hi_ = D.min(0), D.max(0); steps = (hi_ - lo_) / 255
    record("ST int8 d=768", "st", dense_rank(Q, (i8.astype(np.float32) + 128) * steps + lo_), 768, dict(dim=768, bits=8))
    sb = np.where(D > 0, 1.0, -1.0).astype(np.float32)
    record("ST binary d=768 (asym)", "st", dense_rank(Q, sb), 96, dict(dim=768, bits=1))
    record("ST binary d=768 (sym)", "st", dense_rank(np.where(Q > 0, 1.0, -1.0).astype(np.float32), sb), 96, dict(dim=768, bits=1))

    for centered in (False, True):
        for d in DIMS:
            X, Qt = trunc(D, d), trunc(Q, d)
            mu = corpus_mean(X) if centered else None
            for bits in ((4, 3, 2, 1) if centered else (8, 4, 3, 2, 1)):
                q = Quantizer(d, bits=bits, seed=0, rotation="rht", mean=mu)
                name = f"remex{' centered' if centered else ''} {bits}-bit d={d}"
                record(name, "remex-c" if centered else "remex", dense_rank(Qt, q.decode(q.encode(X))), d * bits / 8, dict(dim=d, bits=bits))
        print(f"  [{time.time() - t0:.0f}s]", flush=True)

    for centered in (False, True):
        mu = corpus_mean(D) if centered else None
        q = Quantizer(768, bits=8, seed=0, rotation="rht", mean=mu)
        cv = q.encode(D)
        for bits in (4, 2, 1):
            record(f"remex{' centered' if centered else ''} 8->{bits}-bit nested d=768", "nested",
                   dense_rank(Q, q.decode(cv, precision=bits)), 768 * bits / 8, dict(dim=768, bits=bits))

    for bits in (2, 1):
        for seed in (0, 1):
            q = Quantizer(768, bits=bits, seed=seed, rotation="haar")
            record(f"remex {bits}-bit d=768 haar s{seed}", "seed", dense_rank(Q, q.decode(q.encode(D))), 768 * bits / 8, dict(dim=768, bits=bits))

    for d in DIMS:
        X, Qt = trunc(D, d), trunc(Q, d)
        for k, mode in ([(1, "asym"), (1, "sym"), (2, "asym")] if d in (768, 512) else [(1, "asym")]):
            rq = StackedSignBitQuantizer(d, k, seed=0, rotation="rht")
            R = rq._rotation_matrix
            Qr = Qt @ R
            if mode == "sym":
                Qr = np.where(Qr > 0, 1.0, -1.0).astype(np.float32)
            record(f"remax k={k} d={d} ({mode})", "remax", dense_rank(Qr, signs_pm1(rq.encode(X), k * d)), k * d / 8, dict(dim=d, bits=k))
    print(f"  [{time.time() - t0:.0f}s]", flush=True)

    meta = dict(corpus=corpus, n_docs=n, n_queries=len(qids), centering_norm_of_mean=pred,
                random_pair_cos_mean=round(float(rp.mean()), 3), seconds=round(time.time() - t0, 1))
    json.dump(dict(meta=meta, rows=rows), open(HERE / f"results_{corpus}.json", "w"), indent=1)
    np.savez_compressed(HERE / f"results_{corpus}_perquery.npz", qids=np.array(qids), **perq)


if __name__ == "__main__":
    for c in sys.argv[1:] or ["scifact", "nfcorpus"]:
        main(c)
