#!/usr/bin/env python3
"""Seconds-long check that RESULTS.md / ERRORS.md still agree with the artifacts.

1. ARTIFACT INTEGRITY  both corpora have the same arms; bytes follow from tokens/doc.
2. PROSE vs ARTIFACT   every number the findings quote is recomputed from the JSON /
                       per-query npz and must appear in the prose as written.
3. PAIRED DELTAS       the head-to-head table's deltas are recomputed as plain means of
                       per-query differences (no bootstrap code shared with bench.py).

Does not re-derive the science: rebuilding the encodings is encode.py (~19 min).
Run: python3 recheck.py   (non-zero exit on any failure)
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
C = ["scifact", "nfcorpus"]
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


res = {c: json.load(open(HERE / f"results_{c}.json")) for c in C}
ext = {c: json.load(open(HERE / f"results_extra_{c}.json")) for c in C}
pq = {c: dict(np.load(HERE / f"results_{c}_perquery.npz")) for c in C}
row = {c: {r["arm"]: r for r in res[c]["rows"] + ext[c]["rows"]} for c in C}
nd = lambda c, a: row[c][a]["ndcg10"]
dl = lambda c, a: row[c][a]["delta"]
norm_ws = lambda t: re.sub(r"\s+", " ", t)
prose = norm_ws((HERE / "RESULTS.md").read_text() + (HERE / "ERRORS.md").read_text())

# 1. integrity
check([r["arm"] for r in res["scifact"]["rows"]] == [r["arm"] for r in res["nfcorpus"]["rows"]], "arm lists differ across corpora")
for c in C:
    tpd = res[c]["meta"]["n_tokens"] / res[c]["meta"]["n_docs"]
    for bits in (8, 4, 3, 2, 1):
        b = row[c][f"late remex {bits}-bit"]["bytes_per_doc"]
        check(abs(b - tpd * 64 * bits / 8) < 1.0, f"{c} remex {bits}-bit bytes {b} vs {tpd * 64 * bits / 8:.1f}")
    check(abs(row[c]["late fp32"]["bytes_per_doc"] - tpd * 256) < 1.0, f"{c} fp32 bytes")
    n = res[c]["meta"]["n_tokens"]
    check(row[c]["late plaid 2-bit"]["K"] == 2 ** int(np.floor(np.log2(16 * np.sqrt(n)))), f"{c} plaid K")
    check(f"K = {row[c]['late plaid 2-bit']['K']:,}" in prose, f"prose missing K for {c}")

# 2. prose numbers, formatted exactly as the prose writes them
f3 = lambda x: f"{x:+.3f}".replace("-", "−")
u3 = lambda x: f"{abs(x):.3f}"
claims = [
    (f"1-bit costs {f3(dl('scifact', 'late remex 1-bit'))}", "F1 scifact 1-bit"),
    (f"−{u3(dl('nfcorpus', 'late remex 1-bit'))} on NFCorpus", "F1 nfcorpus 1-bit"),
    (f"2-bit costs {f3(dl('scifact', 'late remex 2-bit'))} and {f3(dl('nfcorpus', 'late remex 2-bit'))}", "F1 2-bit"),
    (f"norm {ext['scifact']['geometry']['mean_norm_of_mean']:.3f} (SciFact) / {ext['nfcorpus']['geometry']['mean_norm_of_mean']:.3f}", "F1 mean norm"),
    (f"cosine {ext['scifact']['geometry']['random_pair_cos_mean']:.3f} on average, {ext['scifact']['geometry']['random_pair_cos_p05']:.3f}", "F1 pair cos"),
    (f"falls to {ext['scifact']['geometry']['centered_pair_cos_mean']:.3f}", "F1 centered cos"),
    (f"scored {nd('scifact', 'late remex 1-bit haar s0'):.4f} and {nd('scifact', 'late remex 1-bit haar s1'):.4f}", "F1 seed swing"),
    (f"4-bit: {f3(dl('scifact', 'late remex 4-bit centered'))} / {f3(dl('nfcorpus', 'late remex 4-bit centered'))}".replace("+0.000", "−0.000"), "F2 4-bit"),
    (f"2-bit: {f3(dl('scifact', 'late remex 2-bit centered'))} / {f3(dl('nfcorpus', 'late remex 2-bit centered'))}", "F2 2-bit"),
    (f"1-bit: {f3(dl('scifact', 'late remex 1-bit centered'))} / {f3(dl('nfcorpus', 'late remex 1-bit centered'))}", "F2 1-bit"),
    (f"spread is {abs(nd('scifact', 'late remex 1-bit centered haar s1') - nd('scifact', 'late remex 1-bit centered haar s0')):.3f} (SciFact) and "
     f"{abs(nd('nfcorpus', 'late remex 1-bit centered haar s1') - nd('nfcorpus', 'late remex 1-bit centered haar s0')):.3f}", "F2 centered spread"),
    (f"against {abs(nd('scifact', 'late remex 1-bit haar s1') - nd('scifact', 'late remex 1-bit haar s0')):.3f} and "
     f"{abs(nd('nfcorpus', 'late remex 1-bit haar s1') - nd('nfcorpus', 'late remex 1-bit haar s0')):.3f}", "F2 uncentered spread"),
    (f"{row['scifact']['late remex 2-bit centered']['bytes_per_doc']:,.0f} vs {row['scifact']['late plaid 2-bit']['bytes_per_doc']:,.0f} B/doc", "F3 bytes"),
    (f"scores {nd('scifact', 'late plaid centroid-only'):.3f} on SciFact and\n     {nd('nfcorpus', 'late plaid centroid-only'):.3f}", "F3 centroid-only"),
    (f"pool2 at fp32 costs {f3(dl('scifact', 'late fp32 pool2'))} on SciFact\n   and {f3(dl('nfcorpus', 'late fp32 pool2'))}", "F6 pool2"),
    (f"k=1 asym scores {nd('scifact', 'late remax k=1 (asym)'):.4f}", "F5 remax"),
    (f"nDCG@10 {nd('scifact', 'late remax k=1 (asym) centered'):.3f} (SciFact) and {nd('nfcorpus', 'late remax k=1 (asym) centered'):.3f}", "E2 collapse"),
]
for s, tag in claims:
    check(norm_ws(s) in prose, f"prose missing [{tag}]: {s!r}")

# 3. paired deltas recomputed without the bootstrap
for c in C:
    allpq = {**pq[c]}
    for p in ext[c]["paired"]:
        if p["a"] in allpq and p["b"] in allpq:
            m = float((allpq[p["a"]] - allpq[p["b"]]).mean())
            check(abs(m - p["delta"]) < 1e-4, f"{c} paired {p['a']} vs {p['b']}: {m:.4f} vs {p['delta']}")

if fails:
    print("RECHECK FAILED\n  " + "\n  ".join(fails)); sys.exit(1)
print(f"recheck ok: {len(claims)} prose claims, paired deltas, byte arithmetic")
