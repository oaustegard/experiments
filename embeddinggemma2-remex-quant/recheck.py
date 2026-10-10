#!/usr/bin/env python3
"""Seconds-long check that RESULTS.md agrees with the artifacts.

1. TABLE vs ARTIFACT  every table row's nDCG@10, Δ and CI is read from results_<corpus>.json
                      and must appear in RESULTS.md exactly as written.
2. PAIRED DELTAS      every comparison quoted in Findings is recomputed as a plain mean of
                      per-query differences from the npz (no bootstrap code) and must match.
3. INTEGRITY          bytes = dim * bits / 8 for every coded arm; fp32 d=768 is the reference.

Does not re-derive the science: rebuilding the encodings is encode.py (~1.7 h).
Run: python3 recheck.py   (non-zero exit on any failure)
"""
from __future__ import annotations

import json, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


md = (HERE / "RESULTS.md").read_text().replace("−", "-")
res = {c: json.load(open(HERE / f"results_{c}.json")) for c in ("scifact", "nfcorpus")}
pq = {c: np.load(HERE / f"results_{c}_perquery.npz") for c in res}
rows = {c: {r["arm"]: r for r in res[c]["rows"]} for c in res}

# 1. table rows
n_rows = 0
for line in md.splitlines():
    m = re.match(r"\| (.+?) \| (\d+) \| \d+× \| (.+?) \| (.+?) \|$", line)
    if not m or m.group(1) == "arm":
        continue
    arm = m.group(1); n_rows += 1
    for c, cell in (("scifact", m.group(3)), ("nfcorpus", m.group(4))):
        r = rows[c].get(arm)
        check(r is not None, f"{arm} missing from {c}")
        if r is None:
            continue
        check(f"{r['bytes_per_doc']:.0f}" == m.group(2), f"{c} {arm} bytes {r['bytes_per_doc']} vs {m.group(2)}")
        want = f"{r['ndcg10']:.4f} (ref)" if "delta" not in r else \
            f"{r['ndcg10']:.4f} · {r['delta']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
        check(cell == want, f"{c} {arm}: table '{cell}' vs artifact '{want}'")
check(n_rows >= 20, f"only {n_rows} table rows parsed")
check(f"{len(rows['scifact'])} arms per corpus" in md, "arm count")

# 2. paired deltas quoted in Findings: (a, b, corpus, value as written)
PAIRS = [
    ("remex centered 4-bit d=768", "fp32 d=128", "scifact", "+0.0573"),
    ("remex centered 4-bit d=768", "fp32 d=128", "nfcorpus", "+0.0663"),
    ("remex centered 2-bit d=512", "fp32 d=128", "scifact", "+0.0492"),
    ("remex centered 2-bit d=512", "fp32 d=128", "nfcorpus", "+0.0567"),
    ("remex 2-bit d=768", "remex 3-bit d=512", "scifact", "-0.0014"),
    ("remex centered 2-bit d=768", "remex 3-bit d=512", "nfcorpus", "-0.0032"),
    ("remex centered 2-bit d=512", "remex centered 4-bit d=256", "scifact", "+0.0039"),
    ("remex centered 2-bit d=512", "remex centered 4-bit d=256", "nfcorpus", "+0.0120"),
    ("remex centered 1-bit d=768", "remex 1-bit d=768", "scifact", "+0.0185"),
    ("remex centered 1-bit d=768", "remex 1-bit d=768", "nfcorpus", "+0.0169"),
    ("ST binary d=768 (asym)", "remex centered 1-bit d=768", "scifact", "-0.0105"),
    ("ST binary d=768 (asym)", "remex centered 1-bit d=768", "nfcorpus", "-0.0009"),
    ("remax k=1 d=768 (asym)", "remex centered 1-bit d=768", "scifact", "-0.0332"),
    ("remax k=1 d=768 (asym)", "remex centered 1-bit d=768", "nfcorpus", "-0.0179"),
    ("remex centered 8->2-bit nested d=768", "remex centered 2-bit d=768", "scifact", "+0.0032"),
    ("remex centered 8->2-bit nested d=768", "remex centered 2-bit d=768", "nfcorpus", "-0.0046"),
    ("remex 8->2-bit nested d=768", "remex 2-bit d=768", "nfcorpus", "-0.0141"),
]
for a, b, c, v in PAIRS:
    d = float((pq[c][a] - pq[c][b]).mean())
    check(f"{d:+.4f}" == v, f"{c} {a} - {b}: {d:+.4f} vs written {v}")
    check(v in md, f"{v} ({c} {a} - {b}) not in RESULTS.md")

# 3. integrity
for c in res:
    check(res[c]["rows"][0]["arm"] == "fp32 d=768" and "delta" not in res[c]["rows"][0], f"{c}: reference row")
    for r in res[c]["rows"]:
        if r["family"] in ("remex", "remex-c", "nested", "seed", "mrl", "remax"):
            check(abs(r["bytes_per_doc"] - r["dim"] * r["bits"] / 8) < 0.11, f"{c} {r['arm']} bytes")
    check(res[c]["meta"]["n_queries"] == len(pq[c]["qids"]), f"{c} query count")

print(f"{n_rows} table rows, {len(PAIRS)} paired deltas checked")
if fails:
    print("\n".join("FAIL " + f for f in fails)); sys.exit(1)
print("recheck ok")
