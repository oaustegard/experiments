#!/usr/bin/env python3
"""Seconds-long check that RESULTS.md agrees with the artifacts.

1. PROSE vs ARTIFACT  every nDCG@10 and Δ the table quotes is read from the result JSONs
                      and must appear in RESULTS.md as written.
2. PAIRED DELTAS      paired comparisons between bench.py arms are recomputed as plain
                      means of per-query differences from the npz (no bootstrap code).
3. INTEGRITY          bytes follow from tokens/doc; fp32 is the reference row.

Does not re-derive the science: rebuilding the encodings is encode.py (~2.3 h).
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


res = json.load(open(HERE / "results_scifact.json"))
ext = json.load(open(HERE / "results_extra_scifact.json"))
ren = json.load(open(HERE / "results_renest_scifact.json"))
pq = dict(np.load(HERE / "results_scifact_perquery.npz"))
row = {r["arm"]: r for r in res["rows"] + ext["rows"]}
prose = re.sub(r"\s+", " ", (HERE / "RESULTS.md").read_text()).replace("−", "-")

# 1. prose vs artifact
for arm, r in row.items():
    if "centered" in arm and "(asym)" in arm and "q-centered" not in arm:
        continue  # superseded harness-bug rows (ERRORS.md #1)
    check(f"{r['ndcg10']:.4f}" in prose, f"{arm}: nDCG@10 {r['ndcg10']:.4f} not in RESULTS.md")
for r in ren["rows"]:
    check(f"{r['ndcg10']:.4f}"[:-1] in prose or f"{r['ndcg10']:.3f}" in prose,
          f"renest {r['arm']}: nDCG@10 {r['ndcg10']:.4f} not in RESULTS.md")
check(f"{res['meta']['centering_predictor']:.3f}" in prose, "centering predictor not in prose")
check(f"{res['meta']['random_pair_cosine']:.3f}" in prose, "random-pair cosine not in prose")

# 2. paired deltas, recomputed without the bootstrap
for c in ext["paired"]:
    if c["a"] in pq and c["b"] in pq:
        d = float(np.mean(pq[c["a"]] - pq[c["b"]]))
        check(abs(d - c["delta"]) < 5e-5, f"paired {c['a']} - {c['b']}: {d:+.4f} vs stored {c['delta']:+.4f}")
        check(f"{d:+.3f}" in prose, f"paired {c['a']} - {c['b']} {d:+.3f} not in RESULTS.md")
for name in ("remex 2-bit centered", "remex 1-bit centered", "plaid 1-bit"):
    d = float(np.mean(pq[name] - pq["fp32"]))
    check(abs(d - row[name]["delta"]) < 5e-5, f"{name}: Δ recompute {d:+.4f} vs {row[name]['delta']:+.4f}")

# 3. integrity
tpd = res["meta"]["tokens_per_doc"]
check(abs(row["fp32"]["bytes_per_doc"] - tpd * 128 * 4) < 1, "fp32 bytes")
check(abs(row["remex 2-bit centered"]["bytes_per_doc"] - tpd * 128 * 2 / 8) < 1, "2-bit bytes")
check(res["rows"][0]["arm"] == "fp32", "fp32 is not the reference row")

if fails:
    print("\n".join("FAIL " + f for f in fails)); sys.exit(1)
print(f"recheck OK: {len(row)} arms, {len(ext['paired'])} paired comparisons, {len(ren['rows'])} renest rows")
