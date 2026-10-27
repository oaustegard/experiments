#!/usr/bin/env python3
"""Seconds-long check that RESULTS.md agrees with the committed artifacts.

1. PAIRED DELTAS  every pooled and arm-vs-arm delta is recomputed from
                  results/perquery.json as a plain mean of per-query differences
                  (no bootstrap) and must match results/summary.json / paired.json.
2. PROSE          the headline deltas appear in RESULTS.md as written.
3. INTEGRITY      bytes per token follow from the codec definitions.

Does not re-derive the science: encoding and scoring are worker.py in the jail (hours).
Run: python3 recheck.py   (non-zero exit on any failure)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results"
fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


pq = json.loads((R / "perquery.json").read_text())
summary = json.loads((R / "summary.json").read_text())
paired = json.loads((R / "paired.json").read_text())
prose = re.sub(r"\s+", " ", (HERE / "RESULTS.md").read_text())


def mean_diff(a, b):
    qs = sorted(set(pq[a]) & set(pq[b]))
    return float(np.mean([pq[a][q] - pq[b][q] for q in qs])), len(qs)


# 1. paired deltas
for arm, row in summary["pooled"].items():
    d, n = mean_diff(arm, "fp32")
    check(abs(d - row["delta"]) < 6e-5 and n == row["n"], f"pooled {arm}: {d:+.4f}/{n} vs {row['delta']:+.4f}/{row['n']}")
for key, row in paired.items():
    a, b = key.split("__")
    d, n = mean_diff(a, b)
    check(abs(d - row["delta"]) < 6e-5 and n == row["n"], f"{a} vs {b}: {d:+.4f} vs {row['delta']:+.4f}")

# 2. prose
for arm, txt in [("plaid-rx-2", "-0.001 [-0.006, +0.005]"), ("plaid-2", "-0.031"), ("plaid-4", "-0.008"),
                 ("plaid-n-2", "-0.018")]:
    check(txt in prose, f"{arm}: '{txt}' not in RESULTS.md")
    check(f"{summary['pooled'][arm]['delta']:+.3f}".replace("+", "") in prose
          or f"{summary['pooled'][arm]['delta']:.3f}" in prose, f"{arm}: stored delta not quoted")
check("+0.031 [+0.016, +0.045]" in prose, "plaid-rx-2 vs plaid-2 not quoted")
for ds in ["scifact", "nfcorpus", "fiqa", "scidocs"]:
    rho = summary[ds]["plaid-rx-2"]["spearman"]
    check(0.989 <= round(rho, 3) <= 0.991, f"{ds} plaid-rx-2 Spearman {rho} outside the quoted 0.989-0.991")

# 3. integrity: bits*128/8 + 4 B centroid id (+2 B length)
arms = json.loads((R / "arms_nfcorpus.json").read_text())
for arm, want in [("plaid-2", 36), ("plaid-4", 68), ("plaid-rx-2", 38), ("plaid-rx-1", 22), ("remex-c-2", 34)]:
    check(arms[arm]["bytes_per_token"] == want, f"{arm} bytes {arms[arm]['bytes_per_token']} != {want}")

if fails:
    print("FAIL\n  " + "\n  ".join(fails))
    sys.exit(1)
print(f"ok: {len(summary['pooled'])} pooled deltas, {len(paired)} paired deltas, prose and bytes agree")
