#!/usr/bin/env python3
"""Named paired comparisons for RESULTS.md (per-query nDCG@10, paired bootstrap)."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "neomme-remex-quant"))
from neomme_quant import paired_bootstrap  # noqa: E402

PAIRS = [
    ("remex centered 1-bit d=768", "remex 1-bit d=768"),
    ("remex centered 1-bit d=768", "ST binary d=768 (asym)"),
    ("remex centered 1-bit d=768", "remax k=1 d=768 (asym)"),
    ("remex centered 2-bit d=768", "remex 2-bit d=768"),
    ("remex centered 2-bit d=512", "fp32 d=128"),
    ("remex centered 4-bit d=768", "fp32 d=128"),
    ("remex centered 8->2-bit nested d=768", "remex centered 2-bit d=768"),
    ("remex 8->2-bit nested d=768", "remex 2-bit d=768"),
    ("remex centered 1-bit d=512", "remex centered 2-bit d=256"),
    ("remex centered 2-bit d=512", "remex centered 4-bit d=256"),
]
for c in sys.argv[1:] or ["scifact", "nfcorpus"]:
    pq = np.load(HERE / f"results_{c}_perquery.npz")
    print(f"## {c}")
    for a, b in PAIRS:
        m, lo, hi, w, l = paired_bootstrap(pq[a] - pq[b])
        print(f"  {a:38s} - {b:30s} {m:+.4f} [{lo:+.4f},{hi:+.4f}]  W/L {w}/{l}")
