#!/usr/bin/env python3
"""Shipped pack sizes (raw and gzip -9) against the upstream ONNX files -> results/packs.json."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = {"tiny_det": 1780590, "tiny_rec": 4462639, "small_det": 9880512, "small_rec": 21159378}
SCHEME = {"tiny_det": "r127", "tiny_rec": "r127", "small_det": "r63", "small_rec": "r127"}


def sz(p):
    b = p.read_bytes()
    return len(b), len(gzip.compress(b, 9))


out = {"models": {}, "totals": {}}
for name, up in UPSTREAM.items():
    size, kind = name.split("_")
    d = ROOT / "web" / kind / "models" / size
    files = [d / "weights.bin", d / "manifest.json"] + ([d / "dict.txt"] if kind == "rec" else [])
    w_raw, w_gz = sz(d / "weights.bin")
    tot_raw = tot_gz = 0
    for f in files:
        r, g = sz(f)
        tot_raw += r
        tot_gz += g
    out["models"][name] = {
        "scheme": SCHEME[name], "upstream_onnx_bytes": up,
        "weights_raw_bytes": w_raw, "weights_gzip_bytes": w_gz,
        "shipped_raw_bytes": tot_raw, "shipped_gzip_bytes": tot_gz,
        "shipped_gzip_over_upstream": round(tot_gz / up, 4),
        "upstream_over_shipped_gzip": round(up / tot_gz, 2),
    }
for size in ("tiny", "small"):
    m = [out["models"][f"{size}_{k}"] for k in ("det", "rec")]
    u = sum(x["upstream_onnx_bytes"] for x in m)
    g = sum(x["shipped_gzip_bytes"] for x in m)
    r = sum(x["shipped_raw_bytes"] for x in m)
    out["totals"][size] = {"upstream_onnx_bytes": u, "shipped_raw_bytes": r, "shipped_gzip_bytes": g,
                           "upstream_over_shipped_gzip": round(u / g, 2)}
(ROOT / "results" / "packs.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["totals"], indent=1))
