#!/usr/bin/env python3
"""Collect the headline numbers from results/*.json into results/summary.json, keyed for RESULTS.md's
@evidence refs. Run after any check or bridge job: python3 bench/summarize.py"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"
load = lambda n: json.loads((R / n).read_text())
self_ = load("selftest-swiftshader.json")["res"]
modes = load("modes-swiftshader.json")["res"]
b1, b2, w3 = load("bench-mac-1.json"), load("bench-mac-2.json"), load("walls-mac-3.json")
rows = {f"{r['kernel']}_{r['nx']}x{r['ny']}_wg{r['wg']}_b{r.get('block', 512)}": r for r in b1["outputs"]["rows"] + b2["outputs"]["rows"] if "medianMs" in r}
disc = modes["disc"]
first6 = [abs(d["fMeasured"] / disc[0]["fMeasured"] / (d["bessel"] / disc[0]["bessel"]) - 1) * 100 for d in disc[:6]]
out = {
    "device": {"bench": b1["device"], "adapter": b1["adapter"], "ua": b1["ua"]},
    "accuracy": {"swiftshaderWorstDb": round(max(r["rmsDb"] for r in self_["runs"]), 1),
                 "macDb": round(max(c["rmsDb"] for c in b1["outputs"]["check"]), 1)},
    "modes": {"rectMaxCentsVsDiscrete": modes["maxCentsRectDiscrete"],
              "rectMaxCentsVsContinuous": max(abs(r["centsVsContinuous"]) for r in modes["rect"]),
              "discFirst6RatioErrPct": round(max(first6), 2), "discAllRatioErrPct": round(modes["maxRatioErrPct"], 2),
              "discCentsRange": [min(d["cents"] for d in disc if d["cents"] is not None), max(d["cents"] for d in disc if d["cents"] is not None)]},
    "mac": {k: {"medianMs": r["medianMs"], "p90Ms": r["p90Ms"], "load": r["load"]} for k, r in rows.items()},
    "macWalls": {f"{r['nx']}x{r['ny']}": {"wallsMedianMs": r["wallsMedianMs"], "drawMedianMs": r["drawMedianMs"]} for r in w3["outputs"]["rows"]},
}
(R / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
