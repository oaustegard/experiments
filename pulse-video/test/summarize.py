#!/usr/bin/env python3
"""Collects the headline numbers from results/{filter,rate,bench-mac}.json into results/summary.json."""
import json
import statistics
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
f = json.loads((R / "filter.json").read_text())
rate = json.loads((R / "rate.json").read_text())["byName"]
mac = json.loads((R / "bench-mac.json").read_text())["outputs"]

pulse = [n for n, r in rate.items() if r["expect"] is not None]
drifting = [n for n in pulse if n != "72-steady"]
none = [n for n, r in rate.items() if r["expect"] is None]
none_snr = [w["pos"]["snr"] for n in none for w in rate[n].get("log", [])]
summary = {
    "filter": {"maxFiltErr": max(c["filt"] for c in f["cases"]), "maxFiltErrE7": round(1e7 * max(c["filt"] for c in f["cases"]), 2), "maxCompLevels": max(c["comp"] for c in f["cases"])},
    "drift": {
        "posMAEmax": max(rate[n]["posMAE"] for n in drifting),
        "evmMAEmax": max(rate[n]["evmMAE"] for n in drifting),
        "greenMAEmax": max(rate[n]["greenMAE"] for n in drifting),
        "expCorrMin": min(rate[n]["meanExpCorr"] for n in drifting),
        "expCorrMax": max(rate[n]["meanExpCorr"] for n in drifting),
    },
    "flicker": {"noPulseEvmBpmMedian": statistics.median(w["evm"]["bpm"] for w in rate["none-flicker"]["log"]), "noPulsePosBpm": [w["pos"]["bpm"] for w in rate["none-flicker"]["log"]], "hz": 1.3},
    "finalEvmBpm": {n: rate[n]["final"]["evm"]["bpm"] for n in ("96", "128", "72-faint", "72-flicker")},
    "steady": {k: rate["72-steady"][k] for k in ("posMAE", "evmMAE", "greenMAE", "meanExpCorr")},
    "snr": {"pulseMeanMin": min(rate[n]["meanPosSNR"] for n in pulse), "noPulseMax": max(none_snr), "noPulseWindows": len(none_snr)},
    "cases": {n: {k: rate[n][k] for k in ("expect", "posMAE", "evmMAE", "greenMAE", "meanPosSNR", "meanExpCorr")} for n in rate},
    "mac": {"check": {**mac["check"], "maxErrE7": round(1e7 * mac["check"]["maxErr"], 2)}, **{f'{r["w"]}x{r["h"]}': r for r in mac["rows"]}},
}
(R / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: summary[k] for k in ("filter", "drift", "steady", "snr")}, indent=1))
