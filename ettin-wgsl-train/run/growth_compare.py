"""WGSL vs PyTorch on adapter-library's growth protocol, 10 seeds a side: is the gap at n = 64 and 300 real?

    python3 -I run/growth_compare.py BRIDGE_N64_DIR BRIDGE_N300_DIR PY_GROWTH_DIR    # -> results/growth_seeds.json

WGSL: decider-lora-creator's bridge_lora.js on the Mac (its training loop, the merge of the shipped spikes as the
start, the packed base). PyTorch: run/growth_python.py on a Colab L4 (adapter-library's loop, the merge of the
rebuilt round-1 spikes, the fp32 base). Both use revealed(invoice_processing, seed, n), so seed s sees the same
labelled cases on both sides and the comparison is paired by seed. Test: the 100 invoice test cases, 500 decisions.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
AL = HERE.parent / "adapter-library" / "results"
CR = HERE.parent / "decider-lora-creator" / "results" / "bridge_creator-lora-2.json"
T975 = {9: 2.262, 8: 2.306, 7: 2.365}   # Student t, two-sided 95%


def bridge_arms(d):
    r = json.loads((Path(d) / "result.json").read_text())
    o = r.get("outputs", r)
    return {(a["n"], a["seed"]): a["test_accuracy"] for a in o["arms"]}, {k: r.get(k) for k in ("device", "adapter", "runMs")}


def main():
    wg = {}
    dev = None
    for d in sys.argv[1:3]:
        a, dev = bridge_arms(d)
        wg |= a
    py = {}
    for f in Path(sys.argv[3]).glob("G-warm-invoice_processing-n*-s*.json"):
        j = json.loads(f.read_text())
        py[(j["n"], j["seed"])] = j["test"]["accuracy"]
    out = {"wgsl_device": dev, "by_n": {}}
    for n in (64, 300):
        seeds = sorted(s for (m, s) in wg if m == n and (n, s) in py)
        w = np.array([wg[(n, s)] for s in seeds])
        p = np.array([py[(n, s)] for s in seeds])
        d = w - p
        k = len(d)
        se = d.std(ddof=1) / math.sqrt(k)
        t = T975.get(k - 1, 2.262)
        welch_se = math.sqrt(w.var(ddof=1) / k + p.var(ddof=1) / k)
        out["by_n"][str(n)] = {
            "seeds": seeds, "wgsl": w.tolist(), "pytorch": p.tolist(),
            "wgsl_mean": float(w.mean()), "wgsl_sd": float(w.std(ddof=1)), "pytorch_mean": float(p.mean()), "pytorch_sd": float(p.std(ddof=1)),
            "paired_diff_mean": float(d.mean()), "paired_diff_se": float(se),
            "paired_ci95": [float(d.mean() - t * se), float(d.mean() + t * se)], "paired_t": float(d.mean() / se) if se else None,
            "welch_se": float(welch_se), "wgsl_wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "pytorch_wins": int((d < 0).sum())}
    # reproducibility of the two seeds published before this round
    cr = json.loads(CR.read_text())["outputs"]["arms"]
    out["repro"] = {
        "wgsl_vs_creator_lora_2": {f"n{a['n']}-s{a['seed']}": [a["test_accuracy"], wg.get((a["n"], a["seed"]))] for a in cr if a["n"] in (64, 300)},
        "pytorch_vs_adapter_library": {f"n{n}-s{s}": [json.loads((AL / f"G-warm-invoice_processing-n{n}-s{s}.json").read_text())["test"]["accuracy"], py.get((n, s))]
                                       for n in (64, 300) for s in (0, 1)}}
    (HERE / "results" / "growth_seeds.json").write_text(json.dumps(out, indent=1))
    for n, r in out["by_n"].items():
        print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("wgsl", "pytorch", "seeds")})
    print(json.dumps(out["repro"]))


if __name__ == "__main__":
    main()
