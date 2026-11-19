"""Multiply-accumulates per input column for every distinct tessdata network spec, by layer.

    python3 run/flops.py   # -> results/flops.json

Per 3 input pixel columns (one output timestep) of a line H px high: conv+FC over 3*H pixels,
the y-LSTM over H/3 steps, then one step of each x-LSTM and the output layer (C classes, taken
as 111 where the inventory has no class count). Shows where the time goes and what each
architecture costs relative to eng best.
"""
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
inv = json.load(open(ROOT / "results/inventory.json"))
specs = Counter(r["spec"] for r in inv if r.get("spec"))
out = {}
for spec, n in specs.items():
    m = re.match(r"\[1,(\d+),0,1Ct3,3,16Mp3,3Lfys(\d+)Lfx(\d+)Lrx(\d+)Lfx(\d+)O1c\d+\]", spec)
    if not m:
        continue
    H, y, x1, x2, x3 = map(int, m.groups())
    C = 111
    lstm = lambda ni, ns: 4 * ns * (ni + ns + 1)  # noqa: E731
    layers = {
        "conv": 3 * H * 16 * 10,
        "ylstm": (H // 3) * lstm(16, y),
        "x1": lstm(y, x1), "x2": lstm(x1, x2), "x3": lstm(x2, x3),
        "out": C * (x3 + 1),
    }
    tot = sum(layers.values())
    out[spec] = {"models": n, "macs_per_step": tot,
                 "share": {k: round(v / tot, 3) for k, v in layers.items()},
                 "sequential_share": round((layers["x1"] + layers["x2"] + layers["x3"]) / tot, 3)}
base = out["[1,36,0,1Ct3,3,16Mp3,3Lfys64Lfx96Lrx96Lfx512O1c1]"]["macs_per_step"]
for v in out.values():
    v["vs_eng_best"] = round(v["macs_per_step"] / base, 3)
(ROOT / "results/flops.json").write_text(json.dumps(out, indent=1))
for k, v in sorted(out.items(), key=lambda kv: -kv[1]["macs_per_step"]):
    print(f"{k:52s} n={v['models']:3d} {v['macs_per_step']/1e6:6.2f} MMAC/step  x3 {v['share']['x3']:.2f}  seq {v['sequential_share']:.2f}  vs best-eng {v['vs_eng_best']}")
