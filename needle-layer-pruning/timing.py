#!/usr/bin/env python3
"""Interleaved timing of the control against the surviving cut.

Latency is the reason to prune at all, and `METHODS.md` records that a busy
4-core container inflates Needle latency by an order of magnitude — the 25-arm
sweep is exactly that situation, so its per-arm medians are not usable. This
alternates the two arms so any drift in box load falls on both equally.

    python3 timing.py --reps 3 --ctl /tmp/t_ctl.cact --cut /tmp/t_cut.cact
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def one(weights: str, label: str, layers: int, engram: str, cut: str) -> tuple[float, float]:
    subprocess.run([sys.executable, str(HERE / "eval_pruned.py"), "--weights", weights,
                    "--label", label, "--layers", str(layers), "--engram", engram, "--cut", cut],
                   cwd=HERE, stdout=subprocess.DEVNULL)
    d = json.loads((HERE / f"results_{label}.json").read_text())
    lat = [r["latency_ms"] for r in d["rows"]]
    return statistics.median(lat), d["summary"]["tool_acc_routable"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--cut", required=True)
    a = ap.parse_args()

    arms = {"control": (a.ctl, 27, "(2, 15)", "none"),
            "cut4_at09": (a.cut, 23, "(2, 11)", "[9,13)")}
    out: dict[str, list[float]] = {k: [] for k in arms}
    for rep in range(a.reps):
        for name, (w, layers, engram, cut) in arms.items():
            med, acc = one(w, f"timing_{name}_r{rep}", layers, engram, cut)
            out[name].append(med)
            print(f"rep {rep}  {name:10} median {med:7.1f} ms   routable {acc:.3f}", flush=True)

    print()
    meds = {k: statistics.median(v) for k, v in out.items()}
    for k, v in out.items():
        print(f"{k:10} medians {['%.0f' % x for x in v]}  ->  median-of-medians {meds[k]:.0f} ms")
    c, p = meds["control"], meds["cut4_at09"]
    print(f"\n27 -> 23 layers (-14.8% depth): {c:.0f} -> {p:.0f} ms, {100 * (1 - p / c):+.1f}%")
    (HERE / "timing.json").write_text(json.dumps(
        {"reps": a.reps, "medians_ms": out, "median_of_medians_ms": meds}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
