#!/usr/bin/env python3
"""Check every number quoted in RESULTS.md against the result artifacts.

    python3 recheck.py        # exits non-zero on the first disagreement

No network and no model: it re-derives the claims from `results_*.json`,
`importance_4.json`, `timing.json` and `throughput_*.json` rather than trusting
the prose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from analyze import destroys_site, load, mcnemar, routable  # noqa: E402

FAILURES: list[str] = []


def eq(label: str, got, want, tol: float = 5e-4) -> None:
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    if not ok:
        FAILURES.append(f"{label}: RESULTS.md says {want}, artifacts say {got}")
    print(f"  {'ok ' if ok else 'FAIL'} {label:56} {got}")


def acc(label: str) -> float:
    return load(label)["summary"]["tool_acc_routable"]


def main() -> int:
    print("the control validates the export path")
    eq("control routable", acc("control"), 0.6111)
    eq("control tool", load("control")["summary"]["tool_acc"], 0.6129)
    eq("control refusal", load("control")["summary"]["refusal_acc"], 0.625)
    eq("control args", load("control")["summary"]["args_acc_routable"], 0.537)
    eq("re-timed control reproduces it", acc("control_timed"), acc("control"))
    for r in range(3):
        eq(f"timing rep {r} control reproduces it", acc(f"timing_control_r{r}"), acc("control"))
    print("the four-layer sweep")
    four = {
        9: 0.574, 12: 0.500, 7: 0.481, 13: 0.463, 8: 0.444, 10: 0.407, 4: 0.389,
        11: 0.389, 14: 0.370, 6: 0.352, 5: 0.241, 18: 0.185, 1: 0.167, 23: 0.167,
        16: 0.148, 17: 0.148, 3: 0.130, 2: 0.111, 19: 0.056, 15: 0.037, 20: 0.018,
        0: 0.000, 21: 0.000, 22: 0.000,
    }
    for start, want in sorted(four.items()):
        eq(f"cut4 [{start},{start + 4}) routable", acc(f"cut4_at{start:02d}"), want, tol=6e-4)
    eq("24 positions measured", len(four), 24)
    within = [s for s, v in four.items() if v - four_base() >= -0.05]
    eq("positions within 0.05 of control", within, [9])
    dead = sorted(s for s, v in four.items() if v <= 0.05)
    eq("positions effectively dead (<=0.05)", dead, [0, 15, 20, 21, 22])
    eq("five of them", len(dead), 5)
    sig = sum(1 for s in four if mcnemar(routable(load("control")),
                                         routable(load(f"cut4_at{s:02d}")))[2] < 0.05)
    eq("positions significant at p<0.05", sig, 20)

    print("the surviving arm is a null")
    ao, bo, p = mcnemar(routable(load("control")), routable(load("cut4_at09")))
    eq("[9,13) discordant pairs", (ao, bo), (10, 8))
    eq("[9,13) p", round(p, 2), 0.81, tol=6e-3)
    eq("[9,13) delta", round(acc("cut4_at09") - four_base(), 3), -0.037, tol=1e-3)

    print("Engram-site loss is not the dominant term")
    eq("[12,16) destroys site 15", destroys_site(12, 4), (15,))
    eq("[13,17) destroys site 15", destroys_site(13, 4), (15,))
    eq("[12,16) routable", acc("cut4_at12"), 0.500, tol=6e-4)
    eq("[13,17) routable", acc("cut4_at13"), 0.463, tol=6e-4)
    eq("[21,25) destroys nothing", destroys_site(21, 4), ())
    eq("[22,26) destroys nothing", destroys_site(22, 4), ())
    eq("arms leaving both sites", sum(1 for s in four if not destroys_site(s, 4)), 17)

    print("the heuristic")
    imp = json.loads((HERE / "importance_4.json").read_text())
    eq("Spearman", round(imp["spearman"], 3), 0.061, tol=6e-4)
    eq("heuristic top pick", imp["heuristic_pick"], 20)
    eq("its measured accuracy", imp["measured"]["20"], 0.0185, tol=6e-4)
    eq("sweep best", imp["measured_best"], 9)
    lowest6 = sorted(imp["angular"], key=lambda k: imp["angular"][k])[:6]
    eq("six lowest-distance blocks", [int(k) for k in lowest6], [20, 19, 18, 7, 21, 22])
    for k, dist, meas in [("20", 0.0385, 0.0185), ("19", 0.0403, 0.0556), ("18", 0.0462, 0.1852),
                          ("7", 0.0503, 0.4815), ("21", 0.0503, 0.0), ("22", 0.0541, 0.0)]:
        eq(f"  block [{k},{int(k) + 4}) distance", round(imp["angular"][k], 4), dist, tol=6e-5)
        eq(f"  block [{k},{int(k) + 4}) measured", imp["measured"][k], meas, tol=6e-4)
    rank_of_9 = sorted(imp["angular"], key=lambda k: imp["angular"][k]).index("9") + 1
    eq("heuristic rank of the actual best", rank_of_9, 10)

    print("deeper cuts")
    eight = {s: acc(f"cut8_at{s:02d}") for s in range(6, 12)}
    eq("8-layer best", round(max(eight.values()), 4), 0.1481, tol=6e-4)
    eq("8-layer best position", max(eight, key=eight.get), 10)
    eq("8-layer worst", round(min(eight.values()), 4), 0.0185, tol=6e-4)
    eq("12-layer routable", acc("cut12_at08"), 0.0)
    eq("12-layer refuses everything", load("cut12_at08")["summary"]["refusal_acc"], 1.0)

    print("what pruning buys")
    tp = {k: json.loads((HERE / f"throughput_{k}.json").read_text())
          for k in ("control", "cut4_at09")}
    eq("control prefill tok/s", tp["control"]["median_prefill_tps"], 423.9, tol=0.05)
    eq("pruned prefill tok/s", tp["cut4_at09"]["median_prefill_tps"], 518.6, tol=0.05)
    eq("prefill gain", round(100 * (tp["cut4_at09"]["median_prefill_tps"]
                                    / tp["control"]["median_prefill_tps"] - 1), 1), 22.3, tol=0.06)
    eq("control decode tok/s", tp["control"]["median_decode_tps"], 177.8, tol=0.05)
    eq("pruned decode tok/s", tp["cut4_at09"]["median_decode_tps"], 228.8, tol=0.05)
    eq("decode gain", round(100 * (tp["cut4_at09"]["median_decode_tps"]
                                   / tp["control"]["median_decode_tps"] - 1), 1), 28.7, tol=0.06)
    t = json.loads((HERE / "timing.json").read_text())
    eq("interleaved control median-of-medians", round(t["median_of_medians_ms"]["control"]), 1227)
    eq("interleaved pruned median-of-medians", round(t["median_of_medians_ms"]["cut4_at09"]), 1227)
    eq("linear-in-depth expectation", round(100 * (27 / 23 - 1), 1), 17.4, tol=0.06)
    k5 = {k: json.loads((HERE / f"timing_k5_{k}.json").read_text())
          for k in ("control", "cut4_at09")}
    eq("k=5 control median ms", k5["control"]["median_ms"], 247.0, tol=0.05)
    eq("k=5 pruned median ms", k5["cut4_at09"]["median_ms"], 354.4, tol=0.05)

    print("cross-checks against the siblings")
    from _lib.paths import experiment
    nb = json.loads((experiment("needle-bsky") / "results_tuned-min.json").read_text())
    eq("needle-bsky tuned-min == control", nb["summary"]["tool_acc_routable"], acc("control"))
    ntn = json.loads((experiment("needle-tool-naming") / "results_canon_flat.json").read_text())
    eq("needle-tool-naming canon == control", ntn["summary"]["tool_acc_routable"], acc("control"))
    return finish()


def four_base() -> float:
    return acc("control")


def finish() -> int:
    print()
    if FAILURES:
        print(f"{len(FAILURES)} disagreement(s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("RESULTS.md agrees with the artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
