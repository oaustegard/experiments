#!/usr/bin/env python3
"""Route every query in evalset.jsonl through each schema arm and score it.

    python3 eval.py                      # all four arms, writes results_<arm>.json
    python3 eval.py --arms tuned         # one arm
    python3 eval.py --repeat 2           # determinism check

Scoring, per item:

* **tool** — the predicted name is in the item's accepted `tool` list. An item
  with an empty list must produce no call at all (Needle's empty-call refusal).
* **args** — every key/value in the item's `args` is present and equal after
  normalisation (`@` stripped, lowercased, whitespace collapsed).
* **invented** — the call carries an argument that is neither expected nor
  listed in `evidenced`, i.e. a value the query never licensed.

Nothing here touches the network: routing is one `complete()` turn.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from needle_bsky.router import ARMS, Router  # noqa: E402


def norm(v) -> str:
    return " ".join(str(v).strip().lstrip("@").lower().split())


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score_one(item: dict, d) -> dict:
    accepted = item["tool"]
    refuse_expected = not accepted
    refused = d.tool is None

    if refuse_expected:
        tool_ok = refused
        args_ok = tool_ok
        invented: list[str] = []
    else:
        tool_ok = (not refused) and d.tool in accepted
        expected_args = item.get("args", {})
        args_ok = tool_ok and all(
            k in d.arguments and norm(d.arguments[k]) == norm(v) for k, v in expected_args.items()
        )
        licensed = set(expected_args) | set(item.get("evidenced", []))
        invented = sorted(k for k, v in d.arguments.items() if k not in licensed and v not in (None, "", [], {}))

    return {
        "id": item["id"],
        "cat": item["cat"],
        "query": item["query"],
        "expected": accepted,
        "got": d.tool,
        "arguments": d.arguments,
        "tool_ok": tool_ok,
        "args_ok": args_ok,
        "invented": invented,
        "confidence": d.confidence,
        "decision": d.decision,
        "latency_ms": round(d.latency_ms, 1),
        "reasoning": d.reasoning,
    }


def summarize(rows: list[dict]) -> dict:
    on = [r for r in rows if r["expected"]]
    off = [r for r in rows if not r["expected"]]
    conf_ok = [r["confidence"] for r in rows if r["tool_ok"] and r["confidence"] is not None]
    conf_bad = [r["confidence"] for r in rows if not r["tool_ok"] and r["confidence"] is not None]

    def frac(xs, key):
        return round(sum(1 for x in xs if x[key]) / len(xs), 4) if xs else None

    return {
        "n": len(rows),
        "tool_acc": frac(rows, "tool_ok"),
        "tool_acc_routable": frac(on, "tool_ok"),
        "refusal_acc": frac(off, "tool_ok"),
        "args_acc_routable": frac(on, "args_ok"),
        "invented_rate": round(sum(1 for r in on if r["invented"]) / len(on), 4) if on else None,
        "mean_conf_correct": round(statistics.mean(conf_ok), 4) if conf_ok else None,
        "mean_conf_wrong": round(statistics.mean(conf_bad), 4) if conf_bad else None,
        "median_latency_ms": round(statistics.median(r["latency_ms"] for r in rows), 1),
        "mean_latency_ms": round(statistics.mean(r["latency_ms"] for r in rows), 1),
    }


def gate_sweep(rows: list[dict], steps=None) -> list[dict]:
    """Precision/coverage of the act-gate as the confidence threshold moves.

    Refusals are excluded: the gate is about whether to execute a call, and a
    refusal executes nothing at any threshold.
    """
    calls = [r for r in rows if r["got"] is not None and r["confidence"] is not None]
    steps = steps or [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    out = []
    for t in steps:
        acted = [r for r in calls if r["confidence"] >= t]
        acc = round(sum(1 for r in acted if r["tool_ok"]) / len(acted), 4) if acted else None
        out.append(
            {
                "threshold": t,
                "acted": len(acted),
                "coverage": round(len(acted) / len(calls), 4) if calls else None,
                "precision_tool": acc,
                "precision_args": round(sum(1 for r in acted if r["args_ok"]) / len(acted), 4) if acted else None,
                "escalated_but_correct": sum(1 for r in calls if r["confidence"] < t and r["tool_ok"]),
            }
        )
    return out


def run_arm(arm: str, items: list[dict], repeat: int = 1) -> dict:
    t0 = time.perf_counter()
    r = Router(arm=arm, threshold=0.0)  # gate applied in analysis, not here
    init_s = time.perf_counter() - t0

    runs = []
    for _ in range(repeat):
        runs.append([score_one(it, r.route(it["query"])) for it in items])

    rows = runs[0]
    identical = all(
        [(x["got"], json.dumps(x["arguments"], sort_keys=True)) for x in runs[0]]
        == [(y["got"], json.dumps(y["arguments"], sort_keys=True)) for y in other]
        for other in runs[1:]
    )
    return {
        "arm": arm,
        "n_tools": len(r.schemas),
        "init_seconds": round(init_s, 2),
        "repeat": repeat,
        "deterministic": identical if repeat > 1 else None,
        "summary": summarize(rows),
        "gate_sweep": gate_sweep(rows),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--evalset", default=str(HERE / "evalset.jsonl"))
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()

    items = load_items(Path(a.evalset))
    print(f"{len(items)} queries, arms {a.arms}")
    for arm in a.arms:
        res = run_arm(arm, items, a.repeat)
        Path(a.out_dir, f"results_{arm}.json").write_text(json.dumps(res, indent=1))
        s = res["summary"]
        print(
            f"{arm:10} tool {s['tool_acc']:.3f}  routable {s['tool_acc_routable']:.3f}  "
            f"refuse {s['refusal_acc']:.3f}  args {s['args_acc_routable']:.3f}  "
            f"invented {s['invented_rate']:.3f}  conf ok/bad {s['mean_conf_correct']}/{s['mean_conf_wrong']}  "
            f"median {s['median_latency_ms']:.0f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
