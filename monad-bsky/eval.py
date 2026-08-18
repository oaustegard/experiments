#!/usr/bin/env python3
"""Score a Monad checkpoint on the same 62 queries the Needle arms used.

    python3 eval.py --model model            # the base model, zero-shot
    python3 eval.py --model tuned --label tuned
    python3 eval.py --model tuned-e1 --label tuned-e1

Scoring is imported from `needle-bsky/eval.py` so the two experiments cannot
drift apart: same accepted-tool lists, same argument normalisation, same
invented-argument rule. One column is added — `status` — because Monad decodes
unconstrained and can return something that is not a call at all, which the
grammar-constrained arm cannot.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from monad_bsky.runner import (
    MonadRouter,
    load_eval_items,
    load_eval_schemas,
    score_one,
)


def summarize(rows: list[dict]) -> dict:
    on = [r for r in rows if r["expected"]]
    off = [r for r in rows if not r["expected"]]

    def frac(xs, key):
        return round(sum(1 for x in xs if x[key]) / len(xs), 4) if xs else None

    status = Counter(r["status"] for r in rows)
    return {
        "n": len(rows),
        "tool_acc": frac(rows, "tool_ok"),
        "tool_acc_routable": frac(on, "tool_ok"),
        "refusal_acc": frac(off, "tool_ok"),
        "args_acc_routable": frac(on, "args_ok"),
        "invented_rate": round(sum(1 for r in on if r["invented"]) / len(on), 4) if on else None,
        "parse_ok_rate": round((status["ok"] + status["refused"]) / len(rows), 4),
        "status_counts": dict(status),
        "hallucinated_tool_rate": round(
            sum(1 for r in rows if r["got"] is not None and r["got"] not in DECLARED) / len(rows), 4
        ),
        "median_latency_ms": round(statistics.median(r["latency_ms"] for r in rows), 1),
        "mean_new_tokens": round(statistics.mean(r["new_tokens"] for r in rows), 1),
        "prompt_tokens": rows[0]["prompt_tokens"] if rows else None,
    }


DECLARED: set[str] = set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(HERE / "model"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--arm", default="tuned-min", help="which schema wording to declare")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    a = ap.parse_args()

    label = a.label or Path(a.model).name
    schemas = load_eval_schemas(a.arm)
    DECLARED.update(s["name"] for s in schemas)
    items = load_eval_items()

    model_dir = a.model if Path(a.model).is_absolute() else str(HERE / a.model)
    r = MonadRouter(model_dir=model_dir, schemas=schemas, max_new_tokens=a.max_new_tokens)

    rows = []
    for i, it in enumerate(items, 1):
        rows.append(score_one(it, r.route(it["query"])))
        if i % 20 == 0:
            print(f"  {i}/{len(items)}", flush=True)

    res = {
        "label": label,
        "model": model_dir,
        "arm": a.arm,
        "n_tools": len(schemas),
        "summary": summarize(rows),
        "rows": rows,
    }
    (HERE / f"results_{label}.json").write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(
        f"{label:12} tool {s['tool_acc']:.3f}  routable {s['tool_acc_routable']:.3f}  "
        f"refuse {s['refusal_acc']:.3f}  args {s['args_acc_routable']:.3f}  "
        f"invented {s['invented_rate']:.3f}  parse-ok {s['parse_ok_rate']:.3f}  "
        f"median {s['median_latency_ms']:.0f}ms"
    )
    print(f"  status {s['status_counts']}  hallucinated-tool {s['hallucinated_tool_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
