#!/usr/bin/env python3
"""Route `needle-bsky/evalset.jsonl` through one naming variant and score it.

    python3 eval_names.py --variant canon --mode flat
    python3 eval_names.py --variant separated --mode oracle

One variant per process, deliberately: the Needle engine holds a single global
session and re-runs `needle_init` whenever a different agent is used, so
switching catalogues in-process is both slower and, for tuned weights,
irreversible (`needle-bsky/RESULTS.md`, METHODS.md). `run_all.py` drives the
subprocesses.

Scoring is `needle-bsky/eval.py` unchanged — the emitted name is translated back
to its catalogue name first, so every variant is scored against the same
`evalset.jsonl` expectations.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from names import NEEDLE_BSKY, VARIANTS, variant  # noqa: E402

sys.path.insert(0, str(NEEDLE_BSKY))
from eval import gate_sweep, load_items, score_one, summarize  # noqa: E402
from needle_bsky.router import Router  # noqa: E402
from oracle import K, subset_for  # noqa: E402

ORACLE_SEED = 20260818  # the seed needle-bsky's oracle arms used


class _Router(Router):
    """A Needle agent over one variant's schemas, with no confidence gate.

    Mirrors `needle-bsky/oracle.py`'s `_OneShotRouter`: same `route()` contract
    and the same one-turn `complete()`, but the schemas are supplied rather than
    loaded from an arm name, which is the whole manipulation here.
    """

    def __init__(self, schemas: list[dict]):
        import needle

        self.arm = "naming"
        self.threshold = 0.0  # gate applied in analysis, as in needle-bsky
        self.schemas = schemas
        self.agent = needle.Needle(tools=schemas)


def _canonicalize(decision, to_canonical: dict[str, str]):
    """Map the emitted tool name back to its catalogue name.

    An emitted name absent from the map is a hallucinated one; it is kept
    verbatim so the scorer counts it wrong rather than silently dropping it.
    """
    if decision.tool is None:
        return decision
    return replace(decision, tool=to_canonical.get(decision.tool, decision.tool))


def run_flat(schemas, to_canonical, items):
    r = _Router(schemas)
    return [score_one(it, _canonicalize(r.route(it["query"]), to_canonical)) for it in items]


def run_oracle(schemas, to_canonical, items):
    """Each query gets its own five-tool catalogue containing the right answer.

    The subset is drawn over *catalogue* names with `needle-bsky`'s seed and
    rule, then mapped to this variant's schemas, so every variant sees the same
    five tools for the same query and the contrast stays a naming contrast.
    """
    by_canonical = {to_canonical[s["name"]]: s for s in schemas}
    canonical_schemas = [{"name": n} for n in by_canonical]
    rng = random.Random(ORACLE_SEED)
    rows = []
    for it in items:
        picked = subset_for(it, canonical_schemas, rng)
        sub = [by_canonical[s["name"]] for s in picked]
        rows.append(score_one(it, _canonicalize(_Router(sub).route(it["query"]), to_canonical)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--mode", default="flat", choices=("flat", "oracle"))
    ap.add_argument("--evalset", default=str(NEEDLE_BSKY / "evalset.jsonl"))
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()

    items = load_items(Path(a.evalset))
    schemas, to_canonical = variant(a.variant)
    runner = run_flat if a.mode == "flat" else run_oracle

    runs = [runner(schemas, to_canonical, items) for _ in range(a.repeat)]
    rows = runs[0]
    identical = all(
        [(x["got"], json.dumps(x["arguments"], sort_keys=True)) for x in rows]
        == [(y["got"], json.dumps(y["arguments"], sort_keys=True)) for y in other]
        for other in runs[1:]
    )

    res = {
        "variant": a.variant,
        "mode": a.mode,
        "n_tools": len(schemas) if a.mode == "flat" else K,
        "names": {s["name"]: to_canonical[s["name"]] for s in schemas},
        "repeat": a.repeat,
        "deterministic": identical if a.repeat > 1 else None,
        "summary": summarize(rows),
        "gate_sweep": gate_sweep(rows),
        "rows": rows,
    }
    out = HERE / f"results_{a.variant}_{a.mode}.json"
    out.write_text(json.dumps(res, indent=1))
    s = res["summary"]

    def num(key, spec=".3f"):
        # refusal_acc is None on an evalset with no off-topic items.
        return format(s[key], spec) if s[key] is not None else "  n/a"

    print(
        f"{a.variant:12} {a.mode:6} routable {num('tool_acc_routable')}  "
        f"tool {num('tool_acc')}  refuse {num('refusal_acc')}  "
        f"args {num('args_acc_routable')}  "
        f"conf ok/bad {s['mean_conf_correct']}/{s['mean_conf_wrong']}  "
        f"median {num('median_latency_ms', '.0f')}ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
