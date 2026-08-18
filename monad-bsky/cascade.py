#!/usr/bin/env python3
"""Retry cascade: Needle alone, then Needle+Monad agreement, then a rewrite.

Oskar's proposal. A single gate trades coverage for precision along one curve;
a cascade can accept on different evidence at each tier and keep the queries a
strict gate would have thrown away.

    tier 1   Needle's own confidence >= t_hi          -> accept
    tier 2   else Needle and Monad name the same tool -> accept
    tier 3   else re-ask, with Monad in the loop      -> accept if it clears
    else     escalate to a larger model

Tiers 1 and 2 are post-processing over committed rows. Tier 3 needs Needle to
run again on a rewritten input, which this script does.

    python3 cascade.py                 # tiers 1-2 only, no model
    python3 cascade.py --tier3         # runs Needle again on the rewrites

Two rewrites are tested, because the obvious one is not available:

* `trace` — Monad's own `<think>` derivation as the restated ask. Free, already
  stored in the results rows.
* `hint` — the original query with Monad's chosen tool appended as a suggestion.
  This keeps the user's literal text, which matters: Monad corrupts identifiers
  *inside its reasoning* (`simonwillison.net` becomes `simanwillander.net`), so
  any rewrite that regenerates the request destroys the strings the tool call
  needs.

A third rewrite — base Monad paraphrasing the request — was probed and does not
exist as a capability: the model analyses the rewrite instruction and never
reaches the request, the same failure it shows on zero-shot routing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

NEEDLE = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE))

T_HI = (0.5, 0.6, 0.7, 0.8, 0.9)


def load_rows(path: Path) -> dict:
    return {r["id"]: r for r in json.loads(path.read_text())["rows"]}


def load_items() -> dict:
    return {
        json.loads(x)["id"]: json.loads(x)
        for x in (NEEDLE / "evalset.jsonl").read_text().splitlines()
        if x.strip()
    }


def tiers(ids, nrows, mrows, t_hi, t3_ok=None):
    """Partition ids into tier1 / tier2 / tier3 / escalated, in order."""
    t1, t2, t3, esc = [], [], [], []
    for i in ids:
        c = nrows[i].get("confidence")
        if nrows[i]["got"] is not None and c is not None and c >= t_hi:
            t1.append(i)
        elif nrows[i]["got"] == mrows[i]["got"]:
            t2.append(i)
        elif t3_ok is not None and i in t3_ok:
            t3.append(i)
        else:
            esc.append(i)
    return t1, t2, t3, esc


def prec(sel, rows):
    return round(sum(rows[i]["tool_ok"] for i in sel) / len(sel), 4) if sel else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier3", action="store_true", help="run Needle again on the rewrites")
    ap.add_argument("--t-hi", type=float, default=0.7, help="tier-1 threshold for the tier-3 run")
    a = ap.parse_args()

    items = load_items()
    ids = sorted(items)
    nrows = load_rows(NEEDLE / "results_two_stage_heuristic.json")
    mrows = load_rows(HERE / "results_tuned-e2.json")
    out: dict = {"n": len(ids)}

    # --- tiers 1-2 across thresholds ---------------------------------------
    table = []
    for t in T_HI:
        t1, t2, _, esc = tiers(ids, nrows, mrows, t)
        acc = t1 + t2
        table.append(
            {
                "t_hi": t,
                "tier1_n": len(t1),
                "tier1_precision": prec(t1, nrows),
                "tier2_n": len(t2),
                "tier2_precision": prec(t2, nrows),
                "cumulative_coverage": round(len(acc) / len(ids), 4),
                "cumulative_precision": prec(acc, nrows),
                "escalated": len(esc),
                "escalated_needle_would_have_been_right": sum(nrows[i]["tool_ok"] for i in esc),
            }
        )
    out["two_tier"] = table

    # single-gate baselines on the same denominator, for comparison
    conf_only = []
    for t in T_HI:
        sel = [i for i in ids if nrows[i]["got"] is not None and (nrows[i].get("confidence") or 0) >= t]
        conf_only.append(
            {"t": t, "coverage": round(len(sel) / len(ids), 4), "precision": prec(sel, nrows)}
        )
    same = [i for i in ids if nrows[i]["got"] == mrows[i]["got"]]
    out["baselines"] = {
        "confidence_only": conf_only,
        "agreement_only": {"coverage": round(len(same) / len(ids), 4), "precision": prec(same, nrows)},
        "no_gate": {"coverage": 1.0, "precision": prec(ids, nrows)},
    }

    if a.tier3:
        out["tier3"] = run_tier3(ids, items, nrows, mrows, a.t_hi)

    (HERE / "results_cascade.json").write_text(json.dumps(out, indent=1))

    print(f"CASCADE over {len(ids)} queries (tier 1: Needle confidence; tier 2: agreement)")
    print(f"{'t_hi':>5}{'tier1':>7}{'prec':>7}{'tier2':>7}{'prec':>7}{'coverage':>10}{'precision':>11}{'escalated':>11}")
    for r in table:
        p1 = f"{r['tier1_precision']:.3f}" if r["tier1_precision"] is not None else "   -  "
        p2 = f"{r['tier2_precision']:.3f}" if r["tier2_precision"] is not None else "   -  "
        print(
            f"{r['t_hi']:>5}{r['tier1_n']:>7}{p1:>7}{r['tier2_n']:>7}{p2:>7}"
            f"{r['cumulative_coverage']:>10.3f}{r['cumulative_precision']:>11.3f}{r['escalated']:>11}"
        )

    print("\nsingle gates on the same denominator:")
    for r in conf_only:
        print(f"  confidence >= {r['t']}      coverage {r['coverage']:.3f}  precision {r['precision']:.3f}")
    b = out["baselines"]["agreement_only"]
    print(f"  agreement alone       coverage {b['coverage']:.3f}  precision {b['precision']:.3f}")
    print(f"  no gate               coverage 1.000  precision {out['baselines']['no_gate']['precision']:.3f}")

    if a.tier3:
        print("\nTIER 3 — re-ask with Monad in the loop")
        for k, v in out["tier3"].items():
            if k == "examples":
                continue
            print(
                f"  {k:22} accepted {v['accepted']:2d} of {v['eligible']:2d} escalated  "
                f"precision-of-accepted {v['precision_of_accepted']}  "
                f"cascade coverage {v['cascade_coverage']:.3f} precision {v['cascade_precision']:.3f}"
            )
    return 0


def run_tier3(ids, items, nrows, mrows, t_hi: float) -> dict:
    """Re-route the escalated queries through Needle on a rewritten input."""
    sys.path.insert(0, str(NEEDLE))
    from needle_bsky.grouped import GroupedRouter

    _, _, _, esc = tiers(ids, nrows, mrows, t_hi)
    router = GroupedRouter(arm="tuned-min", threshold=0.0, stage1="heuristic")

    variants = {
        "trace_as_query": lambda i: (mrows[i].get("reasoning") or "").strip() or items[i]["query"],
        "query_plus_hint": lambda i: (
            f"{items[i]['query']} (suggested tool: {mrows[i]['got']})"
            if mrows[i].get("got")
            else items[i]["query"]
        ),
    }

    res = {}
    for name, rewrite in variants.items():
        accepted, examples = [], []
        for i in esc:
            d = router.route(rewrite(i))
            conf = d.confidence
            ok = d.tool is not None and d.tool in items[i]["tool"] if items[i]["tool"] else d.tool is None
            if conf is not None and conf >= t_hi and d.tool is not None:
                accepted.append((i, ok))
            if len(examples) < 6:
                examples.append({"id": i, "rewritten": rewrite(i)[:110], "got": d.tool, "conf": conf, "ok": ok})
        t1, t2, _, _ = tiers(ids, nrows, mrows, t_hi)
        base = t1 + t2
        cascade_right = sum(nrows[i]["tool_ok"] for i in base) + sum(1 for _, ok in accepted if ok)
        cascade_n = len(base) + len(accepted)
        res[name] = {
            "eligible": len(esc),
            "accepted": len(accepted),
            "precision_of_accepted": round(sum(1 for _, ok in accepted if ok) / len(accepted), 4)
            if accepted
            else None,
            "cascade_coverage": round(cascade_n / len(ids), 4),
            "cascade_precision": round(cascade_right / cascade_n, 4),
            "examples": examples,
        }
    return res


if __name__ == "__main__":
    raise SystemExit(main())
