#!/usr/bin/env python3
"""Do the two models combine into something better than either alone?

Both answered the same 62 queries and the per-query rows are committed, so every
combination below is post-processing over existing artifacts. No model runs, no
network, no new decoding — which also means no combination here can be credited
with a latency it did not pay. Where a combination needs both models it costs
the sum of both, and that is stated.

    python3 synergy.py            # tables to stdout
    python3 synergy.py --json     # machine-readable, used by recheck.py

Seven questions:

1. **Complementarity.** How often is at least one of them right? An ensemble
   cannot beat that ceiling.
2. **Agreement gate.** When they independently name the same tool, how often is
   it right? That is a confidence signal that needs no confidence head — which
   matters because fine-tuning destroys Needle's.
3. **Confidence transfer.** Does Needle's calibrated score predict *Monad's*
   correctness on the same query?
4. **Name snapping.** Monad invents undeclared tool names. Snapping to the
   nearest declared name by edit distance costs nothing and needs no second
   model; it bounds how much of Monad's deficit is a missing grammar.
5. **Split the work.** Monad chooses the tool, Needle supplies the arguments —
   the division the copy probe implies.
6. **Fallback.** Does Monad rescue the queries Needle refuses or gets wrong?
7. **Per-category dispatch.** The ceiling of routing each category to whichever
   model wins it. Fitted to this eval by construction; reported as a bound.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

NEEDLE = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE))


def norm(v) -> str:
    return " ".join(str(v).strip().lstrip("@").lower().split())


def load_rows(path: Path) -> dict:
    return {r["id"]: r for r in json.loads(path.read_text())["rows"]}


def load_items() -> dict:
    return {
        json.loads(x)["id"]: json.loads(x)
        for x in (NEEDLE / "evalset.jsonl").read_text().splitlines()
        if x.strip()
    }


def declared_names() -> list[str]:
    from needle_bsky.router import load_schemas

    return [s["name"] for s in load_schemas("tuned-min")]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(item: dict, tool: str | None, args: dict) -> tuple[bool, bool]:
    accepted = item["tool"]
    if not accepted:
        ok = tool is None
        return ok, ok
    tool_ok = tool is not None and tool in accepted
    args_ok = tool_ok and all(
        k in args and norm(args[k]) == norm(v) for k, v in item.get("args", {}).items()
    )
    return tool_ok, args_ok


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    a_only = sum(1 for x, y in pairs if x and not y)
    b_only = sum(1 for x, y in pairs if y and not x)
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(a_only, b_only)
    return {"a_only": a_only, "b_only": b_only, "p": round(min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n), 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    items = load_items()
    names = declared_names()
    N = {
        "needle-base": load_rows(NEEDLE / "results_tuned-min.json"),
        "needle-2stage": load_rows(NEEDLE / "results_two_stage_heuristic.json"),
    }
    M = {
        "monad-e2": load_rows(HERE / "results_tuned-e2.json"),
        "monad-e3": load_rows(HERE / "results_tuned-e3.json"),
    }
    ids = sorted(items)
    out: dict = {}

    # --- 1. complementarity -------------------------------------------------
    comp = {}
    for nk, nrows in N.items():
        for mk, mrows in M.items():
            both = sum(1 for i in ids if nrows[i]["tool_ok"] and mrows[i]["tool_ok"])
            n_only = sum(1 for i in ids if nrows[i]["tool_ok"] and not mrows[i]["tool_ok"])
            m_only = sum(1 for i in ids if mrows[i]["tool_ok"] and not nrows[i]["tool_ok"])
            neither = len(ids) - both - n_only - m_only
            comp[f"{nk} x {mk}"] = {
                "both": both,
                f"{nk}_only": n_only,
                f"{mk}_only": m_only,
                "neither": neither,
                "union": round((both + n_only + m_only) / len(ids), 4),
                f"{nk}_alone": round((both + n_only) / len(ids), 4),
                f"{mk}_alone": round((both + m_only) / len(ids), 4),
            }
    out["complementarity"] = comp

    # --- 2. agreement gate --------------------------------------------------
    agree = {}
    for nk, nrows in N.items():
        for mk, mrows in M.items():
            same = [i for i in ids if nrows[i]["got"] == mrows[i]["got"]]
            diff = [i for i in ids if nrows[i]["got"] != mrows[i]["got"]]
            agree[f"{nk} x {mk}"] = {
                "n_agree": len(same),
                "coverage": round(len(same) / len(ids), 4),
                "precision_when_agreed": round(sum(nrows[i]["tool_ok"] for i in same) / len(same), 4) if same else None,
                "n_disagree": len(diff),
                f"{nk}_right_when_disagreed": sum(nrows[i]["tool_ok"] for i in diff),
                f"{mk}_right_when_disagreed": sum(mrows[i]["tool_ok"] for i in diff),
                "both_wrong_when_disagreed": sum(
                    1 for i in diff if not nrows[i]["tool_ok"] and not mrows[i]["tool_ok"]
                ),
            }
    out["agreement_gate"] = agree

    # --- 3. does Needle's confidence predict Monad? -------------------------
    conf_rows = [
        (N["needle-base"][i].get("confidence"), N["needle-base"][i]["tool_ok"], M["monad-e2"][i]["tool_ok"])
        for i in ids
        if N["needle-base"][i].get("confidence") is not None
    ]
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    out["confidence_transfer"] = {
        "n": len(conf_rows),
        "mean_needle_conf_when_needle_right": mean([c for c, nr, _ in conf_rows if nr]),
        "mean_needle_conf_when_needle_wrong": mean([c for c, nr, _ in conf_rows if not nr]),
        "mean_needle_conf_when_monad_right": mean([c for c, _, mr in conf_rows if mr]),
        "mean_needle_conf_when_monad_wrong": mean([c for c, _, mr in conf_rows if not mr]),
    }
    for t in (0.4, 0.6, 0.8):
        hi = [(nr, mr) for c, nr, mr in conf_rows if c >= t]
        out["confidence_transfer"][f"above_{t}"] = {
            "n": len(hi),
            "needle_acc": mean([1.0 if x else 0.0 for x, _ in hi]),
            "monad_acc": mean([1.0 if y else 0.0 for _, y in hi]),
        }

    # --- 4. snap Monad's tool name to the nearest declared one --------------
    snapped = {}
    for mk, mrows in M.items():
        rows, fixed, broke = [], [], []
        for i in ids:
            r = mrows[i]
            tool = r["got"]
            if tool is not None and tool not in names:
                best = min(names, key=lambda n: (levenshtein(tool, n), n))
                dist = levenshtein(tool, best)
                new = best if dist <= max(3, len(tool) // 3) else tool
            else:
                new = tool
            t_ok, ar_ok = score(items[i], new, r["arguments"])
            if t_ok and not r["tool_ok"]:
                fixed.append((i, r["got"], new))
            if r["tool_ok"] and not t_ok:
                broke.append((i, r["got"], new))
            rows.append((i, t_ok, ar_ok))
        on = [x for x in rows if items[x[0]]["tool"]]
        snapped[mk] = {
            "routable_before": round(sum(mrows[i]["tool_ok"] for i in ids if items[i]["tool"]) / len(on), 4),
            "routable_after": round(sum(1 for x in on if x[1]) / len(on), 4),
            "queries_fixed": fixed,
            "queries_broken": broke,
        }
    out["name_snapping"] = snapped

    # --- 5. Monad chooses, Needle supplies the arguments --------------------
    split = {}
    for nk, nrows in N.items():
        for mk, mrows in M.items():
            rows = []
            for i in ids:
                tool = mrows[i]["got"]
                args = nrows[i]["arguments"] if nrows[i]["got"] == tool else mrows[i]["arguments"]
                rows.append(score(items[i], tool, args))
            on = [(i, r) for i, r in zip(ids, rows) if items[i]["tool"]]
            split[f"{mk} tool + {nk} args"] = {
                "routable": round(sum(1 for _, r in on if r[0]) / len(on), 4),
                "args": round(sum(1 for _, r in on if r[1]) / len(on), 4),
                "note": "Needle's arguments are only available where it chose the same tool",
                "n_shared_tool": sum(1 for i in ids if nrows[i]["got"] == mrows[i]["got"] and mrows[i]["got"]),
            }
    out["split_roles"] = split

    # --- 6. Monad as fallback where Needle fails ---------------------------
    fb = {}
    for nk, nrows in N.items():
        for mk, mrows in M.items():
            # only a legal policy if the trigger is observable at run time:
            # Needle refusing, or Needle's confidence below a threshold.
            refused = [i for i in ids if nrows[i]["got"] is None and items[i]["tool"]]
            rescued = [i for i in refused if mrows[i]["tool_ok"]]
            fb[f"{nk} -> {mk} on refusal"] = {
                "needle_refused_a_routable_query": len(refused),
                "monad_rescued": len(rescued),
                "ids": rescued,
            }
            low = [
                i
                for i in ids
                if items[i]["tool"]
                and nrows[i].get("confidence") is not None
                and nrows[i]["confidence"] < 0.6
                and not nrows[i]["tool_ok"]
            ]
            fb[f"{nk} -> {mk} below conf 0.6"] = {
                "needle_wrong_and_unsure": len(low),
                "monad_right_there": sum(1 for i in low if mrows[i]["tool_ok"]),
            }
    out["fallback"] = fb

    # --- 7. per-category dispatch ceiling ----------------------------------
    cats = sorted({items[i]["cat"] for i in ids})
    disp = {}
    total = 0
    for c in cats:
        cid = [i for i in ids if items[i]["cat"] == c]
        n_acc = sum(N["needle-2stage"][i]["tool_ok"] for i in cid)
        m_acc = sum(M["monad-e2"][i]["tool_ok"] for i in cid)
        pick = "needle" if n_acc >= m_acc else "monad"
        total += max(n_acc, m_acc)
        disp[c] = {"n": len(cid), "needle": n_acc, "monad": m_acc, "pick": pick}
    out["category_dispatch"] = {
        "per_category": disp,
        "overall": round(total / len(ids), 4),
        "needle_2stage_alone": round(sum(N["needle-2stage"][i]["tool_ok"] for i in ids) / len(ids), 4),
        "caveat": "fitted to this eval by construction; an upper bound, not a result",
    }

    # --- 8. agreement gate against Needle's own confidence gate ------------
    nrows = N["needle-2stage"]
    mrows = M["monad-e2"]
    calls = [i for i in ids if nrows[i]["got"] is not None and nrows[i].get("confidence") is not None]
    sweep = []
    for t in (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        act = [i for i in calls if nrows[i]["confidence"] >= t]
        if act:
            sweep.append(
                {
                    "threshold": t,
                    "coverage": round(len(act) / len(calls), 4),
                    "precision": round(sum(nrows[i]["tool_ok"] for i in act) / len(act), 4),
                }
            )
    same_calls = [i for i in calls if nrows[i]["got"] == mrows[i]["got"]]
    combo = {}
    for t in (0.4, 0.6):
        both = [i for i in same_calls if nrows[i]["confidence"] >= t]
        if both:
            combo[f"agree_and_conf_{t}"] = {
                "coverage": round(len(both) / len(calls), 4),
                "precision": round(sum(nrows[i]["tool_ok"] for i in both) / len(both), 4),
            }
    out["gate_comparison"] = {
        "n_calls": len(calls),
        "needle_confidence_sweep": sweep,
        "agreement": {
            "coverage": round(len(same_calls) / len(calls), 4),
            "precision": round(sum(nrows[i]["tool_ok"] for i in same_calls) / len(same_calls), 4),
        },
        "agreement_and_confidence": combo,
    }

    if a.json:
        print(json.dumps(out, indent=1))
        return 0

    print("1. COMPLEMENTARITY — can an ensemble help at all?")
    for k, v in out["complementarity"].items():
        nk, mk = k.split(" x ")
        print(
            f"  {k:32} both {v['both']:2d}  {nk}-only {v[nk + '_only']:2d}  "
            f"{mk}-only {v[mk + '_only']:2d}  neither {v['neither']:2d}   union {v['union']:.3f}"
        )

    print("\n2. AGREEMENT GATE — precision when they independently name the same tool")
    for k, v in out["agreement_gate"].items():
        nk, mk = k.split(" x ")
        print(
            f"  {k:32} agree on {v['n_agree']:2d}/62 ({v['coverage']:.2f})  "
            f"precision {v['precision_when_agreed']:.3f}   "
            f"disagreements: {nk} right {v[nk + '_right_when_disagreed']:2d}, "
            f"{mk} right {v[mk + '_right_when_disagreed']:2d}, both wrong {v['both_wrong_when_disagreed']:2d}"
        )

    print("\n3. CONFIDENCE TRANSFER — does Needle's head say anything about Monad?")
    c = out["confidence_transfer"]
    print(f"  needle conf | needle right {c['mean_needle_conf_when_needle_right']}   wrong {c['mean_needle_conf_when_needle_wrong']}")
    print(f"  needle conf | monad  right {c['mean_needle_conf_when_monad_right']}   wrong {c['mean_needle_conf_when_monad_wrong']}")
    for t in (0.4, 0.6, 0.8):
        h = c[f"above_{t}"]
        print(f"    needle conf >= {t}: n={h['n']:2d}  needle acc {h['needle_acc']}  monad acc {h['monad_acc']}")

    print("\n4. NAME SNAPPING — nearest declared name by edit distance, no second model")
    for k, v in out["name_snapping"].items():
        print(f"  {k:10} routable {v['routable_before']:.3f} -> {v['routable_after']:.3f}   "
              f"fixed {len(v['queries_fixed'])}  broken {len(v['queries_broken'])}")
        for i, before, after in v["queries_fixed"]:
            print(f"      {i:11} {before!r} -> {after!r}")

    print("\n5. SPLIT ROLES — Monad chooses the tool, Needle supplies arguments")
    for k, v in out["split_roles"].items():
        print(f"  {k:36} routable {v['routable']:.3f}  args {v['args']:.3f}  (shared tool on {v['n_shared_tool']} queries)")

    print("\n6. FALLBACK — does Monad rescue what Needle misses?")
    for k, v in out["fallback"].items():
        if "refusal" in k:
            print(f"  {k:40} needle refused {v['needle_refused_a_routable_query']:2d}  monad rescued {v['monad_rescued']:2d}")
        else:
            print(f"  {k:40} needle wrong+unsure {v['needle_wrong_and_unsure']:2d}  monad right there {v['monad_right_there']:2d}")

    print("\n7. PER-CATEGORY DISPATCH CEILING (fitted — an upper bound)")
    d = out["category_dispatch"]
    for c_, v in d["per_category"].items():
        if v["needle"] != v["monad"]:
            print(f"  {c_:14} n={v['n']}  needle {v['needle']}  monad {v['monad']}  -> {v['pick']}")
    print(f"  overall {d['overall']:.3f} against needle-2stage alone {d['needle_2stage_alone']:.3f}")

    print("\n8. GATE COMPARISON — agreement against Needle's own confidence head")
    g = out["gate_comparison"]
    print(f"  over the {g['n_calls']} queries where needle-2stage emitted a call")
    for r in g["needle_confidence_sweep"]:
        print(f"    conf >= {r['threshold']:.1f}   coverage {r['coverage']:.3f}  precision {r['precision']:.3f}")
    ag = g["agreement"]
    print(f"    AGREEMENT     coverage {ag['coverage']:.3f}  precision {ag['precision']:.3f}")
    for k, v in g["agreement_and_confidence"].items():
        print(f"    {k:13} coverage {v['coverage']:.3f}  precision {v['precision']:.3f}")

    (HERE / "results_synergy.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
