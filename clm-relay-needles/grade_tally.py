"""Grade Custody Register results: final holder accuracy and exact handover counts over 40 assets."""
from __future__ import annotations

import json
from pathlib import Path

from gen_tally import episode

RES = Path(__file__).resolve().parent / "results_tally"
JEV_USD_PER_MTOK = 0.042


def grade_one(r: dict) -> dict:
    ep = episode(r["n"], r["seed"])
    got = r["answer"].get("parsed") or {}
    A = ep["assets"]
    hold = sum(str((got.get(a) or {}).get("holder", "")).strip() == ep["final_holder"][a] for a in A)
    cnt = [(got.get(a) or {}).get("count") for a in A]
    exact = sum(isinstance(c, int) and c == ep["final_count"][a] for a, c in zip(A, cnt))
    errs = [abs(c - ep["final_count"][a]) for a, c in zip(A, cnt) if isinstance(c, int)]
    # what the carried text could support at best: assets whose current holder appears in it
    steps = r["steps"]
    ag = [s["agent"] for s in steps if "agent" in s]
    return {"cond": r["cond"], "n": r["n"], "seed": r["seed"], "events": sum(ep["final_count"].values()),
            "parsed": bool(got), "holder_acc": hold / len(A), "count_exact": exact / len(A),
            "count_mae": round(sum(errs) / len(errs), 2) if errs else None, "answered": len(errs),
            "carried_tokens": steps[-1]["carried_tokens"],
            "relay_usd": round(sum(a.get("cost", 0) for a in ag), 3),
            "answer_usd": round(r["answer"].get("cost", 0), 3),
            "jev_usd": round(sum(s.get("jev_tokens", 0) for s in steps) * JEV_USD_PER_MTOK / 1e6, 4),
            "truncations": sum("truncated_from" in a for a in ag),
            "agent_failures": sum(not a.get("ok") for a in ag)}


def main() -> None:
    rows = [grade_one(json.loads(p.read_text())) for p in sorted(RES.glob("*__*.json"))]
    rows.sort(key=lambda r: (r["n"], r["cond"], r["seed"]))
    (RES / "summary.json").write_text(json.dumps(rows, indent=1))
    print("| chunks | events | cond | holder acc | count exact | count MAE | carried tok | relay $ | answer $ | Jev $ | cut at cap |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['n']} | {r['events']} | {r['cond']} | {r['holder_acc']:.3f} | {r['count_exact']:.3f} | "
              f"{r['count_mae']} | {r['carried_tokens']} | {r['relay_usd']} | {r['answer_usd']} | {r['jev_usd']} | "
              f"{r['truncations']} |")


if __name__ == "__main__":
    main()
