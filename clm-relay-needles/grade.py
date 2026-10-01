"""Grade relay results: exact-line needle recall, corrupted needles, kept noise, carried size, cost."""
from __future__ import annotations

import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from gen import episode

RES = Path(__file__).resolve().parent / "results"
JEV_USD_PER_MTOK = 0.042  # TypeSafe bills input tokens only (METHODS.md)
ID = re.compile(r"^\[[^\]#]+#")


def grade_one(r: dict) -> dict:
    ep = episode(r["n"], r["seed"], r["variant"])
    needles = set(ep["needles"])
    allines = {l for c in ep["chunks"] for l in c.splitlines() if l.startswith("[")}
    final = [l.strip() for l in r["final"].splitlines()]
    fset = set(final)
    hit = needles & fset
    ids_in_final = {m.group(0) for l in final if (m := ID.match(l))}
    corrupted = [n for n in needles - fset if ID.match(n).group(0) in ids_in_final]
    noise = (allines - needles) & fset
    steps = r["steps"]
    ag = [s["agent"] for s in steps if "agent" in s]
    return {"cond": r["cond"], "variant": r["variant"], "n": r["n"], "seed": r["seed"],
            "needles": len(needles), "recall": len(hit) / len(needles), "corrupted": len(corrupted),
            "noise_kept": len(noise), "final_tokens": steps[-1]["carried_tokens"],
            "cost_usd": round(sum(a.get("cost", 0) for a in ag), 4),
            "agent_in_tokens": sum(a.get("in", 0) for a in ag), "agent_out_tokens": sum(a.get("out", 0) for a in ag),
            "agent_failures": sum(not a.get("ok") for a in ag), "truncations": sum("truncated_from" in a for a in ag),
            "jev_tokens": sum(s.get("jev_tokens", 0) for s in steps),
            "jev_usd": round(sum(s.get("jev_tokens", 0) for s in steps) * JEV_USD_PER_MTOK / 1e6, 4)}


def main() -> None:
    rows = [grade_one(json.loads(p.read_text())) for p in sorted(RES.glob("*__*.json"))]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["n"], r["cond"])].append(r)
    table = []
    for (v, n, c), rs in sorted(groups.items()):
        table.append({"variant": v, "n_chunks": n, "cond": c, "seeds": len(rs),
                      "recall_mean": st.mean(r["recall"] for r in rs),
                      "recall_each": [round(r["recall"], 4) for r in rs],
                      "corrupted": sum(r["corrupted"] for r in rs), "noise_kept": sum(r["noise_kept"] for r in rs),
                      "final_tokens_mean": round(st.mean(r["final_tokens"] for r in rs)),
                      "cost_usd": round(sum(r["cost_usd"] for r in rs), 3),
                      "jev_usd": round(sum(r["jev_usd"] for r in rs), 4),
                      "truncations": sum(r["truncations"] for r in rs),
                      "agent_failures": sum(r["agent_failures"] for r in rs)})
    (RES / "summary.json").write_text(json.dumps({"episodes": rows, "table": table}, indent=1))
    print("| variant | chunks | cond | seeds | recall | per seed | corrupted | noise kept | final tok | subagent $ | Jev $ | cut at cap |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in table:
        print(f"| {t['variant']} | {t['n_chunks']} | {t['cond']} | {t['seeds']} | {t['recall_mean']:.3f} | "
              f"{t['recall_each']} | {t['corrupted']} | {t['noise_kept']} | {t['final_tokens_mean']} | "
              f"{t['cost_usd']} | {t['jev_usd']} | {t['truncations']} |")


if __name__ == "__main__":
    main()
