#!/usr/bin/env python3
"""Which rung (model x effort) solved which functions, and what the unsolved tail looks like.

    levels.py FINAL.json [--out runs/report/levels.json]

A function is credited to the first merged worker whose gate flips list it (gate_fixed); a function
that turned PASS without appearing in any flips list (a shared callee or type fixed it as a side
effect) is "collateral". Each solved function carries its size in the target object, its project,
and the diff category it had just before it passed (from the gate's per-commit results). The
unsolved tail carries the same, plus its final frac. Rungs: <model>/<effort>, effort "" = default."""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "swarm" / "state"
sys.path.insert(0, str(ROOT / "tests" / "lib"))
import objcmp  # noqa: E402


def sizes() -> dict[str, int]:
    out = {}
    for sd in (ROOT / "tests" / "suites").iterdir():
        o = objcmp.Obj(str(sd / "target.o"))
        for n, f in o.funcs.items():
            out[f"{sd.name}/{n}"] = f.size + sum(p.size for p in f.parts)
    return out


def bucket(n: int) -> str:
    return "<64B" if n < 64 else "64-255B" if n < 256 else "256-1023B" if n < 1024 else ">=1KB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("final")
    ap.add_argument("--out", default=str(ROOT / "runs" / "report" / "levels.json"))
    a = ap.parse_args()
    final = json.loads(Path(a.final).read_text())
    size = sizes()
    last = {}
    for line in (STATE / "ledger.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["id"]] = r
    merged = sorted((r for r in last.values() if r.get("status") == "merged"), key=lambda r: r.get("t_gate") or 0)
    credit: dict[str, dict] = {}
    for r in merged:
        for t in r.get("gate_fixed") or []:
            credit.setdefault(t, r)
    # the category each test had just before it first passed, from the gate's per-commit results in time order
    commits = []
    for f in (STATE / "results").glob("*.json"):
        p = subprocess.run(["git", "-C", str(ROOT), "show", "-s", "--format=%ct", f.stem], capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            commits.append((float(p.stdout), f))
    commits.sort()
    prev_reason: dict[str, str] = {}
    first_pass: dict[str, float] = {}
    for ts, f in commits:
        for t, v in json.loads(f.read_text()).items():
            if v.get("status") == "PASS":
                first_pass.setdefault(t, ts)
            elif t not in first_pass:
                prev_reason[t] = (v.get("reason") or "").split(":", 1)[0]
    rung = lambda r: f"{r.get('model') or 'haiku'}/{r.get('effort') or 'default'}"
    rows = []
    for t, v in final.items():
        row = {"test": t, "target": "lua" if t.startswith("lua") else "neat", "proj": t.split("-", 1)[0],
               "size": size.get(t), "bucket": bucket(size.get(t) or 0), "status": v["status"],
               "frac": v.get("frac"), "category_before": prev_reason.get(t, "?")}
        if v["status"] == "PASS":
            r = credit.get(t)
            row["rung"] = rung(r) if r else "collateral"
            row["feature"] = bool(r and r.get("feature"))
        else:
            row["category_now"] = (v.get("reason") or "").split(":", 1)[0]
        rows.append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=0))
    for target in ("neat", "lua"):
        rs = [r for r in rows if r["target"] == target]
        print(f"\n== {target}: {sum(r['status'] == 'PASS' for r in rs)}/{len(rs)} matched")
        by = collections.defaultdict(list)
        for r in rs:
            by[r.get("rung", "UNSOLVED")].append(r)
        for k in sorted(by, key=lambda k: (k == "UNSOLVED", k)):
            g = by[k]
            med = statistics.median([r["size"] or 0 for r in g])
            bk = collections.Counter(r["bucket"] for r in g)
            cat = collections.Counter(r.get("category_now") or r["category_before"] for r in g).most_common(4)
            print(f"  {k:22s} n={len(g):4d} median size {med:6.0f}B  sizes {dict(bk)}  categories {cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
