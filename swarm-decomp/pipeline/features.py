#!/usr/bin/env python3
"""What kinds of function Haiku solved at which effort, and what it never solved.

    features.py runs/report/levels.json [--out runs/report/features.json]

Per target function, from its disassembly in tests/suites/<suite>/target.o:
  insns        instruction count
  calls        direct calls
  jumptable    an indirect jmp (a switch compiled to a jump table)
  fp           SSE floating-point arithmetic (not the pxor/movups gcc uses to zero memory)
  loop         a backward branch
  offsets      distinct struct displacements off a register (how much layout the source must get right)
  stack        stack frame size (sub $N, %rsp)
  clone        a gcc clone (.part/.isra/.constprop)
Then solve rates by feature bucket, and cost and flips per rung from the ledger."""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "lib"))
import objcmp  # noqa: E402

FP = re.compile(r"^(add|sub|mul|div|sqrt|max|min|cvt\w*|ucomi|comi)s[sd]$|^cvt")
DISP = re.compile(r"(-?0x[0-9a-f]+|\d+)\(%r(?!ip|sp|bp)")


def feats(obj: objcmp.Obj, name: str) -> dict:
    b = objcmp.body(obj, name)
    ins = b.insns
    branch_back = any(i.mnem.startswith("j") and not i.ref and re.fullmatch(r"0x[0-9a-f]+", i.ops or "")
                      and int(i.ops, 16) <= i.off for i in ins)
    stack = 0
    for i in ins[:8]:
        m = re.match(r"\$(0x[0-9a-f]+|\d+), %rsp", i.ops)
        if i.mnem.startswith("sub") and m:
            stack = int(m.group(1), 0)
    return {"insns": len(ins), "calls": sum(1 for i in ins if i.mnem.startswith("call")),
            "jumptable": any(i.mnem.startswith("jmp") and "*" in i.ops for i in ins),
            "fp": any(FP.match(i.mnem) for i in ins), "loop": branch_back,
            "offsets": len({m for i in ins for m in DISP.findall(i.ops)}), "stack": stack,
            "clone": bool(re.search(r"\.(part|isra|constprop)\.", name))}


def bucket(v, edges):
    for e in edges:
        if v < e:
            return f"<{e}"
    return f">={edges[-1]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("levels")
    ap.add_argument("--out", default=str(ROOT / "runs" / "report" / "features.json"))
    a = ap.parse_args()
    rows = json.loads(Path(a.levels).read_text())
    objs: dict[str, objcmp.Obj] = {}
    for r in rows:
        suite, name = r["test"].split("/", 1)
        if suite not in objs:
            objs[suite] = objcmp.Obj(str(ROOT / "tests" / "suites" / suite / "target.o"))
        r.update(feats(objs[suite], name))
    Path(a.out).write_text(json.dumps(rows, indent=0))

    def table(title, key):
        print(f"\n{title}")
        for target in ("neat", "lua"):
            g = collections.defaultdict(lambda: [0, 0])
            for r in rows:
                if r["target"] == target:
                    k = key(r)
                    g[k][0] += 1
                    g[k][1] += r["status"] == "PASS"
            print(f"  {target}: " + "  ".join(f"{k}: {s}/{n} ({100 * s / n:.0f}%)" for k, (n, s) in sorted(g.items(), key=lambda kv: str(kv[0]))))
    table("solve rate by instruction count", lambda r: bucket(r["insns"], [16, 48, 128, 384]))
    table("solve rate by direct calls", lambda r: bucket(r["calls"], [1, 3, 8, 16]))
    table("solve rate by distinct struct offsets", lambda r: bucket(r["offsets"], [1, 4, 10, 20]))
    for f in ("jumptable", "fp", "loop", "clone"):
        table(f"solve rate by {f}", lambda r, f=f: r[f])

    # cost and flips per rung, from the ledger
    last = {}
    for line in (ROOT / "swarm" / "state" / "ledger.jsonl").read_text().splitlines():
        if line.strip():
            x = json.loads(line)
            last[x["id"]] = x
    rung = collections.defaultdict(lambda: {"workers": 0, "merged": 0, "flips": 0, "partial": 0, "usd": 0.0})
    for x in last.values():
        if x.get("status") == "launched":
            continue
        k = f"{x.get('model') or 'haiku'}/{x.get('effort') or 'default'}" + ("/feature" if x.get("feature") else "")
        d = rung[k]
        d["workers"] += 1
        d["merged"] += x.get("status") == "merged"
        d["flips"] += len(x.get("gate_fixed") or [])
        d["partial"] += x.get("status") == "merged" and not x.get("gate_fixed")
        d["usd"] += x.get("cost_usd") or 0
    print("\ncost per rung (all workers, both targets)")
    for k, d in sorted(rung.items()):
        per = d["usd"] / d["flips"] if d["flips"] else float("nan")
        print(f"  {k:24s} workers {d['workers']:5d}  merged {d['merged']:4d}  flips {d['flips']:4d}  partial {d['partial']:3d}  "
              f"${d['usd']:7.2f}  $/worker {d['usd'] / max(1, d['workers']):.3f}  $/flip {per:.3f}")
    (Path(a.out).parent / "rungs.json").write_text(json.dumps(rung, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
