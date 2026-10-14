#!/usr/bin/env python3
"""Dispatch-time nearness on the haiku-cc sibling ledgers: what is known before a worker starts?

Reads swarm/state/ledger.jsonl from the hcc/stats/s1..s3 branches. Outcome per task that reached
the apply step: conflict, or merged. Files are result.files_changed minus selftest files (always new).

  1. signature component (text before the first colon) shared with a live peer
  2. prior attempts on the same targets/key/rebase chain: how often they exist and predict files
  3. the hold rule on rebase tasks: a live peer at launch changes a file the conflicted patch changed
  4. the same rule on first attempts with an oracle (their own final files), as an upper bound
  5. what the hold costs: time until the last near peer lands

    python3 dispatch.py --repo <experiments checkout>
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess


def load(repo: str) -> dict:
    L: dict = {}
    for s in ("s1", "s2", "s3"):
        out = subprocess.run(["git", "-C", repo, "show", f"origin/hcc/stats/{s}:swarm/state/ledger.jsonl"],
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            r = json.loads(line)
            r["node"] = s
            L[r["id"]] = {**L.get(r["id"], {}), **r}
    return L


def files(r: dict) -> set[str]:
    return {f for f in (r.get("result") or {}).get("files_changed", []) if "/selftest/" not in f}


def end(r: dict):
    return r.get("t_gate") or r.get("t_done")


def conflict(r: dict) -> bool:
    return "conflict applying" in (r.get("reason") or "")


def reached(r: dict) -> bool:
    return conflict(r) or r["status"] == "merged"


def live(L: dict, r: dict) -> list[dict]:
    t0 = r["t_launch"]
    return [o for o in L.values() if o["node"] == r["node"] and o["id"] != r["id"]
            and o["t_launch"] <= t0 < (end(o) or 1e12)]


def split(rows, label):
    for flag in (True, False):
        b = [c for f, c in rows if f == flag]
        print(f"  {label if flag else 'no ' + label:>24}: n={len(b):3d} conflicted {sum(b):3d} "
              f"({sum(b) / max(len(b), 1):.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    L = load(ap.parse_args().repo)
    R = [r for r in L.values() if reached(r)]
    print(f"{len(L)} tasks, {len(R)} reached apply, {sum(map(conflict, R))} conflicted")

    print("1. live peer with the same signature component")
    split([(any(o["signature"].split(":")[0] == r["signature"].split(":")[0] for o in live(L, r)), conflict(r))
           for r in R], "same component")

    T = sorted(L.values(), key=lambda r: r["t_launch"])
    jac, have, withf = [], 0, 0
    for i, r in enumerate(T):
        f = files(r)
        if not f:
            continue
        withf += 1
        prev = [o for o in T[:i] if (end(o) or 1e12) < r["t_launch"] and (
            set(o["targets"]) & set(r["targets"]) or o["key"] == r["key"] or r.get("rebase_of") == o["id"])]
        pf = set().union(*map(files, prev)) if prev else set()
        if pf:
            have += 1
            jac.append(len(f & pf) / len(f | pf))
    print(f"2. prior attempt known at dispatch: {have}/{withf} tasks; mean Jaccard with final files "
          f"{st.mean(jac):.2f}; rebase tasks {sum(1 for r in L.values() if r.get('rebase_of'))}")

    rows, waits = [], []
    for r in L.values():
        src = L.get(r.get("rebase_of") or "")
        if not src:
            continue
        near = [o for o in live(L, r) if files(o) & files(src)]
        if near and all(end(o) for o in near):
            waits.append(max(end(o) for o in near) - r["t_launch"])
        if reached(r):
            rows.append((bool(near), conflict(r)))
    print("3. rebase tasks: a live peer changes a file the conflicted patch changed")
    split(rows, "near peer (hold)")
    print("4. first attempts, oracle files")
    split([(any(files(o) & files(r) for o in live(L, r)), conflict(r)) for r in R if not r.get("rebase_of")],
          "near peer (oracle)")
    print(f"5. hold cost: {len(waits)} rebases would wait; median {st.median(waits):.0f} s, "
          f"p90 {sorted(waits)[int(0.9 * len(waits))]:.0f} s")


if __name__ == "__main__":
    main()
