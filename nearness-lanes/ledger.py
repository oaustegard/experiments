#!/usr/bin/env python3
"""Backtest file-level nearness on the real haiku-cc sibling ledgers (s1-s3).

A rejected candidate's diff was never kept, only its files_changed, so this is file level only.
For each candidate that reached the apply step (merged, or rejected with "conflict applying"):
  target   main it was applied onto: the hub commit named in a rebase conflict, else the last
           sibling merge at or before its gate time
  moved    files changed base..target; window = merges in between
  overlap  files_changed & moved (selftest files excluded: always new files)
Also a dispatch-time predictor: does any task launched while this one ran, or merged in its
window, share its signature component (the text before the first colon)?

Ledgers come from the hcc/stats/s1..s3 branches. s2 kept no merges.jsonl, and sibling mains also
moved by federation restacks the merges log does not record, so a candidate whose reconstructed
window is empty has an unknown target; the summary reports the rows with a known window.

    python3 ledger.py --repo <experiments checkout> --out ledger.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

REB = re.compile(r"rebase onto hcc/main ([0-9a-f]{7,40})")


def git(repo, *a):
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None


def lines(repo: str, node: str, name: str) -> list[str]:
    return (git(repo, "show", f"origin/hcc/stats/{node}:swarm/state/{name}") or "").splitlines()


def last(ls: list[str]) -> dict:
    out = {}
    for l in ls:
        r = json.loads(l)
        out[r["id"]] = {**out.get(r["id"], {}), **r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for s in ("s1", "s2", "s3"):
        led = last(lines(a.repo, s, "ledger.jsonl"))
        merges = [json.loads(l) for l in lines(a.repo, s, "merges.jsonl")]
        for r in led.values():
            reason = r.get("reason") or ""
            confl = "conflict applying" in reason
            if not (confl or r["status"] == "merged") or not r.get("t_gate") and not confl:
                continue
            files = [f for f in (r.get("result") or {}).get("files_changed", []) if "/selftest/" not in f]
            if not files:
                continue
            m = REB.search(reason)
            if m:
                target = m[1]
            else:
                t = r.get("t_gate") or r.get("t_done")
                before = [x for x in merges if x["time"] <= t and x["task"] != r["id"]]
                target = before[-1]["commit"] if before else r["base"]
            moved = git(a.repo, "diff", "--name-only", r["base"], target)
            if moved is None:
                continue
            moved = set(moved.split())
            win = git(a.repo, "rev-list", "--count", f"{r['base']}..{target}")
            comp = r["signature"].split(":")[0]
            t0, t1 = r["t_launch"], r.get("t_gate") or r.get("t_done") or r["t_launch"]
            peers = [o for o in led.values() if o["id"] != r["id"] and o["t_launch"] < t1
                     and (o.get("t_done") or 1e12) > t0]
            rows.append({"node": s, "id": r["id"], "conflict": confl, "window": int(win or 0),
                         "nfiles": len(files), "overlap": sorted(set(files) & moved),
                         "comp": comp, "peer_same_comp": sum(o["signature"].split(":")[0] == comp for o in peers),
                         "npeers": len(peers)})
    json.dump(rows, open(a.out, "w"), indent=0)
    print(f"{len(rows)} candidates reached apply; {sum(r['window'] == 0 for r in rows)} with no known target dropped")
    rows = [r for r in rows if r["window"] > 0]
    n, c = len(rows), sum(r["conflict"] for r in rows)
    fl = [r for r in rows if r["overlap"]]
    tp = sum(r["conflict"] for r in fl)
    print(f"{n} candidates reached apply, {c} conflicted")
    print(f"file overlap with what moved: flags {len(fl)} ({100*len(fl)/n:.0f}%), precision {tp/len(fl):.2f}, recall {tp/c:.2f}")
    for lo, hi in ((1, 5), (6, 20), (21, 60), (61, 10**6)):
        b = [r for r in rows if lo <= r["window"] <= hi]
        if b:
            print(f"  window {lo}-{hi}: n={len(b)} conflict rate {sum(r['conflict'] for r in b)/len(b):.2f}, "
                  f"overlap rate {sum(bool(r['overlap']) for r in b)/len(b):.2f}")


if __name__ == "__main__":
    main()
