#!/usr/bin/env python3
"""Test stage: four conditions x RUNS runs over the test split, all with Haiku.

  A  baseline reviewer prompt (paper-watch/PROCEDURE.md profile + output format)
  B  A plus the frozen Meta-Skill
  C  B plus an evolver at the terminal checkpoint (KEEP, or EVOLVE: one send-back notice and an
     optional standing instruction). Standing instructions are reset for every paper.
     C reuses B's draft for the same paper and run, so C - B is the evolver's effect on identical drafts.
  D  like C, but standing instructions accumulate across papers in first_seen order within a run,
     and the reviewer drafts under them. Each D run is one sequential chain.

Checkpointed per (condition, run, paper) in cache/runs/. Resumable.
"""
import json, sys, threading
from concurrent.futures import ThreadPoolExecutor

import common as c

RUNS = 3
WORKERS = 8
OUT = c.HERE / "cache" / "runs"
META = None


def unit_path(cond, run, pid):
    return OUT / cond / f"r{run}" / f"{pid}.json"


def draft(pid, meta, standing):
    r = None
    for _ in range(2):
        r = c.call("haiku", c.reviewer_system(meta, standing), c.paper_block(pid))
        if r and c.parse_rating(r["text"]):
            break
    return r


def evolve(pid, meta, standing, dr):
    """Terminal checkpoint. Returns (evolver_call, decision_json, sendback_call, new_instruction)."""
    ev, j = None, None
    for _ in range(2):
        ev = c.call("haiku", c.evolver_system(meta, standing), c.evolver_user(pid, dr["text"]))
        j = c.parse_json(ev["text"]) if ev else None
        if j and str(j.get("decision", "")).upper() in ("KEEP", "EVOLVE"):
            break
        j = None
    if not j or str(j["decision"]).upper() != "EVOLVE":
        return ev, j, None, None
    instr = (j.get("standing_instruction") or "").strip() or None
    notice = (j.get("notice") or "").strip() or "Re-check the rating against the paper text."
    sb_standing = list(standing) + ([instr] if instr else [])
    sb = None
    for _ in range(2):
        sb = c.call("haiku", c.reviewer_system(meta, sb_standing), c.sendback_user(pid, dr["text"], notice))
        if sb and c.parse_rating(sb["text"]):
            break
    return ev, j, sb, instr


def record(cond, run, pid, dr, ev=None, j=None, sb=None, standing_before=(), standing_after=(), reused=False):
    draft_rating = c.parse_rating(dr["text"]) if dr else None
    final_call = sb if (sb and c.parse_rating(sb["text"])) else dr
    calls = [dr, ev, sb]  # a reused draft still counts: deployed C pays for its own draft
    rec = {"pid": pid, "cond": cond, "run": run,
           "draft_rating": draft_rating, "final_rating": c.parse_rating(final_call["text"]) if final_call else None,
           "decision": (str(j["decision"]).upper() if j else (None if cond in "CD" else "n/a")),
           "evolver": j, "draft": dr, "evolver_call": ev, "sendback": sb,
           "draft_reused_from_B": reused,
           "standing_before": list(standing_before), "standing_after": list(standing_after),
           "tokens": sum(c.tokens(x) for x in calls),
           "cost_usd": sum((x or {}).get("cost_usd") or 0 for x in calls)}
    c.save(unit_path(cond, run, pid), rec)
    print(cond, run, pid, rec["draft_rating"], "->", rec["final_rating"], rec["decision"], flush=True)
    return rec


def unit_ab(cond, run, pid):
    p = unit_path(cond, run, pid)
    if p.exists():
        return json.loads(p.read_text())
    meta = META if cond == "B" else None
    return record(cond, run, pid, draft(pid, meta, ()))


def unit_c(run, pid):
    p = unit_path("C", run, pid)
    if p.exists():
        return json.loads(p.read_text())
    b = unit_ab("B", run, pid)
    dr = b["draft"]
    if not dr:
        return record("C", run, pid, None)
    ev, j, sb, instr = evolve(pid, META, (), dr)
    return record("C", run, pid, dr, ev, j, sb, (), [instr] if instr else [], reused=True)


def chain_d(run, test):
    try:
        _chain_d(run, test)
    except Exception:
        import traceback
        traceback.print_exc()


def _chain_d(run, test):
    standing = []
    for pid in test:
        p = unit_path("D", run, pid)
        if p.exists():
            standing = json.loads(p.read_text())["standing_after"]
            continue
        dr = draft(pid, META, standing)
        if not dr:
            record("D", run, pid, None, standing_before=standing, standing_after=standing)
            continue
        ev, j, sb, instr = evolve(pid, META, standing, dr)
        after = standing + ([instr] if instr else [])
        record("D", run, pid, dr, ev, j, sb, standing, after)
        standing = after


def main(conds):
    global META
    META = (c.HERE / "meta_skill.md").read_text()
    _, test = c.split()
    threads = []
    if "D" in conds:
        for run in range(1, RUNS + 1):
            t = threading.Thread(target=chain_d, args=(run, test))
            t.start()
            threads.append(t)
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = []
        for run in range(1, RUNS + 1):
            for pid in test:
                for cond in ("A", "B"):
                    if cond in conds:
                        futs.append(ex.submit(unit_ab, cond, run, pid))
        for f in futs:
            f.result()
        if "C" in conds:
            list(ex.map(lambda a: unit_c(*a), [(r, p) for r in range(1, RUNS + 1) for p in test]))
    for t in threads:
        t.join()
    print("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ABCD")
