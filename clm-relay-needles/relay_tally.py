"""Relay on the Custody Register task (gen_tally.py): a running result, carried under the same 24k cap.

Conditions
  state     a Haiku 4.5 `claude -p` subagent per chunk updates state.md, seeded with the initial
            register. It can overwrite counts and holders in place.
  jev       Jev keeps handover lines verbatim (p >= 0.5); over the cap the lowest-scored lines go.
  jev_fifo  same, but over the cap the oldest lines go (keeps the most recent handovers).
  both      each chunk, Jev scores first; the subagent sees which lines Jev flagged and updates the
            register (state.md, 8k reserved); Jev's flagged lines also go to a verbatim store of the
            most recent handovers (16k reserved). The answer call reads both, register first.

Every condition ends with the same answer call: Haiku 4.5, no tools, given the task (with the
initial register) and whatever was carried, returns JSON {asset: {"holder", "count"}}. For the Jev
conditions that call has to replay the kept lines itself; nothing computed the register earlier.

    python3 relay_tally.py --conds state,jev,jev_fifo --sizes 32,96 --seeds 0 --workers 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import relay
from gen_tally import episode, task
from relay import CAP_TOKENS, ENC, JEV_KEEP, MODEL, jev_scores, ntok, subagent

HERE = Path(__file__).resolve().parent
GUARD = False   # --guard: reject register edits that drop rows or lower counts (Phase 3 finding)
# "both": the register and Jev's verbatim store get reserved shares of the cap, so neither can
# squeeze the other (the Phase 1 failure, ERRORS.md #2).
BOTH_STATE_CAP, BOTH_JEV_CAP = 8_000, CAP_TOKENS - 8_000
WORK, RES = HERE / "work_tally", HERE / "results_tally"


def fit(kept: list[list], cap: int, policy: str) -> list[list]:
    """kept = [[line, score, order], ...]"""
    while kept and ntok("\n".join(k[0] for k in kept)) > cap:
        kept.remove(min(kept, key=lambda k: k[1] if policy == "score" else k[2]))
    return kept


REG_ROW = re.compile(r"^\|\s*(\S+)\s*\|\s*([A-Za-z]+)\s*\|\s*(\d+)\s*\|", re.M)


def register_violation(prev: str, new: str, assets: list[str], max_new: int) -> str | None:
    """Edit gate for the register: every asset keeps a row, no count goes down, and the counts
    cannot rise by more than the chunk could hold. Returns the reason, or None if the edit passes."""
    before = {a: int(c) for a, _, c in REG_ROW.findall(prev)}
    after = {a: int(c) for a, _, c in REG_ROW.findall(new)}
    missing = [a for a in assets if a not in after]
    if missing:
        return f"rows missing for {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
    down = [f"{a} {before[a]}->{after[a]}" for a in assets if a in before and after[a] < before[a]]
    if down:
        return f"counts went down ({', '.join(down[:5])}{' ...' if len(down) > 5 else ''}); a count only ever increases"
    rise = sum(after[a] - before.get(a, 0) for a in assets)
    if rise > max_new:
        return f"counts rose by {rise} in total, more than the {max_new} lines in this chunk"
    return None


ANSWER_SYS = "You answer from the material given. Output only the JSON object requested, nothing else."


def answer(task_text: str, carried: str, assets: list[str]) -> dict:
    prompt = (f"TASK:\n{task_text}\n\nThe stream has ended. Everything you carried forward is below.\n"
              f"<<<CARRIED BEGIN>>>\n{carried}\n<<<CARRIED END>>>\n\n"
              f"Report the final register as one JSON object mapping each of the {len(assets)} assets to "
              f'{{"holder": <name>, "count": <number of handovers since the start>}}. Assets: {", ".join(assets)}')
    cmd = ["claude", "-p", "--model", MODEL, "--tools", "", "--setting-sources", "", "--strict-mcp-config",
           "--no-session-persistence", "--system-prompt", ANSWER_SYS, "--output-format", "json"]
    for attempt in range(3):
        p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900)
        try:
            d = json.loads(p.stdout)
            txt = d.get("result") or ""
            m = re.search(r"\{.*\}", txt, re.S)
            return {"parsed": json.loads(m.group(0)) if m else None, "raw": txt[:4000],
                    "cost": d.get("total_cost_usd", 0)}
        except (json.JSONDecodeError, AttributeError):
            time.sleep(10 * (attempt + 1))
    return {"parsed": None, "raw": (p.stdout or p.stderr)[-2000:], "cost": 0}


def run_episode(cond: str, n: int, seed: int, log) -> Path:
    label = cond + ("+guard" if GUARD else "")
    out = RES / f"{label}__{n}__{seed}.json"
    if out.exists():
        return out
    ep = episode(n, seed)
    tk = task(ep)
    wd = WORK / f"{label}__{n}__{seed}"
    wd.mkdir(parents=True, exist_ok=True)
    prog_f = wd / "progress.json"
    prog = json.loads(prog_f.read_text()) if prog_f.exists() else {"next": 0, "steps": [], "kept": [], "order": 0}
    sf = wd / "state.md"
    if not sf.exists():
        sf.write_text("| asset | holder | count |\n|---|---|---|\n"
                      + "".join(f"| {a} | {p} | 0 |\n" for a, p in ep["initial"].items()))
    for c in range(prog["next"], n):
        chunk, step = ep["chunks"][c], {"chunk": c}
        if cond.startswith("jev") or cond == "both":
            sc, jt = jev_scores(tk, chunk)
            order = {l: i for i, l in enumerate(chunk.splitlines())}
            new = [[l, s, prog["order"] + order[l]] for l, s in sc.items() if s >= JEV_KEEP]
            prog["order"] += 1000
            cap, rule = ((BOTH_JEV_CAP, "fifo") if cond == "both"
                         else (CAP_TOKENS, "score" if cond == "jev" else "fifo"))
            prog["kept"] = fit(prog["kept"] + new, cap, rule)
            prog["kept"].sort(key=lambda k: k[2])  # chronological, so the answerer can replay
            (wd / "kept.md").write_text("\n".join(k[0] for k in prog["kept"]))
            step.update(jev_new=len(new), jev_tokens=jt)
            step["carried_tokens"] = ntok((wd / "kept.md").read_text())
        if not cond.startswith("jev"):
            cap = BOTH_STATE_CAP if cond == "both" else CAP_TOKENS
            used = ntok(sf.read_text())
            note = f"state.md may hold at most {cap} tokens; it holds {used} now. Anything past the limit is cut off."
            jnote = ""
            if cond == "both":
                flagged = [l for l in chunk.splitlines() if sc.get(l, 0) >= JEV_KEEP]
                jnote = ("A line filter flagged these lines of this chunk as likely handovers, in order. It is "
                         "usually right but can miss one or flag a non-handover, so check the chunk too:\n"
                         + ("\n".join(flagged) if flagged else "(none)"))
            prev = sf.read_text()
            r = subagent(tk, chunk, wd, note, jnote)
            if GUARD:
                why = register_violation(prev, sf.read_text(), ep["assets"], len(chunk.splitlines()) - 1)
                if why:
                    sf.write_text(prev)
                    r2 = subagent(tk, chunk, wd, note, jnote + f"\nYour last attempt at this chunk was rejected and "
                                  f"undone: {why}. state.md is back to its previous version; update it again.")
                    why2 = register_violation(prev, sf.read_text(), ep["assets"], len(chunk.splitlines()) - 1)
                    if why2:
                        sf.write_text(prev)
                    r = {**r2, "cost": r.get("cost", 0) + r2.get("cost", 0), "in": r.get("in", 0) + r2.get("in", 0),
                         "out": r.get("out", 0) + r2.get("out", 0), "gate_rejected": why, "gate_retry_failed": why2}
            s = sf.read_text()
            if ntok(s) > cap:
                sf.write_text(ENC.decode(ENC.encode(s)[:cap]))
                r["truncated_from"] = ntok(s)
            step["agent"] = r
            step["carried_tokens"] = ntok(sf.read_text()) + step.get("carried_tokens", 0)
        prog["steps"].append(step)
        prog["next"] = c + 1
        prog_f.write_text(json.dumps(prog))
        log(f"tally {label} n={n} s={seed} chunk {c + 1}/{n} carried={step['carried_tokens']}")
    if cond == "both":
        carried = ("REGISTER (authoritative; maintained step by step over the whole stream):\n" + sf.read_text()
                   + "\n\nMOST RECENT HANDOVER LINES (verbatim, oldest first; for reference only):\n"
                   + (wd / "kept.md").read_text())
    else:
        carried = (wd / "kept.md").read_text() if cond.startswith("jev") else sf.read_text()
    ans = answer(tk, carried, ep["assets"])
    out.write_text(json.dumps({"cond": label, "n": n, "seed": seed, "model": MODEL, "carried": carried,
                               "answer": ans, "steps": prog["steps"]}))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="state,jev,jev_fifo")
    ap.add_argument("--sizes", default="32,96")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--guard", action="store_true")
    a = ap.parse_args()
    global GUARD
    GUARD = a.guard
    RES.mkdir(exist_ok=True)
    lock = threading.Lock()

    def log(msg: str) -> None:
        with lock:
            print(time.strftime("%H:%M:%S"), msg, flush=True)

    jobs = sorted([(c, int(n), int(s)) for n in a.sizes.split(",") for s in a.seeds.split(",")
                   for c in a.conds.split(",")], key=lambda j: -j[1])
    failed = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_episode, *j, log): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                log(f"finished {futs[f]} -> {f.result().name}")
            except Exception as e:
                failed += 1
                log(f"FAILED {futs[f]}: {type(e).__name__}: {e}")
    import grade_tally
    grade_tally.main()
    print(f"incomplete: {failed} episodes failed; rerun to resume" if failed
          else "done -> results_tally/summary.json", flush=True)


if __name__ == "__main__":
    main()
