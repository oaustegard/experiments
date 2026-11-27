"""Score every context chunk of every usable delegation with Jev: the teacher labels.

Uses the shipped filter's flat scorer (jevfilter.score at 4x view), which asks Jev
one question per chunk: does it hold information the task needs? Flat rather
than hierarchical so that every chunk gets a score to distil from.

Usable delegations: not `[no-context]` (the hook's opt-out, used by fan-out
workers whose instructions live in files), and at most 3 per session with the
same first 40 characters (templated fan-outs).

Usage: EVAL_DATA=<private dir> python3 teach.py [--limit N] [--workers 4]
Appends one row per delegation to <EVAL_DATA>/labels.jsonl and skips rows
already there, so it resumes after an interruption.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

sys.path.insert(0, str(experiment("subagent-context-filter")))
import jevfilter  # noqa: E402

DATA = Path(os.environ.get("EVAL_DATA") or HERE / "data")


def usable(rows):
    seen, out = collections.Counter(), []
    for r in rows:
        if r["task"].lstrip().startswith("[no-context]"):
            continue
        k = (r["session"], r["task"][:40])
        seen[k] += 1
        if seen[k] <= 3:
            out.append(r)
    return out


def key(r):
    return f"{r['session']}:{r['deleg_chunk']}"


def label(r, chunks_cache):
    chunks = chunks_cache[r["session"]]
    ctx = [c for c in chunks if c["id"] < r["deleg_chunk"]
           and not (c["kind"] == "tool" and c["id"] == r["deleg_chunk"])]
    ctx = [c for c in ctx if not (c["kind"] == "tool" and (
        c.get("tool") in ("Agent", "Task") or c.get("tool", "").endswith(("create_session", "send_message"))))]
    t0 = time.time()
    try:
        scores, stats = jevfilter.score(r["task"], ctx, scale=4.0)
        return {"key": key(r), "scores": {str(k): v for k, v in scores.items()}, "stats": stats,
                "n_ctx": len(ctx), "seconds": round(time.time() - t0, 1)}
    except jevfilter.FilterError as e:
        return {"key": key(r), "error": str(e)[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rows = usable([json.loads(l) for l in open(DATA / "delegations.jsonl")])
    out = DATA / "labels.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["key"] for l in open(out) if '"error"' not in l}
    todo = [r for r in rows if key(r) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(rows)} usable, {len(done)} done, {len(todo)} to label", flush=True)
    cache = {s: json.loads((DATA / "chunks" / f"{s}.json").read_text())
             for s in {r["session"] for r in todo}}
    tok, n, errs, t0 = 0, 0, 0, time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex, open(out, "a") as f:
        futs = [ex.submit(label, r, cache) for r in todo]
        for fut in cf.as_completed(futs):  # write as each finishes: one slow delegation must not hold the rest
            res = fut.result()
            f.write(json.dumps(res) + "\n"); f.flush()
            n += 1
            if "error" in res:
                errs += 1
            else:
                tok += res["stats"]["jev_input_tokens"]
            if n % 10 == 0 or n == len(todo):
                print(f"{n}/{len(todo)} errors={errs} jev_tokens={tok:,} (${tok * 0.042e-6:.2f}) "
                      f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
