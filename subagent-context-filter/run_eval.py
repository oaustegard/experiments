"""End-to-end eval: does a Jev-filtered transcript let a fresh subagent do the task?

Arms (same executor model, tool-less, one turn, isolated from settings/CLAUDE.md):
  none   task only                                   floor: what leaks without context
  jev1   task + Jev-selected chunks, condensed view   one window for most sessions
  jev4   task + Jev-selected chunks, 4x view          paged into ~30k-token windows
  full   task + the whole transcript                  proxy for a fork
  brief  task + a brief written by the same model from the whole transcript
         (the orchestrator-writes-the-background baseline; writer cost is counted)

Score: each task's facts are regexes over the deliverable (searched with re.I).
Usage: python3 run_eval.py [--arms none,jev1,...] [--limit N]
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jevfilter as jf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(DATA, "results")
EMPTY = os.path.join(DATA, "empty")
MODEL = "sonnet"
BUDGET = 20_000

EXEC_PROMPT = """You are a subagent. The parent agent delegated this task to you:

TASK: {task}

Write the deliverable now, in plain text. Work only from the context below. If a fact you need is not in it, say which fact is missing instead of guessing.

{context}"""

BRIEF_PROMPT = """You are the orchestrator of the session transcript below. You are about to delegate this task to a fresh subagent that cannot see the transcript:

TASK: {task}

Write the brief you would pass to that subagent: the background, facts and specifics it needs to do the task well. Output only the brief.

SESSION TRANSCRIPT:
{context}"""


def claude(prompt: str, timeout: int = 900) -> dict:
    t = time.time()
    p = subprocess.run(
        ["claude", "-p", "--safe-mode", "--tools", "", "--strict-mcp-config", "--no-session-persistence",
         "--model", MODEL, "--output-format", "json"],
        input=prompt, capture_output=True, text=True, cwd=EMPTY, timeout=timeout)
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"result": "", "error": (p.stderr or p.stdout)[-500:], "wall_s": time.time() - t}
    # modelUsage sums every model call in the run; top-level usage is only the last one
    mu = d.get("modelUsage") or {}
    inp = sum(m.get("inputTokens", 0) + m.get("cacheReadInputTokens", 0) + m.get("cacheCreationInputTokens", 0)
              for m in mu.values())
    out = sum(m.get("outputTokens", 0) for m in mu.values())
    return {"result": d.get("result", ""), "cost_usd": d.get("total_cost_usd"), "wall_s": round(time.time() - t, 1),
            "input_tokens": inp, "output_tokens": out, "num_turns": d.get("num_turns"), "is_error": d.get("is_error")}


def full_context(chunks):
    return "Full transcript of the parent session:\n\n" + "\n\n".join(jf.render(c, max_chars=10**9) for c in chunks)


def score_facts(text: str, facts: list) -> list:
    return [bool(re.search(f["regex"], text or "", re.I)) for f in facts]


def run_task(task: dict, arms: list) -> dict:
    sid = task["task_id"].rsplit("-", 1)[0]
    chunks = json.load(open(os.path.join(DATA, "chunks", f"{sid}.json")))
    rec = {"task_id": task["task_id"], "arms": {}}
    for arm in arms:
        path = os.path.join(OUT, f"{task['task_id']}.{arm}.json")
        if os.path.exists(path):
            rec["arms"][arm] = json.load(open(path)); continue
        r = {}
        try:
            if arm == "none":
                ctx = "(no context was passed)"
            elif arm in ("jev1", "jev4"):
                ctx, st = jf.build_context(task["task"], chunks, BUDGET, scale=1.0 if arm == "jev1" else 4.0)
                st["must_recall"] = [i in st["kept_ids"] for i in task["must_chunks"]]
                st.pop("scores")
                r["filter"] = st
            elif arm == "full":
                ctx = full_context(chunks)
            elif arm == "brief":
                w = claude(BRIEF_PROMPT.format(task=task["task"], context=full_context(chunks)))
                r["writer"] = {k: v for k, v in w.items() if k != "result"}
                ctx = "Brief from the parent agent:\n\n" + w["result"]
            r["context_tokens"] = jf.tokens(ctx)
            ex = claude(EXEC_PROMPT.format(task=task["task"], context=ctx))
            r["executor"] = {k: v for k, v in ex.items() if k != "result"}
            r["deliverable"] = ex["result"]
            r["facts"] = score_facts(ex["result"], task["facts"])
        except Exception as e:  # recorded, never silently scored as a pass or a fail
            r["error"] = f"{type(e).__name__}: {e}"[:400]
        json.dump(r, open(path, "w"))
        rec["arms"][arm] = r
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="none,jev1,jev4,full,brief")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(EMPTY, exist_ok=True)
    tasks = [t for f in sorted(glob.glob(os.path.join(DATA, "tasks", "*.json"))) for t in json.load(open(f))]
    if a.limit:
        tasks = tasks[: a.limit]
    arms = a.arms.split(",")
    print(f"{len(tasks)} tasks x {arms}", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(lambda t: run_task(t, arms), tasks):
            line = "  ".join(f"{k}:{sum(v.get('facts') or [])}/{len(v.get('facts') or [])}"
                             if "error" not in v else f"{k}:ERR" for k, v in rec["arms"].items())
            print(rec["task_id"], line, flush=True)


if __name__ == "__main__":
    main()
