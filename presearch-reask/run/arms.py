"""Run the five arms on Haiku 5.5 at --effort low through `claude -p`.

    python3 run/arms.py [--arms CB,PS,AG3,AG4,PSAG3] [--limit N] [--workers 8]

Each (arm, question) result is appended to results/raw.jsonl as soon as it
finishes; a rerun skips pairs already there. Each round is one stateless
`claude -p` call carrying the transcript so far.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bm25 import BM25, render  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "results" / "raw.jsonl"
MODEL, EFFORT, TOPK, TIMEOUT = "haiku", "low", 5, 180

ARMS = {  # name: (pre-search, tool budget or None)
    "CB": (False, None), "PS": (True, None),
    "AG3": (False, 3), "AG4": (False, 4), "PSAG3": (True, 3),
}

ANSWER_RULE = ("The answer is a short span (a name, date, number or phrase), not a sentence. "
               "Write it as <answer>short answer</answer>. If unsure, give your best guess anyway.")
SYS_ONESHOT = "You answer multi-hop questions. " + ANSWER_RULE
SYS_TOOL = ("You answer multi-hop questions using a search tool over a fixed collection of Wikipedia "
            "paragraphs. To search, write <search>your query</search> and stop: the results come back in "
            "the next message. You may search at most {budget} times. When you know the answer, write it. "
            + ANSWER_RULE + " Write only one tag per turn, either <search> or <answer>.")
REASK = "You have no turns left. Give your final answer now as <answer>short answer</answer>."

TAG = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
_lock = threading.Lock()


class CallError(Exception):
    pass


def claude(system: str, prompt: str) -> dict:
    argv = ["claude", "-p", "--model", MODEL, "--effort", EFFORT, "--output-format", "json",
            "--system-prompt", system, "--tools", "", "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", prompt]
    env = dict(os.environ, FORCE_PROMPT_CACHING_5M="1")
    last = None
    for attempt in range(3):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT, cwd="/tmp", env=env)
            out = json.loads(r.stdout)
            if out.get("is_error") or not isinstance(out.get("result"), str):
                raise CallError(str(out.get("result"))[:300])
            return {"text": out["result"], "cost": out.get("total_cost_usd") or 0.0,
                    "api_ms": out.get("duration_api_ms") or 0}
        except (subprocess.TimeoutExpired, ValueError, OSError, CallError) as e:
            last = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(5 * (attempt + 1))
    raise CallError(last)


def first_tag(text: str):
    m = TAG.search(text)
    return (m.group(1), m.group(2).strip(), text[:m.end()]) if m else (None, None, text)


def run_one(arm: str, q: dict, kb: BM25) -> dict:
    pre, budget = ARMS[arm]
    head = f"Question: {q['question']}\n"
    rec = {"arm": arm, "id": q["id"], "hops": q["hops"], "rounds": [], "searches": [], "cost": 0.0,
           "api_ms": 0, "retrieved": []}
    if pre:
        hits = kb.search(q["question"], TOPK)
        rec["retrieved"] += [h["id"] for h in hits]
        head += f"\n[pre-search] Results of a search for the question itself:\n<results>\n{render(hits)}\n</results>\n"

    def call(system, prompt):
        r = claude(system, prompt)
        rec["cost"] += r["cost"]
        rec["api_ms"] += r["api_ms"]
        return r["text"]

    if budget is None:
        text = call(SYS_ONESHOT, head + ("\nAnswer using the paragraphs above and what you know." if pre else
                                         "\nAnswer from what you know."))
        kind, val, _ = first_tag(text)
        rec["rounds"].append(text)
        rec["answer_pre"] = rec["answer"] = val if kind == "answer" else None
        rec["no_answer"] = kind != "answer"
        return rec

    system = SYS_TOOL.format(budget=budget)
    transcript, left, answer = "", budget, None
    failures = {"over_budget": 0, "no_tag": 0}
    for turn in range(budget + 1):
        prompt = head + transcript + f"\nSearches left: {left}. Your turn {turn + 1}:"
        text = call(system, prompt)
        kind, val, shown = first_tag(text)
        rec["rounds"].append(text)
        transcript += f"\nYour turn {turn + 1}:\n{shown.strip()}\n"
        if kind == "answer":
            answer = val
            break
        if kind == "search":
            if left > 0:
                left -= 1
                hits = kb.search(val, TOPK)
                rec["searches"].append(val)
                rec["retrieved"] += [h["id"] for h in hits]
                transcript += f"<results>\n{render(hits)}\n</results>\n"
            else:
                failures["over_budget"] += 1
                transcript += "(No searches left; the search was not run.)\n"
        else:
            failures["no_tag"] += 1
            transcript += "(No <search> or <answer> tag found.)\n"
    rec["failures"] = failures
    rec["answer_pre"] = answer
    rec["no_answer"] = answer is None
    if answer is None:  # the final-answer turn
        text = call(system, head + transcript + "\n" + REASK)
        kind, val, _ = first_tag(text)
        rec["reask_text"] = text
        answer = val if kind == "answer" else None
    rec["answer"] = answer
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    qs = [json.loads(l) for l in (HERE / "data" / "eval.jsonl").open()]
    if a.limit:
        qs = qs[:a.limit]
    done = set()
    if RAW.exists():
        done = {(r["arm"], r["id"]) for r in map(json.loads, RAW.open()) if "error" not in r}
    kb = BM25(HERE / "data" / "corpus.jsonl")
    tasks = [(arm, q) for arm in a.arms.split(",") for q in qs if (arm, q["id"]) not in done]
    print(f"{len(tasks)} tasks ({len(done)} already done)", flush=True)
    t0, n = time.time(), [0, 0]

    def work(task):
        arm, q = task
        try:
            rec = run_one(arm, q, kb)
        except CallError as e:
            rec = {"arm": arm, "id": q["id"], "error": str(e)}
        with _lock:
            with RAW.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n[0] += 1
            n[1] += "error" in rec
            if n[0] % 50 == 0 or n[0] == len(tasks):
                print(f"{n[0]}/{len(tasks)} errors {n[1]} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, tasks))


if __name__ == "__main__":
    main()
