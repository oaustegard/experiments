"""Round 2: Haiku research subagents with live WebSearch/WebFetch, with and without a pre-search block.

    python3 run/web_arms.py [--limit N] [--workers 6]

Fixture: the first 50 2-hop, 33 3-hop and 17 4-hop questions of data/eval.jsonl (round 1's random
sample). Arms, both `claude -p --model haiku` at default effort with WebSearch and WebFetch allowed:
  WEB    the question only
  PSWEB  the question plus the raw tool result of one WebSearch on the question verbatim, run first by
         a separate call (the orchestrator's pre-search; its cost is added to the arm)
Results go to results/web_raw.jsonl, one line per (arm, question), checkpointed.
"""
import argparse
import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "results" / "web_raw.jsonl"
STRATA = {"2hop": 50, "3hop": 33, "4hop": 17}
TIMEOUT = 420
SYSTEM = ("You are a research subagent. Use web search to find the answer to the question you are given. "
          "When done, end your reply with <answer>short answer</answer>, where the tag holds only the short "
          "answer span (a name, date, number or phrase) and nothing else. If unsure, give your best guess.")
PRESEARCH_SYSTEM = ("Call the WebSearch tool exactly once, with the user's message copied verbatim as the query "
                    "and no other parameters changed. Then reply with the single word DONE.")
LEAK = re.compile(r"musique|huggingface\.co/datasets|question_decomposition|answer_aliases", re.I)
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
_lock = threading.Lock()


def run_claude(system: str, prompt: str, tools: str) -> dict:
    argv = ["claude", "-p", "--model", "haiku", "--output-format", "stream-json", "--verbose",
            "--tools", tools, "--allowedTools", tools, "--setting-sources", "", "--strict-mcp-config",
            "--no-session-persistence", "--max-turns", "16", "--system-prompt", system, prompt]
    t0 = time.time()
    r = subprocess.run(argv, capture_output=True, text=True, timeout=TIMEOUT, cwd="/tmp", stdin=subprocess.DEVNULL)
    out = {"tool_uses": [], "tool_results": [], "text": "", "cost": 0.0, "wall_s": round(time.time() - t0, 1),
           "error": None}
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "assistant":
            for c in d["message"]["content"]:
                if c["type"] == "tool_use":
                    out["tool_uses"].append({"name": c["name"], "input": c["input"]})
                elif c["type"] == "text":
                    out["text"] += c["text"] + "\n"
        elif d.get("type") == "user":
            for c in d["message"]["content"] if isinstance(d["message"]["content"], list) else []:
                if isinstance(c, dict) and c.get("type") == "tool_result":
                    v = c["content"]
                    out["tool_results"].append(v if isinstance(v, str) else json.dumps(v))
        elif d.get("type") == "result":
            out["cost"] = d.get("total_cost_usd") or 0.0
            out["turns"] = d.get("num_turns")
            if d.get("is_error"):
                out["error"] = str(d.get("result"))[:300]
    if not out["text"] and not out["tool_uses"]:
        out["error"] = out["error"] or f"no output; stderr {r.stderr[:300]}"
    return out


def presearch(question: str) -> dict:
    r = run_claude(PRESEARCH_SYSTEM, question, "WebSearch")
    searches = [u for u in r["tool_uses"] if u["name"] == "WebSearch"]
    ok = len(searches) >= 1 and searches[0]["input"].get("query", "").strip() == question.strip() and r["tool_results"]
    return {"ok": bool(ok), "query": searches[0]["input"].get("query") if searches else None,
            "result": r["tool_results"][0] if r["tool_results"] else None, "cost": r["cost"], "error": r["error"]}


def run_one(arm: str, q: dict) -> dict:
    rec = {"arm": arm, "id": q["id"], "hops": q["hops"], "cost": 0.0}
    prompt = f"Question: {q['question']}\n"
    if arm == "PSWEB":
        ps = None
        for _ in range(2):
            ps = presearch(q["question"])
            rec["cost"] += ps["cost"]
            if ps["ok"]:
                break
        rec["presearch"] = {k: ps[k] for k in ("ok", "query", "error")}
        rec["presearch_leak"] = bool(ps["result"] and LEAK.search(ps["result"]))
        rec["presearch_result"] = ps["result"]
        if not ps["ok"]:
            rec["error"] = f"pre-search failed: {ps['error'] or ps['query']}"
            return rec
        prompt += ("\n[pre-search] The orchestrator ran one web search for the question itself before handing "
                   f"it to you. Its raw results:\n<results>\n{ps['result']}\n</results>\n")
    r = run_claude(SYSTEM, prompt, "WebSearch,WebFetch")
    rec["cost"] += r["cost"]
    m = ANSWER.findall(r["text"])
    rec.update({"answer": m[-1].strip() if m else None, "text": r["text"][-1500:], "wall_s": r["wall_s"],
                "turns": r.get("turns"), "searches": [u["input"].get("query") for u in r["tool_uses"] if u["name"] == "WebSearch"],
                "fetches": [u["input"].get("url") for u in r["tool_uses"] if u["name"] == "WebFetch"],
                "search_modes": [u["input"].get("mode") for u in r["tool_uses"] if u["name"] == "WebSearch"],
                "leak": any(LEAK.search(t) for t in r["tool_results"])})
    if r["error"]:
        rec["error"] = r["error"]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    allq = [json.loads(l) for l in (HERE / "data" / "eval.jsonl").open()]
    qs = []
    for h, n in STRATA.items():
        qs += [q for q in allq if q["hops"] == h][:n]
    if a.limit:
        qs = qs[:a.limit]
    done = set()
    if RAW.exists():
        done = {(r["arm"], r["id"]) for r in map(json.loads, RAW.open()) if "error" not in r}
    tasks = [(arm, q) for q in qs for arm in ("WEB", "PSWEB") if (arm, q["id"]) not in done]
    print(f"{len(tasks)} tasks ({len(done)} done)", flush=True)
    t0, n = time.time(), [0, 0, 0.0]

    def work(task):
        arm, q = task
        try:
            rec = run_one(arm, q)
        except subprocess.TimeoutExpired:
            rec = {"arm": arm, "id": q["id"], "error": "timeout"}
        with _lock:
            with RAW.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            n[0] += 1
            n[1] += "error" in rec
            n[2] += rec.get("cost", 0.0)
            if n[0] % 10 == 0 or n[0] == len(tasks):
                print(f"{n[0]}/{len(tasks)} errors {n[1]} cost ${n[2]:.2f} {time.time() - t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, tasks))


if __name__ == "__main__":
    main()
