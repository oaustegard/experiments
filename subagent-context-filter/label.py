"""Relabel the eval: one tool-less Sonnet labeller per session writes 4 tasks with fact regexes.

Same instructions as the 2026-09-23 Workflow labellers (recovered from that
session's transcript); the rendered session is passed inline instead of read
with tools. The September labels lived in the gitignored data/ and are gone.
Validation: every regex compiles, matches its own evidence, does not match its
task text, and every must_chunk id exists. A failing task is dropped, not fixed.

Usage: EVAL_DATA=<dir> python3 label.py
"""
import concurrent.futures as cf
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import DATA, claude  # noqa: E402

PROMPT = """You are labelling an evaluation dataset. Below is a Claude Code session transcript split into chunks, each starting with a marker like [17]. Read it in full.

Author 4 tasks that the session's orchestrator could plausibly delegate to a fresh subagent AT THE END of the session. A task must:
- depend on specific facts that appear ONLY in this transcript (numbers, ids, paths, names, decisions, measured results, error causes) — a capable model without the transcript must not be able to produce them;
- be phrased the way a busy orchestrator would phrase it: refer to things by name ("the xr search latency numbers", "the CLAUDE.md hierarchy decision") and NEVER restate the facts themselves in the task text;
- produce a short written deliverable (a summary, a message, a config snippet, a command) whose correctness can be checked by regexes over the text.
Spread the tasks across different parts of the session (early, middle, late), and make at least one need facts from two or more distant chunks.

For each task give 2-4 required facts, each with a Python regex (searched case-insensitively) that matches a correct deliverable and would NOT match a vague or generic one. Keep regexes tolerant of formatting (e.g. 1,200 vs 1200; optional units) but specific to the fact. Each regex must match its own evidence string. List must_chunks: every chunk id that contains a required fact (all occurrences if a fact is repeated).

Do not include secrets, tokens or "[REDACTED]" values as facts.

Output ONLY a JSON array, no prose, with objects:
{{"task_id": "{s}-1", "task": "...", "facts": [{{"name": "...", "regex": "...", "evidence": "short quote from the transcript"}}], "must_chunks": [ids]}}

TRANSCRIPT:
{text}"""


def validate(t: dict, ids: set) -> list[str]:
    errs = []
    for f in t.get("facts", []):
        try:
            rx = re.compile(f["regex"], re.I)
        except re.error as e:
            errs.append(f"bad regex {f['name']}: {e}"); continue
        if not rx.search(f.get("evidence", "")):
            errs.append(f"regex misses own evidence: {f['name']}")
        if rx.search(t["task"]):
            errs.append(f"fact leaks into task: {f['name']}")
    if not 2 <= len(t.get("facts", [])) <= 4:
        errs.append("fact count")
    if not t.get("must_chunks") or any(i not in ids for i in t["must_chunks"]):
        errs.append("must_chunks")
    return errs


def label(s: str) -> str:
    out = os.path.join(DATA, "tasks", f"{s}.json")
    if os.path.exists(out):
        return f"{s}: exists"
    ids = {c["id"] for c in json.load(open(os.path.join(DATA, "chunks", f"{s}.json")))}
    text = open(os.path.join(DATA, "rendered", f"{s}.txt")).read()
    r = claude(PROMPT.format(s=s, text=text), timeout=1500)
    m = re.search(r"\[.*\]", r["result"], re.S)
    if not m:
        return f"{s}: no JSON ({r.get('error', '')[:200]})"
    tasks, notes = [], []
    for i, t in enumerate(json.loads(m.group(0)), 1):
        t["task_id"] = f"{s}-{i}"
        errs = validate(t, ids)
        if errs:
            notes.append(f"{t['task_id']} dropped: {errs}")
        else:
            tasks.append(t)
    json.dump(tasks, open(out, "w"), indent=1)
    return f"{s}: {len(tasks)} tasks, {sum(len(t['facts']) for t in tasks)} facts " + "; ".join(notes)


if __name__ == "__main__":
    os.makedirs(os.path.join(DATA, "tasks"), exist_ok=True)
    sessions = sorted(f[:-5] for f in os.listdir(os.path.join(DATA, "chunks")))
    with cf.ThreadPoolExecutor(8) as ex:
        for line in ex.map(label, sessions):
            print(line, flush=True)
