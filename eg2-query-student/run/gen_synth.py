"""Synthetic search queries written by Haiku from the corpus posts.

Usage:
  python3 -I run/gen_synth.py tasks            # writes out/synth/tasks.jsonl
  python3 $HEADLESS run out/synth/tasks.jsonl --work out/synth/run --model haiku --tools "" ...
  python3 -I run/gen_synth.py collect          # writes out/queries/synth_{train,test}.jsonl

Posts are split once (seed 0) into disjoint train and test pools, so no test query comes
from a post a training query came from. One query per post, 50 posts per child. Posts
shorter than 40 characters are skipped (too little to ask about).
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYN = ROOT / "out" / "synth"
N_TRAIN, N_TEST, PER = 24000, 1500, 50

SYSTEM = """You write search queries for a retrieval benchmark over Bluesky posts about AI, \
machine learning, tech and science. For each numbered post, write ONE query that a person \
searching a Bluesky search box might type, for which that post would be a good result.

Vary the style across the batch, roughly:
- 40% short keyword queries, 2-5 words (e.g. "claude coding agent costs")
- 40% natural questions (e.g. "do LLMs actually reason or just pattern match?")
- 20% topical requests (e.g. "posts about copyright lawsuits against AI labs")

Rules: describe what the post is about in the searcher's own words; do not copy more than \
three consecutive words from the post; no @handles, URLs or hashtags; English only; at most \
20 words. If a post is not in English or has no searchable topic, still write a query about \
its gist. Return every post's index exactly once."""

SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {
        "type": "object",
        "properties": {"i": {"type": "integer"}, "q": {"type": "string"}},
        "required": ["i", "q"]}}},
    "required": ["queries"],
}


def posts():
    rows = [json.loads(l) for l in (ROOT / "out" / "posts.jsonl").open()]
    return [r for r in rows if r["text"] and len(r["text"]) >= 40]


def tasks():
    rows = posts()
    random.Random(0).shuffle(rows)
    split = {"train": rows[:N_TRAIN], "test": rows[N_TRAIN:N_TRAIN + N_TEST]}
    SYN.mkdir(parents=True, exist_ok=True)
    with (SYN / "tasks.jsonl").open("w") as f:
        for name, rs in split.items():
            for b in range(0, len(rs), PER):
                batch = rs[b:b + PER]
                body = "\n\n".join(f"[{r['i']}] {r['text'][:600]}" for r in batch)
                f.write(json.dumps({"id": f"{name}-{b // PER:04d}", "system_prompt": SYSTEM,
                                    "json_schema": SCHEMA, "prompt": "Posts:\n\n" + body}) + "\n")
    print(json.dumps({k: len(v) for k, v in split.items()}))


def collect():
    tasks_ = {json.loads(l)["id"]: json.loads(l) for l in (SYN / "tasks.jsonl").open()}
    out = {"train": [], "test": []}
    bad = 0
    for tid, t in tasks_.items():
        p = SYN / "run" / "results" / f"{tid}.json"
        if not p.exists():
            bad += 1
            continue
        res = json.loads(p.read_text())
        # Post texts can hold blank lines, so find the "[i] " markers rather than split on them.
        want = {int(i) for i in re.findall(r"(?:^|\n\n)\[(\d+)\] ", t["prompt"])}
        seen = set()
        for q in (res.get("structured_output") or {}).get("queries", []):
            if q["i"] in want and q["i"] not in seen and q["q"].strip():
                seen.add(q["i"])
                out[tid.split("-")[0]].append({"id": f"post{q['i']}", "q": q["q"].strip(),
                                               "kind": "synth", "post": q["i"]})
    for name, rows in out.items():
        with (ROOT / "out" / "queries" / f"synth_{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps({k: len(v) for k, v in out.items()} | {"missing_tasks": bad}))


if __name__ == "__main__":
    {"tasks": tasks, "collect": collect}[sys.argv[1]]()
