#!/usr/bin/env python3
"""Does a model pick the right skill among the overlapping subagent skills?

    python3 run.py queries-tasks                 # queries/tasks.jsonl: Sonnet writes queries from skill bodies
    python3 run.py queries                       # queries.json from the run (gold = source skill), then edit by hand
    python3 run.py select CATALOGUE.json TAG     # select-TAG/tasks.jsonl: each model picks a skill per query
    python3 run.py score TAG                     # accuracy per model and per gold skill, confusions

Each tasks.jsonl runs through dispatching-subagents/scripts/headless_fanout.py. The
selector sees every skill's name and description, as the harness lists them,
and nothing else.
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS = Path("/home/user/claude-skills")
SOURCES = ["dispatching-subagents", "agent-routing", "orchestrating-agents", "delegating-with-context",
           "flowing", "invoking-gemini", "deciding-with-confidence"]
MODELS = ["haiku", "sonnet", "opus"]
PER_SKILL = 6

QUERY_PROMPT = """Below is the body of an agent skill: the instructions an AI coding agent reads after \
choosing it. Write {n} requests that this skill is the right one for, as a person or an orchestrating \
agent would phrase them in Claude Code. Vary them: some short, some with a concrete situation, some where \
the need is implied rather than named. Do not use the skill's name or reuse distinctive phrases from its \
text; describe the situation in your own words. Skip requests that another skill could equally own.

SKILL BODY:
{body}"""

QUERY_SCHEMA = {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
                "required": ["queries"]}

SELECT_SYSTEM = """You are Claude Code, working in a cloud container. Before acting on a request you may \
load at most one skill. These skills are available, as name: description.

{catalogue}"""

SELECT_PROMPT = """Request: {query}

Which one skill would you load first for this request? Give its exact name, or "none" if no skill fits. \
Also give the skill you would load second, or "none"."""

SELECT_SCHEMA = {"type": "object", "properties": {"skill": {"type": "string"}, "second": {"type": "string"}},
                 "required": ["skill", "second"]}


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def structured(path: Path) -> dict | None:
    return json.loads(path.read_text()).get("structured_output") if path.exists() else None


def body(name: str) -> str:
    text = (SKILLS / name / "SKILL.md").read_text()
    return text.split("---", 2)[2][:9000] if text.startswith("---") else text[:9000]


def queries_tasks() -> None:
    tasks = [{"id": f"q-{s}", "model": "sonnet", "tools": "", "json_schema": QUERY_SCHEMA,
              "system_prompt": "You write evaluation requests. Reply only through the structured output.",
              "prompt": QUERY_PROMPT.format(n=PER_SKILL, body=body(s))} for s in SOURCES]
    jsonl(HERE / "queries" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} query tasks")


def queries() -> None:
    out = []
    for s in SOURCES:
        q = structured(HERE / "queries" / "run" / "results" / f"q-{s}.json") or {}
        for i, text in enumerate(q.get("queries", [])[:PER_SKILL]):
            out.append({"id": f"{s}-{i}", "query": text, "gold": [s], "source": "generated"})
    (HERE / "queries.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} queries -> queries.json (review gold by hand)")


def select(catalogue: Path, tag: str) -> None:
    cat = json.loads(catalogue.read_text())
    names = sorted(cat)
    random.Random(7).shuffle(names)  # no skill sits first by alphabet
    listing = "\n".join(f"- {n}: {cat[n]}" for n in names)
    qs = json.loads((HERE / "queries.json").read_text())
    tasks = [{"id": f"{m}--{q['id']}", "model": m, "tools": "", "json_schema": SELECT_SCHEMA,
              "system_prompt": SELECT_SYSTEM.format(catalogue=listing),
              "prompt": SELECT_PROMPT.format(query=q["query"])} for m in MODELS for q in qs]
    jsonl(HERE / f"select-{tag}" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} select tasks ({len(qs)} queries x {len(MODELS)} models)")


def score(tag: str) -> dict:
    qs = {q["id"]: q for q in json.loads((HERE / "queries.json").read_text())}
    res = HERE / f"select-{tag}" / "run" / "results"
    per_model = collections.defaultdict(lambda: [0, 0, 0])  # top1, top2, n
    per_gold = collections.defaultdict(lambda: [0, 0])
    confusions = collections.Counter()
    rows = []
    for f in sorted(res.glob("*.json")):
        m, qid = f.stem.split("--", 1)
        r = structured(f) or {}
        q = qs[qid]
        pick, second = (r.get("skill") or "").strip(), (r.get("second") or "").strip()
        ok1, ok2 = pick in q["gold"], pick in q["gold"] or second in q["gold"]
        per_model[m][0] += ok1
        per_model[m][1] += ok2
        per_model[m][2] += 1
        per_gold[q["gold"][0]][0] += ok1
        per_gold[q["gold"][0]][1] += 1
        if not ok1:
            confusions[(q["gold"][0], pick)] += 1
        rows.append({"model": m, "query": qid, "pick": pick, "second": second, "ok": ok1})
    out = {"tag": tag,
           "models": {m: {"top1": round(a / n, 3), "top2": round(b / n, 3), "n": n} for m, (a, b, n) in per_model.items()},
           "per_gold": {g: f"{a}/{n}" for g, (a, n) in sorted(per_gold.items())},
           "confusions": [[g, p, c] for (g, p), c in confusions.most_common()],
           "rows": rows}
    (HERE / f"score-{tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("models", "per_gold")}, indent=1))
    print("confusions:", out["confusions"][:15])
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "queries-tasks":
        queries_tasks()
    elif cmd == "queries":
        queries()
    elif cmd == "select":
        select(Path(sys.argv[2]), sys.argv[3])
    elif cmd == "score":
        score(sys.argv[2])
