#!/usr/bin/env python3
"""A/B test: does a stored codex page improve an answer drawn from recall?

Arm A is the store as it was before the codex: recall's top 10 with codex pages
dropped. Arm B is the store as it is now: recall's top 10, pages included. The
same model answers from each arm's memories alone. A blind judge then compares
the two answers against every memory in the question's topic, once in each
order. Questions where B retrieved no page give identical contexts, and they
serve as the noise floor.

    python3 ab.py DATA questions [ROUND]   # DATA/ab[ROUND]/questions/tasks.jsonl
    python3 ab.py DATA answers [ROUND]     # runs recall, DATA/ab[ROUND]/answers/tasks.jsonl
    python3 ab.py DATA judge [ROUND]       # DATA/ab[ROUND]/judge/tasks.jsonl
    python3 ab.py DATA report              # every round pooled, DATA/ab/report.json

Round 1 (ab/) takes 20 topics and alternates synthesis and fact questions. Later
rounds (ab2/, ...) take 30 unused topics and keep only synthesis questions, the
kind round 1 showed pages change.

Each tasks.jsonl runs with dispatching-subagents/scripts/headless_fanout.py
(claude -p children, no tools, structured output). DATA is the private build
directory (claude-workspace .codex/). No memory text is written outside it.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

PAGE = "CODEX PAGE:"
N_TOPICS = 20
SEED = 20261011
SAMPLE = 30           # memories shown to the question writer
TOP = 10              # recall depth each arm answers from
PILOT = {"t005", "t011", "t059", "t045", "t024", "t017", "t003", "t040", "t014", "t100"}

QUESTION_PROMPT = """You are writing test questions for a personal memory assistant. Below are {n} memories, \
oldest first, from one subject in its owner's memory store. Today is 2026-10-11.

Write two questions the owner might plausibly ask the assistant about this subject, in their own words:
1. "synthesis": answering it well needs at least three of these memories, ideally from different dates, \
for example the current state of something, how it changed, or what is still unresolved. Ask about the \
present state where the memories show a change.
2. "fact": one specific detail (a number, a name, a decision, a date) stated in one memory.

Phrase each question the way a person would ask in conversation. Do not quote distinctive strings \
from the memories, mention memory ids, or say "according to the memories". For each, give the correct \
answer as the memories support it (latest memory wins where they disagree), and the ids it rests on.

MEMORIES:
{memories}"""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "synthesis": {"type": "object", "properties": {
            "question": {"type": "string"}, "answer": {"type": "string"},
            "ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["question", "answer", "ids"]},
        "fact": {"type": "object", "properties": {
            "question": {"type": "string"}, "answer": {"type": "string"},
            "ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["question", "answer", "ids"]},
    },
    "required": ["synthesis", "fact"],
}

ANSWER_SYSTEM = ("You are a personal assistant with a long-term memory store. Answer the owner's question "
                 "from the retrieved memories below and nothing else. Today is 2026-10-11. Where memories "
                 "disagree, prefer the most recent and say so. Where they do not cover the question, say "
                 "what is missing. Answer in at most 150 words.")

ANSWER_PROMPT = "Question: {question}\n\nRetrieved memories (most relevant first):\n{memories}"

JUDGE_PROMPT = """You are grading two answers to a question put to a personal memory assistant. Today is \
2026-10-11. The REFERENCE below is every memory the store holds on this subject, oldest first, plus a \
reference answer. The reference answer was written from a sample of those memories and can be incomplete; \
the memories are the ground truth. Where memories disagree, the most recent holds.

Grade each answer on what it states:
- errors: count statements the REFERENCE contradicts, including stating an old value as current \
when a later memory changed it. A statement the reference does not cover is not an error.
- covered: of the points the reference answer makes, how many the answer gets right (integer, 0 to {points}).
- current: true when the answer gives the latest state the REFERENCE supports, false when it gives \
an outdated one or none.
Then pick the better answer for the owner: fewer errors first, then more points covered, then currency. \
"tie" only when neither is better on any of the three.

QUESTION: {question}

REFERENCE ANSWER ({points} points): {gold}

REFERENCE MEMORIES:
{reference}

ANSWER X:
{x}

ANSWER Y:
{y}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "object", "properties": {"errors": {"type": "integer"}, "covered": {"type": "integer"},
                                              "current": {"type": "boolean"}},
              "required": ["errors", "covered", "current"]},
        "y": {"type": "object", "properties": {"errors": {"type": "integer"}, "covered": {"type": "integer"},
                                              "current": {"type": "boolean"}},
              "required": ["errors", "covered", "current"]},
        "better": {"type": "string", "enum": ["x", "y", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["x", "y", "better", "reason"],
}


def fmt(m: dict) -> str:
    tags = m.get("tags") or []
    if isinstance(tags, str):
        tags = json.loads(tags or "[]")
    return (f"--- [{m['id'][:8]}] {m.get('type')} | {str(m.get('created_at') or m.get('t') or '')[:10]} | "
            f"tags: {', '.join(tags[:8])}\n{(m.get('summary') or '').strip()}")


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def structured(work: Path, tid: str) -> dict | None:
    f = work / "run" / "results" / f"{tid}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("structured_output")


def topic_memories(data: Path) -> tuple[dict, dict]:
    mems = {m["id"]: m for m in map(json.loads, (data / "memories.jsonl").open())}
    topics = {t["topic"]: t for t in json.loads((data / "topics.json").read_text())}
    by_topic = {k: sorted((mems[i] for i in t["ids"] if i in mems), key=lambda m: str(m.get("created_at")))
                for k, t in topics.items()}
    return by_topic, mems


def rdir(data: Path, rnd: int) -> Path:
    return data / ("ab" if rnd == 1 else f"ab{rnd}")


def used_topics(data: Path, rnd: int) -> set[str]:
    used = set()
    for r in range(1, rnd):
        f = rdir(data, r) / "questions" / "tasks.jsonl"
        used |= {json.loads(x)["id"][2:] for x in f.open()} if f.exists() else set()
    return used


def questions(data: Path, rnd: int = 1) -> None:
    by_topic, _ = topic_memories(data)
    skip = PILOT | used_topics(data, rnd)
    pool = sorted(k for k, v in by_topic.items() if k not in skip and len(v) >= 8)
    picked = sorted(random.Random(SEED + rnd - 1).sample(pool, N_TOPICS if rnd == 1 else 30))
    tasks = []
    for k in picked:
        ms = by_topic[k]
        if len(ms) > SAMPLE:
            ms = sorted(random.Random(f"{SEED}{k}").sample(ms, SAMPLE), key=lambda m: str(m.get("created_at")))
        tasks.append({"id": f"q-{k}", "prompt": QUESTION_PROMPT.format(n=len(ms), memories="\n\n".join(map(fmt, ms))),
                      "json_schema": QUESTION_SCHEMA, "tools": "",
                      "system_prompt": "You write evaluation questions. Reply only through the structured output."})
    jsonl(rdir(data, rnd) / "questions" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} question tasks: {picked}")


def remembering():
    sys.path.insert(0, "/home/user/claude-workspace/scripts")
    import memory_audit
    return memory_audit._remembering()


def answers(data: Path, rnd: int = 1) -> None:
    rem = remembering()
    qdir = rdir(data, rnd) / "questions"
    items = []
    for line in (qdir / "tasks.jsonl").open():
        tid = json.loads(line)["id"]
        q = structured(qdir, tid)
        if not q:
            print(f"skip {tid}: no questions")
            continue
        for kind in ("synthesis", "fact"):
            items.append({"topic": tid[2:], "kind": kind, **q[kind]})
    if rnd == 1:  # alternate kinds across topics so the 20 questions split 10/10
        chosen = [it for i, it in enumerate(items) if (i // 2) % 2 == (0 if it["kind"] == "synthesis" else 1)]
    else:
        chosen = [it for it in items if it["kind"] == "synthesis"]
    tasks, meta = [], []
    for n, it in enumerate(chosen):
        hits = list(rem.recall(it["question"], n=TOP + 6, raw=True, use_cache=False))
        arm_b = hits[:TOP]
        arm_a = [m for m in hits if not (m.get("summary") or "").startswith(PAGE)][:TOP]
        pages = [m["id"][:8] for m in arm_b if (m.get("summary") or "").startswith(PAGE)]
        qid = f"r{rnd}a{n:02d}" if rnd > 1 else f"a{n:02d}"
        meta.append({**it, "qid": qid, "pages_in_b": pages, "same_context": [m["id"] for m in arm_a] ==
                     [m["id"] for m in arm_b], "a_ids": [m["id"][:8] for m in arm_a],
                     "b_ids": [m["id"][:8] for m in arm_b],
                     "a_chars": sum(len(fmt(m)) for m in arm_a), "b_chars": sum(len(fmt(m)) for m in arm_b)})
        for arm, ms in (("A", arm_a), ("B", arm_b)):
            tasks.append({"id": f"{qid}-{arm}", "tools": "", "system_prompt": ANSWER_SYSTEM,
                          "prompt": ANSWER_PROMPT.format(question=it["question"],
                                                         memories="\n\n".join(map(fmt, ms)))})
    jsonl(rdir(data, rnd) / "answers" / "tasks.jsonl", tasks)
    (rdir(data, rnd) / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(meta)} questions, {sum(m['kind'] == 'synthesis' for m in meta)} synthesis; "
          f"pages in B: {sum(bool(m['pages_in_b']) for m in meta)}; identical contexts: "
          f"{sum(m['same_context'] for m in meta)}")


def answer_text(rd: Path, tid: str) -> str | None:
    f = rd / "answers" / "run" / "results" / f"{tid}.json"
    return json.loads(f.read_text()).get("result") if f.exists() else None


def judge(data: Path, rnd: int = 1) -> None:
    by_topic, _ = topic_memories(data)
    rd = rdir(data, rnd)
    meta = json.loads((rd / "meta.json").read_text())
    tasks = []
    for m in meta:
        a, b = answer_text(rd, f"{m['qid']}-A"), answer_text(rd, f"{m['qid']}-B")
        if not (a and b):
            print(f"skip {m['qid']}: missing answer")
            continue
        points = max(1, m["answer"].count(".") + m["answer"].count(";"))
        ref = "\n\n".join(map(fmt, by_topic[m["topic"]]))
        for order, (x, y) in (("ab", (a, b)), ("ba", (b, a))):
            tasks.append({"id": f"{m['qid']}-{order}", "tools": "", "model": "opus", "json_schema": JUDGE_SCHEMA,
                          "system_prompt": "You are a strict, fair grader. Reply only through the structured output.",
                          "prompt": JUDGE_PROMPT.format(question=m["question"], gold=m["answer"], points=points,
                                                        reference=ref, x=x, y=y)})
    jsonl(rd / "judge" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} judge tasks")


def report(data: Path) -> None:
    rows = []
    rnd = 1
    while (rdir(data, rnd) / "meta.json").exists():
        rows += judged_rows(rdir(data, rnd), rnd)
        rnd += 1
    report_rows(data, rows)


def judged_rows(rd: Path, rnd: int) -> list[dict]:
    meta = json.loads((rd / "meta.json").read_text())
    jdir = rd / "judge"
    rows = []
    for m in meta:
        per = {}
        for order in ("ab", "ba"):
            j = structured(jdir, f"{m['qid']}-{order}")
            if not j:
                continue
            a, b = (j["x"], j["y"]) if order == "ab" else (j["y"], j["x"])
            win = {"x": order[0], "y": order[1], "tie": "tie"}[j["better"]]
            per[order] = {"A": a, "B": b, "winner": win.upper() if win != "tie" else "tie", "reason": j["reason"]}
        rows.append({k: m[k] for k in ("qid", "topic", "kind", "question", "pages_in_b", "same_context",
                                       "a_chars", "b_chars")} | {"round": rnd, "judged": per})
    return rows


def report_rows(data: Path, rows: list[dict]) -> None:
    stored = {v[:8]: k for k, v in json.loads((data / "store" / "stored.json").read_text()).items()}
    for r in rows:
        r["own_page"] = any(stored.get(p) == r["topic"] for p in r["pages_in_b"])

    def tally(sel):
        votes = [v["winner"] for r in sel for v in r["judged"].values()]
        agree = sum(1 for r in sel if len(r["judged"]) == 2 and
                    len({v["winner"] for v in r["judged"].values()}) == 1)
        def mean(arm, key):
            vals = [int(v[arm][key]) for r in sel for v in r["judged"].values()]
            return round(statistics.mean(vals), 2) if vals else None
        return {"questions": len(sel), "votes": {k: votes.count(k) for k in ("A", "B", "tie")},
                "orders_agree": agree, "errors_A": mean("A", "errors"), "errors_B": mean("B", "errors"),
                "covered_A": mean("A", "covered"), "covered_B": mean("B", "covered"),
                "current_A": mean("A", "current"), "current_B": mean("B", "current")}

    differ = [r for r in rows if not r["same_context"]]
    out = {"all": tally(rows), "page_retrieved": tally([r for r in differ if r["pages_in_b"]]),
           "identical_context": tally([r for r in rows if r["same_context"]]),
           "synthesis_own_page": tally([r for r in differ if r["kind"] == "synthesis" and r["own_page"]]),
           "synthesis_other_change": tally([r for r in differ if r["kind"] == "synthesis" and not r["own_page"]]),
           "fact": tally([r for r in differ if r["kind"] == "fact"]), "rows": rows}
    (data / "ab" / "report.json").write_text(json.dumps(out, indent=1))
    for k in [k for k in out if k != "rows"]:
        print(k, json.dumps(out[k]))


if __name__ == "__main__":
    data, cmd = Path(sys.argv[1]), sys.argv[2]
    if cmd == "report":
        report(data)
    else:
        {"questions": questions, "answers": answers, "judge": judge}[cmd](data, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
