#!/usr/bin/env python3
"""Write input files and fanout task lists for each codex wave.

    python3 tasks.py sessions DATA     # wave A: session cards from transcript digests
    python3 tasks.py writers DATA      # wave B: one page per topic (memories + assigned session cards)
    python3 tasks.py verifiers DATA    # wave C: check every citation on every page

Each command writes DATA/<wave>/{inputs/,out/,tasks.jsonl,schema.json}; run it
with claude-workspace scripts/fanout.py using subagent_type "auditor" (Haiku,
read-only tools, writes only its OUTPUT file).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cluster import tags  # noqa: E402

BATCH_CHARS = 100_000
TODAY = "2026-10-11"
SECTIONS = ["Muninn: identity, memory and operating rules",
            "Harness: Claude Code, containers, hooks, GitHub access",
            "Skills and tooling",
            "Research, experiments and paper reviews",
            "Projects (repos and products)",
            "Writing and publishing",
            "Oskar: preferences, life, work",
            "World: news and other people's work",
            "Autonomous time: perch, fly, inbox and routines"]

CARD_PROMPT = """[no-context]
You are summarising work sessions between Oskar (a human) and Muninn (an AI assistant with persistent memory) for an internal wiki. Today is {today}.

Input file: {inp}
It holds {n} session transcripts. Each starts with a header line:
=== SESSION <8-char id> | <start date> | <title>
Lines starting OSKAR are Oskar's messages; MUNINN are the assistant's replies; [tools: ...] lines list tool calls made in between.

Read the whole file with the Read tool. If Read says the output was truncated, keep reading with offset until you have seen the last session, {last}.

For each session write one card:
- "sid": the 8-char id from the header
- "date": the start date from the header
- "title": a plain 4-10 word name for what the session was about
- "asked": what Oskar asked for, 1-3 sentences, in your own words
- "did": what Muninn actually did and built, 2-5 sentences; name repos, files, commits and numbers that appear in the transcript
- "outcome": how it ended: shipped, abandoned, blocked (by what), or still open
- "corrections": a list of each time Oskar corrected Muninn or pushed back, one sentence each with what was wrong; [] if none
- "lessons": a list of rules or findings the session established that a later session should know, one sentence each; [] if none

Write only what the transcript says. Do not guess at work that is not shown.

Output: write ONE JSON object to {out}: {{"cards": [...]}} with one card per session, in file order.

Reply with exactly one line: sessions read: <count>; last session: <id>; cards written: <count>
"""

WRITER_PROMPT = """[no-context]
You are writing one page of the codex: an internal wiki compiled from the memory store of Muninn (an AI assistant) and its work sessions with Oskar (its human). Today is {today}.

Input file: {inp}
It holds {n} sources about one topic. Each starts with a header line:
=== MEMORY <8-char id> | <type> | created <YYYY-MM-DD> | tags: ...
=== SESSION <8-char id> | <date> | <title>   (a summary card of a work session)
A memory that starts "[AUDIT 2026-10-09: ...] WAS: ... NOW: ..." was checked against live state: the NOW sentence is true today and the WAS sentence is not.

Read the whole file with the Read tool. If Read says the output was truncated, keep reading with offset until you have seen the last source, {last}.

Then write the page:
- "title": a plain name for the topic, 2-6 words (a label, not a verdict)
- "section": exactly one of: {sections}
- "summary": 2-3 sentences: what this topic is and where it stands today
- "body": markdown, 400-1200 words. Use ## subheadings that name their content. Describe the current state first, then how it got there. Where sources disagree, the newer source wins; say what changed and when.
- "timeline": up to 12 dated events, oldest first: {{"date": "YYYY-MM-DD", "event": "...", "cite": ["m:xxxxxxxx"]}}
- "open": unresolved items, pending work or open questions the sources mention and no later source closes: {{"item": "...", "cite": [...]}}
- "contradictions": places where sources conflict and no later source settles it: {{"what": "...", "cite": [...]}}
- "terms": 3-10 names of things on this page another page might mention (repos, tools, people, concepts)

Citation rules:
- Every sentence in the body that states a fact ends with one or more citations in square brackets: [m:1a2b3c4d] for a memory, [s:1a2b3c4d] for a session. Use the 8-char ids from the headers, exactly.
- Cite only a source that says the thing. A sentence with no source that supports it does not go on the page.
- Write dates as YYYY-MM-DD, never "recently" or "last week".
- Plain declarative prose. No praise, no hype, no "interestingly".

Output: write ONE JSON object to {out} with the fields title, section, summary, body, timeline, open, contradictions, terms.

Reply with exactly one line: sources read: <count>; last source: <id>; citations used: <count>
"""

VERIFY_PROMPT = """[no-context]
You are fact-checking one page of an internal wiki against its sources. Today is {today}.

Page: {page} (JSON with fields title, section, summary, body, timeline, open, contradictions, terms)
Sources: {inp} holds {n} sources, each starting with a header line "=== MEMORY <8-char id> | ..." or "=== SESSION <8-char id> | ...". The page cites them as [m:<id>] and [s:<id>].

Read the page, then read the whole sources file with the Read tool (keep reading with offset until you reach the last source, {last}).

Check every sentence of the body and every timeline, open and contradictions entry:
- "supported": a cited source says it.
- "fixable": a cited source says something close; rewrite the sentence to match the source exactly, or swap in the right citation.
- "unsupported": no source in the file says it. Delete it.
A citation to an id that is not in the sources file counts as no citation.

Output: write ONE JSON object to {out} with:
- the same fields as the page (title, section, summary, body, timeline, open, contradictions, terms), corrected: unsupported text deleted, fixable text rewritten, everything else unchanged
- "checks": {{"checked": <count>, "supported": <count>, "fixed": <count>, "removed": <count>}}
- "changes": a list, one per fixed or removed item: {{"was": "<original text, at most 200 chars>", "action": "fixed" or "removed", "why": "<one sentence>"}}

Reply with exactly one line: checked <a>; supported <b>; fixed <c>; removed <d>
"""

CARD_SCHEMA = {"type": "object", "required": ["cards"], "properties": {"cards": {"type": "array", "minItems": 1,
               "items": {"type": "object", "required": ["sid", "date", "title", "asked", "did", "outcome"]}}}}
PAGE_FIELDS = ["title", "section", "summary", "body", "timeline", "open", "contradictions", "terms"]
PAGE_SCHEMA = {"type": "object", "required": PAGE_FIELDS,
               "properties": {"section": {"enum": SECTIONS}, "body": {"type": "string", "minLength": 400}}}
VERIFY_SCHEMA = {**PAGE_SCHEMA, "required": PAGE_FIELDS + ["checks", "changes"]}


def _jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _wave(data: Path, name: str) -> Path:
    w = data / name
    (w / "inputs").mkdir(parents=True, exist_ok=True)
    (w / "out").mkdir(exist_ok=True)
    return w


def _task(wave: str, i: str, prompt: str, out: Path) -> dict:
    return {"id": f"{wave}-{i}", "prompt": prompt, "out": str(out.resolve()), "subagent_type": "auditor",
            "model": "haiku", "description": f"codex {wave} {i}"}


def sessions(data: Path) -> int:
    w = _wave(data, "cards")
    index = [json.loads(line) for line in (data / "sessions.jsonl").open()]
    batches, cur, size = [], [], 0
    for r in index:
        if cur and size + r["chars"] > BATCH_CHARS:
            batches.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += r["chars"]
    batches.append(cur)
    tasks = []
    for i, b in enumerate(batches):
        f = w / "inputs" / f"c{i:02d}.txt"
        f.write_text("\n\n".join(f"=== SESSION {r['sid']} | {(r['start'] or '')[:10]} | {r['title'] or ''}\n"
                                 + (data / "sessions" / f"{r['sid']}.txt").read_text() for r in b))
        out = w / "out" / f"c{i:02d}.json"
        tasks.append(_task("cards", f"c{i:02d}", CARD_PROMPT.format(today=TODAY, inp=f.resolve(), n=len(b),
                                                                    last=b[-1]["sid"], out=out.resolve()), out))
    _jsonl(w / "tasks.jsonl", tasks)
    (w / "schema.json").write_text(json.dumps(CARD_SCHEMA))
    print(f"{len(tasks)} card tasks over {len(index)} sessions")
    return 0


def card_text(c: dict) -> str:
    lines = [f"Asked: {c['asked']}", f"Did: {c['did']}", f"Outcome: {c['outcome']}"]
    lines += [f"Correction: {x}" for x in c.get("corrections") or []]
    lines += [f"Lesson: {x}" for x in c.get("lessons") or []]
    return "\n".join(lines)


def assign_cards(data: Path, topics: list[dict]) -> dict[str, list[dict]]:
    """Each session card goes to the (up to) 3 topics whose centroid it is closest to."""
    from model2vec import StaticModel
    cards = [c for f in sorted((data / "cards" / "out").glob("*.json")) for c in json.loads(f.read_text())["cards"]]
    model = StaticModel.from_pretrained("minishlab/potion-base-32M")
    v = model.encode([c["title"] + "\n" + card_text(c) for c in cards])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cent = np.array([t["centroid"] for t in topics])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    sims = v @ cent.T
    out: dict[str, list[dict]] = {t["topic"]: [] for t in topics}
    for ci, c in enumerate(cards):
        best = np.argsort(-sims[ci])[:3]
        for j in best:
            if sims[ci, j] >= sims[ci, best[0]] - 0.05:
                out[topics[j]["topic"]].append(c)
    return out


def writers(data: Path) -> int:
    w = _wave(data, "pages")
    topics = json.loads((data / "topics.json").read_text())
    mems = {m["id"]: m for m in map(json.loads, (data / "memories.jsonl").open())}
    cards = assign_cards(data, topics)
    tasks = []
    for t in topics:
        blocks = []
        for mid in sorted(t["ids"], key=lambda i: mems[i].get("created_at") or mems[i].get("t") or ""):
            m = mems[mid]
            blocks.append(f"=== MEMORY {mid[:8]} | {m['type']} | created {(m.get('created_at') or m.get('t') or '')[:10]}"
                          f" | tags: {', '.join(tags(m))}\n{m['summary'].strip()}")
        for c in cards[t["topic"]]:
            blocks.append(f"=== SESSION {c['sid']} | {c['date']} | {c['title']}\n{card_text(c)}")
        f = w / "inputs" / f"{t['topic']}.txt"
        f.write_text("\n\n".join(blocks))
        last = blocks[-1].split("|")[0].split()[-1]
        out = w / "out" / f"{t['topic']}.json"
        tasks.append(_task("page", t["topic"], WRITER_PROMPT.format(
            today=TODAY, inp=f.resolve(), n=len(blocks), last=last, out=out.resolve(),
            sections="; ".join(f'"{s}"' for s in SECTIONS)), out))
    _jsonl(w / "tasks.jsonl", tasks)
    (w / "schema.json").write_text(json.dumps(PAGE_SCHEMA))
    n_cards = sum(len(v) for v in cards.values())
    print(f"{len(tasks)} page tasks; {n_cards} session-card placements")
    return 0


def verifiers(data: Path) -> int:
    w = _wave(data, "verified")
    tasks = []
    for page in sorted((data / "pages" / "out").glob("*.json")):
        topic = page.stem
        inp = data / "pages" / "inputs" / f"{topic}.txt"
        text = inp.read_text()
        n = text.count("\n=== ") + 1
        last = text.rsplit("\n=== ", 1)[-1].split("|")[0].split()[-1]
        out = w / "out" / f"{topic}.json"
        tasks.append(_task("verify", topic, VERIFY_PROMPT.format(
            today=TODAY, page=page.resolve(), inp=inp.resolve(), n=n, last=last, out=out.resolve()), out))
    _jsonl(w / "tasks.jsonl", tasks)
    (w / "schema.json").write_text(json.dumps(VERIFY_SCHEMA))
    print(f"{len(tasks)} verify tasks")
    return 0


if __name__ == "__main__":
    cmd, data = sys.argv[1], Path(sys.argv[2])
    raise SystemExit({"sessions": sessions, "writers": writers, "verifiers": verifiers}[cmd](data))
