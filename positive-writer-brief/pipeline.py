#!/usr/bin/env python3
"""Build task files for each stage and collect the children's outputs.

Stages, in order:
  draft  arm A and arm B drafts, N per task per arm          -> drafts/{A,B}/<task>__<k>.md
  edit   the declauding editor pass on every draft           -> finals/{A,B}/<task>__<k>.md
         (A + edit is "A-final"; B + edit is arm C)
  fresh  a newcomer read of every draft and every final      -> scores/fresh/<stage>__<arm>__<task>__<k>.json
  judge  blind rating, one call per task per stage per judge -> scores/judge/<judge>__<stage>__<task>.json

    python3 pipeline.py prep STAGE        # writes runs/STAGE/tasks.jsonl
    python3 pipeline.py collect STAGE     # copies runs/STAGE/results into the tree
Run each stage with dispatching-subagents' headless_fanout.py (see README).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTX = HERE / "context"
RUNS = HERE / "runs"
TASKS = json.loads((HERE / "tasks.json").read_text())
N_DRAFTS = 4
ARMS = ("A", "B")
DRAFTER = "claude-opus-5-5"
EDITOR = "claude-opus-5-5"
READER = "claude-sonnet-5-5"
JUDGES = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5"}

PERSONA = (
    "You are Muninn, an AI agent with persistent memory who works with Oskar Austegard "
    "and writes under your own byline. Before drafting anything a stranger will read, "
    "your instructions have you load the documents below. You have loaded them.\n\n"
)


def ctx(name: str) -> str:
    return (CTX / name).read_text()


def system_prompt(arm: str) -> str:
    if arm == "A":
        return (PERSONA
                + "=== muninn-voice-signature (config entry) ===\n\n" + ctx("voice_signature.md")
                + "\n\n=== declauding skill, SKILL.md ===\n\n" + ctx("declauding_SKILL.md"))
    if arm == "B":
        return PERSONA + "=== writer brief ===\n\n" + (HERE / "brief_b.md").read_text()
    raise ValueError(arm)


def task_block(t: dict) -> str:
    facts = "\n".join(f"- {f}" for f in t["facts"])
    return (f"Surface: {t['surface']}\nReader: {t['audience']}\n"
            f"Task: {t['instruction']}\n\nFacts (use these; do not add others):\n{facts}\n")


def draft_prompt(t: dict) -> str:
    return ("Write the deliverable described below. Put the complete text, in Markdown, in the "
            "`draft` field and nothing else.\n\n" + task_block(t))


EDITOR_SYSTEM_HEAD = (
    "You are Muninn's editor. A draft written for a stranger to read is about to ship, and you "
    "run the declauding pass on it, as the skill below describes, then check it against the "
    "voice signature. The skill's reference files register.md and exceptions.md are included.\n\n"
)


def editor_system() -> str:
    return (EDITOR_SYSTEM_HEAD
            + "=== declauding skill, SKILL.md ===\n\n" + ctx("declauding_SKILL.md")
            + "\n\n=== references/register.md ===\n\n" + ctx("declauding_register.md")
            + "\n\n=== references/exceptions.md ===\n\n" + ctx("declauding_exceptions.md")
            + "\n\n=== muninn-voice-signature (config entry) ===\n\n" + ctx("voice_signature.md"))


def lint_text(path: Path) -> str:
    r = subprocess.run([sys.executable, str(CTX / "declaude_lint.py"), str(path)],
                       capture_output=True, text=True)
    return r.stdout.strip() or "(no hits)"


def edit_prompt(t: dict, draft_path: Path) -> str:
    return ("Run the declauding pass on the draft below: workflow steps 1 to 4 and step 6. "
            "Step 2's scan has been run for you and its output is below. The structural review "
            "(2b), the ranker (2a), the newcomer read (4a) and the diff check (5) are not available "
            "in this run; do their checks by eye where you can. Keep every claim the draft makes "
            "and invent none. Put the finished text, in Markdown, in `final`, and one or two "
            "sentences on what you changed and any factual problems in `notes`.\n\n"
            + task_block(t)
            + "\n=== draft ===\n\n" + draft_path.read_text()
            + "\n\n=== declaude_lint.py output ===\n\n" + lint_text(draft_path) + "\n")


FRESH_PROMPT = (
    "You are {audience}. You have never seen this project or any earlier version of it, and you "
    "have only the text below, which is a {surface}. Read it as that reader would.\n\n"
    "List each phrase or passage that would make you stumble: one that leans on something the "
    "text never says (an earlier version, a prior conversation, a name it does not explain), one "
    "whose meaning or instruction you cannot work out, or one that contradicts another part of "
    "the text. Quote the phrase exactly. Leave out style preferences; list only what would stop "
    "or mislead you. An empty list is a valid answer. Then write, in one sentence, the point you "
    "take away from the text.\n\n=== text ===\n\n{text}\n"
)

JUDGE_SYSTEM = """You rate drafts of one piece of writing. All drafts were written from the same task and the same facts. Rate each draft on its own merits against the scales below; use the full range, and give different scores where drafts differ.

register (1-7): how far the prose reads as plain technical writing by a careful person. 7: every finding is stated where it is known, sentences say what things are, paragraphs end on facts, no stock model constructions. 1: saturated with them. Stock constructions include staged reveals and suspense, verdict or teaser headers, aphoristic or moralising closers, "it's not X, it's Y" contrasts, forced triads, announcing that something is important or interesting, drum-roll dashes and one-line paragraphs, filler and stacked hedges, and abstractions acting as agents.

focus (1-7): how well the text serves the reader of this surface. 7: the point the reader needs comes first; where the surface describes a thing (a README, a results Answer, a blog section) it describes the thing as it is now, with no narration of what changed or of the author's path to it; the thing, not the author, is the subject; the facts that matter are all there and nothing minor dominates. For a commit message or PR body the change is the subject by design, and focus means the reader learns what changed and why, first. 1: the reader has to dig for the point, or the text is about the author or about a change the reader never saw.

invented: the number of claims (numbers, names, behaviours, causes) not supported by the facts given.

Return a rating for every draft id."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"ratings": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"},
                       "register": {"type": "integer", "minimum": 1, "maximum": 7},
                       "focus": {"type": "integer", "minimum": 1, "maximum": 7},
                       "invented": {"type": "integer", "minimum": 0}},
        "required": ["id", "register", "focus", "invented"]}}},
    "required": ["ratings"]}

FRESH_SCHEMA = {
    "type": "object",
    "properties": {"stumbles": {"type": "array", "items": {
        "type": "object", "properties": {"quote": {"type": "string"}, "why": {"type": "string"}},
        "required": ["quote", "why"]}},
        "point": {"type": "string"}},
    "required": ["stumbles", "point"]}


def texts(stage: str):
    """(arm, task, k, path) for every text of a stage: 'raw' drafts or 'final' edits."""
    root = HERE / ("drafts" if stage == "raw" else "finals")
    for arm in ARMS:
        for t in TASKS:
            for k in range(N_DRAFTS):
                p = root / arm / f"{t['id']}__{k}.md"
                if p.exists():
                    yield arm, t, k, p


def prep(stage: str) -> Path:
    out = RUNS / stage
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    if stage == "draft":
        for arm in ARMS:
            sp = system_prompt(arm)
            for t in TASKS:
                for k in range(N_DRAFTS):
                    rows.append({"id": f"{arm}__{t['id']}__{k}", "prompt": draft_prompt(t),
                                 "system_prompt": sp, "model": DRAFTER, "tools": "",
                                 "json_schema": {"type": "object",
                                                 "properties": {"draft": {"type": "string"}},
                                                 "required": ["draft"]}})
    elif stage == "edit":
        sp = editor_system()
        for arm, t, k, p in texts("raw"):
            rows.append({"id": f"{arm}__{t['id']}__{k}", "prompt": edit_prompt(t, p),
                         "system_prompt": sp, "model": EDITOR, "tools": "",
                         "json_schema": {"type": "object",
                                         "properties": {"final": {"type": "string"},
                                                        "notes": {"type": "string"}},
                                         "required": ["final", "notes"]}})
    elif stage == "fresh":
        for st in ("raw", "final"):
            for arm, t, k, p in texts(st):
                rows.append({"id": f"{st}__{arm}__{t['id']}__{k}", "model": READER, "tools": "",
                             "prompt": FRESH_PROMPT.format(audience=t["audience"],
                                                           surface=t["surface"], text=p.read_text()),
                             "system_prompt": "You are a careful first-time reader.",
                             "json_schema": FRESH_SCHEMA})
    elif stage == "judge":
        rng = random.Random(20261110)
        key = {}
        for st in ("raw", "final"):
            for t in TASKS:
                items = [(arm, k, p) for arm, tt, k, p in texts(st) if tt["id"] == t["id"]]
                rng.shuffle(items)
                ids = rng.sample(range(100, 1000), len(items))
                block = ""
                for (arm, k, p), i in zip(items, ids):
                    did = f"D{i}"
                    key[f"{st}__{t['id']}__{did}"] = {"arm": arm, "k": k}
                    block += f"\n=== draft {did} ===\n\n{p.read_text()}\n"
                for jname, jmodel in JUDGES.items():
                    rows.append({"id": f"{jname}__{st}__{t['id']}", "model": jmodel, "tools": "",
                                 "system_prompt": JUDGE_SYSTEM, "json_schema": JUDGE_SCHEMA,
                                 "prompt": task_block(t) + "\n" + block})
        (HERE / "scores").mkdir(exist_ok=True)
        (HERE / "scores" / "judge_key.json").write_text(json.dumps(key, indent=1, sort_keys=True))
    else:
        raise SystemExit(f"unknown stage {stage}")
    with (out / "tasks.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} tasks -> {out / 'tasks.jsonl'}")
    return out


def collect(stage: str) -> None:
    res = RUNS / stage / "results"
    n = 0
    for f in sorted(res.glob("*.json")):
        so = json.loads(f.read_text()).get("structured_output")
        if so is None:
            continue
        parts = f.stem.split("__")
        if stage == "draft":
            arm, tid, k = parts
            dst = HERE / "drafts" / arm / f"{tid}__{k}.md"
            text = so["draft"]
        elif stage == "edit":
            arm, tid, k = parts
            dst = HERE / "finals" / arm / f"{tid}__{k}.md"
            text = so["final"]
            notes = HERE / "finals" / "notes"
            notes.mkdir(parents=True, exist_ok=True)
            (notes / f"{arm}__{tid}__{k}.txt").write_text(so.get("notes", "") + "\n")
        elif stage in ("fresh", "judge"):
            dst = HERE / "scores" / stage / f"{f.stem}.json"
            text = json.dumps(so, indent=1, ensure_ascii=False)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text.rstrip() + "\n")
        n += 1
    print(f"collected {n} -> {stage}")


if __name__ == "__main__":
    cmd, stage = sys.argv[1], sys.argv[2]
    {"prep": prep, "collect": collect}[cmd](stage)
