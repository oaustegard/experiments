#!/usr/bin/env python3
"""Work through the contradictions the codex pages surfaced.

Each page lists the points where its sources disagree, with the ids involved
(393 across 149 pages). Most are not errors: a count measured on two dates, or
two figures that measure different things. Some are a memory that a later one
showed wrong, and that memory still reads as current to recall. This pass
finds those and puts a dated correction header on them.

    python3 reconcile.py DATA adjudicate    # DATA/reconcile/adjudicate/tasks.jsonl, one per page
    python3 reconcile.py DATA recheck       # DATA/reconcile/recheck/tasks.jsonl, one per memory_wrong verdict
    python3 reconcile.py DATA plan          # DATA/reconcile/plan.json, a dry run
    python3 reconcile.py DATA snapshot      # read-only repo snapshots for the live checks
    python3 reconcile.py DATA live          # DATA/reconcile/live/tasks.jsonl, one per needs_live_check
    python3 reconcile.py DATA live-recheck  # DATA/reconcile/live-recheck/tasks.jsonl
    python3 reconcile.py DATA apply         # supersede each corrected memory

Adjudicate runs on Haiku and recheck on Sonnet, both through
dispatching-subagents/scripts/headless_fanout.py with no tools: everything a
child needs is in its prompt. A correction is written only when the recheck
confirms it. The header is [RECONCILED date ...] rather than the monthly
audit's [AUDIT date ...]: only one statement was checked, so the memory should
stay in the audit's queue.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

TODAY = "2026-10-11"
ID = re.compile(r"(?:\b([ms]):)?\b([0-9a-f]{8})\b")
KINDS = ["changed_over_time", "different_measures", "memory_wrong", "needs_live_check", "unresolvable"]

ADJUDICATE_PROMPT = """You are reconciling disagreements between notes in a personal memory store. Today is \
{today}. Each item below is a point where the notes cited disagree, with the full text and date of each \
note. Classify every item as exactly one of:

- changed_over_time: each statement was true when its note was written, and the difference is a later \
state or a later measurement (a count, a status, a version). Dates differ and nothing says the earlier \
note was wrong when written.
- different_measures: the statements describe different things (scope, definition, population, unit), so \
both can hold at once.
- memory_wrong: one note states something that another note shows was false or mistaken. The other note \
says so explicitly ("correction", "actually", "was wrong", "not X but Y"), or gives a direct measurement of \
the same thing on the same date. Name the wrong note, quote the wrong statement (WAS, at most 200 \
characters, copied from that note), and write the correct statement (NOW), taken from the note that shows it.
- needs_live_check: the notes alone cannot settle it, but a file in a GitHub repository or a live service \
could. Say what to look at.
- unresolvable: neither the notes nor any check could settle it now.

A later note disagreeing is not enough for memory_wrong: a newer value means changed_over_time unless \
something says the old value was wrong at the time. When unsure between memory_wrong and another kind, \
choose the other kind.

{items}"""

ADJUDICATE_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer"},
            "kind": {"type": "string", "enum": KINDS},
            "explanation": {"type": "string"},
            "wrong_note": {"type": "string"},
            "was": {"type": "string"},
            "now": {"type": "string"},
            "basis": {"type": "array", "items": {"type": "string"}},
            "check": {"type": "string"},
        },
        "required": ["index", "kind", "explanation"],
    }}},
    "required": ["items"],
}

RECHECK_PROMPT = """A first reviewer decided that one note in a personal memory store is wrong and should \
carry a correction. Your job is to try to prove that decision wrong. Today is {today}.

The disagreement: {what}

First reviewer: note {wrong} is wrong. WAS: "{was}" NOW: "{now}". Reason: {explanation}

The notes:
{notes}

Answer "overturned" if the WAS statement was true when its note was written (a later change is not an \
error), if the two notes measure different things, or if no note shows WAS was false. Answer \
"confirmed" if a note shows WAS was false and NOW is the correct statement. Answer \
"confirmed_bad_correction" if WAS was false but NOW is not right; give the correct NOW. Check that WAS \
actually appears in note {wrong}."""

RECHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["confirmed", "confirmed_bad_correction", "overturned"]},
        "now": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def load(data: Path):
    site = json.loads((data / "site" / "codex.json").read_text())
    return site["pages"], site["sources"]


def note(sources: dict, cite: str) -> str:
    s = sources.get(cite)
    if not s:
        return f"--- [{cite}] (not found)"
    label = "note" if s.get("kind") == "memory" else "session summary"
    return f"--- [{cite}] {label}, {s.get('date')}\n{(s.get('text') or '').strip()}"


def structured(path: Path) -> dict | None:
    return json.loads(path.read_text()).get("structured_output") if path.exists() else None


def adjudicate(data: Path) -> None:
    pages, sources = load(data)
    tasks, n = [], 0
    for p in pages:
        cs = p.get("contradictions") or []
        if not cs:
            continue
        blocks = []
        for i, c in enumerate(cs):
            notes = "\n\n".join(note(sources, x) for x in c.get("cite", []))
            blocks.append(f"=== ITEM {i}\nDisagreement: {c['what']}\n\n{notes}")
        n += len(cs)
        tasks.append({"id": f"adj-{p['topic']}", "tools": "", "json_schema": ADJUDICATE_SCHEMA,
                      "system_prompt": "You reconcile notes carefully. Reply only through the structured output.",
                      "prompt": ADJUDICATE_PROMPT.format(today=TODAY, items="\n\n".join(blocks))})
    jsonl(data / "reconcile" / "adjudicate" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} pages, {n} contradictions")


def verdicts(data: Path) -> list[dict]:
    pages, sources = load(data)
    out = []
    for p in pages:
        cs = p.get("contradictions") or []
        if not cs:
            continue
        r = structured(data / "reconcile" / "adjudicate" / "run" / "results" / f"adj-{p['topic']}.json")
        got = {it.get("index"): it for it in (r or {}).get("items", [])}
        for i, c in enumerate(cs):
            v = got.get(i, {"kind": "missing", "explanation": ""})
            out.append({"topic": p["topic"], "index": i, "what": c["what"], "cite": c.get("cite", []), **v})
    return out


def recheck(data: Path) -> None:
    _, sources = load(data)
    rows = verdicts(data)
    print(dict(collections.Counter(r["kind"] for r in rows)))
    tasks = []
    for r in rows:
        if r["kind"] != "memory_wrong":
            continue
        wrong = r.get("wrong_note", "").strip("[]")
        notes = "\n\n".join(note(sources, x) for x in r["cite"])
        tasks.append({"id": f"rc-{r['topic']}-{r['index']}", "tools": "", "model": "sonnet",
                      "json_schema": RECHECK_SCHEMA,
                      "system_prompt": "You are a skeptical reviewer. Reply only through the structured output.",
                      "prompt": RECHECK_PROMPT.format(today=TODAY, what=r["what"], wrong=wrong, was=r.get("was", ""),
                                                      now=r.get("now", ""), explanation=r["explanation"],
                                                      notes=notes)})
    jsonl(data / "reconcile" / "recheck" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} recheck tasks")


LIVE_PROMPT = """You are checking one disagreement between notes in a personal memory store against live \
sources. Today is {today}.

The disagreement: {what}

The notes:
{notes}

A first reviewer could not settle it from the notes and suggested: {check}

Sources you can use:
- Read-only snapshots of the owner's GitHub repositories under {repos}/: each <name>/ is the repository's \
files at HEAD as of today, and <name>.gitlog.txt lists every commit (hash, date, subject, then the paths \
it touched). Repositories: {repo_list}. Use Grep and Read on them; there is no git command.
- WebSearch and WebFetch for public facts, and for issue or pull request pages of public repositories \
(https://github.com/oaustegard/<name>/issues/<n> or /pull/<n>).
You cannot reach Strava, the memory store, or private services; if the question needs them, answer \
could_not_check.

Decide:
- note_wrong: a live source shows that a statement in one note was false when that note was written. Name \
the note, copy the false statement verbatim from it (WAS, at most 200 characters), and write the correct \
statement (NOW). Quote the live evidence verbatim with its file path or URL.
- notes_consistent: the live sources show the statements were each true at their date, or describe \
different things.
- could_not_check: the sources above cannot settle it.
A statement that was true when written and changed later is not wrong. Evidence must come from a live \
source, not from the notes. When unsure, answer could_not_check."""

LIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["note_wrong", "notes_consistent", "could_not_check"]},
        "wrong_note": {"type": "string"}, "was": {"type": "string"}, "now": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "quote": {"type": "string"}}, "required": ["source", "quote"]}},
        "explanation": {"type": "string"},
    },
    "required": ["outcome", "explanation"],
}

LIVE_RECHECK_PROMPT = """A first checker decided, from live sources, that a note in a personal memory store \
is wrong. Try to prove that decision wrong. Today is {today}.

The disagreement: {what}

The notes:
{notes}

First checker: note {wrong} is wrong. WAS: "{was}" NOW: "{now}". Reason: {explanation}
Evidence it cited:
{evidence}

Open every cited source yourself (repository snapshots under {repos}/ with Read and Grep, URLs with \
WebFetch) and confirm each quote is there and says what the checker claims. Answer "overturned" if a quote \
is missing or misread, if WAS was true when its note was written, or if WAS does not appear in note \
{wrong}. Answer "confirmed" if the evidence holds and NOW is right, or "confirmed_bad_correction" with the \
right NOW if WAS was false but NOW is not right."""


def live_tasks_rows(data: Path) -> list[dict]:
    return [r for r in verdicts(data) if r["kind"] == "needs_live_check"]


def live(data: Path) -> None:
    _, sources = load(data)
    repos = data / "reconcile" / "repos"
    present = sorted(p.name for p in repos.iterdir() if p.is_dir())
    tasks = []
    for r in live_tasks_rows(data):
        notes = "\n\n".join(note(sources, x) for x in r["cite"])
        tasks.append({"id": f"lv-{r['topic']}-{r['index']}", "json_schema": LIVE_SCHEMA,
                      "tools": "Read,Grep,Glob,WebSearch,WebFetch", "allow": ["WebSearch", "WebFetch"],
                      "system_prompt": "You check facts against sources. Reply only through the structured output.",
                      "prompt": LIVE_PROMPT.format(today=TODAY, what=r["what"], notes=notes,
                                                   check=r.get("check") or r["explanation"], repos=repos,
                                                   repo_list=", ".join(present))})
    jsonl(data / "reconcile" / "live" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} live-check tasks")


def live_recheck(data: Path) -> None:
    _, sources = load(data)
    repos = data / "reconcile" / "repos"
    tasks, counts = [], collections.Counter()
    for r in live_tasks_rows(data):
        lv = structured(data / "reconcile" / "live" / "run" / "results" / f"lv-{r['topic']}-{r['index']}.json")
        counts[(lv or {}).get("outcome", "missing")] += 1
        if not lv or lv["outcome"] != "note_wrong":
            continue
        notes = "\n\n".join(note(sources, x) for x in r["cite"])
        ev = "\n".join(f"- {e['source']}: \"{e['quote']}\"" for e in lv.get("evidence", []))
        tasks.append({"id": f"lr-{r['topic']}-{r['index']}", "model": "sonnet", "json_schema": RECHECK_SCHEMA,
                      "tools": "Read,Grep,Glob,WebFetch", "allow": ["WebFetch"],
                      "system_prompt": "You are a skeptical reviewer. Reply only through the structured output.",
                      "prompt": LIVE_RECHECK_PROMPT.format(today=TODAY, what=r["what"], notes=notes,
                                                           wrong=lv.get("wrong_note", ""), was=lv.get("was", ""),
                                                           now=lv.get("now", ""), explanation=lv["explanation"],
                                                           evidence=ev or "(none)", repos=repos)})
    jsonl(data / "reconcile" / "live-recheck" / "tasks.jsonl", tasks)
    print(dict(counts), f"-> {len(tasks)} live recheck tasks")


def header(items: list[dict]) -> str:
    lines = [f"[RECONCILED {TODAY}: {len(items)} statement(s) below are contradicted by other notes, live sources "
             "or this note's own data. Read the original with these corrections; each was found by the codex "
             "contradiction pass and confirmed by a second reviewer against the notes cited.]"]
    lines += [f"- WAS: {it['was']} NOW: {it['now']} "
              f"({'per ' + ', '.join(it['basis']) if it['basis'] else 'from this note itself'})" for it in items]
    return "\n".join(lines) + "\n\n"


def plan(data: Path) -> dict:
    mems = {m["id"][:8]: m for m in map(json.loads, (data / "memories.jsonl").open())}
    rows = verdicts(data)
    stats = collections.Counter(r["kind"] for r in rows)
    # Reviewer edits after reading plan.json: {"<id8>": {"now": "..."} | {"skip": true}}
    of = data / "reconcile" / "overrides.json"
    overrides = json.loads(of.read_text()) if of.exists() else {}
    repos = str(data / "reconcile" / "repos") + "/"
    candidates = []  # (pass, row, first-pass verdict with was/now/wrong_note, recheck)
    for r in rows:
        if r["kind"] == "memory_wrong":
            rc = structured(data / "reconcile" / "recheck" / "run" / "results" / f"rc-{r['topic']}-{r['index']}.json")
            candidates.append(("notes", r, r, rc))
        elif r["kind"] == "needs_live_check":
            lv = structured(data / "reconcile" / "live" / "run" / "results" / f"lv-{r['topic']}-{r['index']}.json")
            if lv:
                stats["live_" + lv["outcome"]] += 1
            if lv and lv["outcome"] == "note_wrong":
                lr = structured(data / "reconcile" / "live-recheck" / "run" / "results"
                                / f"lr-{r['topic']}-{r['index']}.json")
                candidates.append(("live", r, lv, lr))
    fixes: dict[str, list[dict]] = collections.defaultdict(list)
    for kind, r, first, rc in candidates:
        v = (rc or {}).get("verdict", "missing")
        stats[f"{kind}_recheck_{v}"] += 1
        if v not in ("confirmed", "confirmed_bad_correction"):
            continue
        ref = ID.search(first.get("wrong_note", ""))
        if ref is None or ref.group(1) == "s":
            stats["skipped_not_a_memory"] += 1  # a session summary is not in the store
            continue
        wrong = ref.group(2)
        m = mems.get(wrong)
        if m is None or first.get("was", "")[:40].strip() not in (m.get("summary") or ""):
            stats["skipped_unmatched_was"] += 1
            continue
        now = rc.get("now") if v == "confirmed_bad_correction" and rc.get("now") else first.get("now", "")
        if now.strip()[:60] in (m.get("summary") or ""):
            stats["skipped_already_corrected"] += 1  # an earlier audit header says the same
            continue
        o = overrides.get(wrong, {})
        if o.get("skip"):
            stats["skipped_by_review"] += 1
            continue
        now = o.get("now", now)
        if kind == "live":
            basis = list(dict.fromkeys(e["source"].replace(repos, "oaustegard/") for e in first.get("evidence", [])))
        else:
            ids = [f"{k or 'm'}:{i}" for b in (first.get("basis") or []) for k, i in ID.findall(b)]
            # keep ids that name a memory or a session; a commit hash matches the pattern too
            basis = [x for x in dict.fromkeys(ids) if x != f"m:{wrong}" and (x[2:] in mems or x.startswith("s:"))]
        fixes[wrong].append({"was": first["was"].strip(), "now": now.strip(), "basis": basis, "pass": kind,
                             "topic": r["topic"], "index": r["index"]})
    supersede = []
    for mid, items in fixes.items():
        m = mems[mid]
        tags = m.get("tags") or []
        tags = json.loads(tags) if isinstance(tags, str) else tags
        supersede.append({"id": m["id"], "type": m["type"], "priority": int(m.get("priority") or 0),
                          "tags": list(dict.fromkeys(tags + ["codex-reconciled"])),
                          "summary": header(items) + (m.get("summary") or ""), "items": items})
    out = {"stats": dict(stats), "supersede": supersede,
           "verdicts": [{k: r.get(k) for k in ("topic", "index", "kind", "explanation", "check")} for r in rows]}
    (data / "reconcile" / "plan.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["stats"]), f"-> {len(supersede)} memories to correct, "
          f"{sum(len(s['items']) for s in supersede)} statements")
    return out


def apply(data: Path) -> None:
    sys.path.insert(0, "/home/user/claude-workspace/scripts")
    import memory_audit
    rem = memory_audit._remembering()
    p = json.loads((data / "reconcile" / "plan.json").read_text())
    done_f = data / "reconcile" / "applied.json"
    done = json.loads(done_f.read_text()) if done_f.exists() else {}
    for s in p["supersede"]:
        if done.get(s["id"]):
            continue
        try:
            done[s["id"]] = str(rem.supersede(s["id"], s["summary"], s["type"], tags=s["tags"],
                                              priority=s["priority"]))
        except Exception as e:  # noqa: BLE001 - record and continue; a rerun retries
            print(f"FAIL {s['id'][:8]} {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
            done[s["id"]] = None
        done_f.write_text(json.dumps(done, indent=1))
    print(f"superseded {sum(1 for v in done.values() if v)}, failed {sum(1 for v in done.values() if not v)}")


REPOS = ["claude-workspace", "claude-skills", "experiments", "remex", "remax", "remax_kb", "remex-mojo",
         "oaustegard.github.io", "llm-as-computer", "muninn-utilities", "eml-sr", "aeyu.io", "muninns-inbox",
         "claude-code-mods", "muninn.austegard.com", "muninn-mcp"]


def snapshot(data: Path) -> None:
    """Clone each repo blobless with history, dump its log, and drop the remote.

    Without a remote a child cannot push or fetch, so the snapshot is read-only
    whatever tools it holds. The log dump carries every commit's date, subject
    and touched paths; the working tree is HEAD as of today.
    """
    import subprocess
    root = data / "reconcile" / "repos"
    root.mkdir(parents=True, exist_ok=True)
    for r in REPOS:
        dest = root / r
        if not dest.exists():
            p = subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--depth", "2000",
                                f"https://github.com/oaustegard/{r}", str(dest)], capture_output=True, text=True,
                               timeout=900)
            if p.returncode:
                print(f"{r}: clone failed: {p.stderr.strip()[-200:]}")
                continue
        log = subprocess.run(["git", "-C", str(dest), "log", "--date=short", "--format=%n%h %ad %s",
                              "--name-status"], capture_output=True, text=True).stdout
        (root / f"{r}.gitlog.txt").write_text(log)
        subprocess.run(["git", "-C", str(dest), "remote", "remove", "origin"], capture_output=True)
        print(f"{r}: {log.count(chr(10) + chr(10))} commits")


if __name__ == "__main__":
    data, cmd = Path(sys.argv[1]), sys.argv[2]
    {"adjudicate": adjudicate, "recheck": recheck, "plan": plan, "apply": apply, "snapshot": snapshot,
     "live": live, "live-recheck": live_recheck}[cmd](data)
