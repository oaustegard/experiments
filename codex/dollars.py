#!/usr/bin/env python3
"""Find and repair memories whose dollar amounts were stripped by shell expansion.

Inside a double-quoted bash string, `$7` is an empty positional parameter, so
"$7.2B" is stored as ".2B", "$210B" as "10B" and "$35k" as "5k"; "$TSLA" vanishes
whole. Round 4 of the codex reconciliation found two such memories. This pass
looks for the rest and restores what a source can restore.

    python3 dollars.py DATA candidates   # heuristic scan of DATA/memories.jsonl
    python3 dollars.py DATA detect       # tasks: Haiku lists damaged spans per candidate
    python3 dollars.py DATA recover      # tasks: Sonnet + web restores each damaged span
    python3 dollars.py DATA recheck      # tasks: a second Sonnet reopens every source
    python3 dollars.py DATA plan         # DATA/dollars/plan.json, a dry run
    python3 dollars.py DATA apply        # supersede each repaired memory

Every task file runs through dispatching-subagents/scripts/headless_fanout.py.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

TODAY = "2026-10-11"

DOT = re.compile(r"(?<![\w$€£¥.,])\.\d+\s?(?:B|M|K|T|bn|tn|k)\b")
MAG = re.compile(r"(?<![\w$€£¥.,\-/])\d[\d,]*(?:\.\d+)?\s?(?:B|M|bn|billion|million|trillion|T)\b")
MONEY = re.compile(r"\b(revenue|budget|raised|raise|funding|spent|spend|valuation|valued|ARR|run-rate|deal|"
                   r"investment|invest|aid|pledge|cost|costs|price|priced|loan|fund|seed|Series [A-F]|IPO|capex|"
                   r"salary|grant|bailout|tariff|fine|settlement|market cap)\b", re.I)

DETECT_PROMPT = """This note was saved through a shell command that may have stripped dollar amounts. \
Inside a double-quoted bash string, "$" followed by a digit is replaced by nothing and the digit is lost: \
"$7.2B" became ".2B", "$210B" became "10B", "$35k" became "5k", "$1.5B+" became ".5B+". "$" followed by \
letters vanishes with them: "$TSLA rose" became " rose".

List every span in the note that was damaged this way. Signals:
- a figure that starts with a dot (".2B", ".5M");
- an amount with no currency where the note otherwise gives currencies, such as "€70B (~0B)";
- a figure off by a leading digit, shown by the note's own arithmetic ("512 x ~5k = ~8M" cannot hold) or \
by being implausible for what it describes;
- a sentence with a hole where a ticker or amount was.
Do not flag intact amounts in euros, pounds, kroner or other currencies, numbers that are not money, or \
figures you merely doubt. Copy each span verbatim (at most 120 characters) so it can be found in the note.

NOTE (written {date}):
{text}"""

DETECT_SCHEMA = {
    "type": "object",
    "properties": {
        "damaged": {"type": "array", "items": {"type": "object", "properties": {
            "span": {"type": "string"}, "reason": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]}},
            "required": ["span", "reason", "confidence"]}},
    },
    "required": ["damaged"],
}

RECOVER_PROMPT = """A note in a personal memory store, written {date}, lost dollar amounts to shell \
expansion: inside a double-quoted bash string "$" plus a digit vanishes with the digit ("$7.2B" became \
".2B", "$210B" became "10B"), and "$" plus letters vanishes with them.

Restore each damaged span below. For each, find the original figure: from the note's own arithmetic or \
other mentions where that settles it, otherwise from a news source or document published around {date} \
about the same event (use WebSearch, then WebFetch the page to read the figure). Write the span as it \
should read, changing only what the expansion removed. If no source settles the figure, answer \
cannot_recover for that span. Quote the evidence verbatim with its URL, or say "arithmetic" and show it.

DAMAGED SPANS:
{spans}

NOTE:
{text}"""

RECOVER_SCHEMA = {
    "type": "object",
    "properties": {"spans": {"type": "array", "items": {"type": "object", "properties": {
        "span": {"type": "string"},
        "outcome": {"type": "string", "enum": ["restored", "cannot_recover", "not_damaged"]},
        "restored": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "quote": {"type": "string"}}, "required": ["source", "quote"]}},
    }, "required": ["span", "outcome"]}}},
    "required": ["spans"],
}

RECHECK_PROMPT = """A checker restored a dollar amount that shell expansion had stripped from a note \
written {date} ("$" plus a digit vanishes with the digit inside a double-quoted bash string). Try to prove \
the restoration wrong.

Damaged span: "{span}"
Restored as: "{restored}"
Evidence cited:
{evidence}

Context in the note: "{context}"

Open each cited URL with WebFetch, or redo the arithmetic, and confirm the figure. Check that the restored \
span differs from the damaged one only by what expansion removes (a "$", a leading digit, a ticker). \
Answer "confirmed", "confirmed_bad_correction" with the right restored span, or "overturned"."""

RECHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["confirmed", "confirmed_bad_correction", "overturned"]},
        "restored": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
}


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def structured(path: Path) -> dict | None:
    return json.loads(path.read_text()).get("structured_output") if path.exists() else None


def mems(data: Path) -> dict:
    return {m["id"]: m for m in map(json.loads, (data / "memories.jsonl").open())}


def candidates(data: Path) -> None:
    out = []
    for m in mems(data).values():
        s = m.get("summary") or ""
        if DOT.search(s) or ("$" not in s and MAG.search(s) and MONEY.search(s)):
            out.append(m["id"])
    (data / "dollars").mkdir(exist_ok=True)
    (data / "dollars" / "candidates.json").write_text(json.dumps(out))
    print(f"{len(out)} candidates")


def detect(data: Path) -> None:
    ms = mems(data)
    ids = json.loads((data / "dollars" / "candidates.json").read_text())
    tasks = [{"id": f"dt-{i[:8]}", "tools": "", "json_schema": DETECT_SCHEMA,
              "system_prompt": "You inspect text for a specific corruption. Reply only through the structured output.",
              "prompt": DETECT_PROMPT.format(date=str(ms[i].get("created_at"))[:10], text=ms[i]["summary"])}
             for i in ids]
    jsonl(data / "dollars" / "detect" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} detect tasks")


# Memories that quote damaged text on purpose: the corrected copy of the Korea
# note (its header already restores the figures) and the session log that
# describes the bug.
EXCLUDE = {"037bfa0e", "c22e7706"}
# The remembering library's write-time check (muninn-utilities ShellDamageWarning).
SIGNATURE = re.compile(r"(?<![\w$€£¥.,])\.\d+\s?(?:B|M|K|T|bn|tn)\b|(?<![\w$€£¥.,])0\d+(?:\.\d+)?\s?(?:B|M|K|T|bn|k)\b"
                       r"|\(~?0[BMKT]\)|(?<![\w$€£¥\d.,]),\d{3}\b")


def damaged(data: Path) -> dict[str, list[dict]]:
    """Damaged spans per memory: Haiku's high and medium findings, plus signature hits it missed."""
    ms = mems(data)
    by8 = {i[:8]: i for i in ms}
    out = {}
    for f in sorted((data / "dollars" / "detect" / "run" / "results").glob("dt-*.json")):
        r = structured(f) or {}
        mid = by8[f.stem[3:]]
        spans = [d for d in r.get("damaged", []) if d["confidence"] in ("high", "medium")
                 and d["span"].strip() and d["span"].strip() in ms[mid]["summary"]]
        if spans:
            out[mid] = spans
    for mid, m in ms.items():
        found = {d["span"] for d in out.get(mid, [])}
        for x in SIGNATURE.finditer(m.get("summary") or ""):
            if not any(x.group(0) in f for f in found):
                text = m["summary"]
                ctx = text[max(0, x.start() - 40): x.end() + 20].strip()
                out.setdefault(mid, []).append({"span": ctx, "reason": "shell-damage signature", "confidence": "high"})
    return {k: v for k, v in out.items() if k[:8] not in EXCLUDE}


def recover(data: Path) -> None:
    ms = mems(data)
    dmg = damaged(data)
    tasks = []
    for mid, spans in dmg.items():
        listing = "\n".join(f"- \"{d['span']}\" ({d['reason']})" for d in spans)
        tasks.append({"id": f"rv-{mid[:8]}", "model": "sonnet", "json_schema": RECOVER_SCHEMA,
                      "tools": "WebSearch,WebFetch", "allow": ["WebSearch", "WebFetch"],
                      "system_prompt": "You restore damaged figures from sources. Reply only through the structured output.",
                      "prompt": RECOVER_PROMPT.format(date=str(ms[mid].get("created_at"))[:10], spans=listing,
                                                      text=ms[mid]["summary"])})
    jsonl(data / "dollars" / "recover" / "tasks.jsonl", tasks)
    print(f"{sum(len(v) for v in dmg.values())} damaged spans in {len(dmg)} memories -> {len(tasks)} recover tasks")


def restored(data: Path) -> list[dict]:
    ms = mems(data)
    by8 = {i[:8]: i for i in ms}
    out = []
    for f in sorted((data / "dollars" / "recover" / "run" / "results").glob("rv-*.json")):
        mid = by8[f.stem[3:]]
        for n, s in enumerate((structured(f) or {}).get("spans", [])):
            out.append({"mid": mid, "n": n, **s})
    return out


def recheck(data: Path) -> None:
    ms = mems(data)
    tasks = []
    for r in restored(data):
        if r["outcome"] != "restored" or not r.get("restored"):
            continue
        text = ms[r["mid"]]["summary"]
        i = text.find(r["span"])
        context = text[max(0, i - 200): i + len(r["span"]) + 200] if i >= 0 else r["span"]
        ev = "\n".join(f"- {e['source']}: \"{e['quote']}\"" for e in r.get("evidence", [])) or "(none)"
        tasks.append({"id": f"rk-{r['mid'][:8]}-{r['n']}", "model": "sonnet", "json_schema": RECHECK_SCHEMA,
                      "tools": "WebFetch", "allow": ["WebFetch"],
                      "system_prompt": "You are a skeptical reviewer. Reply only through the structured output.",
                      "prompt": RECHECK_PROMPT.format(date=str(ms[r["mid"]].get("created_at"))[:10], span=r["span"],
                                                      restored=r["restored"], evidence=ev, context=context)})
    jsonl(data / "dollars" / "recheck" / "tasks.jsonl", tasks)
    print(f"{len(tasks)} recheck tasks")


def header(items: list[dict], unrecovered: list[str]) -> str:
    lines = [f"[REPAIRED {TODAY}: shell expansion stripped dollar amounts from this note when it was saved "
             "(\"$7.2B\" became \".2B\"). Read the original with these restorations; each figure was traced to "
             "a source and confirmed by a second reviewer.]"]
    lines += [f"- WAS: {it['span']} NOW: {it['restored']} (per {it['source']})" for it in items]
    if unrecovered:
        lines.append("- Damaged, figure not recovered: " + "; ".join(f'"{u}"' for u in unrecovered))
    return "\n".join(lines) + "\n\n"


def plan(data: Path) -> dict:
    ms = mems(data)
    stats = collections.Counter()
    fixes: dict[str, list[dict]] = collections.defaultdict(list)
    lost: dict[str, list[str]] = collections.defaultdict(list)
    for r in restored(data):
        stats["span_" + r["outcome"]] += 1
        if r["outcome"] == "cannot_recover":
            lost[r["mid"]].append(r["span"])
            continue
        if r["outcome"] != "restored":
            continue
        rk = structured(data / "dollars" / "recheck" / "run" / "results" / f"rk-{r['mid'][:8]}-{r['n']}.json")
        v = (rk or {}).get("verdict", "missing")
        stats["recheck_" + v] += 1
        if v == "overturned" or v == "missing":
            lost[r["mid"]].append(r["span"])
            continue
        new = rk["restored"] if v == "confirmed_bad_correction" and rk.get("restored") else r["restored"]
        src = next((e["source"] for e in r.get("evidence", [])), "arithmetic")
        fixes[r["mid"]].append({"span": r["span"], "restored": new, "source": src[:160]})
    supersede = []
    for mid in sorted(set(fixes) | set(lost)):
        m = ms[mid]
        tags = m.get("tags") or []
        tags = json.loads(tags) if isinstance(tags, str) else tags
        supersede.append({"id": mid, "type": m["type"], "priority": int(m.get("priority") or 0),
                          "tags": list(dict.fromkeys(tags + ["shell-dollar-repaired"])),
                          "summary": header(fixes.get(mid, []), lost.get(mid, [])) + (m.get("summary") or ""),
                          "restored": fixes.get(mid, []), "unrecovered": lost.get(mid, [])})
    out = {"stats": dict(stats), "supersede": supersede}
    (data / "dollars" / "plan.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["stats"]), f"-> {len(supersede)} memories, "
          f"{sum(len(s['restored']) for s in supersede)} restored spans, "
          f"{sum(len(s['unrecovered']) for s in supersede)} flagged unrecovered")
    return out


def apply(data: Path) -> None:
    sys.path.insert(0, "/home/user/claude-workspace/scripts")
    import memory_audit
    rem = memory_audit._remembering()
    p = json.loads((data / "dollars" / "plan.json").read_text())
    done_f = data / "dollars" / "applied.json"
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


if __name__ == "__main__":
    data, cmd = Path(sys.argv[1]), sys.argv[2]
    {"candidates": candidates, "detect": detect, "recover": recover, "recheck": recheck, "plan": plan,
     "apply": apply}[cmd](data)
