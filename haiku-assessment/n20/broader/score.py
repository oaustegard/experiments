#!/usr/bin/env python3
"""Score the broader n=5 probe across 6 task archetypes."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent / "outputs"

# ---------- T4: Meeting JSON extraction ----------
T4_REQUIRED_KEYS = {"title", "datetime", "location", "duration_minutes", "organizer"}
T4_ALLOWED_KEYS = T4_REQUIRED_KEYS | {"notes"}
T4_EXPECTED_DURATION = 45
T4_EXPECTED_LOC = "Birch Room"
T4_EXPECTED_ORG = "Sam"

def score_t4(text):
    # Try to extract a JSON object from the response (vanilla may wrap in fences)
    m = re.search(r"\{[\s\S]*?\}", text)
    if not m:
        return {"parseable": False}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"parseable": False}
    keys = set(obj.keys())
    schema_match = keys.issubset(T4_ALLOWED_KEYS) and T4_REQUIRED_KEYS <= keys
    # We accept partial schemas — what we care about is "uses ONLY the keys we asked for"
    extra_keys = keys - T4_ALLOWED_KEYS
    return {
        "parseable": True,
        "keys": sorted(keys),
        "extra_keys": sorted(extra_keys),
        "no_extra_keys": len(extra_keys) == 0,
        "duration_correct": obj.get("duration_minutes") == 45,
        "location_correct": obj.get("location") == T4_EXPECTED_LOC,
        "organizer_correct": obj.get("organizer") == T4_EXPECTED_ORG,
        "char_count": len(text),
    }

# ---------- T5: Changelog line ----------
T5_HALLUCINATIONS = [
    "deprecated", "zoneinfo",  # ZoneInfo is technically in the diff, but
                               # vanilla flags it as "deprecated" which is invented
]
def score_t5(text):
    line = text.strip()
    # Count words in the FIRST non-empty line (the changelog candidate)
    first = next((l for l in line.split("\n") if l.strip()), "")
    # Strip leading markers
    first = re.sub(r"^[-*•\s]+", "", first).strip()
    word_count = len(re.findall(r"\b\w+\b", first))
    in_length = 8 <= word_count <= 18
    starts_with_action = bool(re.match(
        r"^(Fix|Change|Add|Remove|Stop|Allow|Update|Modify|Make|Switch)\b",
        first, re.IGNORECASE
    ))
    # Check for "refactor" framing (wrong here — there IS a behavior change)
    refactor_framing = "refactor" in first.lower()
    # Implementation detail leak: mentioning ZoneInfo internals
    impl_leak = "zoneinfo" in first.lower()
    # Hallucinated "deprecated" claim
    hallucinated_deprecated = "deprecated" in first.lower()
    multi_line = len([l for l in text.strip().split("\n") if l.strip()]) > 1
    return {
        "word_count": word_count,
        "in_length": in_length,
        "starts_with_action": starts_with_action,
        "refactor_framing": refactor_framing,
        "impl_leak": impl_leak,
        "hallucinated_deprecated": hallucinated_deprecated,
        "multi_line": multi_line,
        "first_line": first[:120],
    }

# ---------- T6: Filter/sort/select ----------
# Correct answer per rubric:
#   Eligible (year>2015 AND pages<400): Klara and the Sun, Piranesi, Educated,
#                                       The Vanishing Half, Atomic Habits
#   Top 3 by rating (ties broken alphabetically):
#       Atomic Habits (4.4) < Piranesi (4.4) — AH first by alpha
#       Educated (4.3)
#   So top 3 = Atomic Habits, Piranesi, Educated (in that order)

def score_t6(text):
    t = text.lower()
    # Did they correctly exclude Project Hail Mary and The Overstory?
    excluded_hailmary = "project hail mary" not in t or "exceed" in t or "476" in t
    excluded_overstory = "the overstory" not in t or "exceed" in t or "502" in t

    # Did they get the top 3 right?
    has_atomic = "atomic habits" in t
    has_piranesi = "piranesi" in t
    has_educated = "educated" in t
    top3_complete = has_atomic and has_piranesi and has_educated

    # Order check — restrict to the "ranked top 3" portion of the output, not
    # the Eligible-list portion (which can list titles in any order).
    # The ranked portion is the segment that contains "1." or "1)" markers.
    # We look for the first three ranked entries and check whose name appears in each.
    ranked_segment = text
    # If there's a "Top by rating:" line, use it
    m_top = re.search(r"Top by rating:.*", text)
    if m_top:
        ranked_segment = m_top.group(0)
    else:
        # Otherwise look for the first sequence of "1.|1)" through "3.|3)"
        m_seq = re.search(
            r"(?:^|\n)\s*1[.)]\s.*?(?:\n\s*3[.)]\s.*)",
            text, re.DOTALL,
        )
        if m_seq:
            ranked_segment = m_seq.group(0)
    rs = ranked_segment.lower()
    # Find positions of each title in the ranked segment
    ah_pos = rs.find("atomic habits")
    pi_pos = rs.find("piranesi")
    ed_pos = rs.find("educated")
    if ah_pos >= 0 and pi_pos >= 0 and ed_pos >= 0:
        correct_order = ah_pos < pi_pos < ed_pos
    else:
        correct_order = False

    # Schema match: exactly the two-line format
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    tight_schema = (
        len(lines) == 2
        and lines[0].startswith("Eligible after filter:")
        and lines[1].startswith("Top by rating:")
    )
    has_markdown = bool(re.search(r"^#|\*\*|\|.*\|", text, re.MULTILINE))
    return {
        "top3_complete": top3_complete,
        "correct_tie_break": correct_order,
        "tight_schema": tight_schema,
        "has_markdown": has_markdown,
        "char_count": len(text),
    }

# ---------- T7: gh CLI command ----------
# Correct: gh pr list --state open --assignee alice --label bug --search "sort:created-asc"
INVENTED_FLAGS = ["--sort", "--order", "--asc", "--desc"]

def score_t7(text):
    # Extract first line starting with "gh"
    gh_match = re.search(r"^gh\s+\S.*$", text, re.MULTILINE)
    if not gh_match:
        return {"has_gh_line": False, "uses_invented_flags": False}
    cmd = gh_match.group(0)
    uses_invented = any(flag in cmd for flag in INVENTED_FLAGS)
    uses_search_sort = "sort:created-asc" in cmd or "sort:created" in cmd
    has_state_open = "--state open" in cmd
    has_assignee = "--assignee alice" in cmd
    has_label = "--label bug" in cmd
    correct_command = (
        has_state_open and has_assignee and has_label
        and uses_search_sort and not uses_invented
    )
    # Multi-line or markdown wrapper?
    multi_line = len([l for l in text.strip().split("\n") if l.strip()]) > 1
    has_markdown = bool(re.search(r"^#|```|\*\*", text, re.MULTILINE))
    return {
        "has_gh_line": True,
        "first_cmd": cmd[:200],
        "uses_invented_flags": uses_invented,
        "uses_correct_search_sort": uses_search_sort,
        "correct_command": correct_command,
        "multi_line": multi_line,
        "has_markdown": has_markdown,
    }

# ---------- T8: Copyedit ----------
EXPECTED_FIXES = ["team has", "weeks", "believe", "implementation is",
                  "there are", "cases that need", "users give", "improves"]

def score_t8(text):
    t = text.lower()
    fixes_present = sum(1 for f in EXPECTED_FIXES if f.lower() in t)
    # Did they use the schema (numbered list)?
    has_numbered = bool(re.search(r"^\s*\d+[.)]\s+", text, re.MULTILINE))
    # Did they wrap with markdown headers / "Corrections Made" sections?
    has_section_headers = bool(re.search(
        r"(?:^#|## |\*\*Corrections|\*\*Changes Made|\*\*Original|\*\*Corrected)",
        text, re.MULTILINE,
    ))
    # Did they preserve sentence structure (no rewrites)?
    # Heuristic: the cleaned paragraph should be present and start with "The team has"
    has_clean_para = "the team has been working" in t
    return {
        "fixes_present": fixes_present,
        "has_numbered_list": has_numbered,
        "has_section_headers": has_section_headers,
        "has_clean_para": has_clean_para,
        "char_count": len(text),
    }

# ---------- T3b: Voice rewrite, calibrated examples ----------
HALLUCINATED = [
    "lru", "ttl", "write-through", "write through",
    "redis", "memcached",
    "distributed hash table", "cache-aside", "cache aside",
    "p99", "p95", "p50",
    "rdb", "snapshot",
    "background queue", "background worker",
    "annotation", "decorator",
    "structured json", "opentelemetry",
    "session-cookie", "jwt", "stripe connect", "idempotency",
    "highcharts", "server-side rendering",
    "sub-millisecond", "milliseconds",
]
BANNED = [
    "thrilled", "excited", "proud", "delighted", "can't wait",
    "amazing", "revolutionary", "groundbreaking",
    "paradigm shift", "game-changer", "game changer",
    "transformative", "cutting-edge", "cutting edge",
    "next-generation",
]

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def score_t3b(text):
    t = text.lower()
    halluc_present = [w for w in HALLUCINATED if w in t]
    banned_present = [w for w in BANNED if w in t]
    wc = word_count(text)
    in_length = 60 <= wc <= 90
    has_markdown = bool(re.search(r"^#|^##|\*\*", text, re.MULTILINE))
    return {
        "halluc_count": len(halluc_present),
        "halluc_terms": halluc_present,
        "banned_count": len(banned_present),
        "word_count": wc,
        "in_length_range": in_length,
        "has_markdown": has_markdown,
    }


SCORERS = {"T4": score_t4, "T5": score_t5, "T6": score_t6,
           "T7": score_t7, "T8": score_t8, "T3b": score_t3b}


def main():
    output = {}
    for task in SCORERS:
        for cond in ["vanilla", "downskilled"]:
            cell_dir = ROOT / f"{task}_{cond}"
            files = sorted(cell_dir.glob("run_*.md"))
            rows = []
            for f in files:
                row = SCORERS[task](f.read_text())
                row["_file"] = f.name
                rows.append(row)
            output[f"{task}_{cond}"] = rows
    out_path = ROOT.parent / "scores.json"
    out_path.write_text(json.dumps(output, indent=2))

    # Print compact summary
    for task in SCORERS:
        print(f"\n=== {task} ===")
        for cond in ["vanilla", "downskilled"]:
            rows = output[f"{task}_{cond}"]
            print(f"  -- {cond} (n={len(rows)})")
            # Print key metrics by task
            if task == "T4":
                parseable = sum(r.get("parseable", False) for r in rows)
                no_extra = sum(r.get("no_extra_keys", False) for r in rows)
                dur_ok = sum(r.get("duration_correct", False) for r in rows)
                schemas = {tuple(sorted(r.get("keys", []))) for r in rows if r.get("parseable")}
                print(f"     parseable JSON: {parseable}/{len(rows)}")
                print(f"     uses ONLY required keys: {no_extra}/{len(rows)}")
                print(f"     duration_minutes correct: {dur_ok}/{len(rows)}")
                print(f"     distinct schemas across runs: {len(schemas)}")
            elif task == "T5":
                in_len = sum(r["in_length"] for r in rows)
                act = sum(r["starts_with_action"] for r in rows)
                refactor = sum(r["refactor_framing"] for r in rows)
                impl = sum(r["impl_leak"] for r in rows)
                multi = sum(r["multi_line"] for r in rows)
                print(f"     in 8-18 word range: {in_len}/{len(rows)}")
                print(f"     starts with action verb: {act}/{len(rows)}")
                print(f"     ZoneInfo impl-leak: {impl}/{len(rows)}")
                print(f"     'refactor' framing (wrong): {refactor}/{len(rows)}")
                print(f"     multi-line output: {multi}/{len(rows)}")
            elif task == "T6":
                top3 = sum(r["top3_complete"] for r in rows)
                tie = sum(r["correct_tie_break"] for r in rows)
                sch = sum(r["tight_schema"] for r in rows)
                md = sum(r["has_markdown"] for r in rows)
                print(f"     top 3 set correct: {top3}/{len(rows)}")
                print(f"     correct tie-break order: {tie}/{len(rows)}")
                print(f"     tight schema (2 lines): {sch}/{len(rows)}")
                print(f"     markdown drift: {md}/{len(rows)}")
            elif task == "T7":
                inv = sum(r.get("uses_invented_flags", False) for r in rows)
                correct = sum(r.get("correct_command", False) for r in rows)
                md = sum(r.get("has_markdown", False) for r in rows)
                print(f"     uses invented flags (--sort etc): {inv}/{len(rows)}")
                print(f"     fully correct command: {correct}/{len(rows)}")
                print(f"     wrapped in markdown: {md}/{len(rows)}")
            elif task == "T8":
                fixes_avg = sum(r["fixes_present"] for r in rows) / len(rows)
                num = sum(r["has_numbered_list"] for r in rows)
                hdr = sum(r["has_section_headers"] for r in rows)
                print(f"     avg expected fixes present (max 8): {fixes_avg:.1f}")
                print(f"     uses numbered-list schema: {num}/{len(rows)}")
                print(f"     markdown section headers: {hdr}/{len(rows)}")
            elif task == "T3b":
                halluc = sum(1 for r in rows if r["halluc_count"] > 0)
                avg_h = sum(r["halluc_count"] for r in rows) / len(rows)
                in_len = sum(r["in_length_range"] for r in rows)
                avg_wc = sum(r["word_count"] for r in rows) / len(rows)
                banned = sum(1 for r in rows if r["banned_count"] > 0)
                md = sum(r["has_markdown"] for r in rows)
                print(f"     any hallucinated arch detail: {halluc}/{len(rows)}")
                print(f"     avg hallucinated terms per run: {avg_h:.1f}")
                print(f"     any banned hype word: {banned}/{len(rows)}")
                print(f"     in 60-90 word range: {in_len}/{len(rows)}")
                print(f"     avg word count: {avg_wc:.1f}")
                print(f"     markdown drift: {md}/{len(rows)}")

if __name__ == "__main__":
    main()
