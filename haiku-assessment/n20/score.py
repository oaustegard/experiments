#!/usr/bin/env python3
"""Score n=20 Haiku outputs across 3 tasks x 2 conditions.

Per-task scoring rubrics:

TASK 1 (triage): Per-ticket category correctness against down-skilled
rubric (A=BUG, B=QUESTION, C=SPAM, D=BUG). Vanilla can also be
"reasonable" but we score against the same rubric for consistency.
Plus format compliance: 4-line schema with reason.

TASK 2 (code review): Found mutation bug? Stayed in scope (no fix
section)? Hallucinated extra bugs?

TASK 3 (voice rewrite): Banned words present? Length 60-90? Invented
architectural details?
"""

import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent / "outputs"

# ---------- TASK 1 SCORING ----------

# Rubric-correct categories. B is QUESTION per rubric.
TASK1_GOLD = {"A": "BUG", "B": "QUESTION", "C": "SPAM", "D": "BUG"}
# B as FEATURE_REQUEST is also defensible without the rubric — track separately
TASK1_GOLD_LOOSE = {"A": "BUG", "B": {"QUESTION", "FEATURE_REQUEST"},
                    "C": "SPAM", "D": "BUG"}

CATEGORIES = ["BUG", "FEATURE_REQUEST", "QUESTION", "SPAM"]

def extract_task1_classifications(text):
    """Return dict {ticket_id: category} for tickets A,B,C,D.

    Strategy: split the text into per-ticket blocks. A block for ticket X
    starts at the first line mentioning "Ticket X" (with markdown
    decoration tolerated) and runs until the next ticket mention or EOF.
    Within that block, find the first category keyword.
    """
    result = {}
    # Find positions of each ticket-header line. Tolerate markdown
    # decoration, blockquotes, list markers, and table column separators.
    # The ticket id must be either preceded by "Ticket " or appear after
    # markdown decoration / column separator with whitespace.
    header_re = re.compile(
        r"^[\s>*#\-|]*(?:\*\*)?\s*(?:Ticket\s+)?([ABCD])\b",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(header_re.finditer(text))
    if not matches:
        return result
    # Build block ranges per ticket
    for i, m in enumerate(matches):
        tid = m.group(1).upper()
        if tid in result:
            continue
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        cat_match = re.search(
            r"\b(BUG|FEATURE_REQUEST|FEATURE\s+REQUEST|QUESTION|SPAM)\b",
            block,
            re.IGNORECASE,
        )
        if cat_match:
            cat = cat_match.group(1).upper().replace(" ", "_")
            result[tid] = cat
    return result

def score_task1(text):
    cls = extract_task1_classifications(text)
    strict_correct = sum(
        1 for tid, gold in TASK1_GOLD.items()
        if cls.get(tid) == gold
    )
    loose_correct = 0
    for tid, gold in TASK1_GOLD_LOOSE.items():
        got = cls.get(tid)
        if isinstance(gold, set):
            if got in gold:
                loose_correct += 1
        elif got == gold:
            loose_correct += 1

    # Format: did they produce a compact 4-line schema?
    # Count non-empty lines that contain a ticket id + category
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    schema_lines = sum(
        1 for l in lines
        if re.match(r"^(?:\*\*|#)?\s*(?:ticket\s+)?[ABCD]\b", l, re.IGNORECASE)
        and any(c in l.upper() for c in CATEGORIES)
    )
    # "Tight schema" = exactly 4 such lines AND no big formatting (no **, no headers)
    has_markdown = bool(re.search(r"\*\*|^#", text, re.MULTILINE))
    tight_schema = (schema_lines == 4) and not has_markdown

    return {
        "B_classification": cls.get("B", "MISSING"),
        "strict_correct": strict_correct,        # /4 per rubric
        "loose_correct": loose_correct,          # /4 if B-as-FR counted
        "schema_lines": schema_lines,
        "tight_schema": tight_schema,
        "char_count": len(text),
        "has_markdown": has_markdown,
    }


# ---------- TASK 2 SCORING ----------

def score_task2(text):
    t = text.lower()
    # Found the mutation bug?
    found_mutation = any(kw in t for kw in [
        "mutat", "alias", "reference", "modif", "does not copy",
        "doesn't copy", "shares the same dict", "shared dict",
        "shared reference", "modifies the input", "modifies the original",
        "modifies the caller", "in-place",
    ])

    # Added unsolicited "fix" or "suggested" section?
    added_fix = any(kw in t for kw in [
        "fix:", "**fix", "## fix", "suggested fix",
        "the fix is", "to fix", "to fix this",
        "result = default_prefs.copy()",
        "copy.deepcopy", ".copy()",
        "should be:", "should use",
        "improved version", "corrected version",
        "here's the fix", "here's how to fix",
    ])

    # Mentioned None-skipping as a bug? (false positive — contract doesn't
    # explicitly promise None passes through; this is a borderline call)
    flagged_none_handling = any(kw in t for kw in [
        "none values are skipped",
        "filters out none",
        "ignores none",
        "none filter",
        "skips none values",
        "won't allow none",
    ])

    # Verbose? > 400 chars
    char_count = len(text)
    verbose = char_count > 400

    # Multiple bugs claimed? (down-skilling caps at 4)
    # Count numbered/bulleted items at line start
    finding_lines = re.findall(
        r"(?:^|\n)\s*(?:\d+[\.\)]|[-*])\s+",
        text,
    )
    num_findings = len(finding_lines)

    return {
        "found_mutation": found_mutation,
        "added_fix": added_fix,
        "flagged_none_handling": flagged_none_handling,
        "verbose": verbose,
        "char_count": char_count,
        "num_findings_listed": num_findings,
    }


# ---------- TASK 3 SCORING ----------

BANNED_WORDS = [
    "thrilled", "excited", "proud", "delighted", "can't wait",
    "amazing", "revolutionary", "groundbreaking",
    "paradigm shift", "game-changer", "game changer",
    "transformative", "cutting-edge", "cutting edge",
    "next-generation", "next generation",
    # also catch second-person hype
    "you'll love", "wait until you see",
]

# Loosely promotional words also worth tracking
SOFT_PROMO = [
    "transform", "leap forward", "ground-up",
    "we believe", "we hope", "we think", "raving",
    "elegance", "elegant", "innovative",
]

# Architectural facts NOT present in the input (hallucination markers).
# Input only says "caching layer" with no implementation details.
HALLUCINATED_DETAILS = [
    "lru", "ttl", "write-through", "write through",
    "redis", "memcached",
    "distributed hash table", "consistent hashing",
    "cache-aside", "cache aside",
    "p99", "p95", "p50",  # specific percentiles not in input
    "rdb", "snapshot",
    "in-memory", "in memory",  # input doesn't say
    "sub-millisecond", "milliseconds",
    "background queue", "background worker",
    "annotation", "decorator",
    "batch operation",
    "opentelemetry", "structured json",
    "session-cookie", "jwt",
    "highcharts", "server-side rendering",
]

def word_count(text):
    return len(re.findall(r"\b\w+\b", text))

def score_task3(text):
    t = text.lower()
    banned_present = [w for w in BANNED_WORDS if w in t]
    soft_promo_present = [w for w in SOFT_PROMO if w in t]
    hallucinated = [w for w in HALLUCINATED_DETAILS if w in t]

    wc = word_count(text)
    in_length = 60 <= wc <= 90

    return {
        "banned_count": len(banned_present),
        "banned_present": banned_present,
        "soft_promo_count": len(soft_promo_present),
        "soft_promo_present": soft_promo_present,
        "hallucinated_count": len(hallucinated),
        "hallucinated_present": hallucinated,
        "word_count": wc,
        "in_length_range": in_length,
    }


# ---------- MAIN ----------

def score_cell(task, condition):
    cell_dir = ROOT / f"task{task}_{condition}"
    files = sorted(cell_dir.glob("run_*.md"))
    scorer = {1: score_task1, 2: score_task2, 3: score_task3}[task]
    results = []
    for f in files:
        text = f.read_text()
        score = scorer(text)
        score["_file"] = f.name
        results.append(score)
    return results


def summarize_task1(rows):
    n = len(rows)
    strict_total = sum(r["strict_correct"] for r in rows)
    loose_total = sum(r["loose_correct"] for r in rows)
    tight_schema = sum(1 for r in rows if r["tight_schema"])
    markdown = sum(1 for r in rows if r["has_markdown"])
    b_dist = defaultdict(int)
    for r in rows:
        b_dist[r["B_classification"]] += 1
    avg_chars = sum(r["char_count"] for r in rows) / n
    return {
        "n": n,
        "strict_accuracy": strict_total / (n * 4),
        "loose_accuracy": loose_total / (n * 4),
        "tight_schema_rate": tight_schema / n,
        "markdown_rate": markdown / n,
        "B_distribution": dict(b_dist),
        "avg_chars": avg_chars,
    }

def summarize_task2(rows):
    n = len(rows)
    return {
        "n": n,
        "found_mutation_rate": sum(r["found_mutation"] for r in rows) / n,
        "added_fix_rate": sum(r["added_fix"] for r in rows) / n,
        "flagged_none_rate": sum(r["flagged_none_handling"] for r in rows) / n,
        "verbose_rate": sum(r["verbose"] for r in rows) / n,
        "avg_chars": sum(r["char_count"] for r in rows) / n,
        "avg_findings_listed": sum(r["num_findings_listed"] for r in rows) / n,
    }

def summarize_task3(rows):
    n = len(rows)
    any_banned = sum(1 for r in rows if r["banned_count"] > 0)
    any_hallucination = sum(1 for r in rows if r["hallucinated_count"] > 0)
    any_soft_promo = sum(1 for r in rows if r["soft_promo_count"] > 0)
    in_length = sum(1 for r in rows if r["in_length_range"])
    return {
        "n": n,
        "any_banned_rate": any_banned / n,
        "avg_banned_count": sum(r["banned_count"] for r in rows) / n,
        "any_soft_promo_rate": any_soft_promo / n,
        "avg_soft_promo_count": sum(r["soft_promo_count"] for r in rows) / n,
        "any_hallucination_rate": any_hallucination / n,
        "avg_hallucinated_count": sum(r["hallucinated_count"] for r in rows) / n,
        "in_length_rate": in_length / n,
        "avg_word_count": sum(r["word_count"] for r in rows) / n,
    }


def main():
    summarizers = {1: summarize_task1, 2: summarize_task2, 3: summarize_task3}
    output = {}
    for task in [1, 2, 3]:
        for cond in ["vanilla", "downskilled"]:
            rows = score_cell(task, cond)
            output[f"task{task}_{cond}"] = {
                "summary": summarizers[task](rows),
                "per_run": rows,
            }

    out_path = Path(__file__).parent / "scores.json"
    out_path.write_text(json.dumps(output, indent=2))

    # Print human-readable summary
    for k, v in output.items():
        print(f"\n=== {k} ===")
        for sk, sv in v["summary"].items():
            print(f"  {sk}: {sv}")

if __name__ == "__main__":
    main()
