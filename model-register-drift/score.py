#!/usr/bin/env python3
"""Score writing samples against the declauding register.

Runs declaude_lint.py over every sample, normalises hit counts per 1000 words,
and adds the prose-shape measures the linter reports as text rather than JSON.

    python3 score.py samples/ [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

LINT = Path("/mnt/skills/user/declauding/scripts/declaude_lint.py")

# Which linter categories belong to which family in references/register.md.
STAGING = {
    "negation-first", "significance", "agency", "deferred-noun", "announce",
    "locator", "staging", "em-dash", "aphorism", "rhetorical-q",
    "self-grading", "humility", "throat-clearing", "rtfm",
}
FLAT = {
    "copula", "participle", "triad", "false-range", "list-shape", "chatbot",
    "filler", "gap-fill", "diff-anchored", "subjectless", "hyphenation",
    "spec-ese", "dev-cliche", "slop", "editorializing", "time-inflation",
}
STRUCTURE = {"header", "typography", "cadence", "reuse", "density"}

CODE_FENCE = re.compile(r"^```.*?^```", re.M | re.S)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def prose_of(text: str) -> str:
    """Body prose only: no fences, no headers, no frontmatter."""
    text = CODE_FENCE.sub("", text)
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(lines)


def measure(path: Path) -> dict:
    raw = path.read_text()
    prose = prose_of(raw)
    words = len(prose.split())

    out = subprocess.run(
        [sys.executable, str(LINT), str(path), "--json"],
        capture_output=True, text=True,
    )
    hits = json.loads(out.stdout)["hits"] if out.stdout.strip() else []

    by_cat: dict[str, int] = {}
    for h in hits:
        by_cat[h["category"]] = by_cat.get(h["category"], 0) + 1

    # Structure blocks are per-document findings, not per-sentence tics, so they
    # are counted but kept out of the density figure.
    tic_hits = [h for h in hits if h["category"] not in STRUCTURE]
    per_k = lambda n: round(n * 1000 / words, 2) if words else 0.0

    sentences = [s for s in SENT_SPLIT.split(prose) if s.strip()]
    lens = sorted(len(s.split()) for s in sentences)
    paras = [p.strip() for p in prose.split("\n\n") if p.strip()]
    one_liners = sum(
        1 for p in paras
        if len(SENT_SPLIT.split(p)) == 1 and not p.startswith(("-", "*", "|", ">"))
    )

    headers = [ln.strip() for ln in raw.splitlines() if ln.lstrip().startswith("#")]

    return {
        "sample": path.stem,
        "words": words,
        "sentences": len(sentences),
        "median_sentence_words": lens[len(lens) // 2] if lens else 0,
        "short_sentences_pct": round(
            100 * sum(1 for n in lens if n <= 5) / len(lens), 1) if lens else 0.0,
        "em_dashes": prose.count("—"),
        "em_dash_per_150w": round(prose.count("—") * 150 / words, 2) if words else 0.0,
        "bold_spans": len(re.findall(r"\*\*[^*]+\*\*", prose)),
        "headers": len(headers),
        "one_sentence_paragraphs": one_liners,
        "total_hits": len(hits),
        "tic_hits": len(tic_hits),
        "tics_per_1k": per_k(len(tic_hits)),
        "staging_per_1k": per_k(sum(n for c, n in by_cat.items() if c in STAGING)),
        "flat_per_1k": per_k(sum(n for c, n in by_cat.items() if c in FLAT)),
        "structure_findings": sum(n for c, n in by_cat.items() if c in STRUCTURE),
        "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="samples")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = [measure(p) for p in sorted(Path(args.dir).glob("*.md"))]
    rows.sort(key=lambda r: r["tics_per_1k"])

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    cols = [
        ("sample", 22), ("words", 6), ("tic_hits", 5), ("tics_per_1k", 6),
        ("staging_per_1k", 8), ("flat_per_1k", 6), ("structure_findings", 6),
        ("em_dash_per_150w", 8), ("short_sentences_pct", 7),
        ("one_sentence_paragraphs", 5),
    ]
    head = ["sample", "words", "hits", "per1k", "staging", "flat", "struct",
            "dash/150", "short%", "1-sent"]
    print("  ".join(h.ljust(w) for (_, w), h in zip(cols, head)))
    print("  ".join("-" * w for _, w in cols))
    for r in rows:
        print("  ".join(str(r[k]).ljust(w) for k, w in cols))

    print("\ncategories by sample")
    for r in rows:
        top = ", ".join(f"{c} {n}" for c, n in list(r["by_category"].items())[:8])
        print(f"  {r['sample']}: {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
