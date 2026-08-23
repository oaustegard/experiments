#!/usr/bin/env python3
"""Proxies for the register entries declaude_lint.py cannot reach.

Entries 7/41 (verdict headers), 12 (aphoristic closer) and 39 (welded epigram)
are structural. Regex cannot judge them, but each has a measurable shape:

  verdict header   a header that is a full clause — has a finite verb
  bare closer      the last sentence of a paragraph carrying no number, no
                   backticked identifier and no proper noun; a sentence that
                   states a general truth rather than a fact about this system
  welded clause    a sentence whose second half, after ", and"/", so"/"; ",
                   likewise carries no number, identifier or proper noun

These over-fire. They are counted, not trusted: every hit is printed so the
call can be checked against references/register.md by eye.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FENCE = re.compile(r"^```.*?^```", re.M | re.S)
TABLE = re.compile(r"^\|.*$", re.M)
SENT = re.compile(r"(?<=[.!?])\s+")
VERB = re.compile(
    r"\b(is|are|was|were|has|have|had|does|do|did|can|could|will|would|"
    r"makes?|made|took|take|got|get|adds?|added|removed?|means?|lives?|"
    r"doesn't|don't|didn't|isn't|wasn't|aren't|won't)\b", re.I)
# A concrete anchor: a digit, a `code` span, or a Capitalised word mid-sentence.
ANCHOR = re.compile(r"\d|`[^`]+`|(?<!^)(?<![.!?] )\b[A-Z][a-zA-Z]{2,}")
WELD = re.compile(r",\s+(?:and|so|but)\s+|;\s+")


def strip(text: str) -> str:
    return TABLE.sub("", FENCE.sub("", text))


def bare(sentence: str) -> bool:
    """No number, identifier or proper noun — a general claim, not a fact."""
    return not ANCHOR.search(sentence.strip())


def analyse(path: Path) -> dict:
    raw = path.read_text()
    body = strip(raw)

    headers, verdicts = [], []
    for ln in raw.splitlines():
        if ln.lstrip().startswith("#"):
            h = ln.lstrip("#").strip()
            headers.append(h)
            if VERB.search(h):
                verdicts.append(h)

    paras = [
        p.strip() for p in body.split("\n\n")
        if p.strip() and not p.lstrip().startswith(("#", "-", "*", "|", "1.", "2.", "3."))
    ]

    closers, welds = [], []
    for p in paras:
        sents = [s.strip() for s in SENT.split(p) if s.strip()]
        if sents and bare(sents[-1]) and 4 <= len(sents[-1].split()) <= 25:
            closers.append(sents[-1])
        for s in sents:
            parts = WELD.split(s)
            if len(parts) > 1 and bare(parts[-1]) and 4 <= len(parts[-1].split()) <= 20:
                welds.append(parts[-1].rstrip("."))

    words = len(body.split())
    return {
        "sample": path.stem, "words": words,
        "headers": len(headers), "verdict_headers": len(verdicts),
        "bare_closers": len(closers), "welded_clauses": len(welds),
        "structural_per_1k": round(
            (len(verdicts) + len(closers) + len(welds)) * 1000 / words, 2),
        "_verdicts": verdicts, "_closers": closers, "_welds": welds,
    }


def main() -> int:
    rows = [analyse(p) for p in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "samples").glob("*.md"))]
    rows.sort(key=lambda r: r["structural_per_1k"])

    cols = [("sample", 12), ("words", 6), ("headers", 8), ("verdict_headers", 8),
            ("bare_closers", 8), ("welded_clauses", 7), ("structural_per_1k", 6)]
    head = ["sample", "words", "headers", "verdict", "closers", "welds", "per1k"]
    print("  ".join(h.ljust(w) for (_, w), h in zip(cols, head)))
    print("  ".join("-" * w for _, w in cols))
    for r in rows:
        print("  ".join(str(r[k]).ljust(w) for k, w in cols))

    for r in rows:
        print(f"\n--- {r['sample']} ---")
        for label, key in (("verdict header", "_verdicts"),
                           ("bare closer", "_closers"),
                           ("welded clause", "_welds")):
            for s in r[key]:
                print(f"  [{label}] {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
