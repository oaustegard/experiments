#!/usr/bin/env python3
"""Check that numbers in a writeup trace to the artifacts they came from.

A sentence carries a ref to the file and key its number was read from:

    Balanced accuracy rose to 0.778. <!-- @evidence[[results/final.json#tabular.bal_acc]] -->

Each ref resolves against the directory of the file it sits in and ends in one
of four states:

  resolved   the target and key exist, and the value matches a number in the
             claim text at that number's displayed precision (a percentage
             also matches a stored fraction, and `a/b`, `a of b` or `a in b` matches a stored a/b)
  anchored   the target exists but there is nothing to compare: no locator, a
             line-range or unknown locator, a non-numeric or container value,
             or a claim with no number in it
  dangling   the target file or the key does not exist
  mismatch   the key resolves to a number and no number in the claim matches it

dangling and mismatch exit 1. The claim text is the line the ref sits on with
every ref removed; a ref alone on its line checks the nearest non-empty line
above it. Refs inside fenced code blocks and `inline code` are ignored, so
documentation of the syntax does not trip the check.

Locators, by target type:
  .json         dotted path `a.b.0.c`, or JSON pointer `/a/b/0/c` for keys with dots
  .jsonl        same, first segment is the 0-based record index
  .csv / .tsv   `row(col=val,col2=val).col` — must select exactly one row
  anything      `L12` or `L12-L20` — the lines must exist (anchored)

  python3 _lib/evidence.py <experiment>/RESULTS.md [...]
  python3 _lib/evidence.py --all          # every */RESULTS.md in the repo
  python3 _lib/evidence.py --json FILE    # machine-readable report
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REF = re.compile(r"@evidence\[\[([^\]#]+?)(?:#([^\]]*))?\]\]")
COMMENT_WITH_REF = re.compile(r"<!--[^>]*?@evidence\[\[[^\]]*\]\][^>]*?-->")
NUMBER = re.compile(r"(?<![\w.])[-−]?\d[\d,]*(?:\.\d+)?(%?)")
FRACTION = re.compile(r"(?<![\w.])(\d+)\s*(?:/|\bof\b|\bin\b)\s*(\d+)(?![\w.])")
ROW = re.compile(r"^row\((.*)\)\.(.+)$")
LINES = re.compile(r"^L(\d+)(?:-L?(\d+))?$")
FAILING = ("dangling", "mismatch")


@dataclass
class Ref:
    file: str
    line: int
    target: str
    locator: str | None
    status: str
    detail: str


class Missing(Exception):
    """A target or key that does not exist."""


def _strip_code(text: str) -> list[str]:
    """Lines with fenced blocks blanked and inline code spans removed, numbering kept."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            out.append("")
            continue
        out.append("" if fenced else re.sub(r"`[^`]*`", "", line))
    return out


def _walk(value, parts: list[str], locator: str):
    for p in parts:
        if isinstance(value, dict) and p in value:
            value = value[p]
        elif isinstance(value, list) and re.fullmatch(r"-?\d+", p) and -len(value) <= int(p) < len(value):
            value = value[int(p)]
        else:
            raise Missing(f"key {p!r} not found on the way to {locator!r}")
    return value


def _split(locator: str) -> list[str]:
    if locator.startswith("/"):
        return [p.replace("~1", "/").replace("~0", "~") for p in locator[1:].split("/")]
    return locator.split(".")


def _lookup(path: Path, locator: str):
    """Return (value, comparable). comparable=False means the locator is opaque here."""
    m = LINES.match(locator)
    if m:
        n = sum(1 for _ in path.open(errors="replace"))
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if not 1 <= lo <= hi <= n:
            raise Missing(f"lines {lo}-{hi} outside 1-{n}")
        return None, False
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _walk(json.loads(path.read_text()), _split(locator), locator), True
    if suffix == ".jsonl":
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return _walk(records, _split(locator), locator), True
    if suffix in (".csv", ".tsv"):
        m = ROW.match(locator)
        if not m:
            return None, False
        conds = dict(c.split("=", 1) for c in m.group(1).split(",") if "=" in c)
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t" if suffix == ".tsv" else ","))
        if rows and (bad := [k for k in [*conds, m.group(2)] if k not in rows[0]]):
            raise Missing(f"no column {bad[0]!r}")
        hits = [r for r in rows if all(r.get(k) == v for k, v in conds.items())]
        if len(hits) != 1:
            raise Missing(f"row({m.group(1)}) selects {len(hits)} rows, need 1")
        return hits[0][m.group(2)], True
    return None, False


def _as_number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _matches(value: float, claim: str) -> bool:
    for m in FRACTION.finditer(claim):
        if int(m.group(2)) and abs(value - int(m.group(1)) / int(m.group(2))) <= 1e-6:
            return True
    for m in NUMBER.finditer(claim):
        s = m.group(0).rstrip("%").replace(",", "").replace("−", "-")
        decimals = len(s.split(".")[1]) if "." in s else 0
        shown = float(s)
        if not decimals and not m.group(1) and 0 < abs(value) < 1:
            continue  # a rate strictly between 0 and 1 is never written as a bare integer
        tol = 0.5 * 10 ** -decimals + 1e-9 * max(1.0, abs(shown))
        candidates = [value, value * 100] if m.group(1) else [value]
        if any(abs(c - shown) <= tol for c in candidates):
            return True
    return False


def check_file(doc: Path) -> list[Ref]:
    lines = _strip_code(doc.read_text())
    refs = []
    for i, line in enumerate(lines):
        for m in REF.finditer(line):
            target, locator = m.group(1).strip(), (m.group(2) or "").strip() or None
            claim = REF.sub("", COMMENT_WITH_REF.sub("", line)).strip()
            if not re.search(r"\w", claim):
                claim = next((l for l in reversed(lines[:i]) if l.strip()), "")
            refs.append(Ref(str(doc), i + 1, target, locator, *_judge(doc, target, locator, claim)))
    return refs


def _judge(doc: Path, target: str, locator: str | None, claim: str) -> tuple[str, str]:
    path = (doc.parent / target).resolve()
    if not path.is_file():
        return "dangling", f"no file {target}"
    if locator is None:
        return "anchored", "file exists; no locator"
    try:
        value, comparable = _lookup(path, locator)
    except Missing as e:
        return "dangling", str(e)
    except (json.JSONDecodeError, UnicodeDecodeError, csv.Error) as e:
        return "dangling", f"unreadable {target}: {e}"
    if not comparable:
        return "anchored", "locator not value-bearing for this file type"
    number = _as_number(value)
    if number is None:
        if isinstance(value, str) and value and value.lower() in claim.lower():
            return "resolved", f"text {value!r} appears in claim"
        return "anchored", f"non-numeric value {value!r:.60}"
    if not NUMBER.search(claim):
        return "anchored", f"value {value!r}; claim has no number to compare"
    if _matches(number, claim):
        return "resolved", f"value {value!r} matches claim"
    return "mismatch", f"artifact has {value!r}; claim reads {claim[:120]!r}"


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    args = [a for a in argv if a != "--json"]
    root = Path(__file__).resolve().parents[1]
    docs = sorted(root.glob("*/RESULTS.md")) if args == ["--all"] else [Path(a) for a in args]
    if not docs:
        print(__doc__.strip().split("\n\n")[0])
        return 2
    refs = [r for d in docs for r in check_file(d)]
    if as_json:
        print(json.dumps([asdict(r) for r in refs], indent=2))
    else:
        for r in refs:
            if r.status in FAILING or args != ["--all"]:
                loc = f"#{r.locator}" if r.locator else ""
                print(f"{r.status:<9} {r.file}:{r.line}  {r.target}{loc}  — {r.detail}")
        counts = {s: sum(r.status == s for r in refs) for s in ("resolved", "anchored", *FAILING)}
        print(f"{len(refs)} refs in {len(docs)} files: " + ", ".join(f"{n} {s}" for s, n in counts.items()))
    return 1 if any(r.status in FAILING for r in refs) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
