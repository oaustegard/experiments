#!/usr/bin/env python3
"""Mechanical checks on written or verified pages, independent of any model.

For each page: words, citations, citations whose id is not in the topic's
input file (invented or mistyped ids), and body sentences with no citation.

    python3 qa.py DATA pages|verified     # prints per-page rows and totals as JSON lines
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

CITE = re.compile(r"\[([ms]):([0-9a-f]{8})\]")
HEAD = re.compile(r"^=== (MEMORY|SESSION) ([0-9a-f]{8}) ", re.M)


def sentences(body: str) -> list[str]:
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out += [s for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", line.lstrip("-* ")) if len(s) > 25]
    return out


def check(page: dict, sources: str) -> dict:
    known = {("m" if k == "MEMORY" else "s") + ":" + i for k, i in HEAD.findall(sources)}
    blob = json.dumps({k: page.get(k) for k in ("body", "timeline", "open", "contradictions")})
    cites = [f"{k}:{i}" for k, i in CITE.findall(blob)]
    bad = sorted({c for c in cites if c not in known})
    sents = sentences(page.get("body", ""))
    uncited = [s for s in sents if not CITE.search(s)]
    return {"words": len(page.get("body", "").split()), "cites": len(cites), "distinct": len(set(cites)),
            "unknown_ids": bad, "sentences": len(sents), "uncited": len(uncited),
            "coverage": round(len(set(cites) & known) / max(1, len(known)), 3)}


def main(data: Path, stage: str) -> int:
    rows = []
    for f in sorted((data / stage / "out").glob("*.json")):
        try:
            page = json.loads(f.read_text())
        except json.JSONDecodeError:
            rows.append({"topic": f.stem, "error": "bad json"})
            continue
        r = {"topic": f.stem, **check(page, (data / "pages" / "inputs" / f"{f.stem}.txt").read_text())}
        rows.append(r)
        print(json.dumps(r))
    ok = [r for r in rows if "error" not in r]
    tot = {k: sum(r[k] for r in ok) for k in ("words", "cites", "sentences", "uncited")}
    tot["unknown_ids"] = sum(len(r["unknown_ids"]) for r in ok)
    tot["pages"] = len(ok)
    tot["mean_coverage"] = round(sum(r["coverage"] for r in ok) / max(1, len(ok)), 3)
    print(json.dumps({"TOTAL": tot}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), sys.argv[2]))
