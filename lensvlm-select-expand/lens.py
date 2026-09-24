#!/usr/bin/env python3
"""The Expand tool, plus the ledger that enforces select-before-expand.

A subagent works through a batch of documents. Per document it must:
  1. Read the sheet PNG (Claude's Read tool) and look at it.
  2. commit  - record an image-only answer and a ranked page guess. Locked.
  3. expand  - get the full text of one page (at most MAX_EXPAND per doc,
               refused before commit, logged).
  4. final   - record the answer after expansion. Locked.

Usage:
  lens.py hello  <run> --model <model id from your system prompt>
  lens.py commit <run> <doc> --answer "..." --pages 7,3
  lens.py expand <run> <doc> <page>
  lens.py final  <run> <doc> --answer "..."
  lens.py closed <run> <doc> --answer "..."   (closed-book arm: no document)
"""
import argparse
import base64
import json
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_EXPAND = 3


def ledger_path(run):
    p = HERE / "runs" / run / "ledger.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def events(run):
    p = ledger_path(run)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def log(run, **ev):
    ev["t"] = round(time.time(), 1)
    with ledger_path(run).open("a") as f:
        f.write(json.dumps(ev) + "\n")


def pages_for(doc):
    texts = json.loads(zlib.decompress(base64.b64decode((HERE / "data" / "pages.bin").read_bytes())))
    if doc not in texts:
        sys.exit(f"unknown doc {doc}")
    return texts[doc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["hello", "commit", "expand", "final", "closed"])
    ap.add_argument("run")
    ap.add_argument("doc", nargs="?")
    ap.add_argument("page", nargs="?", type=int)
    ap.add_argument("--answer")
    ap.add_argument("--pages")
    ap.add_argument("--model")
    a = ap.parse_args()
    evs = [e for e in events(a.run) if e.get("doc") == a.doc] if a.doc else []
    kinds = [e["kind"] for e in evs]

    if a.cmd == "hello":
        log(a.run, kind="hello", model=a.model)
        print("ok")
    elif a.cmd == "commit":
        if "commit" in kinds:
            sys.exit("already committed for this doc; commit is final")
        if not a.answer or not a.pages:
            sys.exit("commit needs --answer and --pages")
        pages = [int(x) for x in a.pages.replace(" ", "").split(",") if x]
        log(a.run, kind="commit", doc=a.doc, answer=a.answer, pages=pages)
        print(f"committed. You may now expand up to {MAX_EXPAND} pages of {a.doc}.")
    elif a.cmd == "expand":
        if "commit" not in kinds:
            sys.exit("refused: commit your image-only answer and page guess first")
        if "final" in kinds:
            sys.exit("refused: already finalized")
        n = kinds.count("expand")
        if n >= MAX_EXPAND:
            sys.exit(f"refused: expansion budget ({MAX_EXPAND}) spent for {a.doc}")
        texts = pages_for(a.doc)
        if not a.page or not 1 <= a.page <= len(texts):
            sys.exit(f"page must be 1..{len(texts)}")
        log(a.run, kind="expand", doc=a.doc, page=a.page)
        print(f"Text content of {a.doc} page P{a.page} ({MAX_EXPAND - n - 1} expansions left):\n")
        print(texts[a.page - 1])
    elif a.cmd == "final":
        if "commit" not in kinds:
            sys.exit("refused: commit first")
        if "final" in kinds:
            sys.exit("already finalized")
        if not a.answer:
            sys.exit("final needs --answer")
        log(a.run, kind="final", doc=a.doc, answer=a.answer)
        print("finalized.")
    elif a.cmd == "closed":
        if "closed" in kinds:
            sys.exit("already answered")
        if not a.answer:
            sys.exit("closed needs --answer")
        log(a.run, kind="closed", doc=a.doc, answer=a.answer)
        print("recorded.")


if __name__ == "__main__":
    main()
