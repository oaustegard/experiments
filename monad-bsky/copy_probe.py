#!/usr/bin/env python3
"""Separate "picked the right tool" from "copied the identifier correctly".

The tuned-Monad errors look like two different failures wearing one label.
Some are routing (`get_followers` where `get_following` was wanted). The rest
are transcription: `austegard.com` came back as `afethew.com`, `jetstream` as
`jetforek`, `at://did:plc:s3cqfxbcwnvvyrsttl3wivgp` as
`at://did:plc:s3cqfxbcwnvvirior`. Monad's tokenizer holds 8,192 pieces learned
from English prose, so a handle or a DID shatters into many subwords and the
model regenerates it approximately.

A grammar-constrained decoder cannot make that mistake about a *tool name* —
the grammar admits only declared names — but it can about a free-string
argument, so this measures both models on the same footing.

    python3 copy_probe.py         # writes results_copy_probe.json

Scored over every routable eval item whose expected arguments contain a literal
span of the query: did the emitted value equal the expected one exactly, whether
or not the tool was right?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

NEEDLE = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE))


def literal_args(item: dict) -> dict:
    """Expected arguments whose value appears verbatim in the query."""
    q = item["query"].lower()
    return {k: v for k, v in item.get("args", {}).items() if isinstance(v, str) and v.lower().lstrip("@") in q}


def probe(rows: list[dict], items: dict) -> dict:
    from eval import norm

    total = hit = 0
    misses = []
    for r in rows:
        it = items.get(r["id"])
        if it is None:
            continue
        for k, v in literal_args(it).items():
            total += 1
            got = r["arguments"].get(k)
            # Credit the value wherever it landed: a value copied correctly into
            # the wrong key is a routing/schema error, not a copying one.
            values = {norm(x) for x in r["arguments"].values() if isinstance(x, str)}
            if got is not None and norm(got) == norm(v) or norm(v) in values:
                hit += 1
            else:
                misses.append({"id": r["id"], "arg": k, "want": v, "got": got})
    return {
        "n_literal_args": total,
        "verbatim_copies": hit,
        "copy_accuracy": round(hit / total, 4) if total else None,
        "misses": misses,
    }


def main() -> int:
    items = {
        json.loads(x)["id"]: json.loads(x)
        for x in (NEEDLE / "evalset.jsonl").read_text().splitlines()
        if x.strip()
    }

    arms = [
        ("needle-base", NEEDLE / "results_tuned-min.json"),
        ("needle-lora", NEEDLE / "results_finetuned.json"),
        ("needle-2stage", NEEDLE / "results_two_stage_heuristic.json"),
        ("monad-e1", HERE / "results_tuned-e1.json"),
        ("monad-e2", HERE / "results_tuned-e2.json"),
        ("monad-e3", HERE / "results_tuned-e3.json"),
    ]
    out = {}
    for label, path in arms:
        if not path.exists():
            continue
        rows = json.loads(path.read_text())["rows"]
        out[label] = probe(rows, items)
        r = out[label]
        print(f"{label:15} verbatim copy {r['verbatim_copies']:3d}/{r['n_literal_args']:3d} = {r['copy_accuracy']:.3f}")

    (HERE / "results_copy_probe.json").write_text(json.dumps(out, indent=1))
    print("\nMonad misses (first 12):")
    for m in out.get("monad-e3", {}).get("misses", [])[:12]:
        print(f"  {m['id']:11} {m['arg']:16} want {m['want']!r:52} got {m['got']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
