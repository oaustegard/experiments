#!/usr/bin/env python3
"""How often does the request actually contain the thing it refers to?

`monad-bsky`'s eval queries always carried the handle, post URI or DID the tool
needed, so a structural router always had something to condition on. Real
requests in a session do not: the referent has usually been established a few
turns earlier and the sentence says "it".

This measures that directly — the share of routable queries carrying any
structural cue at all, and the share carrying the specific cue their gold tool's
required arguments need.

    python3 context_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

from catalogue import load as load_catalogue
from cues import cues, extract

HERE = Path(__file__).resolve().parent

SPLITS = {
    "family A (fitted)": HERE / "data" / "family_a.jsonl",
    "family B (held-out)": HERE / "data" / "family_b.jsonl",
    "wild (hand-authored)": HERE / "wild.jsonl",
}


def main() -> int:
    cat = load_catalogue("session")
    hdr = f"{'split':<24}{'n':>5}{'any cue':>10}{'owner/repo':>12}{'all required':>14}"
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for name, path in SPLITS.items():
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        on = [r for r in rows if r.get("label")]
        any_cue = has_or = complete = 0
        for r in on:
            c = cues(r["query"])
            any_cue += any(c.values())
            has_or += c["owner_repo"]
            spec = cat[r["label"].split("::")[0]]
            got = extract(r["query"])
            complete += all(k in got or k == "method" for k in spec["required"])
        n = len(on)
        out[name] = {"n": n, "any_cue": round(any_cue / n, 4),
                     "owner_repo": round(has_or / n, 4),
                     "all_required_extractable": round(complete / n, 4)}
        print(f"{name:<24}{n:>5}{any_cue / n:>10.3f}{has_or / n:>12.3f}{complete / n:>14.3f}")
    (HERE / "results_context.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
