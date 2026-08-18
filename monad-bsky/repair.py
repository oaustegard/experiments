#!/usr/bin/env python3
"""Keep Monad's tool choice, take its argument strings away from it.

The copy probe says tuned Monad transcribes an identifier out of the request
correctly 51% of the time, against 78–90% for the grammar-constrained arm. The
misses are not routing errors — `austegard.com` came back as `afethew.com` and
`jetstream` as `jetforek`. Nothing about that is going to be fixed by more
training; epoch 3 copies *worse* than epoch 1.

So don't ask it to copy. This takes the tool name the model chose and fills the
arguments from the query with the same kind of structural extractor the
`needle-bsky` two-stage router uses for its group pick — a handle, a post URI, a
feed URI, a DID, a bare number for `limit` — and re-scores. Everything here is
post-processing over an existing results file; no model runs.

    python3 repair.py --results results_tuned-e3.json

**The extractor was written after reading the failures**, so its accuracy on
this eval is fitted to this distribution to an unknown degree, exactly as the
`needle-bsky` stage-1 regex was. What generalises is the shape: the model
chooses, deterministic code transcribes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

NEEDLE = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE))

_POST_URI = re.compile(r"(?:https?://bsky\.app/profile/[^\s]+/post/[^\s]+|at://did:[^\s]+/app\.bsky\.feed\.post/[^\s]+)")
_FEED_URI = re.compile(
    r"(?:https?://bsky\.app/profile/[^\s]+/(?:lists|feed)/[^\s]+|at://did:[^\s]+/app\.bsky\.(?:feed\.generator|graph\.list)/[^\s]+)"
)
_DID = re.compile(r"did:plc:[a-z0-9]+", re.IGNORECASE)
_HANDLE = re.compile(r"@?\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+\b", re.IGNORECASE)
_NUMBER = re.compile(r"\b(\d{1,3})\b")
_AFTER_COLON = re.compile(r":\s*(.+)$", re.DOTALL)

def _first(rx, q):
    m = rx.search(q)
    return m.group(0) if m else None


def _handle(q):
    for m in _HANDLE.finditer(q):
        tok = m.group(0).lstrip("@")
        if tok.startswith(("http", "at://", "bsky.app")):
            continue
        return tok
    return None


def _int(q):
    m = _NUMBER.search(q)
    return int(m.group(1)) if m else None


def _after_colon(q):
    m = _AFTER_COLON.search(q)
    return m.group(1).strip() if m else None


# Which extractor serves which argument name. Defined after the helpers so the
# names it references exist.
FILLERS = {
    "post_uri_or_url": lambda q: _first(_POST_URI, q),
    "feed_uri": lambda q: _first(_FEED_URI, q),
    "actor": lambda q: _first(_DID, q) or _handle(q),
    "handle": _handle,
    "limit": _int,
    "duration": _int,
    "text": _after_colon,
}


def repair_args(query: str, args: dict) -> dict:
    """Refill every argument the extractor knows how to fill; keep the rest."""
    out = {}
    for k, v in args.items():
        filler = FILLERS.get(k)
        if filler is None:
            out[k] = v
            continue
        got = filler(query)
        out[k] = got if got is not None else v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_tuned-e3.json"))
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    from eval import norm

    src = json.loads(Path(a.results).read_text())
    items = {
        json.loads(x)["id"]: json.loads(x)
        for x in (NEEDLE / "evalset.jsonl").read_text().splitlines()
        if x.strip()
    }

    rows = []
    for r in src["rows"]:
        it = items[r["id"]]
        new_args = repair_args(r["query"], r["arguments"])
        accepted = it["tool"]
        tool_ok = r["tool_ok"]
        if not accepted:
            args_ok = r["got"] is None
        else:
            args_ok = tool_ok and all(
                k in new_args and norm(new_args[k]) == norm(v) for k, v in it.get("args", {}).items()
            )
        rows.append({**r, "arguments": new_args, "args_ok": args_ok})

    on = [r for r in rows if r["expected"]]
    summary = {
        **src["summary"],
        "args_acc_routable": round(sum(1 for r in on if r["args_ok"]) / len(on), 4),
    }
    label = a.label or (src["label"] + "-repaired")
    out = {**src, "label": label, "summary": summary, "rows": rows, "repair": "regex argument fill"}
    (HERE / f"results_{label}.json").write_text(json.dumps(out, indent=1))

    print(f"{label}")
    print(f"  tool accuracy unchanged by construction: routable {summary['tool_acc_routable']:.3f}")
    print(f"  args routable {src['summary']['args_acc_routable']:.3f} -> {summary['args_acc_routable']:.3f}")
    still = [(r["id"], r["got"], r["arguments"]) for r in on if r["tool_ok"] and not r["args_ok"]]
    print(f"  tool right, args still wrong ({len(still)}):")
    for x in still[:8]:
        print("   ", x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
