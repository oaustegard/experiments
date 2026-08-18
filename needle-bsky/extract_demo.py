#!/usr/bin/env python3
"""The other half of the interface: pull structured records out of post text.

Needle treats extraction as tool calling with one declared tool, so the same
45M model that routes a query can also read a fetched post and return a typed
record. This runs it over real posts from a live Bluesky account feed and prints
what came back.

    python3 extract_demo.py --handle austegard.com --limit 10

**This is a capture, not a measurement.** There is no labelled ground truth
here, so nothing in RESULTS.md claims an extraction accuracy. It exists to show
the second door on the same engine, and to give a reader real inputs and real
outputs rather than a synthetic example.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from needle_bsky import catalogue  # noqa: F401  (puts the skill scripts on sys.path)

MINIMAL_RECORD = {
    "name": "post_record",
    "description": "What one Bluesky post is about.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "the main thing the post is about, in a few words"}
        },
        "required": ["subject"],
    },
}

POST_RECORD = {
    "name": "post_record",
    "description": "What one Bluesky post is about.",
    "parameters": {
        "type": "object",
        "properties": {
            "subject": {"type": "string", "description": "the main thing the post is about, in a few words"},
            "mentions_handle": {"type": "string", "description": "a handle the post mentions, if any"},
            "links_to": {"type": "string", "description": "a domain or URL the post links to, if any"},
            "language": {"type": "string", "description": "two-letter code for the language the post is written in"},
        },
        "required": ["subject"],
    },
}


# Constructed posts, not fetched ones: each carries the field values literally,
# to separate "the model cannot extract" from "the field asked for a summary
# rather than a span". Needle's documented contract is that arguments contain
# only values evidenced by the input, so a `subject` field is outside it by
# construction and these are inside it.
SPANNY = [
    "Bokbytterkveld torsdag 19:00 på Kulturhuset, meld deg på innen 20. august.",
    "Release 0.4.2 is out, fixes the ICE restart bug, download at github.com/oaustegard/remex",
    "Standup moved to 09:30 CET starting Monday, room B2.",
    "Selling a Wahoo Kickr Core for 4200 NOK, pickup in Oslo this weekend.",
    "Talk accepted: 'Small models, big catalogues' at ATmosphere Conf, March 14, Seattle.",
    "Rain all week in Bergen, 12C on Thursday, 9C by Saturday.",
]

EVENT_RECORD = {
    "name": "event_record",
    "description": "A dated or priced thing announced in a post.",
    "parameters": {
        "type": "object",
        "properties": {
            "what": {"type": "string", "description": "the name of the thing, copied from the post"},
            "when": {"type": "string", "description": "the date or time written in the post"},
            "where": {"type": "string", "description": "the place written in the post"},
            "amount": {"type": "string", "description": "a price or quantity written in the post"},
        },
        "required": ["what"],
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handle", default="austegard.com")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "extract_demo.txt"))
    a = ap.parse_args()

    import bsky
    import needle

    posts = [p for p in bsky.get_user_posts(a.handle, limit=a.limit) if (p.get("text") or "").strip()]
    texts = [" ".join(p["text"].split()) for p in posts]

    lines = [f"# extraction over {len(posts)} live posts from @{a.handle}", ""]
    for label, schema in (("four fields", POST_RECORD), ("subject only", MINIMAL_RECORD)):
        agent = needle.Needle(tools=[schema])
        confs, refused = [], 0
        lines.append(f"## declared: {label} ({', '.join(sorted(schema['parameters']['properties']))})")
        lines.append("")
        for text in texts:
            agent.reset()
            r = agent.complete(text)
            calls = r.get("function_calls") or []
            got = calls[0]["arguments"] if calls else None
            if got is None:
                refused += 1
            if r.get("confidence") is not None:
                confs.append(r["confidence"])
            lines.append(f"post: {text[:180]}")
            lines.append(f"  -> {json.dumps(got, ensure_ascii=False)}  (confidence {r.get('confidence')})")
            lines.append("")
        mean = sum(confs) / len(confs) if confs else 0.0
        lines.append(f"  mean confidence {mean:.4f}   refused {refused}/{len(texts)}")
        lines.append("")

    agent = needle.Needle(tools=[EVENT_RECORD])
    confs, refused = [], 0
    lines.append("## constructed posts, span-shaped fields (what, when, where, amount)")
    lines.append("")
    for text in SPANNY:
        agent.reset()
        r = agent.complete(text)
        calls = r.get("function_calls") or []
        got = calls[0]["arguments"] if calls else None
        if got is None:
            refused += 1
        if r.get("confidence") is not None:
            confs.append(r["confidence"])
        lines.append(f"post: {text}")
        lines.append(f"  -> {json.dumps(got, ensure_ascii=False)}  (confidence {r.get('confidence')})")
        lines.append("")
    mean = sum(confs) / len(confs) if confs else 0.0
    lines.append(f"  mean confidence {mean:.4f}   refused {refused}/{len(SPANNY)}")
    lines.append("")

    out = "\n".join(lines)
    Path(a.out).write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
