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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handle", default="austegard.com")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "extract_demo.txt"))
    a = ap.parse_args()

    import bsky
    import needle

    posts = [p for p in bsky.get_user_posts(a.handle, limit=a.limit) if (p.get("text") or "").strip()]
    agent = needle.Needle(tools=[POST_RECORD])

    lines = [f"# extraction over {len(posts)} live posts from @{a.handle}", ""]
    for p in posts:
        text = " ".join(p["text"].split())
        agent.reset()
        r = agent.complete(text)
        calls = r.get("function_calls") or []
        got = calls[0]["arguments"] if calls else None
        lines.append(f"post: {text[:200]}")
        lines.append(f"  -> {json.dumps(got, ensure_ascii=False)}  (confidence {r.get('confidence')})")
        lines.append("")

    out = "\n".join(lines)
    Path(a.out).write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
