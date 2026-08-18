#!/usr/bin/env python3
"""The null model: route all 18 tools with no model at all.

`needle-bsky` found that a ~20-line regex beat a Needle turn by 35 points at
picking one of five *groups*, and the two-stage router that followed still used
Needle to pick the tool inside the group. Oskar's question is the obvious next
one: if deterministic code is better at the classification, what does it score
doing the whole job?

This is the baseline neither experiment measured, and it is the one that decides
whether the model layer earns its place on this catalogue at all.

    python3 regex_only.py

**Provenance, stated plainly.** These rules were written after months — well,
two days — of staring at this eval's failures, so their accuracy here is fitted
to this distribution to an unknown degree, exactly as the stage-1 group regex
was. They are written from the tool descriptions and the structural shape of the
requests, not from a lookup of the eval's answers, but that is a claim about
intent, not a guarantee. A held-out query set is the only thing that would
settle it, and there isn't one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

NEEDLE = experiment("needle-bsky")
sys.path.insert(0, str(NEEDLE))

POST_URI = re.compile(r"(?:https?://bsky\.app/profile/[^\s]+/post/[^\s]+|at://did:[^\s]+/app\.bsky\.feed\.post/[^\s]+)")
FEED_URI = re.compile(
    r"(?:https?://bsky\.app/profile/[^\s]+/(?:lists|feed)/[^\s]+|at://did:[^\s]+/app\.bsky\.(?:feed\.generator|graph\.list)/[^\s]+)"
)
DID = re.compile(r"did:plc:[a-z0-9]+", re.IGNORECASE)
HANDLE = re.compile(r"@?\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)+\b", re.IGNORECASE)

# Ordered rules. First match wins; each is (tool, pattern, requires).
# `requires` is a structural precondition, so a keyword alone cannot fire a rule
# that needs a post URI.
RULES: list[tuple[str, str, str | None]] = [
    # writes and other things no declared tool serves
    ("__refuse__", r"\b(post this|send a dm|dm |delete|block|mute|unfollow|follow (?!ers|ing)|reply to|like this|upload)\b", None),
    # a specific post
    ("get_likes", r"\b(like|likes|liked|likers)\b", "post"),
    ("get_reposts", r"\b(repost|reposts|reposted|reposter|boost|boosted|shared)\b", "post"),
    ("get_quotes", r"\b(quote|quotes|quoted|quoting|subtweet)\b", "post"),
    ("get_thread", r"\b(thread|repl(y|ies|ied)|conversation|discussion)\b", "post"),
    ("get_thread", r"", "post"),  # a bare post reference is a thread read
    # a feed or list
    ("get_feed_posts", r"", "feed"),
    # the network itself
    ("atproto_status", r"\b(down|outage|broken|offline|up\b|appview|app view|relay|working|healthy)\b", None),
    ("resolve_identity", r"\b(did|pds|hosts?|hosted|hosting|identifier|resolve)\b", None),
    ("extract_keywords", r"\b(key ?(terms|words)|keywords|main terms|pull the key)\b", None),
    # the follow graph
    ("get_following", r"\b(follows|following|subscribes)\b", None),
    ("get_followers", r"\b(followers|follower|audience|who follows)\b", None),
    # accounts
    ("analyze_account", r"\b(about|summari[sz]e|themes|characteri[sz]e|analy[sz]e|what is .* mostly)\b", "handle"),
    ("get_profile", r"\b(profile|bio|account page|how many|display name|look up the account)\b", "handle"),
    ("search_users", r"\b(accounts?|users?|people|whose)\b", None),
    ("get_user_posts", r"", "handle"),  # a bare handle is a timeline read
    # discovery
    ("sample_firehose", r"\b(firehose|live stream|stream|watch|listen)\b", None),
    ("get_trending", r"\b(how many posts|counts?|volumes?|busiest|rank)\b", None),
    ("get_trending_topics", r"\b(trending|hot right now|popular|topics?)\b", None),
    ("search_posts", r"", None),  # the fallback: it is a content search
]


def has(kind: str, q: str) -> bool:
    if kind == "post":
        return bool(POST_URI.search(q))
    if kind == "feed":
        return bool(FEED_URI.search(q))
    if kind == "handle":
        return bool(DID.search(q)) or _bare_handle(q) is not None
    return True


def _bare_handle(q: str) -> str | None:
    for m in HANDLE.finditer(q):
        tok = m.group(0).lstrip("@")
        if tok.startswith(("http", "at://", "bsky.app")):
            continue
        return tok
    return None


def route(q: str) -> str | None:
    for tool, pat, req in RULES:
        if req and not has(req, q):
            continue
        if pat and not re.search(pat, q, re.IGNORECASE):
            continue
        return None if tool == "__refuse__" else tool
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="regex-only")
    a = ap.parse_args()

    from needle_bsky.router import load_schemas

    from eval import norm
    from repair import repair_args

    schemas = {s["name"]: s for s in load_schemas("tuned-min")}
    items = [
        json.loads(x) for x in (NEEDLE / "evalset.jsonl").read_text().splitlines() if x.strip()
    ]

    rows, lat = [], []
    for it in items:
        t0 = time.perf_counter()
        tool = route(it["query"])
        args = {}
        if tool:
            req = schemas[tool]["parameters"].get("required", [])
            args = {k: v for k, v in repair_args(it["query"], dict.fromkeys(req)).items() if v is not None}
        dt = (time.perf_counter() - t0) * 1000
        lat.append(dt)
        accepted = it["tool"]
        if not accepted:
            tool_ok = tool is None
            args_ok = tool_ok
        else:
            tool_ok = tool is not None and tool in accepted
            args_ok = tool_ok and all(
                k in args and norm(args[k]) == norm(v) for k, v in it.get("args", {}).items()
            )
        rows.append(
            {
                "id": it["id"],
                "cat": it["cat"],
                "query": it["query"],
                "expected": accepted,
                "got": tool,
                "arguments": args,
                "tool_ok": tool_ok,
                "args_ok": args_ok,
                "invented": [],
                "latency_ms": round(dt, 4),
            }
        )

    on = [r for r in rows if r["expected"]]
    off = [r for r in rows if not r["expected"]]
    summary = {
        "n": len(rows),
        "tool_acc": round(sum(r["tool_ok"] for r in rows) / len(rows), 4),
        "tool_acc_routable": round(sum(r["tool_ok"] for r in on) / len(on), 4),
        "refusal_acc": round(sum(r["tool_ok"] for r in off) / len(off), 4),
        "args_acc_routable": round(sum(r["args_ok"] for r in on) / len(on), 4),
        "invented_rate": 0.0,
        "median_latency_ms": round(sorted(lat)[len(lat) // 2], 4),
    }
    (HERE / f"results_{a.label}.json").write_text(
        json.dumps({"label": a.label, "summary": summary, "rows": rows}, indent=1)
    )
    s = summary
    print(
        f"{a.label}: tool {s['tool_acc']:.3f}  routable {s['tool_acc_routable']:.3f}  "
        f"refuse {s['refusal_acc']:.3f}  args {s['args_acc_routable']:.3f}  "
        f"median {s['median_latency_ms']:.3f} ms"
    )
    bad = [(r["id"], r["expected"], r["got"]) for r in rows if not r["tool_ok"]]
    print(f"  errors ({len(bad)}):")
    for b in bad:
        print("   ", b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
