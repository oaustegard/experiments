#!/usr/bin/env python3
"""Two-stage routing: a five-way group choice, then a five-tool agent.

Two measurements from the flat arms motivate this. Declaring a sixth tool costs
a fixed ~750 ms per turn, and giving the model a five-tool catalogue that
contains the right answer is worth +11 to +17 points of routable accuracy. Both
say the same thing: keep every agent at five tools.

So route in two turns. Stage 1 declares five *group* tools, each of which just
names a family of reads and takes the query through unchanged. Stage 2 declares
the ≤5 real tools in the chosen group. Neither agent is ever above the
retrieval threshold, so neither pays the retrieval cost, and stage 2 sees a
catalogue small enough to be in the oracle regime.

    python3 two_stage.py                  # writes results_two_stage.json
    python3 two_stage.py --arm tuned      # which wording the leaf tools use

Cost of a wrong group is total: the right tool is not in stage 2's catalogue at
all, so a group error is unrecoverable by construction. The group table below
therefore reports stage-1 accuracy separately.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eval import load_items, score_one, summarize
from needle_bsky.router import Decision, load_schemas

# Five groups, none above five tools. Grouped by the object the user is asking
# about, which is the distinction the flat arms were worst at.
GROUPS: dict[str, list[str]] = {
    "account": ["get_profile", "get_user_posts", "analyze_account", "search_users"],
    "follow_graph": ["get_followers", "get_following"],
    "one_post": ["get_thread", "get_likes", "get_reposts", "get_quotes"],
    "find_content": ["search_posts", "get_trending", "get_trending_topics", "sample_firehose", "get_feed_posts"],
    "plumbing": ["resolve_identity", "atproto_status", "extract_keywords"],
}

GROUP_SCHEMAS = [
    {
        "name": "account",
        "description": "The request is about one named account or about finding accounts: their profile, their own posts, what they are about, or searching for accounts.",
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "the request, unchanged"}},
            "required": ["request"],
        },
    },
    {
        "name": "follow_graph",
        "description": "The request is about who follows an account or who an account follows.",
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "the request, unchanged"}},
            "required": ["request"],
        },
    },
    {
        "name": "one_post",
        "description": "The request names a specific post by URL or at:// URI and asks about that post: its replies, its likes, its reposts, or posts quoting it.",
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "the request, unchanged"}},
            "required": ["request"],
        },
    },
    {
        "name": "find_content",
        "description": "The request is to find or browse posts by subject rather than by author: searching, trending topics, the live stream, or a named feed or list.",
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "the request, unchanged"}},
            "required": ["request"],
        },
    },
    {
        "name": "plumbing",
        "description": "The request is about the network itself or about a supplied block of text: DIDs and hosting servers, whether atproto is up, or key terms in a passage.",
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "the request, unchanged"}},
            "required": ["request"],
        },
    },
]

GROUP_OF = {tool: g for g, tools in GROUPS.items() for tool in tools}

# --- the cheap stage 1 -------------------------------------------------------
# Structural cues in the query text, not a learned model and not a Needle turn.
# Written after seeing which groups the Needle classifier confused, which is a
# real overfitting risk and is stated in RESULTS.md; the rules are deliberately
# surface-level (does the text contain a post URI, a feed URI, a DID, a handle)
# rather than keyed to individual eval items.
_POST_URI = re.compile(r"(bsky\.app/profile/[^\s]+/post/|app\.bsky\.feed\.post/)", re.IGNORECASE)
_FEED_URI = re.compile(r"(bsky\.app/profile/[^\s]+/(lists|feed)/|app\.bsky\.(feed\.generator|graph\.list)/)", re.IGNORECASE)
_DID = re.compile(r"did:plc:", re.IGNORECASE)
_PLUMBING = re.compile(r"\b(did|pds|appview|app view|relay|plc|down|outage|broken|offline|keywords?|key terms)\b", re.IGNORECASE)
_GRAPH = re.compile(r"\b(follow|follows|follower|followers|following)\b", re.IGNORECASE)
_HANDLE = re.compile(r"\b[a-z0-9-]+(\.[a-z0-9-]+)+\b", re.IGNORECASE)


def heuristic_group(query: str) -> str:
    """Pick a group from surface cues. Microseconds, no model."""
    if _POST_URI.search(query):
        return "one_post"
    if _FEED_URI.search(query):
        return "find_content"
    if _DID.search(query) or _PLUMBING.search(query):
        return "plumbing"
    if _GRAPH.search(query):
        return "follow_graph"
    if _HANDLE.search(query):
        return "account"
    return "find_content"


class TwoStageRouter:
    def __init__(self, arm: str = "tuned-min", stage1: str = "needle"):
        import needle

        self.arm = arm
        self.stage1_kind = stage1
        by_name = {s["name"]: s for s in load_schemas(arm)}
        self.stage1 = needle.Needle(tools=GROUP_SCHEMAS) if stage1 == "needle" else None
        self.stage2 = {g: needle.Needle(tools=[by_name[t] for t in tools]) for g, tools in GROUPS.items()}
        self.schemas = list(by_name.values())

    def route(self, query: str) -> tuple[Decision, str | None]:
        t0 = time.perf_counter()
        if self.stage1_kind == "heuristic":
            group = heuristic_group(query)
        else:
            self.stage1.reset()
            r1 = self.stage1.complete(query)
            calls = r1.get("function_calls") or []
            if not calls:
                dt = (time.perf_counter() - t0) * 1000
                return (
                    Decision(query, None, {}, r1.get("confidence"), "refuse", r1.get("reasoning"), dt),
                    None,
                )
            group = calls[0]["name"]
        agent = self.stage2.get(group)
        if agent is None:
            dt = (time.perf_counter() - t0) * 1000
            return Decision(query, None, {}, r1.get("confidence"), "refuse", "unknown group", dt), group

        agent.reset()
        r2 = agent.complete(query)
        dt = (time.perf_counter() - t0) * 1000
        calls2 = r2.get("function_calls") or []
        if not calls2:
            return (
                Decision(query, None, {}, r2.get("confidence"), "refuse", r2.get("reasoning"), dt),
                group,
            )
        c = calls2[0]
        return (
            Decision(
                query,
                c.get("name"),
                c.get("arguments") or {},
                r2.get("confidence"),
                "act",
                r2.get("reasoning"),
                dt,
            ),
            group,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="tuned-min")
    ap.add_argument("--stage1", default="needle", choices=["needle", "heuristic"])
    ap.add_argument("--evalset", default=str(HERE / "evalset.jsonl"))
    a = ap.parse_args()

    items = load_items(Path(a.evalset))
    r = TwoStageRouter(a.arm, a.stage1)

    rows, group_rows = [], []
    for it in items:
        d, group = r.route(it["query"])
        row = score_one(it, d)
        want = GROUP_OF.get(it["tool"][0]) if it["tool"] else None
        row["group"] = group
        row["group_expected"] = want
        row["group_ok"] = (group == want) if want else (group is None)
        rows.append(row)
        group_rows.append(row)

    routable = [x for x in rows if x["expected"]]
    res = {
        "arm": f"two-stage-{a.stage1}-{a.arm}",
        "stage1": a.stage1,
        "groups": GROUPS,
        "summary": summarize(rows),
        "stage1_group_acc_routable": round(sum(x["group_ok"] for x in routable) / len(routable), 4),
        "stage1_refused_off_topic": round(
            sum(1 for x in rows if not x["expected"] and x["group"] is None) / max(1, len(rows) - len(routable)), 4
        ),
        "rows": rows,
    }
    (HERE / f"results_two_stage_{a.stage1}.json").write_text(json.dumps(res, indent=1))
    s = res["summary"]
    print(
        f"two-stage[{a.stage1}]-{a.arm}: tool {s['tool_acc']:.3f}  routable {s['tool_acc_routable']:.3f}  "
        f"refuse {s['refusal_acc']:.3f}  args {s['args_acc_routable']:.3f}  "
        f"invented {s['invented_rate']:.3f}  median {s['median_latency_ms']:.0f}ms"
    )
    print(f"  stage-1 group accuracy on routable queries: {res['stage1_group_acc_routable']:.3f}")
    print(f"  mean two-turn latency: {statistics.mean(x['latency_ms'] for x in rows):.0f} ms")
    bad = [(x["id"], x["group_expected"], x["group"]) for x in routable if not x["group_ok"]]
    print(f"  group errors ({len(bad)}): {bad[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
