"""Two-stage routing: pick a group of ≤5 tools, then route inside it.

The measured design. Two results from the flat arms motivate it — a five-tool
catalogue containing the right answer beats the flat 18 by 11–17 points, and
declaring a sixth tool costs 3.6x the per-turn latency — and a third one
constrains it: asking Needle itself to choose the group is 24 points *worse*
than declaring all 18, so stage 1 has to be something other than a model turn.

`heuristic_group` is that something. Structural cues in the query text, no model,
microseconds. On the eval set it picks the right group for 87% of routable
queries, and the whole two-stage router lands at 0.722 routable against the flat
arm's 0.611, at 316 ms against 1187 ms.

A stage-1 error is unrecoverable: the right tool is then absent from stage 2's
catalogue entirely. That is the price of the design and it is why stage 1 is
deterministic and inspectable rather than learned.
"""

from __future__ import annotations

import re
import time

from .router import DEFAULT_THRESHOLD, EXECUTOR_ONLY, Decision, load_schemas

# Five groups, none above the five-tool retrieval threshold. Grouped by the
# object the request is about, which is the distinction the flat arms were
# worst at.
GROUPS: dict[str, list[str]] = {
    "account": ["get_profile", "get_user_posts", "analyze_account", "search_users"],
    "follow_graph": ["get_followers", "get_following"],
    "one_post": ["get_thread", "get_likes", "get_reposts", "get_quotes"],
    "find_content": ["search_posts", "get_trending", "get_trending_topics", "sample_firehose", "get_feed_posts"],
    "plumbing": ["resolve_identity", "atproto_status", "extract_keywords"],
}

GROUP_OF = {tool: g for g, tools in GROUPS.items() for tool in tools}

_DESCRIPTIONS = {
    "account": "The request is about one named account or about finding accounts: their profile, their own posts, what they are about, or searching for accounts.",
    "follow_graph": "The request is about who follows an account or who an account follows.",
    "one_post": "The request names a specific post by URL or at:// URI and asks about that post: its replies, its likes, its reposts, or posts quoting it.",
    "find_content": "The request is to find or browse posts by subject rather than by author: searching, trending topics, the live stream, or a named feed or list.",
    "plumbing": "The request is about the network itself or about a supplied block of text: DIDs and hosting servers, whether atproto is up, or key terms in a passage.",
}

GROUP_SCHEMAS = [
    {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {"request": {"type": "string", "description": "the request, unchanged"}},
            "required": ["request"],
        },
    }
    for name, desc in _DESCRIPTIONS.items()
]

# --- stage 1, the cheap one --------------------------------------------------
# Surface structure only: does the text carry a post URI, a feed or list URI, a
# DID, a follow word, a handle-shaped token. Written after reading which groups
# the model-based stage 1 confused, so its 0.870 on this eval is fitted to this
# distribution to an unknown degree; the rules are deliberately structural
# rather than keyed to individual queries.
_POST_URI = re.compile(r"(bsky\.app/profile/[^\s]+/post/|app\.bsky\.feed\.post/)", re.IGNORECASE)
_FEED_URI = re.compile(r"(bsky\.app/profile/[^\s]+/(lists|feed)/|app\.bsky\.(feed\.generator|graph\.list)/)", re.IGNORECASE)
_DID = re.compile(r"did:plc:", re.IGNORECASE)
_PLUMBING = re.compile(r"\b(did|pds|appview|app view|relay|plc|down|outage|broken|offline|keywords?|key terms)\b", re.IGNORECASE)
_GRAPH = re.compile(r"\b(follow|follows|follower|followers|following)\b", re.IGNORECASE)
_HANDLE = re.compile(r"\b[a-z0-9-]+(\.[a-z0-9-]+)+\b", re.IGNORECASE)


def heuristic_group(query: str) -> str:
    """Pick a group from surface cues. No model, microseconds."""
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


class GroupedRouter:
    """Same surface as `Router`: `route(query)` and `call(query)`."""

    def __init__(self, arm: str = "tuned-min", threshold: float = DEFAULT_THRESHOLD, stage1: str = "heuristic"):
        import needle

        self.arm = arm
        self.threshold = threshold
        self.stage1_kind = stage1
        by_name = {s["name"]: s for s in load_schemas(arm)}
        self.schemas = list(by_name.values())
        self.stage1 = needle.Needle(tools=GROUP_SCHEMAS) if stage1 == "needle" else None
        self.stage2 = {g: needle.Needle(tools=[by_name[t] for t in tools]) for g, tools in GROUPS.items()}

    def pick_group(self, query: str) -> tuple[str | None, dict]:
        if self.stage1_kind == "heuristic":
            return heuristic_group(query), {}
        self.stage1.reset()
        r = self.stage1.complete(query)
        calls = r.get("function_calls") or []
        return (calls[0]["name"] if calls else None), r

    def route(self, query: str) -> Decision:
        t0 = time.perf_counter()
        group, r1 = self.pick_group(query)
        if group is None or group not in self.stage2:
            dt = (time.perf_counter() - t0) * 1000
            return Decision(query, None, {}, r1.get("confidence"), "refuse", r1.get("reasoning"), dt)

        agent = self.stage2[group]
        agent.reset()
        r = agent.complete(query)
        dt = (time.perf_counter() - t0) * 1000
        calls = r.get("function_calls") or []
        conf = r.get("confidence")
        if not calls:
            return Decision(query, None, {}, conf, "refuse", r.get("reasoning"), dt, raw={"group": group})
        c = calls[0]
        gate = "act" if (conf is None or conf >= self.threshold) else "escalate"
        return Decision(
            query,
            c.get("name"),
            c.get("arguments") or {},
            conf,
            gate,
            r.get("reasoning"),
            dt,
            prefill_tps=r.get("prefill_tps"),
            decode_tps=r.get("decode_tps"),
            peak_ram_mb=r.get("peak_ram_mb"),
            raw={"group": group},
        )

    def call(self, query: str):
        from . import catalogue

        d = self.route(query)
        if d.decision != "act" or not d.tool:
            return d, None
        fn = catalogue.executors().get(d.tool)
        if fn is None:
            d.decision = "escalate"
            return d, None
        args = {k: v for k, v in d.arguments.items() if k not in EXECUTOR_ONLY}
        d.arguments = args
        return d, fn(**args)
