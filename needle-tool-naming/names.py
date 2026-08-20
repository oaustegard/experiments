"""Six naming variants over the one `needle-bsky` tuned-min catalogue.

Every variant declares the same 18 tools, the same arguments, the same required
sets and the same argument *names*. Only two things move:

* the tool **name**, and
* whether the prose **descriptions** (tool and argument) are the hand-authored
  ones or a constant placeholder.

`variant()` returns `(schemas, to_canonical)` where `to_canonical` maps whatever
name the model emits back to the catalogue name the scorer expects, so a
routing decision stays comparable across variants.

The `separated` names follow one rule stated in PREREG.md before any run:
`<verb>_<distinguishing object>`, the object being the noun phrase a user of
*this* tool would say, with the head noun disambiguating where two tools share a
noun (`follower_count` vs `follower_accounts`). The `adversarial` names are
those same strings cyclically rotated inside the confusable groups below — a
mechanical permutation, not a hand-picked one.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import experiment  # noqa: E402

NEEDLE_BSKY = experiment("needle-bsky")
if str(NEEDLE_BSKY) not in sys.path:
    sys.path.insert(0, str(NEEDLE_BSKY))

PLACEHOLDER_TOOL = "A Bluesky API operation."
PLACEHOLDER_ARG = "value"

# `<verb>_<distinguishing object>`, one per catalogue tool.
SEPARATED = {
    "get_profile": "get_profile_and_follower_count",
    "get_user_posts": "get_posts_by_one_account",
    "search_posts": "search_posts_by_keyword",
    "get_feed_posts": "get_posts_from_a_feed",
    "get_trending": "get_trending_posts",
    "get_trending_topics": "get_trending_topic_names",
    "sample_firehose": "sample_live_firehose",
    "get_thread": "get_thread_replies",
    "get_quotes": "get_posts_quoting_a_post",
    "get_likes": "get_accounts_that_liked",
    "get_reposts": "get_accounts_that_reposted",
    "get_followers": "get_follower_accounts",
    "get_following": "get_followed_accounts",
    "search_users": "search_for_accounts_by_name",
    "analyze_account": "summarize_one_account",
    "extract_keywords": "extract_keywords_from_text",
    "resolve_identity": "resolve_did_and_pds_host",
    "atproto_status": "check_network_outage_status",
}

# Confusable groups. `adversarial` gives each member the SEPARATED name of the
# next member in its group, so the distinguishing term lands on a neighbour.
CONFUSABLE_GROUPS = [
    ["get_profile", "get_followers", "get_following"],
    ["get_likes", "get_reposts", "get_quotes"],
    ["get_trending", "get_trending_topics"],
    ["search_posts", "search_users"],
    ["get_user_posts", "get_feed_posts"],
    ["resolve_identity", "atproto_status"],
    ["analyze_account", "extract_keywords"],
    ["get_thread", "sample_firehose"],
]


def _adversarial() -> dict[str, str]:
    out = {}
    for group in CONFUSABLE_GROUPS:
        for i, name in enumerate(group):
            out[name] = SEPARATED[group[(i + 1) % len(group)]]
    return out


ADVERSARIAL = _adversarial()

VARIANTS = ("canon", "desc-only", "names-only", "neither", "separated", "adversarial")


def _base_schemas() -> list[dict]:
    """The `tuned-min` schemas: hand-authored wording, required arguments only."""
    from needle_bsky.router import load_schemas

    return copy.deepcopy(load_schemas("tuned-min"))


def _opaque_names(schemas: list[dict]) -> dict[str, str]:
    return {s["name"]: f"tool_{i + 1:02d}" for i, s in enumerate(schemas)}


def _strip(schema: dict) -> None:
    schema["description"] = PLACEHOLDER_TOOL
    for prop in schema.get("parameters", {}).get("properties", {}).values():
        if "description" in prop:
            prop["description"] = PLACEHOLDER_ARG


def variant(name: str) -> tuple[list[dict], dict[str, str]]:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; expected one of {VARIANTS}")
    schemas = _base_schemas()

    rename: dict[str, str]
    if name in ("desc-only", "neither"):
        rename = _opaque_names(schemas)
    elif name == "separated":
        rename = dict(SEPARATED)
    elif name == "adversarial":
        rename = dict(ADVERSARIAL)
    else:  # canon, names-only
        rename = {s["name"]: s["name"] for s in schemas}

    strip_descriptions = name in ("names-only", "neither")
    to_canonical = {}
    for s in schemas:
        canonical = s["name"]
        s["name"] = rename[canonical]
        to_canonical[s["name"]] = canonical
        if strip_descriptions:
            _strip(s)

    assert len(to_canonical) == len(schemas), "renaming collided"
    return schemas, to_canonical


if __name__ == "__main__":
    for v in VARIANTS:
        schemas, back = variant(v)
        print(f"\n== {v}  ({len(schemas)} tools)")
        for s in schemas[:4]:
            print(f"   {s['name']:<32} {s['description'][:64]}")
