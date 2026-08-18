"""The Bluesky read surface this experiment routes over.

One list of tool names, one binding from name to the callable that serves it.
Both schema arms (`tools_auto`, `tools_tuned`) declare exactly these names, so
a routing decision is comparable across arms and the executor is shared.

The callables come from two skills:

* `browsing-bluesky/scripts/bsky.py` — everything through the AppView.
* `atprotoing/scripts/atproto.py` — the two reads that have no AppView
  equivalent (identity resolution, layer status).

`get_all_followers` / `get_all_following` are deliberately NOT declared. They
differ from `get_followers` / `get_following` only in pagination, which is a
caller's decision and not recoverable from natural language; declaring both
would manufacture a routing ambiguity that no real caller has.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# The skills mount on Claude Code on the Web. Overridable so this runs anywhere
# the two skill checkouts exist.
SKILLS = Path(os.environ.get("MUNINN_SKILLS_ROOT", "/mnt/skills/user"))
_BSKY = SKILLS / "browsing-bluesky" / "scripts"
_ATPROTO = SKILLS / "atprotoing" / "scripts"

for _p in (str(_BSKY), str(_ATPROTO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The 18 declared tools, in a stable order. Both arms iterate this list.
TOOL_NAMES = [
    "get_profile",
    "get_user_posts",
    "search_posts",
    "get_feed_posts",
    "get_trending",
    "get_trending_topics",
    "sample_firehose",
    "get_thread",
    "get_quotes",
    "get_likes",
    "get_reposts",
    "get_followers",
    "get_following",
    "search_users",
    "analyze_account",
    "extract_keywords",
    "resolve_identity",
    "atproto_status",
]


def _bsky():
    import bsky

    return bsky


def resolve_identity(actor: str):
    """Resolve a handle or DID to its DID, handle and PDS host."""
    import atproto

    return atproto.resolve(actor)


def atproto_status(actor: str | None = None):
    """Report which atproto layer (AppView, PDS, relay, PLC) is currently up."""
    import atproto

    return atproto.status(actor) if actor else atproto.status()


def executors() -> dict:
    """name -> callable. Imported lazily so schema-only work needs no network."""
    b = _bsky()
    return {
        "get_profile": b.get_profile,
        "get_user_posts": b.get_user_posts,
        "search_posts": b.search_posts,
        "get_feed_posts": b.get_feed_posts,
        "get_trending": b.get_trending,
        "get_trending_topics": b.get_trending_topics,
        "sample_firehose": b.sample_firehose,
        "get_thread": b.get_thread,
        "get_quotes": b.get_quotes,
        "get_likes": b.get_likes,
        "get_reposts": b.get_reposts,
        "get_followers": b.get_followers,
        "get_following": b.get_following,
        "search_users": b.search_users,
        "analyze_account": b.analyze_account,
        "extract_keywords": b.extract_keywords,
        "resolve_identity": resolve_identity,
        "atproto_status": atproto_status,
    }
