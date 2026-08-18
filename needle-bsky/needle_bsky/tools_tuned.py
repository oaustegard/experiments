"""Hand-authored schemas for the 18-tool Bluesky catalogue.

The tuned arm. Written for a 45M router rather than for a human reader:

* description is verb-first and names the object that distinguishes this tool
  from its neighbours ("who liked" vs "who reposted" vs "posts that quote");
* every argument carries the phrase a person would actually say;
* arguments a router cannot fill from natural language are not declared at all
  (`transcribe`, `stopwords`, `depth`, `parent_height`) — the executor supplies
  the defaults;
* numeric bounds go in the schema so the decode grammar enforces them.

Compare `tools_auto.py`, which is the same 18 tools with their existing
human-facing docstrings and full signatures, derived by introspection.
"""

from __future__ import annotations

_HANDLE = "a Bluesky handle such as pfrazee.com or austegard.com, with or without a leading @"
_POST = "a bsky.app post URL or an at:// post URI"


def _limit(default: int, desc: str = "how many results to return") -> dict:
    return {"type": "integer", "minimum": 1, "maximum": 100, "description": desc, "default": default}


SCHEMAS: list[dict] = [
    {
        "name": "get_profile",
        "description": "Get one account's profile card: display name, bio, follower count, following count, post count.",
        "parameters": {
            "type": "object",
            "properties": {"handle": {"type": "string", "description": _HANDLE}},
            "required": ["handle"],
        },
    },
    {
        "name": "get_user_posts",
        "description": "Get the recent posts written by one named account.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": _HANDLE},
                "limit": _limit(20, "how many posts to return"),
            },
            "required": ["handle"],
        },
    },
    {
        "name": "search_posts",
        "description": "Search all of Bluesky for posts matching words or a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the words or topic to search for"},
                "author": {"type": "string", "description": "restrict the search to one author's handle"},
                "since": {"type": "string", "description": "earliest date, YYYY-MM-DD"},
                "until": {"type": "string", "description": "latest date, YYYY-MM-DD"},
                "lang": {"type": "string", "description": "two-letter language code such as en or no"},
                "limit": _limit(25, "how many posts to return"),
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_feed_posts",
        "description": "Get the posts in a named custom feed or a user list, given its URL or at:// URI.",
        "parameters": {
            "type": "object",
            "properties": {
                "feed_uri": {
                    "type": "string",
                    "description": "a bsky.app feed or list URL, or an at:// URI for a feed or list",
                },
                "limit": _limit(20, "how many posts to return"),
            },
            "required": ["feed_uri"],
        },
    },
    {
        "name": "get_trending",
        "description": "Get what is trending on Bluesky right now, with post counts and the top accounts per topic.",
        "parameters": {
            "type": "object",
            "properties": {"limit": _limit(10, "how many trending topics to return")},
            "required": [],
        },
    },
    {
        "name": "get_trending_topics",
        "description": "Get just the names of the trending topics and suggested topics, without counts or accounts.",
        "parameters": {
            "type": "object",
            "properties": {"limit": _limit(10, "how many topic names to return")},
            "required": [],
        },
    },
    {
        "name": "sample_firehose",
        "description": "Listen to the live stream of new posts for a number of seconds and report what is being posted.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 120,
                    "description": "how many seconds to listen",
                    "default": 10,
                },
                "filter": {"type": "string", "description": "only keep posts containing this word"},
            },
            "required": [],
        },
    },
    {
        "name": "get_thread",
        "description": "Get one post together with its replies and the posts it is replying to.",
        "parameters": {
            "type": "object",
            "properties": {"post_uri_or_url": {"type": "string", "description": _POST}},
            "required": ["post_uri_or_url"],
        },
    },
    {
        "name": "get_quotes",
        "description": "Get the posts that quote a given post.",
        "parameters": {
            "type": "object",
            "properties": {
                "post_uri_or_url": {"type": "string", "description": _POST},
                "limit": _limit(25, "how many quote posts to return"),
            },
            "required": ["post_uri_or_url"],
        },
    },
    {
        "name": "get_likes",
        "description": "Get the accounts that liked a given post.",
        "parameters": {
            "type": "object",
            "properties": {
                "post_uri_or_url": {"type": "string", "description": _POST},
                "limit": _limit(50, "how many accounts to return"),
            },
            "required": ["post_uri_or_url"],
        },
    },
    {
        "name": "get_reposts",
        "description": "Get the accounts that reposted a given post.",
        "parameters": {
            "type": "object",
            "properties": {
                "post_uri_or_url": {"type": "string", "description": _POST},
                "limit": _limit(50, "how many accounts to return"),
            },
            "required": ["post_uri_or_url"],
        },
    },
    {
        "name": "get_followers",
        "description": "Get the accounts that follow one named account.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": _HANDLE},
                "limit": _limit(50, "how many accounts to return"),
            },
            "required": ["handle"],
        },
    },
    {
        "name": "get_following",
        "description": "Get the accounts that one named account follows.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": _HANDLE},
                "limit": _limit(50, "how many accounts to return"),
            },
            "required": ["handle"],
        },
    },
    {
        "name": "search_users",
        "description": "Find Bluesky accounts whose handle, display name or bio matches a search term.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the name, word or subject to look for in accounts"},
                "limit": _limit(25, "how many accounts to return"),
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_account",
        "description": "Summarise what one account is about by reading its profile and recent posts and pulling out its keywords.",
        "parameters": {
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": _HANDLE},
                "posts_limit": _limit(20, "how many of their posts to read"),
            },
            "required": ["handle"],
        },
    },
    {
        "name": "extract_keywords",
        "description": "Pull the key terms out of a block of text that was supplied directly.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the text to pull key terms from"},
                "top_n": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "how many key terms to return",
                    "default": 10,
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "resolve_identity",
        "description": "Look up the DID and hosting PDS server behind a handle, without using the Bluesky AppView.",
        "parameters": {
            "type": "object",
            "properties": {"actor": {"type": "string", "description": "a handle or a did:plc: identifier"}},
            "required": ["actor"],
        },
    },
    {
        "name": "atproto_status",
        "description": "Check which part of the atproto network is broken right now: the AppView, the relay, a PDS or the PLC directory.",
        "parameters": {
            "type": "object",
            "properties": {"actor": {"type": "string", "description": "optional handle whose own PDS to check as well"}},
            "required": [],
        },
    },
]

assert len({s["name"] for s in SCHEMAS}) == len(SCHEMAS)
