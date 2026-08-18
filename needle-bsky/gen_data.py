#!/usr/bin/env python3
"""Synthesize LoRA training data for the Bluesky catalogue.

Needle's own `needle generate-data` wants an OPENROUTER_API_KEY, which this
container does not have and is not supposed to have. This writes the same JSONL
format from templates instead:

    {"query": ..., "tools": [...], "answers": [{"name":..., "arguments":{...}}],
     "reasoning": ...}

**The confound, stated up front.** Both the training templates and
`evalset.jsonl` were authored by the same writer in the same sitting, so a gain
here is partly "the model learned this author's phrasing", not only "the model
learned the catalogue". Two things hold it down and neither removes it: the
entity pools are disjoint (no handle, post URI, feed URI or search topic that
appears in the eval set is used for training), and no template here reuses an
eval query's verb frame. Read the per-category table in RESULTS.md rather than
the headline number — a template-memorisation win shows up as a uniform lift
across categories including the ones the templates cover least.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- entity pools, disjoint from evalset.jsonl -------------------------------

HANDLES = [
    "danabra.mov", "retr0.id", "nowbreezing.ntw.app", "ewanmorr.bsky.social",
    "laurenshof.online", "cpluspatch.com", "hailey.at", "mackuba.eu",
    "aliceyuan.bsky.social", "tressiemcphd.bsky.social", "nrk.no",
    "kongehuset.no", "smallcircles.social", "tapbots.com", "futur.blue",
]
DIDS = [
    "did:plc:44ybard66vv44zksje25o7dz", "did:plc:z72i7hdynmk6r22z27h6tvur",
    "did:plc:vpkhqolt662uhesyj6nxm7ys", "did:plc:ragtjsm2j2vknwkz3zp4oxrd",
]
POST_URLS = [
    "https://bsky.app/profile/danabra.mov/post/3kwq7ybmhx22h",
    "https://bsky.app/profile/mackuba.eu/post/3l2nvvdcxu72c",
    "https://bsky.app/profile/hailey.at/post/3kzzrmfw2ok2b",
    "at://did:plc:44ybard66vv44zksje25o7dz/app.bsky.feed.post/3kwq7ybmhx22h",
    "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.post/3l2nvvdcxu72c",
]
FEEDS = [
    "https://bsky.app/profile/skyfeed.xyz/feed/catch-up",
    "https://bsky.app/profile/mackuba.eu/lists/3kb2rjyzqbn2y",
    "at://did:plc:ragtjsm2j2vknwkz3zp4oxrd/app.bsky.feed.generator/whats-hot",
    "at://did:plc:vpkhqolt662uhesyj6nxm7ys/app.bsky.graph.list/3l7abcdefgh2z",
]
TOPICS = [
    "rust compiler", "sourdough", "sea ice extent", "labour market data",
    "vision pro", "kubernetes operators", "birdwatching", "eurovision",
    "quantisation", "trail running", "public transit funding", "typst",
]
LANGS = [("norwegian", "no"), ("german", "de"), ("japanese", "ja"), ("french", "fr")]
TEXTS = [
    "the relay dropped every event for nine minutes while the pdses stayed green",
    "kaffe, regn og en sykkeltur langs fjorden i september",
    "quarterly revenue rose while headcount fell for the third consecutive period",
]

# --- templates ---------------------------------------------------------------
# Verb frames deliberately disjoint from evalset.jsonl.

# Templates that need two draws from different pools get a named function; a
# directly-called lambda would do the same job and read worse.


def _lang_search(r):
    t = r.choice(TOPICS)
    lname, lcode = r.choice(LANGS)
    return (f"{lname} posts about {t}", "search_posts", {"query": t, "lang": lcode},
            f"'{t}' -> query; '{lname}' -> lang {lcode}")


def _author_search(r):
    t, h = r.choice(TOPICS), r.choice(HANDLES)
    return (f"{t} posts written by {h}", "search_posts", {"query": t, "author": h},
            f"'{t}' -> query; '{h}' -> author")


def _hose_duration(r):
    d = r.choice([5, 15, 20, 45, 60])
    return (f"tap the stream for {d} seconds", "sample_firehose", {"duration": d},
            f"'{d} seconds' -> duration {d}")


def _hose_filter(r):
    t = r.choice(TOPICS)
    return (f"tap the stream and keep anything about {t}", "sample_firehose", {"filter": t},
            f"'{t}' -> filter")


def _keywords(r):
    t = r.choice(TEXTS)
    return (f"main terms in the following: {t}", "extract_keywords", {"text": t},
            "the supplied passage -> text")


TEMPLATES = [
    # (weight, builder)
    (3, lambda r: (f"grab {(h := r.choice(HANDLES))}'s timeline", "get_user_posts", {"handle": h},
                   f"'{h}' -> handle")),
    (2, lambda r: (f"I want to read what {(h := r.choice(HANDLES))} wrote recently", "get_user_posts",
                   {"handle": h}, f"'{h}' -> handle")),
    (2, lambda r: (f"anything new from {(h := r.choice(HANDLES))}?", "get_user_posts", {"handle": h},
                   f"'{h}' -> handle")),
    (3, lambda r: (f"open the account page for {(h := r.choice(HANDLES))}", "get_profile", {"handle": h},
                   f"'{h}' -> handle")),
    (2, lambda r: (f"how big is {(h := r.choice(HANDLES))}'s audience", "get_profile", {"handle": h},
                   f"'{h}' -> handle")),
    (3, lambda r: (f"dig up posts on {(t := r.choice(TOPICS))}", "search_posts", {"query": t},
                   f"'{t}' -> query")),
    (2, lambda r: (f"is anyone on bluesky discussing {(t := r.choice(TOPICS))}", "search_posts", {"query": t},
                   f"'{t}' -> query")),
    (2, _lang_search),
    (2, _author_search),
    (3, lambda r: (f"whose accounts mention {(t := r.choice(TOPICS))}", "search_users", {"query": t},
                   f"'{t}' -> query")),
    (2, lambda r: (f"look for accounts named {(h := r.choice(HANDLES).split('.')[0])}", "search_users",
                   {"query": h}, f"'{h}' -> query")),
    (3, lambda r: (f"pull the likers of {(p := r.choice(POST_URLS))}", "get_likes", {"post_uri_or_url": p},
                   "the post reference -> post_uri_or_url")),
    (2, lambda r: (f"how did {(p := r.choice(POST_URLS))} do on likes", "get_likes", {"post_uri_or_url": p},
                   "the post reference -> post_uri_or_url")),
    (3, lambda r: (f"pull the reposters of {(p := r.choice(POST_URLS))}", "get_reposts",
                   {"post_uri_or_url": p}, "the post reference -> post_uri_or_url")),
    (2, lambda r: (f"was {(p := r.choice(POST_URLS))} shared by anyone", "get_reposts",
                   {"post_uri_or_url": p}, "the post reference -> post_uri_or_url")),
    (3, lambda r: (f"pull the quote posts on {(p := r.choice(POST_URLS))}", "get_quotes",
                   {"post_uri_or_url": p}, "the post reference -> post_uri_or_url")),
    (2, lambda r: (f"did anybody subtweet {(p := r.choice(POST_URLS))}", "get_quotes",
                   {"post_uri_or_url": p}, "quoting -> get_quotes")),
    (3, lambda r: (f"expand the conversation under {(p := r.choice(POST_URLS))}", "get_thread",
                   {"post_uri_or_url": p}, "the post reference -> post_uri_or_url")),
    (2, lambda r: (f"give me the discussion around {(p := r.choice(POST_URLS))}", "get_thread",
                   {"post_uri_or_url": p}, "the post reference -> post_uri_or_url")),
    (3, lambda r: (f"whose follows make up {(h := r.choice(HANDLES))}'s graph", "get_following",
                   {"handle": h}, f"'{h}' -> handle; follows -> get_following")),
    (2, lambda r: (f"{(h := r.choice(HANDLES))} subscribes to which accounts", "get_following",
                   {"handle": h}, f"'{h}' -> handle")),
    (3, lambda r: (f"whose audience is {(h := r.choice(HANDLES))} made of", "get_followers",
                   {"handle": h}, f"'{h}' -> handle; audience -> get_followers")),
    (2, lambda r: (f"give me {(h := r.choice(HANDLES))}'s follower list", "get_followers", {"handle": h},
                   f"'{h}' -> handle")),
    (3, lambda r: (f"open {(f := r.choice(FEEDS))}", "get_feed_posts", {"feed_uri": f},
                   "the feed reference -> feed_uri")),
    (2, lambda r: (f"what has been posted into {(f := r.choice(FEEDS))}", "get_feed_posts", {"feed_uri": f},
                   "the feed reference -> feed_uri")),
    (3, lambda r: ("tell me the busiest subjects with their volumes", "get_trending", {},
                   "volumes -> get_trending")),
    (2, lambda r: ("rank today's subjects by how many posts they got", "get_trending", {},
                   "post counts -> get_trending")),
    (3, lambda r: ("name the current topics, nothing else", "get_trending_topics", {},
                   "names only -> get_trending_topics")),
    (2, lambda r: ("a bare list of what is popular", "get_trending_topics", {},
                   "bare list -> get_trending_topics")),
    (3, _hose_duration),
    (2, _hose_filter),
    (3, lambda r: (f"where is {(h := r.choice(HANDLES))}'s repo hosted", "resolve_identity", {"actor": h},
                   f"'{h}' -> actor; hosting -> resolve_identity")),
    (2, lambda r: (f"turn {(d := r.choice(DIDS))} into a handle", "resolve_identity", {"actor": d},
                   f"'{d}' -> actor")),
    (2, lambda r: (f"identifier behind {(h := r.choice(HANDLES))}", "resolve_identity", {"actor": h},
                   f"'{h}' -> actor")),
    (3, lambda r: ("anything broken in the network at the moment", "atproto_status", {},
                   "network health -> atproto_status")),
    (2, lambda r: ("are the pdses and the relay healthy", "atproto_status", {},
                   "layer health -> atproto_status")),
    (3, lambda r: (f"characterise {(h := r.choice(HANDLES))} from their posting", "analyze_account",
                   {"handle": h}, f"'{h}' -> handle; characterise -> analyze_account")),
    (2, lambda r: (f"what themes run through {(h := r.choice(HANDLES))}", "analyze_account", {"handle": h},
                   f"'{h}' -> handle")),
    (3, _keywords),
]

OFF_TOPIC = [
    "set a timer for twenty minutes", "convert 40 fahrenheit to celsius",
    "who won the match last night", "translate takk to english",
    "add milk to my shopping list", "call my sister",
    "what's my next calendar event", "rename this file to draft2",
    "reply to that post with a thumbs up", "follow that account for me",
    "mute the word politics", "upload this photo to my profile",
    "how much did I spend on coffee this month", "play something by Susanne Sundfor",
]


def _render(tools: list[dict], keep: str | None, k: int, r: random.Random) -> list[dict]:
    """The k tools this row declares.

    Needle renders at most five tools per turn — retrieval picks them at
    inference time — so a training row that declares all 18 does not match the
    context the model will ever decode against, and at 18 the tool block alone
    is ~1,400 tokens, past `--max-len 1024`. Every label then falls outside the
    window and the run reports `loss 0.0000` while learning nothing. Declare k.
    """
    if k >= len(tools):
        return tools
    by = {t["name"]: t for t in tools}
    pool = [n for n in by if n != keep]
    names = ([keep] if keep else []) + r.sample(pool, k - (1 if keep else 0))
    r.shuffle(names)
    return [by[n] for n in names]


def build(n: int, seed: int, tools: list[dict], off_ratio: float = 0.15, k: int = 5) -> list[dict]:
    r = random.Random(seed)
    pop = [f for w, f in TEMPLATES for _ in range(w)]
    rows = []
    n_off = int(n * off_ratio)
    for _ in range(n - n_off):
        q, name, args, reasoning = r.choice(pop)(r)
        rows.append(
            {
                "query": q,
                "tools": _render(tools, name, k, r),
                "answers": [{"name": name, "arguments": args}],
                "reasoning": reasoning,
            }
        )
    for _ in range(n_off):
        rows.append(
            {
                "query": r.choice(OFF_TOPIC),
                "tools": _render(tools, None, k, r),
                "answers": [],
                "reasoning": "no declared tool serves this",
            }
        )
    r.shuffle(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--arm", default="tuned-min")
    ap.add_argument("-k", type=int, default=5, help="tools declared per row; matches what retrieval renders")
    ap.add_argument("--out", default=str(HERE / "data" / "train.jsonl"))
    a = ap.parse_args()

    import sys

    sys.path.insert(0, str(HERE))
    from needle_bsky.router import load_schemas

    tools = load_schemas(a.arm)
    rows = build(a.n, a.seed, tools, k=a.k)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    names = {}
    for row in rows:
        k = row["answers"][0]["name"] if row["answers"] else "(refuse)"
        names[k] = names.get(k, 0) + 1
    print(f"{len(rows)} rows -> {out}")
    for k in sorted(names, key=lambda x: -names[x]):
        print(f"  {k:22} {names[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
