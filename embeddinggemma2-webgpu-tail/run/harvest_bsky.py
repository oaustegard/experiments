"""Harvest recent posts by the members of a Bluesky list, for the in-browser search corpus.

Usage: python3 -I run/harvest_bsky.py [--list AT_URI] [--per-member 200] [--min-chars 20]

Default list: Oskar's "Artificial (and real) Intelligence"
(at://did:plc:r2whjvupgfw55mllpksnombn/app.bsky.graph.list/3lankcdrlip2f).
For each member: app.bsky.feed.getAuthorFeed with filter=posts_and_author_threads,
newest first, keeping posts the member wrote (no reposts) whose text plus image alt
text has at least --min-chars characters, up to --per-member posts.

Checkpoints one JSON line per member to out/bsky/posts.jsonl; a rerun skips members
already there. Post text stays in out/ (gitignored): the published index holds only
URIs, and the page fetches text from Bluesky when it shows a result.
"""
import argparse
import concurrent.futures
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "bsky"
API = "https://api.bsky.app/xrpc/"
LIST = "at://did:plc:r2whjvupgfw55mllpksnombn/app.bsky.graph.list/3lankcdrlip2f"


def get(method, **params):
    url = API + method + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "muninn-experiments"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 400:            # deleted / suspended actor and the like
                return None
            if attempt == 5:
                raise
        except OSError:
            if attempt == 5:
                raise
        time.sleep(2 ** attempt + random.random())
    return None


def members(list_uri):
    out, cursor = [], None
    while True:
        params = {"list": list_uri, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        d = get("app.bsky.graph.getList", **params)
        for it in d["items"]:
            s = it["subject"]
            out.append({"did": s["did"], "handle": s["handle"]})
        cursor = d.get("cursor")
        if not cursor or not d["items"]:
            return out, d["list"]["name"]


def post_text(post):
    rec = post["record"]
    parts = [rec.get("text", "")]
    emb = rec.get("embed") or {}
    imgs = emb.get("images") or (emb.get("media") or {}).get("images") or []
    parts += [i.get("alt", "") for i in imgs if i.get("alt")]
    return "\n".join(p for p in parts if p).strip()


def harvest(member, per_member, min_chars):
    kept, cursor, pages = [], None, 0
    while len(kept) < per_member and pages < 20:
        params = {"actor": member["did"], "limit": 100, "filter": "posts_and_author_threads"}
        if cursor:
            params["cursor"] = cursor
        d = get("app.bsky.feed.getAuthorFeed", **params)
        pages += 1
        if not d or not d.get("feed"):
            break
        for it in d["feed"]:
            p = it["post"]
            if it.get("reason") or p["author"]["did"] != member["did"]:
                continue
            text = post_text(p)
            if len(text) < min_chars:
                continue
            kept.append({"uri": p["uri"], "created_at": p["record"].get("createdAt"),
                         "langs": p["record"].get("langs"), "text": text})
            if len(kept) >= per_member:
                break
        cursor = d.get("cursor")
        if not cursor:
            break
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=LIST)
    ap.add_argument("--per-member", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ms, name = members(a.list)
    (OUT / "members.json").write_text(json.dumps({"list": a.list, "name": name, "members": ms}, indent=1))
    path = OUT / "posts.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["did"] for line in path.open()}
    print(f"{name}: {len(ms)} members, {len(done)} already harvested", flush=True)
    todo = [m for m in ms if m["did"] not in done]
    with path.open("a") as f, concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(harvest, m, a.per_member, a.min_chars): m for m in todo}
        for i, fut in enumerate(concurrent.futures.as_completed(futs)):
            m = futs[fut]
            posts = fut.result()
            f.write(json.dumps({"did": m["did"], "handle": m["handle"], "posts": posts}) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{len(done) + i + 1}/{len(ms)} {m['handle']}: {len(posts)}", flush=True)
    total = sum(len(json.loads(line)["posts"]) for line in path.open())
    print(f"done: {total} posts", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
