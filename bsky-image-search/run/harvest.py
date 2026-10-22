"""Collect popular image posts from Bluesky's public feeds, for a semantic image-search demo.

Usage: python3 -I run/harvest.py [--per-feed 3000]

Pages each feed in FEEDS through public.api.bsky.app app.bsky.feed.getFeed (no auth) until the
feed ends or --per-feed posts, keeps posts with images (embed images, or record-with-media
images), drops anything labelled adult or graphic (post labels, author labels, self-labels),
and writes one row per image to out/images.jsonl, deduplicated by image blob cid and sorted by
the post's like count: {cid, did, rkey, handle, likes, reposts, feed, alt, thumb, fullsize,
aspect, created}. Post text stays out of git (out/ is ignored); the demo shows images through
Bluesky's CDN URLs. Checkpoints per feed in out/feeds/, so a rerun resumes.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getFeed?"
# Popular general and image-heavy feeds (getPopularFeedGenerators, 2026-10-20). Adult-only feeds
# are left out on purpose.
FEEDS = {
    "whats-hot": "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot",
    "art-whats-hot": "at://did:plc:odqmsar3ikz5ubokya4sempk/app.bsky.feed.generator/aaabfn4o34jdi",
    "artists-trending": "at://did:plc:y7crv2yh74s7qhmtx3mvbgv5/app.bsky.feed.generator/art-new",
    "photography": "at://did:plc:ojisdwhv5b7tzitkipo3yx3f/app.bsky.feed.generator/aaao3rw3oxwku",
    "photography-2": "at://did:plc:y7crv2yh74s7qhmtx3mvbgv5/app.bsky.feed.generator/photography",
    "nature-photography": "at://did:plc:cngpbn4gngkf7oenuzysanng/app.bsky.feed.generator/aaaeu3w7ci7aw",
    "landscape": "at://did:plc:tfv6pt4z4f7f4v7k4p6fbeon/app.bsky.feed.generator/landscapes",
    "street": "at://did:plc:fs4fm7dpz7ineau52hijo2xe/app.bsky.feed.generator/aaafwjbw2yt6g",
    "film-photo": "at://did:plc:tpcyx4ahuaelumrzo5efbtyt/app.bsky.feed.generator/aaapwcekxomxy",
    "cats": "at://did:plc:jfhpnnst6flqway4eaeqzj2a/app.bsky.feed.generator/cats",
    "science": "at://did:plc:jfhpnnst6flqway4eaeqzj2a/app.bsky.feed.generator/for-science",
    "moss": "at://did:plc:x3cya3wkt4n6u4ihmvpsc5if/app.bsky.feed.generator/aaacynbxwimok",
}
BLOCK = {"porn", "sexual", "nudity", "graphic-media", "gore", "sexual-figurative", "!hide", "!warn", "nsfl"}


def get(feed, cursor):
    q = [("feed", feed), ("limit", "100")] + ([("cursor", cursor)] if cursor else [])
    req = urllib.request.Request(API + urllib.parse.urlencode(q), headers={"User-Agent": "muninn-experiments"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 (network: retry, then give up on this feed)
            if attempt == 4:
                print("  give up:", e, flush=True)
                return None
            time.sleep(2 ** attempt)


def labels(post):
    vals = {l.get("val") for l in post.get("labels", [])}
    vals |= {l.get("val") for l in post.get("author", {}).get("labels", [])}
    vals |= {v.get("val") for v in (post.get("record", {}).get("labels") or {}).get("values", [])}
    return vals


def images(post):
    e = post.get("embed") or {}
    return e.get("images") or (e.get("media") or {}).get("images") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-feed", type=int, default=3000)
    a = ap.parse_args()
    (OUT / "feeds").mkdir(parents=True, exist_ok=True)
    for name, uri in FEEDS.items():
        path = OUT / "feeds" / f"{name}.jsonl"
        if path.exists():
            continue
        posts, cursor, pages = [], None, 0
        while len(posts) < a.per_feed:
            d = get(uri, cursor)
            if not d or not d.get("feed"):
                break
            posts += [it["post"] for it in d["feed"]]
            pages += 1
            cursor = d.get("cursor")
            if not cursor:
                break
            time.sleep(0.3)
        path.write_text("".join(json.dumps(p) + "\n" for p in posts))
        print(f"{name}: {len(posts)} posts in {pages} pages", flush=True)

    rows, dropped = {}, 0
    for name in FEEDS:
        for line in (OUT / "feeds" / f"{name}.jsonl").open():
            p = json.loads(line)
            if labels(p) & BLOCK:
                dropped += 1
                continue
            did, rkey = p["uri"].split("/")[2], p["uri"].split("/")[-1]
            for img in images(p):
                cid = img["thumb"].rsplit("/", 1)[-1].split("@")[0]
                if cid in rows:
                    continue
                rows[cid] = {"cid": cid, "did": did, "rkey": rkey, "handle": p["author"].get("handle"),
                             "likes": p.get("likeCount", 0), "reposts": p.get("repostCount", 0), "feed": name,
                             "alt": img.get("alt", ""), "thumb": img["thumb"], "fullsize": img.get("fullsize"),
                             "aspect": img.get("aspectRatio"), "created": p.get("record", {}).get("createdAt")}
    out = sorted(rows.values(), key=lambda r: -r["likes"])
    (OUT / "images.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    with_alt = sum(1 for r in out if r["alt"].strip())
    print(f"{len(out)} images ({with_alt} with alt text), {dropped} labelled posts dropped", flush=True)


if __name__ == "__main__":
    main()
