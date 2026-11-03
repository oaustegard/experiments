"""Page through the Paper Skygest feed (logged-out view) and rank the papers it links.

    python3 fetch_feed.py --pages 15      # -> data/feed_posts.json, data/papers_ranked.json

The feed is personalized for logged-in users; logged out it serves the latest ~120 paper posts
from across Bluesky, so each run merges with data/feed_posts.json and re-ranks. A paper is identified by the first arXiv id or DOI found in the post's
link facets, external embed or text. Papers are ranked by summed likes + reposts + replies + quotes
across every post that links them, then by number of posts.
"""
import argparse, json, re, time, urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEED = "at://did:plc:uaadt6f5bbda6cycbmatcm3z/app.bsky.feed.generator/preprintdigest"
API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getFeed"
ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", re.I)
DOI = re.compile(r"(10\.\d{4,9}/[^\s\"<>?#]+)", re.I)


def urls(post):
    rec = post["record"]
    out = [f["features"][0].get("uri", "") for f in rec.get("facets", []) if f.get("features")]
    e = post.get("embed", {})
    for ext in (e.get("external"), (e.get("media") or {}).get("external")):
        if ext:
            out.append(ext.get("uri", ""))
    out.append(rec.get("text", ""))
    return out


def paper_id(post):
    for u in urls(post):
        m = ARXIV.search(u)
        if m:
            return "arxiv:" + m.group(1)
    for u in urls(post):
        m = DOI.search(u)
        if m:
            d = m.group(1).rstrip(".,);]").lower()
            d = re.sub(r"(v\d+)?(\.full(-text)?|\.abstract|\.pdf)?$", "", d) if d.startswith("10.1101/") else d
            return "doi:" + d
    return None


def muninn_view():
    """The same feed as Muninn's account sees it (personalized: posts from accounts it follows)."""
    import os
    req = urllib.request.Request("https://bsky.social/xrpc/com.atproto.server.createSession", json.dumps(
        {"identifier": os.environ["MUNINN_BSKY_HANDLE"], "password": os.environ["MUNINN_BSKY_APP_PASSWORD"]}).encode(),
        {"Content-Type": "application/json"})
    jwt = json.load(urllib.request.urlopen(req))["accessJwt"]
    req = urllib.request.Request(f"https://bsky.social/xrpc/app.bsky.feed.getFeed?feed={FEED}&limit=100",
                                 headers={"Authorization": "Bearer " + jwt, "atproto-proxy": "did:web:api.bsky.app#bsky_appview"})
    return [x["post"] for x in json.load(urllib.request.urlopen(req))["feed"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=15)
    ap.add_argument("--muninn", action="store_true", help="also merge Muninn's personalized view")
    a = ap.parse_args()
    posts, cur = (muninn_view() if a.muninn else []), None
    for _ in range(a.pages):
        u = f"{API}?feed={FEED}&limit=100" + (f"&cursor={cur}" if cur else "")
        d = None
        for t in range(4):
            try:
                d = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "muninn-experiments"})))
                break
            except Exception as e:  # the feed generator 502s intermittently
                print("retry", t, e, flush=True); time.sleep(2 ** t)
        if d is None:
            break
        posts += [x["post"] for x in d["feed"]]
        cur = d.get("cursor")
        if not cur:
            break
        time.sleep(0.5)
    old = HERE / "data/feed_posts.json"
    if old.exists():  # accumulate snapshots: logged out, the feed is a window of the latest ~120 posts
        posts = json.loads(old.read_text()) + posts
    seen, uniq = set(), []
    for p in posts:
        if p["uri"] not in seen:
            seen.add(p["uri"]); uniq.append(p)
    (HERE / "data/feed_posts.json").write_text(json.dumps(uniq))
    agg = defaultdict(lambda: {"engagement": 0, "posts": 0, "uris": [], "texts": [], "links": []})
    for p in uniq:
        pid = paper_id(p)
        if not pid:
            continue
        g = agg[pid]
        g["engagement"] += sum(p.get(k, 0) for k in ("likeCount", "repostCount", "replyCount", "quoteCount"))
        g["posts"] += 1
        g["uris"].append(p["uri"])
        g["texts"].append(p["record"].get("text", "")[:300])
        g["links"] += [x for x in urls(p)[:-1] if x]
    ranked = sorted(({"id": k, **v} for k, v in agg.items()), key=lambda r: (-r["engagement"], -r["posts"]))
    (HERE / "data/papers_ranked.json").write_text(json.dumps(ranked, indent=1))
    print(f"{len(uniq)} posts, {len(ranked)} papers; kinds:",
          {k: sum(r['id'].startswith(k) for r in ranked) for k in ("arxiv", "doi:10.1101", "doi")})
    for r in ranked[:15]:
        print(r["engagement"], r["posts"], r["id"], r["texts"][0][:70].replace("\n", " "))


if __name__ == "__main__":
    main()
