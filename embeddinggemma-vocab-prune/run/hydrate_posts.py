"""Re-fetch the post text behind the Bluesky index (copied from eg2-query-student, writes .work/posts.jsonl).

Usage: python3 -I run/hydrate_posts.py

The published index (../embeddinggemma2-webgpu-tail/web/bsky) holds only [author, rkey]
per post; the text stayed in that experiment's gitignored out/. This rebuilds it with
app.bsky.feed.getPosts, 25 URIs per call, and the same text rule as harvest_bsky.py
(post text plus image alt text). Writes out/posts.jsonl, one {"i", "uri", "text"} row per
post in index order; a deleted post gets text null. Checkpoints per batch (six in flight); a rerun resumes.
"""
import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "embeddinggemma2-webgpu-tail" / "web" / "bsky"
OUT = ROOT / ".work" / "posts.jsonl"
API = "https://api.bsky.app/xrpc/app.bsky.feed.getPosts?"


def post_text(post):
    rec = post["record"]
    parts = [rec.get("text", "")]
    emb = rec.get("embed") or {}
    imgs = emb.get("images") or (emb.get("media") or {}).get("images") or []
    parts += [i.get("alt", "") for i in imgs if i.get("alt")]
    return "\n".join(p for p in parts if p).strip()


def get(uris):
    url = API + urllib.parse.urlencode([("uris", u) for u in uris])
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "muninn-experiments"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)["posts"]
        except (urllib.error.URLError, TimeoutError) as e:
            time.sleep(2 ** attempt)
            err = e
    raise RuntimeError(f"getPosts failed: {err}")


def main():
    meta = json.loads((SRC / "index.json").read_text())
    authors = meta["authors"]
    posts = json.loads((SRC / "posts.json").read_text())
    uris = [f"at://{authors[a]['did']}/app.bsky.feed.post/{rk}" for a, rk in posts]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in OUT.open()) if OUT.exists() else 0
    starts = range(done, len(uris), 25)
    # Six requests in flight; map() yields in submission order, so rows stay in index order.
    with OUT.open("a") as f, concurrent.futures.ThreadPoolExecutor(6) as ex:
        for s, posts_ in zip(starts, ex.map(lambda s: get(uris[s:s + 25]), starts)):
            batch = uris[s:s + 25]
            got = {p["uri"]: post_text(p) for p in posts_}
            for j, u in enumerate(batch):
                f.write(json.dumps({"i": s + j, "uri": u, "text": got.get(u)}) + "\n")
            f.flush()
            if (s // 25) % 200 == 0:
                print(f"{s + len(batch)}/{len(uris)}", flush=True)
    rows = [json.loads(l) for l in OUT.open()]
    print(json.dumps({"n": len(rows), "missing": sum(r["text"] is None for r in rows)}))


if __name__ == "__main__":
    sys.exit(main())
