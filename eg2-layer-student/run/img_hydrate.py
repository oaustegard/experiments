"""Rebuild alt text (and post text) for the bsky image-search index, whose committed rows hold
only [author, rkey, image cid, likes] (oaustegard.github.io bsky/image-search/data/rows.json).

Usage: python3 -I run/img_hydrate.py

Fetches the site's data files into out/img/site/, then app.bsky.feed.getPosts 25 URIs per call
(six in flight), and for each index row finds the image with the row's cid in the post's embed
(images, or recordWithMedia images). Writes out/img/rows.jsonl, one {"i", "uri", "cid", "alt",
"text"} per index row in index order; a deleted post or missing image gets alt/text null.
Checkpoints per batch; a rerun resumes.
"""
import concurrent.futures
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "img"
SITE = "https://austegard.com/bsky/image-search/data/"
API = "https://api.bsky.app/xrpc/app.bsky.feed.getPosts?"


def fetch(url, tries=6):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "muninn-experiments"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 ** t)


def images(post):
    emb = post["record"].get("embed") or {}
    return emb.get("images") or (emb.get("media") or {}).get("images") or []


def main():
    (OUT / "site").mkdir(parents=True, exist_ok=True)
    for f in ["index.json", "rows.json", "codes.u8.bin", "aux.f32.bin"]:
        if not (OUT / "site" / f).exists():
            (OUT / "site" / f).write_bytes(fetch(SITE + f))
    rl = json.loads((OUT / "site" / "rows.json").read_text())
    rows = [{"i": i, "uri": f"at://{rl['authors'][a][0]}/app.bsky.feed.post/{rkey}", "cid": cid}
            for i, (a, rkey, cid, likes) in enumerate(rl["rows"])]
    path = OUT / "rows.jsonl"
    done = sum(1 for _ in path.open()) if path.exists() else 0
    batches = [rows[s:s + 25] for s in range(done, len(rows), 25)]

    def get(batch):
        uris = sorted({r["uri"] for r in batch})
        posts = json.loads(fetch(API + urllib.parse.urlencode([("uris", u) for u in uris])))["posts"]
        return batch, {p["uri"]: p for p in posts}

    with path.open("a") as f, concurrent.futures.ThreadPoolExecutor(6) as ex:
        for k, (batch, posts) in enumerate(ex.map(get, batches)):
            for r in batch:
                p = posts.get(r["uri"])
                alt = text = None
                if p:
                    text = p["record"].get("text", "")
                    for im in images(p):
                        if (im.get("image") or {}).get("ref", {}).get("$link") == r["cid"]:
                            alt = im.get("alt", "")
                f.write(json.dumps({**r, "alt": alt, "text": text}, ensure_ascii=False) + "\n")
            if k % 40 == 0:
                print(f"{done + (k + 1) * 25}/{len(rows)}", flush=True)
    got = [json.loads(l) for l in path.open()]
    print(json.dumps({"rows": len(got), "post_missing": sum(r["text"] is None for r in got),
                      "with_alt": sum(bool((r["alt"] or "").strip()) for r in got),
                      "alt_25_300": sum(25 <= len((r["alt"] or "").strip()) <= 300 for r in got)}))


if __name__ == "__main__":
    main()
