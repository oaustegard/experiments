"""More caption-domain training text: alt text from other image posts by the index's authors.

Usage: python3 -I run/img_harvest.py [--target 60000] [--per-author 100]

Authors = the bsky image-search index's authors, minus every author with a test-split image
(run/img_common.is_test), so no held-out author's writing is trained on. For each, pages
app.bsky.feed.getAuthorFeed (filter posts_with_media, public API) and keeps image alt texts of
10-400 characters from posts that are not in the index and carry no adult/graphic label (the
bsky-image-search harvest.py block list). Deduplicated by text. Writes out/img/extra_alt.json
(list of strings); checkpoints per author in out/img/harvest/ so a rerun resumes.
"""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "out" / "img"
API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?"
BLOCK = {"porn", "sexual", "nudity", "graphic-media", "gore", "sexual-figurative", "!hide", "!warn", "nsfl"}


def is_test(cid):
    return int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 5 == 0


def get(q, api=None):
    req = urllib.request.Request((api or API) + urllib.parse.urlencode(q), headers={"User-Agent": "muninn-experiments"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if attempt == 4:
                return None
            time.sleep(2 ** attempt)


def labels(post):
    vals = {l.get("val") for l in post.get("labels", [])}
    vals |= {l.get("val") for l in post.get("author", {}).get("labels", [])}
    vals |= {v.get("val") for v in (post.get("record", {}).get("labels") or {}).get("values", [])}
    return vals


def author(did, per, index_uris):
    path = IMG / "harvest" / (did.replace(":", "_") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    out, cursor, seen = [], None, 0
    while seen < per:
        r = get([("actor", did), ("filter", "posts_with_media"), ("limit", "100")] + ([("cursor", cursor)] if cursor else []))
        if not r or not r.get("feed"):
            break
        for item in r["feed"]:
            p = item["post"]
            seen += 1
            if p["uri"] in index_uris or p["author"]["did"] != did or labels(p) & BLOCK:
                continue
            e = p.get("embed") or {}
            for im in e.get("images") or (e.get("media") or {}).get("images") or []:
                a = (im.get("alt") or "").strip()
                if 10 <= len(a) <= 400:
                    out.append(a)
        cursor = r.get("cursor")
        if not cursor:
            break
    path.write_text(json.dumps(out, ensure_ascii=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=60000)
    ap.add_argument("--per-author", type=int, default=100)
    a = ap.parse_args()
    (IMG / "harvest").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (IMG / "rows.jsonl").open()]
    index_uris = {r["uri"] for r in rows}
    did_of = lambda uri: uri.split("/")[2]  # noqa: E731
    test_authors = {did_of(r["uri"]) for r in rows if is_test(r["cid"])}
    authors = sorted({did_of(r["uri"]) for r in rows} - test_authors)
    print(len(authors), "authors (", len(test_authors), "test authors excluded )", flush=True)
    texts, done = set(), 0
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(author, d, a.per_author, index_uris): d for d in authors}
        for f in concurrent.futures.as_completed(futs):
            texts.update(f.result())
            done += 1
            if done % 250 == 0:
                print(done, "authors,", len(texts), "alt texts", flush=True)
            if len(texts) >= a.target:
                for g in futs:
                    g.cancel()
                break
    test_alts = {r["alt"].strip() for r in rows if r["alt"] and is_test(r["cid"])}
    out = sorted(texts - test_alts)
    (IMG / "extra_alt.json").write_text(json.dumps(out, ensure_ascii=False))
    print(len(out), "extra alt texts written (", len(texts & test_alts), "that equal a test alt dropped )")


if __name__ == "__main__":
    main()
