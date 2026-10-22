"""More caption text, round 2: alt text from authors found through the follows of the image
index's non-test authors (run/img_harvest.py covered the index authors themselves).

Usage: python3 -I run/img_harvest2.py [--target 150000] [--seed-authors 1500] [--per-author 100]

New authors = accounts followed by up to --seed-authors non-test index authors
(app.bsky.graph.getFollows, public API, first 100 follows each), minus every index author
(test or not), ranked by how many seed authors follow them. Their image posts are read with
img_harvest.author() (same label filter, 10-400 character alt text, no index posts). Texts
equal to any index alt text or already in extra_alt.json are dropped. Writes
out/img/extra_alt2.json; checkpoints per author under out/img/harvest/.
"""
import argparse
import collections
import concurrent.futures
import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from img_harvest import IMG, author, is_test  # noqa: E402
import img_harvest  # noqa: E402

FOLLOWS = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows?"


def follows(did):
    path = IMG / "follows" / (did.replace(":", "_") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    r = img_harvest.get([("actor", did), ("limit", "100")], api=FOLLOWS)
    out = [f["did"] for f in (r or {}).get("follows", [])]
    path.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150000)
    ap.add_argument("--seed-authors", type=int, default=1500)
    ap.add_argument("--per-author", type=int, default=100)
    a = ap.parse_args()
    (IMG / "follows").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (IMG / "rows.jsonl").open()]
    index_uris = {r["uri"] for r in rows}
    did_of = lambda uri: uri.split("/")[2]  # noqa: E731
    index_authors = {did_of(r["uri"]) for r in rows}
    test_authors = {did_of(r["uri"]) for r in rows if is_test(r["cid"])}
    seeds = sorted(index_authors - test_authors)[: a.seed_authors]
    cnt = collections.Counter()
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        for fl in ex.map(follows, seeds):
            cnt.update(d for d in fl if d not in index_authors)
    new = [d for d, _ in cnt.most_common()]
    print(len(seeds), "seed authors ->", len(new), "new authors", flush=True)
    old = set(json.loads((IMG / "extra_alt.json").read_text()))
    index_alts = {r["alt"].strip() for r in rows if r["alt"]}
    texts, done = set(), 0
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(author, d, a.per_author, index_uris): d for d in new}
        for f in concurrent.futures.as_completed(futs):
            texts.update(t for t in f.result() if t not in old and t not in index_alts)
            done += 1
            if done % 500 == 0:
                print(done, "authors,", len(texts), "new alt texts", flush=True)
            if len(texts) >= a.target:
                for g in futs:
                    g.cancel()
                break
    (IMG / "extra_alt2.json").write_text(json.dumps(sorted(texts), ensure_ascii=False))
    print(len(texts), "alt texts written to extra_alt2.json")


if __name__ == "__main__":
    main()
