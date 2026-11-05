"""Collect Bluesky accounts and their latest posts for the account decider.

    python3 -I run/collect.py [--target 1600]

Sources, each tagged in the output:
  list      members of Oskar's "Artificial (and real) Intelligence" list
  follows   accounts the list members follow (a random sample of members, then of their follows)
  discover  authors in the public "Discover" (whats-hot) feed
  firehose  authors of English posts seen on Jetstream for a few minutes (brings in bots and
            ordinary non-AI accounts, weighted toward active posters)
For each account: the profile (handle, display name, bio, counts) and the latest 20 feed items
(posts, replies, reposts) from the public AppView. Writes data/accounts.jsonl, one line per account,
resume-safe (accounts already in the file are skipped). Post texts are other people's public posts:
data/ is gitignored and results/collect.json keeps counts only.
"""
import argparse
import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "accounts.jsonl"
PUBLIC = "https://public.api.bsky.app/xrpc"
LIST = "at://did:plc:r2whjvupgfw55mllpksnombn/app.bsky.graph.list/3lankcdrlip2f"
WHATS_HOT = "at://did:plc:z72i7hdynmk6r22z27h6tvur/app.bsky.feed.generator/whats-hot"
N_POSTS = 20
S = requests.Session()


def get(ep, params, tries=6):
    for i in range(tries):
        try:
            r = S.get(f"{PUBLIC}/{ep}", params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (400, 404):
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    return None


def list_members():
    dids, cur = [], None
    while True:
        d = get("app.bsky.graph.getList", {"list": LIST, "limit": 100, **({"cursor": cur} if cur else {})})
        if not d:
            break
        dids += [it["subject"]["did"] for it in d.get("items", [])]
        cur = d.get("cursor")
        if not cur:
            break
    return dids


def follows_of(did, limit=100):
    d = get("app.bsky.graph.getFollows", {"actor": did, "limit": limit})
    return [f["did"] for f in (d or {}).get("follows", [])]


def discover_authors(pages=8):
    dids, cur = [], None
    for _ in range(pages):
        d = get("app.bsky.feed.getFeed", {"feed": WHATS_HOT, "limit": 100, **({"cursor": cur} if cur else {})})
        if not d:
            break
        dids += [it["post"]["author"]["did"] for it in d.get("feed", [])]
        cur = d.get("cursor")
        if not cur:
            break
    return dids


async def firehose_authors(seconds=180):
    import websockets
    url = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
    seen = {}
    t0 = time.time()
    async with websockets.connect(url, open_timeout=15, max_size=2 ** 22) as ws:
        while time.time() - t0 < seconds:
            m = json.loads(await ws.recv())
            c = m.get("commit") or {}
            rec = c.get("record") or {}
            if c.get("operation") == "create" and "en" in (rec.get("langs") or []):
                seen[m["did"]] = seen.get(m["did"], 0) + 1
    return seen


def feed_items(did):
    d = get("app.bsky.feed.getAuthorFeed", {"actor": did, "limit": 40, "filter": "posts_with_replies"})
    items = []
    for it in (d or {}).get("feed", []):
        p = it["post"]
        rec = p.get("record", {})
        reason = (it.get("reason") or {}).get("$type", "")
        kind = "repost" if "reasonRepost" in reason else ("reply" if rec.get("reply") else "post")
        if kind == "repost" and p["author"]["did"] == did:
            continue
        emb = (p.get("embed") or {}).get("$type", "")
        items.append({"kind": kind, "text": rec.get("text", ""), "createdAt": rec.get("createdAt", ""),
                      "indexedAt": (it.get("reason") or {}).get("indexedAt") or p.get("indexedAt", ""),
                      "author": p["author"].get("handle") if kind == "repost" else None,
                      "embed": emb.split(".")[-1].split("#")[0] if emb else None,
                      "uri": p.get("uri")})
        if len(items) >= N_POSTS:
            break
    return items


def profiles(dids):
    out = {}
    for i in range(0, len(dids), 25):
        d = get("app.bsky.actor.getProfiles", {"actors": dids[i:i + 25]})
        for p in (d or {}).get("profiles", []):
            out[p["did"]] = {k: p.get(k) for k in ("did", "handle", "displayName", "description",
                                                   "followersCount", "followsCount", "postsCount", "createdAt")}
            out[p["did"]]["labels"] = [l.get("val") for l in p.get("labels", [])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["did"] for l in OUT.open()}
    members = list_members()
    print("list members", len(members), flush=True)
    sample_members = rng.sample(members, min(60, len(members)))
    with ThreadPoolExecutor(8) as ex:
        fl = [d for ds in ex.map(follows_of, sample_members) for d in ds]
    mset = set(members)
    follows = list(dict.fromkeys(d for d in fl if d not in mset))
    rng.shuffle(follows)
    disc = list(dict.fromkeys(discover_authors()))
    fire = asyncio.run(firehose_authors())
    fire_dids = list(fire)
    rng.shuffle(fire_dids)
    print(f"follows {len(follows)} discover {len(disc)} firehose {len(fire_dids)}", flush=True)
    plan = ([("list", d) for d in members] + [("follows", d) for d in follows[:450]]
            + [("discover", d) for d in disc[:400]] + [("firehose", d) for d in fire_dids[:550]])
    seen, todo = set(done), []
    for src, d in plan:
        if d not in seen:
            seen.add(d)
            todo.append((src, d))
    prof = profiles([d for _, d in todo])
    kept = 0
    with OUT.open("a") as f, ThreadPoolExecutor(8) as ex:
        for (src, d), items in zip(todo, ex.map(lambda x: feed_items(x[1]), todo)):
            if d not in prof or len(items) < 5:
                continue
            f.write(json.dumps({"did": d, "source": src, "profile": prof[d], "items": items}) + "\n")
            kept += 1
    n = sum(1 for _ in OUT.open())
    src_counts = {}
    for l in OUT.open():
        s = json.loads(l)["source"]
        src_counts[s] = src_counts.get(s, 0) + 1
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "collect.json").write_text(json.dumps(
        {"accounts": n, "by_source": src_counts, "list_members": len(members), "follows_pool": len(follows),
         "discover_pool": len(disc), "firehose_pool": len(fire_dids), "min_items": 5, "max_items": N_POSTS,
         "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print("accounts", n, src_counts, flush=True)


if __name__ == "__main__":
    main()
