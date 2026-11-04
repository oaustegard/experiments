"""Follow-graph pass: people the list's members follow, who post about AI but are too quiet for search to find.

1. For every current member of "Artificial (and real) Intelligence", their follows (app.bsky.graph.getFollows,
   public, up to MAX_FOLLOWS each).
2. Non-members followed by at least MIN_MEMBERS members are candidates, minus the candidates list and the
   accounts removed on 2026-11-04 (removed.json), most-followed first, at most MAX_CANDS.
3. Each candidate's last 100 posts and replies (no reposts) are scored with the about-AI classifier
   (score.py's model), with reply share, posts per day and days since the last post.
Writes data/follow_candidates.json (names: not committed) and results/follows.json (aggregate).
vet.py then screens and judges them (python3 vet.py data/follow_candidates.json data/follow_vetted.json).
"""
import json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/user/oaustegard.github.io/scripts/answer-search")
from refresh import Ettin  # noqa: E402

PUBLIC = "https://public.api.bsky.app/xrpc"
LIST = "at://did:plc:r2whjvupgfw55mllpksnombn/app.bsky.graph.list/3lankcdrlip2f"
CANDS_LIST = "at://did:plc:r2whjvupgfw55mllpksnombn/app.bsky.graph.list/3mzilji6fkn23"
MAX_FOLLOWS, MIN_MEMBERS, MAX_CANDS = 3000, 12, 500


def get(ep, params, tries=6):
    for k in range(tries):
        try:
            r = requests.get(f"{PUBLIC}/{ep}", params=params, timeout=30)
        except requests.RequestException:
            time.sleep(2 ** k); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (400, 401, 404):
            return None
        time.sleep(2 ** k)
    return None


def list_dids(uri):
    out, cur = [], None
    while True:
        d = get("app.bsky.graph.getList", {"list": uri, "limit": 100, **({"cursor": cur} if cur else {})})
        out += [it["subject"]["did"] for it in d["items"]]
        cur = d.get("cursor")
        if not cur:
            return out


def follows(did):
    out, cur = [], None
    while len(out) < MAX_FOLLOWS:
        d = get("app.bsky.graph.getFollows", {"actor": did, "limit": 100, **({"cursor": cur} if cur else {})})
        if not d:
            break
        out += [f["did"] for f in d.get("follows", [])]
        cur = d.get("cursor")
        if not cur:
            break
    return out


def main():
    members = list_dids(LIST)
    skip = set(members) | set(list_dids(CANDS_LIST)) | set(json.loads((HERE / "data" / "removed.json").read_text()))
    print(f"{len(members)} members; {len(skip)} accounts skipped", flush=True)
    t0 = time.time()
    counts = Counter()
    with ThreadPoolExecutor(6) as ex:
        for k, fl in enumerate(ex.map(follows, members), 1):
            counts.update(set(fl) - skip)
            if k % 50 == 0:
                print(f"  follows: {k}/{len(members)} members, {len(counts)} accounts seen ({time.time() - t0:.0f} s)", flush=True)
    cands = [(did, n) for did, n in counts.most_common() if n >= MIN_MEMBERS][:MAX_CANDS]
    print(f"{len(counts)} non-members followed; {sum(1 for n in counts.values() if n >= MIN_MEMBERS)} by >= {MIN_MEMBERS} members; "
          f"scoring {len(cands)}", flush=True)

    def feed(did):
        d = get("app.bsky.feed.getAuthorFeed", {"actor": did, "limit": 100, "filter": "posts_with_replies"}) or {}
        own = [it["post"] for it in d.get("feed", []) if "reason" not in it and it["post"]["author"]["did"] == did]
        posts = [(datetime.fromisoformat(p["indexedAt"].replace("Z", "+00:00")).timestamp(), (p.get("record") or {}).get("text") or "",
                  bool((p.get("record") or {}).get("reply"))) for p in own]
        return did, [p for p in posts if sum(1 for w in p[1].split() if any(c.isalpha() for c in w)) >= 3], (
            sum(p[2] for p in posts) / len(posts) if posts else 0.0)
    with ThreadPoolExecutor(6) as ex:
        feeds = {did: (ps, rf) for did, ps, rf in ex.map(feed, [d for d, _ in cands])}
    profiles = {}
    dids = [d for d, _ in cands]
    for i in range(0, len(dids), 25):
        d = get("app.bsky.actor.getProfiles", [("actors", x) for x in dids[i:i + 25]])
        for pr in (d or {}).get("profiles", []):
            profiles[pr["did"]] = pr

    src = HERE.parent / "answer-search-1bit" / "data"
    X = np.load(src / "X.npy")
    ids = json.loads((HERE / "data" / "sample.json").read_text())["ids"]
    y = np.array(sum((json.loads((HERE / "data" / f"labels{b}.json").read_text()) for b in range(6)), []))
    clf = LogisticRegression(C=json.loads((HERE / "results" / "model.json").read_text())["C"], max_iter=2000).fit(X[ids], y)
    enc = Ettin(Path("/home/user/oaustegard.github.io/bsky/embedders/models/ettin"))
    flat = [(did, t) for did, (ps, _) in feeds.items() for _, t, _ in ps]
    P = clf.predict_proba(enc.embed([t for _, t in flat]))[:, 1]
    by = {}
    for (did, _), p in zip(flat, P):
        by.setdefault(did, []).append(float(p))
    now = time.time()
    rows = []
    for did, n in cands:
        ps, rf = feeds[did]
        if len(ps) < 10:
            continue
        span = max(1.0, (max(t for t, _, _ in ps) - min(t for t, _, _ in ps)) / 86400)
        share = float(np.mean(by[did]))
        pr = profiles.get(did, {})
        rows.append({"handle": pr.get("handle", did), "did": did, "member_followers": n, "search_hits": 0,
                     "sampled": len(ps), "days": round(span, 1), "posts_per_day": round(len(ps) / span, 2),
                     "last_post_days": round((now - max(t for t, _, _ in ps)) / 86400, 1), "reply_frac_feed": round(rf, 2),
                     "ai_share": round(share, 3), "ai_per_day": round(share * len(ps) / span, 2),
                     "followers": pr.get("followersCount"), "name": pr.get("displayName", ""),
                     "bio": (pr.get("description") or "")[:200].replace("\n", " ")})
    rows.sort(key=lambda r: (-r["member_followers"]))
    (HERE / "data" / "follow_candidates.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (HERE / "results" / "follows.json").write_text(json.dumps({
        "run": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "members": len(members),
        "non_members_followed": len(counts), "min_members": MIN_MEMBERS, "candidates": len(cands), "scored": len(rows),
        "max_follows_per_member": MAX_FOLLOWS, "crawl_s": round(time.time() - t0)}, indent=1))
    print(f"scored {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
