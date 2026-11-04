"""Find Bluesky accounts that post a lot about AI and are not on the "Artificial (and real) Intelligence" list.

1. Authenticated post search (the public AppView refuses searchPosts) for AI_TERMS over the last DAYS days;
   authors not on the list with at least MIN_HITS hits become candidates.
2. Each candidate's last 100 posts and replies (no reposts) are scored with the same about-AI classifier that
   score.py fits (logistic regression on Ettin-17m vectors, 600 labelled posts), giving an AI share and AI
   posts per day, comparable to the members' numbers.
3. Profiles add follower counts and descriptions, to spot bots and news feeds.

Writes data/candidates.json (names: kept out of git) and results/discover.json (aggregate only).
Credentials come from the environment through the browsing-bluesky skill and are never printed.
"""
import json, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/mnt/skills/user/browsing-bluesky/scripts")
import bsky  # noqa: E402
sys.path.insert(0, "/home/user/oaustegard.github.io/scripts/answer-search")
from refresh import Ettin  # noqa: E402

LIST = "at://did:plc:r2whjvupgfw55mllpksnombn/app.bsky.graph.list/3lankcdrlip2f"
PUBLIC = "https://public.api.bsky.app/xrpc"
DAYS, MIN_HITS, PAGES = 14, 3, 3
AI_TERMS = ["LLM", "LLMs", "large language model", "Claude", "ChatGPT", "OpenAI", "Anthropic", "Gemini", "AI agent",
            "agentic", "machine learning", "neural network", "RLHF", "AI safety", "alignment", "interpretability",
            "coding agent", "Claude Code", "Codex", "vibe coding", "open weights", "Llama", "DeepSeek", "Qwen",
            "Mistral", "diffusion model", "transformer", "embeddings", "RAG", "fine-tuning", "AI policy", "AGI",
            "frontier model", "inference", "benchmark", "arxiv", "MCP", "prompt injection", "evals", "GPU"]


def get(url, params, headers=None, tries=5):
    for k in range(tries):
        r = requests.get(url, params=params, headers=headers or {}, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (400, 401, 404):
            return None
        time.sleep(2 ** k)
    return None


def main():
    members = set()
    cur = None
    while True:
        d = get(f"{PUBLIC}/app.bsky.graph.getList", {"list": LIST, "limit": 100, **({"cursor": cur} if cur else {})})
        members |= {it["subject"]["did"] for it in d["items"]}
        cur = d.get("cursor")
        if not cur:
            break
    assert bsky.is_authenticated(), "browsing-bluesky could not authenticate"
    base, headers = bsky._get_base_and_headers()
    since = (datetime.now(timezone.utc) - timedelta(days=DAYS)).strftime("%Y-%m-%d")

    hits, handles, seen = Counter(), {}, set()
    for term in AI_TERMS:
        cur = None
        for _ in range(PAGES):
            q = {"q": f'"{term}" lang:en since:{since}', "limit": 100, "sort": "latest"}
            if cur:
                q["cursor"] = cur
            d = get(f"{base}/app.bsky.feed.searchPosts", q, headers)
            if not d:
                break
            for p in d.get("posts", []):
                if p["uri"] in seen:
                    continue
                seen.add(p["uri"])
                did = p["author"]["did"]
                if did not in members:
                    hits[did] += 1
                    handles[did] = p["author"]["handle"]
            cur = d.get("cursor")
            if not cur:
                break
        print(f"  {term}: {len(seen)} posts so far, {len(hits)} non-member authors", flush=True)
    cands = [did for did, n in hits.items() if n >= MIN_HITS]
    print(f"{len(seen)} search hits, {len(hits)} non-member authors, {len(cands)} with >= {MIN_HITS} hits", flush=True)

    def feed(did):
        d = get(f"{PUBLIC}/app.bsky.feed.getAuthorFeed", {"actor": did, "limit": 100, "filter": "posts_with_replies"})
        if not d:
            return did, []
        out = []
        for it in d.get("feed", []):
            if "reason" in it or it["post"]["author"]["did"] != did:
                continue
            t = (it["post"].get("record") or {}).get("text") or ""
            if sum(1 for w in t.split() if any(ch.isalpha() for ch in w)) >= 3:
                out.append((datetime.fromisoformat(it["post"]["indexedAt"].replace("Z", "+00:00")).timestamp(), t))
        return did, out
    with ThreadPoolExecutor(6) as ex:
        feeds = dict(ex.map(feed, cands))

    profiles = {}
    for i in range(0, len(cands), 25):
        d = get(f"{PUBLIC}/app.bsky.actor.getProfiles", [("actors", x) for x in cands[i:i + 25]])
        for pr in (d or {}).get("profiles", []):
            profiles[pr["did"]] = pr

    # the classifier, refitted exactly as score.py fits it
    import gzip
    src = HERE.parent / "answer-search-1bit" / "data"
    X = np.load(src / "X.npy")
    ids = json.loads((HERE / "data" / "sample.json").read_text())["ids"]
    y = np.array(sum((json.loads((HERE / "data" / f"labels{b}.json").read_text()) for b in range(6)), []))
    clf = LogisticRegression(C=json.loads((HERE / "results" / "model.json").read_text())["C"], max_iter=2000).fit(X[ids], y)
    enc = Ettin(Path("/home/user/oaustegard.github.io/bsky/embedders/models/ettin"))
    flat = [(did, t) for did, ps in feeds.items() for _, t in ps]
    P = clf.predict_proba(enc.embed([t for _, t in flat]))[:, 1] if flat else np.zeros(0)
    by = {}
    for (did, _), p in zip(flat, P):
        by.setdefault(did, []).append(float(p))

    rows = []
    for did in cands:
        ps = feeds.get(did) or []
        if len(ps) < 10:
            continue
        span = max(1.0, (max(t for t, _ in ps) - min(t for t, _ in ps)) / 86400)
        share = float(np.mean(by[did]))
        pr = profiles.get(did, {})
        rows.append({"handle": pr.get("handle", handles[did]), "did": did, "search_hits": hits[did], "sampled": len(ps),
                     "days": round(span, 1), "posts_per_day": round(len(ps) / span, 1), "ai_share": round(share, 3),
                     "ai_per_day": round(share * len(ps) / span, 2), "followers": pr.get("followersCount"),
                     "name": pr.get("displayName", ""), "bio": (pr.get("description") or "")[:200].replace("\n", " ")})
    rows.sort(key=lambda r: -r["ai_per_day"])
    (HERE / "data" / "candidates.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (HERE / "results" / "discover.json").write_text(json.dumps({
        "run": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "terms": len(AI_TERMS), "days": DAYS,
        "search_hits": len(seen), "non_member_authors": len(hits), "candidates": len(cands), "scored": len(rows),
        "members": len(members)}, indent=1))
    print(f"scored {len(rows)} candidates", flush=True)


if __name__ == "__main__":
    main()
