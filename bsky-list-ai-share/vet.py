"""Second pass over discover.py's candidates: keep individuals who talk about AI, drop bots, feeds and marketers.

1. Screen: AI share >= 0.40, <= 40 posts a day, no bot / feed / news words in handle or bio.
2. Re-fetch each survivor's last 100 posts: reply fraction (bots and feeds rarely reply) and six sample posts.
3. A headless Claude Haiku judges, from bio, stats and samples, whether the account is an individual person
   writing substantive commentary about AI (vet_judge.sh); keyed JSON replies.
Writes data/vetted.json (names: not committed).
"""
import json, random, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
PUBLIC = "https://public.api.bsky.app/xrpc"
BOTWORDS = re.compile(r"\b(bot|bots|feed|feeds|news|newsletter|digest|daily|hourly|updates|automated|arxiv|headlines|briefing|"
                      r"aggregat\w*|curated|rss|jobs?|deals|crypto|signals)\b|-bot\b|bot\.", re.I)


def get(url, params, tries=5):
    for k in range(tries):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (400, 401, 404):
            return None
        time.sleep(2 ** k)
    return None


def main():
    rows = json.loads((HERE / "data" / "candidates.json").read_text())
    screened = [r for r in rows if r["ai_share"] >= 0.40 and r["posts_per_day"] <= 40
                and not BOTWORDS.search(r["handle"] + " " + r["name"] + " " + r["bio"])]
    print(f"{len(rows)} candidates, {len(screened)} pass the screen", flush=True)

    def feed(r):
        d = get(f"{PUBLIC}/app.bsky.feed.getAuthorFeed", {"actor": r["did"], "limit": 100, "filter": "posts_with_replies"}) or {}
        own = [it["post"] for it in d.get("feed", []) if "reason" not in it and it["post"]["author"]["did"] == r["did"]]
        texts = [(p.get("record") or {}).get("text") or "" for p in own]
        replies = sum(1 for p in own if (p.get("record") or {}).get("reply"))
        longish = [t for t in texts if len(t) > 80] or texts
        rnd = random.Random(r["did"])
        r["reply_frac"] = round(replies / max(1, len(own)), 2)
        r["samples"] = [" ".join(t.split())[:220] for t in rnd.sample(longish, min(6, len(longish)))]
        return r
    with ThreadPoolExecutor(6) as ex:
        screened = list(ex.map(feed, screened))

    # judge in batches of 25 with a headless Haiku child, keyed by index
    for b in range(0, len(screened), 25):
        batch = screened[b:b + 25]
        blob = "\n\n".join(
            f"[{j}] @{r['handle']} | {r['name']} | bio: {r['bio']} | {r['posts_per_day']} posts/day, "
            f"{int(r['reply_frac'] * 100)}% replies, followers {r['followers']}\n" + "\n".join(f"  - {s}" for s in r["samples"])
            for j, r in enumerate(batch))
        (HERE / "data" / f"vet{b // 25}.txt").write_text(blob)
        out = subprocess.run(["bash", str(HERE / "vet_judge.sh"), str(HERE / "data" / f"vet{b // 25}.txt"), str(len(batch))],
                             capture_output=True, text=True, timeout=600)
        verdict = json.loads(out.stdout)
        for j, r in enumerate(batch):
            v = verdict[str(j)]
            r["person"], r["why"] = bool(v["keep"]), v["why"]
        print(f"  judged {b + len(batch)}/{len(screened)}", flush=True)
    (HERE / "data" / "vetted.json").write_text(json.dumps(screened, indent=1, ensure_ascii=False))
    keep = [r for r in screened if r["person"]]
    print(f"{len(keep)} kept as individuals writing about AI", flush=True)


if __name__ == "__main__":
    main()
