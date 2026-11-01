#!/usr/bin/env python3
"""paper-watch: dedup ledger for @paper.bsky.social's daily "arXiv Upvote Trends Top 30".

  python3 paperwatch.py queue  SEEN.json QUEUE.json   # new + retry-eligible papers since last run
  python3 paperwatch.py record SEEN.json RESULTS.json # merge reviewed papers into the ledger

SEEN.json shape:
  {"last_run": ISO8601, "papers": {"<arxiv id>": {title, first_seen, best_rank, upvotes,
     reviewed, read: "full"|"abstract"|"none", practical, attempts, review}}}
A paper is queued when it is not in the ledger, or when its read is not "full",
practical is not NONE, and attempts < MAX_ATTEMPTS (one retry for a full-text read).
"""
import json, re, sys, urllib.request, urllib.parse
from datetime import datetime, timedelta, timezone

ACTOR = "paper.bsky.social"
API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
ITEM = re.compile(r"^\[(\d+)/(\d+)\]\s+(\d+)\s+Upvotes.*?arXiv:(\d{4}\.\d{4,5})", re.S)
MAX_ATTEMPTS = 2
MAX_PAGES = 12  # 100 posts/page, ~31 posts/day -> ~38 days max lookback


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def fetch_since(since):
    cursor, out = None, []
    for _ in range(MAX_PAGES):
        q = {"actor": ACTOR, "limit": 100, "filter": "posts_no_replies"}
        if cursor:
            q["cursor"] = cursor
        req = urllib.request.Request(API + "?" + urllib.parse.urlencode(q),
                                     headers={"User-Agent": "paper-watch"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
        feed = d.get("feed", [])
        for f in feed:
            p = f["post"]
            rec = p["record"]
            created = datetime.fromisoformat(rec["createdAt"].replace("Z", "+00:00"))
            if created < since:
                return out
            out.append((created, rec, p["uri"]))
        cursor = d.get("cursor")
        if not cursor or not feed:
            break
    return out


ABS = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})")


def parse(posts):
    """The daily summary post ("arXiv Upvote Trends Top 30") links every ranked paper
    in its facets; only some papers also get their own [n/30] post carrying title and
    upvotes. Summary facets are the source of the list, item posts enrich it."""
    papers = {}
    for created, rec, uri in posts:
        text = rec.get("text", "")
        post = "https://bsky.app/profile/%s/post/%s" % (ACTOR, uri.rsplit("/", 1)[-1])
        if text.startswith("arXiv Upvote Trends"):
            raw = text.encode()
            for f in rec.get("facets", []):
                for feat in f.get("features", []):
                    m = ABS.search(feat.get("uri", ""))
                    if not m:
                        continue
                    label = raw[f["index"]["byteStart"]:f["index"]["byteEnd"]].decode(errors="ignore")
                    rank = int(label.split("/")[0]) if label.split("/")[0].isdigit() else None
                    p = papers.setdefault(m[1], {"id": m[1], "title": "", "rank": rank,
                                                 "upvotes": 0, "posted": created.isoformat(), "post": post})
                    if rank and (p["rank"] is None or rank < p["rank"]):
                        p["rank"] = rank
            continue
        m = ITEM.match(text)
        if not m:
            continue
        rank, upvotes, aid = int(m[1]), int(m[3]), m[4]
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        title = lines[1].lstrip("\U0001F195").strip() if len(lines) > 1 else ""
        p = papers.setdefault(aid, {"id": aid, "title": "", "rank": rank, "upvotes": 0,
                                    "posted": created.isoformat(), "post": post})
        p["title"] = p["title"] or title
        p["upvotes"] = max(p["upvotes"], upvotes)
        p["rank"] = min(x for x in (p["rank"], rank) if x)
        p["post"] = post
    return papers


def cmd_queue(seen_path, queue_path):
    seen = load(seen_path, {"last_run": None, "papers": {}})
    last = seen.get("last_run")
    # one-day overlap so a post landing during the previous run is not missed;
    # the ledger makes the overlap harmless
    since = (datetime.fromisoformat(last) - timedelta(days=1)) if last else \
        datetime.now(timezone.utc) - timedelta(days=2)
    found = parse(fetch_since(since))
    queue = []
    for aid, p in found.items():
        prev = seen["papers"].get(aid)
        if prev is None:
            queue.append({**p, "retry": False})
        elif (prev.get("read") != "full" and prev.get("practical") not in ("NONE",)
              and prev.get("attempts", 1) < MAX_ATTEMPTS):
            queue.append({**p, "retry": True, "previous": prev})
    queue.sort(key=lambda p: (p["rank"] or 99, -p["upvotes"]))
    with open(queue_path, "w") as f:
        json.dump({"since": since.isoformat(), "scanned": len(found), "queue": queue}, f, indent=1)
    print("scanned %d papers since %s; queued %d (%d retries)"
          % (len(found), since.isoformat(), len(queue), sum(q["retry"] for q in queue)))


def cmd_record(seen_path, results_path):
    """RESULTS.json: list of {id, title, upvotes, rank, read, practical, review} for this run."""
    seen = load(seen_path, {"last_run": None, "papers": {}})
    results = load(results_path, [])
    today = datetime.now(timezone.utc).date().isoformat()
    for r in results:
        prev = seen["papers"].get(r["id"], {})
        seen["papers"][r["id"]] = {
            "title": r.get("title") or prev.get("title", ""),
            "first_seen": prev.get("first_seen", today),
            "best_rank": min(x for x in (r.get("rank"), prev.get("best_rank")) if x) if (r.get("rank") or prev.get("best_rank")) else None,
            "upvotes": max(r.get("upvotes", 0), prev.get("upvotes", 0)),
            "reviewed": today,
            "read": r["read"],
            "practical": r["practical"],
            "attempts": prev.get("attempts", 0) + 1,
            "review": r["review"],
        }
    seen["last_run"] = datetime.now(timezone.utc).isoformat()
    with open(seen_path, "w") as f:
        json.dump(seen, f, indent=1, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    print("recorded %d; ledger now %d papers" % (len(results), len(seen["papers"])))


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("queue", "record"):
        sys.exit(__doc__)
    {"queue": cmd_queue, "record": cmd_record}[sys.argv[1]](sys.argv[2], sys.argv[3])
