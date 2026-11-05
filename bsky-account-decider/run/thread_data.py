"""Thread-aware training data for a LoRA on the per-post "about AI" question.

    python3 -I run/thread_data.py collect   # accounts -> data/thread_lora/items.jsonl (resume-safe per account)
    python3 -I run/thread_data.py label     # Sonnet, 30 items a call -> data/thread_lora/labels/b*.json (resume-safe)

Accounts: Oskar's AI list, a sample of their follows, Discover authors; accounts in the
reply_thread eval (data/reply_thread) are left out so that set stays a held-out test. Per account,
up to 4 replies (each with its whole thread), 2 posts and 1 repost, so the adapter sees the item
kinds the page scores and plain posts do not drift. Labels follow reply_context_eval's rule: a reply
in a conversation about AI is about AI.
"""
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
from collect import discover_authors, follows_of, get, list_members  # noqa: E402
from reply_context_eval import SYSTEM as REPLY_SYSTEM, thread_of  # noqa: E402
from state import _clean  # noqa: E402

D = ROOT / "data" / "thread_lora"
BATCH = 30
KEEP = {"reply": 4, "post": 2, "repost": 1}

SYSTEM = REPLY_SYSTEM.replace(
    "You annotate Bluesky replies. Each numbered item shows the conversation the reply belongs to, oldest post first (THREAD), then the reply (REPLY); texts cut to 200 characters.",
    "You annotate Bluesky items. A reply is shown with the conversation it belongs to, oldest post first (THREAD), then the reply (REPLY); a post is shown as POST, a repost as REPOST. Texts are cut to 200 characters (300 for the item itself).\n\nFor a POST or REPOST give the probability that it is about AI by the definition below. For a reply, ").replace(
    "\n\nFor every item give the probability that the REPLY is ABOUT AI", "give the probability that the REPLY is ABOUT AI")


def items_of(did):
    d = get("app.bsky.feed.getAuthorFeed", {"actor": did, "limit": 50, "filter": "posts_with_replies"})
    out, n = [], {k: 0 for k in KEEP}
    for it in (d or {}).get("feed", []):
        p, rec = it["post"], it["post"].get("record", {})
        reason = (it.get("reason") or {}).get("$type", "")
        kind = "repost" if "reasonRepost" in reason else ("reply" if rec.get("reply") else "post")
        if (kind == "repost" and p["author"]["did"] == did) or n[kind] >= KEEP[kind]:
            continue
        th = None
        if kind == "reply":
            th = thread_of(p["uri"])
            if not th:
                continue
        emb = (p.get("embed") or {}).get("$type", "")
        out.append({"kind": kind, "text": rec.get("text", ""), "uri": p["uri"], "account": did,
                    "author": p["author"].get("handle") if kind == "repost" else None,
                    "embed": emb.split(".")[-1].split("#")[0] if emb else None, "thread": th})
        n[kind] += 1
        if n == KEEP:
            break
    return out


def collect():
    D.mkdir(parents=True, exist_ok=True)
    f_items = D / "items.jsonl"
    done = {json.loads(l)["account"] for l in f_items.open()} if f_items.exists() else set()
    held = {json.loads(l)["account"] for l in (ROOT / "data" / "reply_thread" / "items.jsonl").open()}
    rng = random.Random(21)
    ai = list_members()
    fol = sorted({f for m in rng.sample(ai, 40) for f in follows_of(m)})
    disc = sorted(set(discover_authors(pages=8)))
    pool = list(dict.fromkeys(ai + rng.sample(fol, min(450, len(fol))) + rng.sample(disc, min(450, len(disc)))))
    src = {a: "list" for a in ai} | {a: "follows" for a in fol if a not in ai} | {a: "discover" for a in disc if a not in ai and a not in fol}
    todo = [a for a in pool if a not in held and a not in done]
    print({"pool": len(pool), "held_out": len(set(pool) & held), "done": len(done), "todo": len(todo)}, flush=True)
    t0, n = time.time(), 0
    with ThreadPoolExecutor(8) as ex, f_items.open("a") as f:
        for k, items in enumerate(ex.map(items_of, todo), 1):
            for it in items:
                f.write(json.dumps({**it, "source": src.get(it["account"], "?")}) + "\n")
                n += 1
            f.flush()
            if k % 100 == 0:
                print(f"{k}/{len(todo)} accounts, {n} items, {time.time() - t0:.0f}s", flush=True)
    print({"accounts": len(todo), "items": n, "s": round(time.time() - t0)}, flush=True)


def show(it):
    body = _clean(it["text"], 300) or "(no text)"
    if it["kind"] == "post":
        return f"   POST: {body}"
    if it["kind"] == "repost":
        return f"   REPOST of @{it['author']}: {body}"
    th = it["thread"][::-1][-12:]
    lines = [f"   - @{a['author']}: {_clean(a['text'], 200) or '(no text)'}" if a["text"] is not None else "   - (post unavailable)"
             for a in th]
    return "   THREAD:\n" + "\n".join(lines) + f"\n   REPLY: {body}"


def label():
    items = list({json.loads(l)["uri"]: json.loads(l) for l in (D / "items.jsonl").open()}.values())
    random.Random(5).shuffle(items)   # mix kinds and accounts within a batch
    batches = [items[i: i + BATCH] for i in range(0, len(items), BATCH)]
    out = D / "labels"
    out.mkdir(exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="label-")

    def run(ib):
        i, b = ib
        f = out / f"b{i:04d}.json"
        if f.exists():
            return "skip"
        prompt = "\n\n".join(f"{k}.\n{show(it)}" for k, it in enumerate(b, 1))
        cmd = ["claude", "-p", "--model", "sonnet", "--output-format", "json", "--setting-sources", "",
               "--strict-mcp-config", "--no-session-persistence", "--tools", "", "--system-prompt", SYSTEM, prompt]
        for attempt in range(3):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, cwd=scratch, timeout=600)
                env = json.loads(r.stdout)
                lab = json.loads(re.search(r"\{[\s\S]*\}", env["result"]).group(0))
                ps = [min(1.0, max(0.0, float(lab[str(k)]))) for k in range(1, len(b) + 1)]
                f.write_text(json.dumps({"labels": {it["uri"]: p for it, p in zip(b, ps)}, "cost_usd": env.get("total_cost_usd")}))
                return "ok"
            except Exception as e:  # noqa: BLE001  retried, then the batch is left unlabelled and reported
                print(f"b{i:04d} attempt {attempt}: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return "fail"

    t0, res = time.time(), []
    with ThreadPoolExecutor(6) as ex:
        for k, r in enumerate(ex.map(run, enumerate(batches)), 1):
            res.append(r)
            if k % 20 == 0:
                print(f"{k}/{len(batches)} batches, {time.time() - t0:.0f}s", flush=True)
    cost = sum(json.loads(f.read_text()).get("cost_usd") or 0 for f in out.glob("b*.json"))
    print({k: res.count(k) for k in set(res)}, {"items": len(items), "cost_usd": round(cost, 2), "s": round(time.time() - t0)}, flush=True)


if __name__ == "__main__":
    {"collect": collect, "label": label}[sys.argv[1]]()
