"""The text the model and the labeller both read for an account, and for a single post.

Keep in step with web/state.js. An account state is the profile header, the account's counts, how
fast it posted, and its latest feed items, each cut to POST_CHARS characters.
"""
from datetime import datetime

POST_CHARS = 140
EMBED = {"images": "image", "video": "video", "external": "link", "record": "quote", "recordWithMedia": "quote"}


def _num(n):
    n = n or 0
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else str(n)


def _clean(t, n=POST_CHARS):
    t = " ".join((t or "").split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def fit_account_state(acct, count_tokens, budget=830, n_posts=12):
    """account_state with trailing items dropped until it is at most `budget` tokens."""
    for n in range(n_posts, 0, -1):
        t = account_state(acct, n)
        if count_tokens(t) <= budget:
            return t
    return account_state(acct, 1)


def account_state(acct, n_posts=12):
    p, items = acct["profile"], acct["items"][:n_posts]
    head = f"@{p['handle']}" + (f" · {_clean(p.get('displayName'), 80)}" if p.get("displayName") else "")
    lines = [head]
    if p.get("description"):
        lines.append("Bio: " + _clean(p["description"], 300))
    created = (p.get("createdAt") or "")[:7]
    lines.append(f"Account: created {created or 'unknown'}, {_num(p.get('followersCount'))} followers, "
                 f"follows {_num(p.get('followsCount'))}, {_num(p.get('postsCount'))} posts")
    ts = [t for t in (_ts(i.get("indexedAt") or i.get("createdAt")) for i in items) if t]
    kinds = {k: sum(i["kind"] == k for i in items) for k in ("post", "reply", "repost")}
    span = (max(ts) - min(ts)).total_seconds() / 86400 if len(ts) > 1 else 0
    lines.append(f"Recent activity: {len(items)} items over {span:.1f} days "
                 f"({kinds['post']} posts, {kinds['reply']} replies, {kinds['repost']} reposts)")
    for k, it in enumerate(items, 1):
        tag = it["kind"] if it["kind"] != "repost" else f"repost of @{it.get('author') or '?'}"
        emb = f" [{EMBED.get(it.get('embed'), it.get('embed'))}]" if it.get("embed") else ""
        lines.append(f"{k}. [{tag}]{emb} {_clean(it.get('text')) or '(no text)'}")
    return "\n".join(lines)


def post_state(item):
    emb = f" [{EMBED.get(item.get('embed'), item.get('embed'))}]" if item.get("embed") else ""
    return f"[{item['kind']}]{emb} {_clean(item.get('text'), 300) or '(no text)'}"
