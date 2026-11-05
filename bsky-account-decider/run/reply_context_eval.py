"""Does showing the replied-to post help the per-post "about AI" question on replies?

    python3 -I run/reply_context_eval.py collect   # fresh replies + their parents -> data/reply_ctx/items.jsonl
    python3 -I run/reply_context_eval.py label     # Sonnet, shown parent and reply -> data/reply_ctx/labels.json
    python3 -I run/reply_context_eval.py score     # shipped page model, each input format -> results/reply_context.json

The shipped model was trained on post_state(item), the reply's text alone, against labels whose
instructions said a short reply with no visible topic is not about AI. Here Sonnet labels replies
seeing the parent, under the rule that a reply taking part in a conversation about AI is about AI,
and the packed page model (NumPy reference of what the browser computes) is scored on three inputs:
the reply alone, the reply then its parent, and the parent then the reply. No retraining.
Post texts stay in gitignored data/; the results file keeps metrics only.
"""
import json
import random
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
sys.path.insert(0, str(ROOT.parent / "browser-decider" / "run"))
from collect import discover_authors, get, list_members  # noqa: E402
from state import _clean, post_state  # noqa: E402

D = ROOT / "data" / "reply_ctx"
PER_ACCOUNT = 6
BATCH = 30


def parent_of(feed_item):
    par = (feed_item.get("reply") or {}).get("parent") or {}
    if not par.get("$type", "").endswith("#postView"):
        return None  # notFoundPost / blockedPost
    return {"author": par["author"].get("handle"), "text": (par.get("record") or {}).get("text", "")}


def replies_of(did):
    d = get("app.bsky.feed.getAuthorFeed", {"actor": did, "limit": 50, "filter": "posts_with_replies"})
    out = []
    for it in (d or {}).get("feed", []):
        if it.get("reason"):
            continue
        p, rec = it["post"], it["post"].get("record", {})
        if not rec.get("reply"):
            continue
        par = parent_of(it)
        if not par:
            continue
        emb = (p.get("embed") or {}).get("$type", "")
        out.append({"kind": "reply", "text": rec.get("text", ""), "uri": p["uri"], "account": did,
                    "embed": emb.split(".")[-1].split("#")[0] if emb else None, "parent": par})
        if len(out) >= PER_ACCOUNT:
            break
    return out


def collect():
    rng = random.Random(11)
    ai = rng.sample(list_members(), 45)
    disc = [a for a in discover_authors(pages=4) if a not in ai]
    disc = rng.sample(disc, min(45, len(disc)))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(replies_of, ai + disc))
    D.mkdir(parents=True, exist_ok=True)
    n = 0
    with (D / "items.jsonl").open("w") as f:
        for src, accts in (("list", ai), ("discover", disc)):
            for did, items in zip(accts, res[: len(ai)] if src == "list" else res[len(ai):]):
                for it in items:
                    f.write(json.dumps({**it, "source": src}) + "\n")
                    n += 1
    print({"accounts": len(ai) + len(disc), "replies": n})


SYSTEM = """You annotate Bluesky replies. Each numbered item shows the post being replied to (PARENT) and the reply (REPLY), texts cut to 300 characters.

For every item give the probability that the REPLY is ABOUT AI (artificial intelligence, machine learning, LLMs or chatbots, AI products, companies, research, policy, safety, art, jobs or energy use, or using AI tools). Read the reply as part of its conversation: a reply that responds to a parent post about AI is about AI, even when the reply's own words do not name AI. A reply whose own text is about AI is about AI whatever the parent says. A reply to a post that is not about AI, whose own text is not about AI either, is not about AI. Mentions of tech, computing or science without AI are not about AI. Reflect your real uncertainty: use values near 1 or 0 only when the evidence is clear.

Reply with ONLY a JSON object mapping each item number to a probability, e.g. {"1": 0.9, "2": 0.05}."""


def label():
    items = [json.loads(l) for l in (D / "items.jsonl").open()]
    batches = [items[i: i + BATCH] for i in range(0, len(items), BATCH)]
    scratch = tempfile.mkdtemp(prefix="label-")

    def call(b):
        prompt = "\n\n".join(f"{k}. PARENT (@{it['parent']['author']}): {_clean(it['parent']['text'], 300) or '(no text)'}\n"
                             f"   REPLY: {_clean(it['text'], 300) or '(no text)'}" for k, it in enumerate(b, 1))
        cmd = ["claude", "-p", "--model", "sonnet", "--output-format", "json", "--setting-sources", "",
               "--strict-mcp-config", "--no-session-persistence", "--tools", "", "--system-prompt", SYSTEM, prompt]
        for _ in range(3):
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=scratch, timeout=600)
            try:
                env = json.loads(r.stdout)
                lab = json.loads(re.search(r"\{[\s\S]*\}", env["result"]).group(0))
                ps = [min(1.0, max(0.0, float(lab[str(k)]))) for k in range(1, len(b) + 1)]
                return ps, env.get("total_cost_usd")
            except Exception as e:  # noqa: BLE001  retried; a batch that never parses stays None
                err = e
        print("batch failed:", type(err).__name__, str(err)[:200])
        return None, None

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(call, batches))
    labels, cost = {}, 0.0
    for b, (ps, c) in zip(batches, res):
        cost += c or 0
        if ps is not None:
            labels.update({it["uri"]: p for it, p in zip(b, ps)})
    (D / "labels.json").write_text(json.dumps({"labels": labels, "cost_usd": cost}))
    print({"labelled": len(labels), "of": len(items), "cost_usd": round(cost, 3)})


def fmt_before(it):
    p = it["parent"]
    return f"In reply to @{p['author']}: {_clean(p['text']) or '(no text)'}\n" + post_state({**it, "parent": None})


# reply_then_parent is what state.post_state (and the page) now builds for a reply with a parent
VARIANTS = {"reply_only": lambda it: post_state({**it, "parent": None}), "reply_then_parent": post_state,
            "parent_then_reply": fmt_before}


def score():
    from build_data import POST_Q
    from common import examples
    from pack import Ref
    import types
    sys.modules.setdefault("transformers", types.SimpleNamespace(AutoModel=None))  # Enc needs only tokenizers
    from train_enc import Enc
    M = ROOT / "web" / "models" / "account"
    man = json.loads((M / "manifest.json").read_text())
    ref, tok = Ref(man, (M / "weights.bin").read_bytes()), Enc(M / "tokenizer.json")
    items = [json.loads(l) for l in (D / "items.jsonl").open()]
    lab = json.loads((D / "labels.json").read_text())["labels"]
    items = list({it["uri"]: it for it in items if it["uri"] in lab}.values())
    gold = np.array([lab[it["uri"]] for it in items])

    def p_true(state):
        e = examples([{"id": "x", "workflow": "post", "state": state, "questions": POST_Q,
                       "gold": {"about_ai": {"type": "noul", "probabilities": {"true": 0.5, "false": 0.5}}}}])[0]
        ids, pos = tok.build(e)
        z = ref.logits(ids, pos) / man["temps"]["noul"]
        z = np.exp(z - z.max())
        return float((z / z.sum())[[n for n, _ in e["options"]].index("true")])

    def metrics(pred, sel):
        g, p = gold[sel], pred[sel]
        pos, neg = g > 0.5, g <= 0.5
        return {"n": int(sel.sum()), "accuracy": round(float(((p > 0.5) == pos).mean()), 4),
                "recall": round(float((p[pos] > 0.5).mean()), 4) if pos.any() else None,
                "false_pos_rate": round(float((p[neg] > 0.5).mean()), 4) if neg.any() else None,
                "mean_abs_err": round(float(np.abs(p - g).mean()), 4)}

    src = np.array([it["source"] for it in items])
    groups = {"all": np.ones(len(items), bool), "list": src == "list", "discover": src == "discover"}
    out = {"replies": len(items), "sonnet_positive": int((gold > 0.5).sum()), "variants": {}}
    for name, f in VARIANTS.items():
        pred = np.array([p_true(f(it)) for it in items])
        out["variants"][name] = {g: metrics(pred, s) for g, s in groups.items()}
    (ROOT / "results" / "reply_context.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"collect": collect, "label": label, "score": score}[sys.argv[1]]()
