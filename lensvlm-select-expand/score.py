#!/usr/bin/env python3
"""Score ledgers: closed-book, image-only (commit), select+expand (final),
selection accuracy and effective compression, per model x ratio.

Answer match: HotpotQA normalisation (lowercase, drop punctuation and
articles). `em` is exact match; `hit` is lenient: exact, or one normalised
string contains the other with the shorter at least 4 chars. Every miss that
`hit` lets through, and every non-exact hit, is listed for hand review with
--review.
"""
import json
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
man = {m["id"]: m for m in json.loads((HERE / "data" / "manifest.json").read_text())}
stats = json.loads((HERE / "data" / "sheet_stats.json").read_text())
img_tok = {(r, d["id"]): d["img_tokens"] for r, v in stats.items() for d in v["per_doc"]}


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


# Hand adjudication, applied identically to every arm and model. Each entry is
# a normalised prefix that is also a correct answer to the question as asked.
#  h02: SSIH and ASUAG are the companies the banks forced to merge; the gold
#       names the merged result.
#  h05: paraphrases of the gold.
#  h06: the same person with his middle name.
#  h09: the 2012 season was the Rams' 75th; naming the year names the season.
#  h10: Nolan and Schrader are both film directors as well as screenwriters.
ACCEPT = {
    "h02": ["ssih", "asuag"],
    "h05": ["ghana national football team", "ghana national team"],
    "h06": ["bernard law montgomery"],
    "h09": ["2012"],
    "h10": ["director", "film director"],
}


def match(pred, gold, doc=None):
    p, g = norm(pred or ""), norm(gold)
    if p == g:
        return "em"
    if doc and any(p.startswith(a) for a in ACCEPT.get(doc, [])):
        return "adj"
    short, long_ = sorted((p, g), key=len)
    if len(short) >= 4 and short in long_:
        return "hit"
    return None


def load(run):
    p = HERE / "runs" / run / "ledger.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


MODELS = ("opus", "sonnet", "gemini", "muse")


def pct(n, d):
    return f"{100 * n / d:5.1f}" if d else "  n/a"


def main(review=False):
    rows = []
    closed = {}
    for model in MODELS:
        for e in load(f"closed-{model}"):
            if e["kind"] == "closed":
                closed[(model, e["doc"])] = e["answer"]
    models_seen = defaultdict(set)
    for model in MODELS:
        for e in load(f"closed-{model}"):
            if e["kind"] == "hello":
                models_seen[model].add(e["model"])
        for r in ("5", "10", "15"):
            per_doc = {}
            for half in ("A", "B"):
                for e in load(f"{model}-r{r}-{half}"):
                    if e["kind"] == "hello":
                        models_seen[model].add(e["model"])
                        continue
                    if e["kind"] == "usage" and e.get("model"):
                        models_seen[model].add(e["model"])
                    if "doc" not in e:
                        continue
                    d = per_doc.setdefault(e["doc"], {"expand": []})
                    if e["kind"] == "commit":
                        d["commit"], d["pages"] = e["answer"], e["pages"]
                    elif e["kind"] == "expand":
                        d["expand"].append(e["page"])
                    elif e["kind"] == "final":
                        d["final"] = e["answer"]
            for doc, d in sorted(per_doc.items()):
                m = man[doc]
                gt = set(m["gt_pages"])
                exp_chars = sum(len(t) for t in pages_text(doc, d["expand"]))
                rows.append({
                    "model": model, "ratio": int(r), "doc": doc,
                    "gold": m["answer"], "closed": closed.get((model, doc)),
                    "commit": d.get("commit"), "final": d.get("final"),
                    "closed_ok": bool(match(closed.get((model, doc)), m["answer"], doc)),
                    "img_ok": bool(match(d.get("commit"), m["answer"], doc)),
                    "final_ok": bool(match(d.get("final"), m["answer"], doc)),
                    "final_em": match(d.get("final"), m["answer"]) == "em",
                    "sel1": bool(d.get("pages")) and d["pages"][0] in gt,
                    "sel_any": bool(set(d.get("pages", [])) & gt),
                    "exp_hit": bool(set(d["expand"]) & gt),
                    "exp_cover": len(set(d["expand"]) & gt) / len(gt),
                    "n_expand": len(d["expand"]),
                    "complete": "final" in d,
                    "ecr": m["text_tokens"] / (img_tok[(r, doc)] + exp_chars / 4),
                })
    (HERE / "results.json").write_text(json.dumps(rows, indent=1))
    print("served models:", {k: sorted(v) for k, v in models_seen.items()})
    for model in MODELS:
        for r in ("5", "10", "15"):
            us = [e for e in load(f"{model}-r{r}-A") if e["kind"] == "usage" and e.get("step") == "commit"]
            if us and model in ("gemini", "muse"):
                key = "img" if model == "gemini" else "in"
                vals = [e[key] for e in us if e.get(key)]
                if vals:
                    print(f"{model} r{r}: commit-turn {'image' if key == 'img' else 'input'} tokens mean "
                          f"{sum(vals) / len(vals):.0f} (Claude formula: {stats[r]['mean_img_tokens']})")
    hdr = f"{'model':7} {'ratio':>5} {'n':>3} {'closed':>6} {'image':>6} {'expand':>6} {'exp-em':>6} {'sel@1':>6} {'sel@k':>6} {'expHit':>6} {'cover':>6} {'#exp':>5} {'ECR':>5} {'img|cbX':>8}"
    print(hdr)
    for model in MODELS:
        for r in (5, 10, 15):
            g = [x for x in rows if x["model"] == model and x["ratio"] == r]
            n = len(g)
            if not n:
                continue
            cbx = [x for x in g if not x["closed_ok"]]
            print(f"{model:7} {r:>5} {n:>3} {pct(sum(x['closed_ok'] for x in g), n):>6} "
                  f"{pct(sum(x['img_ok'] for x in g), n):>6} {pct(sum(x['final_ok'] for x in g), n):>6} "
                  f"{pct(sum(x['final_em'] for x in g), n):>6} {pct(sum(x['sel1'] for x in g), n):>6} "
                  f"{pct(sum(x['sel_any'] for x in g), n):>6} {pct(sum(x['exp_hit'] for x in g), n):>6} "
                  f"{pct(sum(x['exp_cover'] for x in g), n):>6} {sum(x['n_expand'] for x in g) / n:5.2f} "
                  f"{sum(x['ecr'] for x in g) / n:5.1f} "
                  f"{sum(x['img_ok'] for x in cbx)}/{len(cbx):<4}")
    if review:
        print("\nnon-exact matches and misses:")
        for x in rows:
            for k in ("closed", "commit", "final"):
                mt = match(x[k], x["gold"], x["doc"])
                if mt != "em":
                    print(f"  {x['model']:6} r{x['ratio']:<2} {x['doc']} {k:6} {str(mt):4} gold={x['gold']!r} pred={x[k]!r}")


_TEXTS = None


def pages_text(doc, pages):
    global _TEXTS
    if _TEXTS is None:
        import base64, zlib
        _TEXTS = json.loads(zlib.decompress(base64.b64decode((HERE / "data" / "pages.bin").read_bytes())))
    return [_TEXTS[doc][p - 1] for p in pages]


if __name__ == "__main__":
    main(review="--review" in sys.argv)
