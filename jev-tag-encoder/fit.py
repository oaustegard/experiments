"""Step 2: taxonomy fit on 200 mixed docs (100 AG News + 100 20 Newsgroups), plus cost/latency (step 1)."""
import numpy as np

import jev
from common import DATA, boot, ci, fmt, jsonl, save

recs = jsonl(DATA / "mixed.jsonl")
ids = [r["id"] for r in recs]
P = jev.matrix("mixed", "about", ids)
ok = ~np.isnan(P).any(1)
recs = [r for r, k in zip(recs, ok) if k]
P = P[ok]
n = len(P)
Bits = P >= 0.5
fire = Bits.mean(0)
tags = np.array(jev.TAGS, dtype=object)

active = Bits.sum(1)
print(f"docs encoded: {n}/{len(ok)}; active tags/doc mean {active.mean():.2f} median {np.median(active):.0f} "
      f"min {active.min()} max {active.max()}; docs with 0 active: {(active == 0).sum()}")
dead = tags[fire == 0]
print(f"dead tags (never >= 0.5): {len(dead)}/256")
print(f"always-on (fire >= 0.8): {list(tags[fire >= 0.8])}; >= 0.5: {list(tags[fire >= 0.5])}; "
      f"tags firing on >= 5 docs: {int((Bits.sum(0) >= 5).sum())}")
order = np.argsort(-fire)
print("top fire rates:", [(tags[i], round(float(fire[i]), 3)) for i in order[:15]])

# block = one line of tags.txt (16 tags)
blocks = [line.split("|")[0] + "..." for line in (jev.HERE / "tags.txt").read_text().splitlines()]
blk = [{"block": b, "fire_rate_mean": float(fire[16 * k:16 * k + 16].mean()),
        "dead": int((fire[16 * k:16 * k + 16] == 0).sum())} for k, b in enumerate(blocks)]
for b in blk:
    print(f"  {b['block']:<28} mean fire {b['fire_rate_mean']:.3f}  dead {b['dead']}/16")

# correlation over tags that fire (>= 0.5) on at least 5 docs; rarer tags give spurious r ~ 1 from one doc
live = Bits.sum(0) >= 5
C = np.corrcoef(P[:, live].T)
li = np.where(live)[0]
iu = np.triu_indices(len(li), 1)
pairs = sorted(zip(C[iu], li[iu[0]], li[iu[1]]), key=lambda x: -x[0])
top_pairs = [(tags[a], tags[b], round(float(c), 3)) for c, a, b in pairs[:15]]
neg_pairs = [(tags[a], tags[b], round(float(c), 3)) for c, a, b in pairs[-5:]]
print("top correlated pairs:", top_pairs)
print("most anti-correlated:", neg_pairs)

# sanity: source label -> strongest mean tags
by = {}
for r, p in zip(recs, P):
    by.setdefault(r["label"], []).append(p)
sanity = {lab: [(tags[i], round(float(np.mean(v, 0)[i]), 2)) for i in np.argsort(-np.mean(v, 0))[:4]]
          for lab, v in sorted(by.items())}
for lab, top in sanity.items():
    print(f"  {lab:<26} {top}")

# dead across every "about" doc set (mixed + arXiv + SciFact corpus): corpus gap vs a tag that never fires
alld = np.concatenate([m[~np.isnan(m).any(1)] for m in (
    P, jev.matrix("arxiv", "about", [r["id"] for r in jsonl(DATA / "arxiv.jsonl")]),
    jev.matrix("scifact", "about", [r["_id"] for r in jsonl(DATA / "scifact" / "corpus.jsonl")]))])
fire_all = (alld >= 0.5).mean(0)
dead_all = list(tags[fire_all == 0])
print(f"dead across all {len(alld)} docs: {len(dead_all)}/256: {dead_all}")

# step 1: cost/latency across every cached set
allr = []
for f in sorted(jev.CACHE.glob("*.jsonl")):
    allr += [r for r in jev.load(*f.stem.split("__")).values()]
okr = [r for r in allr if "p" in r]
lat = np.array([r["latency_s"] for r in okr])
cost = {"calls_ok": len(okr), "calls_failed": len(allr) - len(okr),
        "waf_blocked": sum(r.get("error") == "blocked" for r in allr),
        "latency_p50": float(np.percentile(lat, 50)), "latency_p95": float(np.percentile(lat, 95)),
        "in_tok_mean": float(np.mean([r["in_tok"] for r in okr])),
        "out_tok_mean": float(np.mean([r["out_tok"] for r in okr])),
        "models": sorted({r["model"] for r in okr})}
print("cost/latency:", cost)
act_ci = ci(boot(lambda i: active[i].mean(), n))[0]
print("active tags/doc", fmt(active.mean(), act_ci))

save("step2_fit", {"n": n, "active_mean": float(active.mean()), "active_ci": act_ci,
                   "zero_active_docs": int((active == 0).sum()),
                   "dead": list(dead), "dead_all_corpora": dead_all, "n_all_corpora": len(alld), "fire": dict(zip(jev.TAGS, fire.round(4).tolist())),
                   "blocks": blk, "top_pairs": top_pairs, "neg_pairs": neg_pairs, "sanity": sanity,
                   "cost": cost})
