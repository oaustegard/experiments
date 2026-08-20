#!/usr/bin/env python3
"""When should nlsh generate a command, and when should it just show the pages?

A 270M model has no reliable command knowledge of its own — the ablation proved
it (0.000 with no sources). So the honest failure mode is NOT "guess from your
own knowledge" (that trains confabulation), it is **abstain and show the
documentation**: "couldn't grok that, here are the pages that look relevant."

That needs a confidence signal to decide generate-vs-abstain. BM25's top score
is the natural one. This measures the calibration on the independent cyber eval:
as a function of a score threshold, how often does retrieval actually contain the
gold utility (so generation could work), versus how often we should decline.

Reported:
* **coverage** — fraction of queries above the threshold (where we'd generate).
* **retrieval precision @ threshold** — of those, how often the gold utility is
  in the top-k the model would see. High precision here = safe to generate.
* **abstain recall@N** — on the queries we decline, is the gold utility among the
  N pages we would show the user? This is the fallback's usefulness.

The goal is a threshold where generation fires only when it will likely succeed,
and everything else degrades to honest, useful pointers.

    python3 calibrate.py --nl ../nl2sh-selfhist/cyber_nl.json --tldr <tldr>/pages
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pleias_gate as G
import retrieve as R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", type=Path, required=True)
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, default=HERE / "data" / "chunks.jsonl")
    ap.add_argument("--k", type=int, default=3, help="sources fed to the model when generating")
    ap.add_argument("--show", type=int, default=5, help="pages shown on abstain")
    ap.add_argument("--out", type=Path, default=HERE / "results_calibrate.json")
    a = ap.parse_args()

    tldr = G.load_tldr(a.tldr)
    data = [r for r in json.loads(a.nl.read_text())
            if r.get("nl") and not r.get("names_utility") and r["utility"] in tldr]
    index = R.Index(R.load_chunks(a.chunks))

    import numpy as np
    per = []
    for r in data:
        s = index.scores(r["nl"])
        # per-utility best-chunk score, ranked
        agg = {}
        for i in np.argsort(s)[::-1]:
            if s[i] <= 0:
                break
            u = index.chunks[int(i)].utility
            agg.setdefault(u, float(s[i]))
        ranked = sorted(agg.items(), key=lambda x: -x[1])
        top1 = ranked[0][1] if ranked else 0.0
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        # Margin, not absolute score: absolute BM25 scales with query length and
        # term rarity and does not separate hits from misses; the top1-vs-top2
        # margin does (median 5.5 when gold is top1, 1.9 when not).
        margin = round(top1 - top2, 3)
        gen_utils = [u for u, _ in ranked[:a.k]]
        show_utils = [u for u, _ in ranked[:a.show]]
        per.append({"gold": r["utility"], "margin": margin,
                    "gold_in_gen": r["utility"] in gen_utils,
                    "gold_at_top1": bool(ranked) and ranked[0][0] == r["utility"],
                    "gold_in_show": r["utility"] in show_utils})

    n = len(per)
    rows = []
    for thr in [0, 1, 2, 3, 5, 8, 12]:
        above = [p for p in per if p["margin"] >= thr]
        below = [p for p in per if p["margin"] < thr]
        rows.append({
            "margin_threshold": thr,
            "coverage": round(len(above) / n, 3),
            "gen_precision_top1": round(sum(p["gold_at_top1"] for p in above) / len(above), 3) if above else None,
            "gen_precision_topk": round(sum(p["gold_in_gen"] for p in above) / len(above), 3) if above else None,
            "abstain_recall_at_show": round(sum(p["gold_in_show"] for p in below) / len(below), 3) if below else None,
        })

    out = {"n": n, "k_generate": a.k, "n_show": a.show,
           "overall_gold_in_topk": round(sum(p["gold_in_gen"] for p in per) / n, 3),
           "overall_gold_in_show": round(sum(p["gold_in_show"] for p in per) / n, 3),
           "margin_when_gold_top1_median": None, "by_threshold": rows}
    import statistics
    hit=[p["margin"] for p in per if p["gold_at_top1"]]
    miss=[p["margin"] for p in per if not p["gold_at_top1"]]
    out["margin_when_gold_top1_median"] = round(statistics.median(hit),2) if hit else None
    out["margin_when_gold_not_top1_median"] = round(statistics.median(miss),2) if miss else None
    a.out.write_text(json.dumps(out, indent=1) + "\n")

    print(f"n={n}  gold@top1 margin median: hit {out['margin_when_gold_top1_median']} "
          f"vs miss {out['margin_when_gold_not_top1_median']}\n")
    print(f"{'margin>=':>9}{'coverage':>10}{'gen_prec@1':>12}{'gen_prec@k':>12}{'abstain_rec@'+str(a.show):>16}")
    for row in rows:
        g1 = "  -  " if row["gen_precision_top1"] is None else f"{row['gen_precision_top1']:.3f}"
        gk = "  -  " if row["gen_precision_topk"] is None else f"{row['gen_precision_topk']:.3f}"
        ar = "  -  " if row["abstain_recall_at_show"] is None else f"{row['abstain_recall_at_show']:.3f}"
        print(f"{row['margin_threshold']:>9}{row['coverage']:>10.3f}{g1:>12}{gk:>12}{ar:>16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
