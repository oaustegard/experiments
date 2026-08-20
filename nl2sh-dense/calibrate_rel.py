#!/usr/bin/env python3
"""A confidence signal for the abstention gate that survives a change of corpus.

The gate in `nlsh.py` generates when the retrieval margin (top1 minus top2 over
per-utility best scores) clears 5, and abstains otherwise. That threshold was
fitted on the security corpus, where BM25 scores run 11-43. Issue #47's live run
put everyday requests through it, where scores run 0.2-2.8, and *everything*
abstained. An absolute margin is in score units, and BM25 score units are a
function of query length and term rarity — so the number does not travel.

This measures four candidate signals on the same queries:

| signal | definition | units |
|---|---|---|
| `margin` | top1 - top2 | score (the incumbent) |
| `ratio` | top2 / top1 | scale-free, lower = more confident |
| `relmargin` | (top1 - top2) / top1 | scale-free, higher = more confident |
| `top1` | top1 | score (the one `calibrate.py` rejected) |

Reported two ways, because a threshold has two separate jobs:

* **Separation, threshold-free.** AUC of the signal against "is the gold utility
  ranked first" — the probability that a random hit outranks a random miss.
  0.5 is a coin flip. This says whether the signal carries information at all,
  without picking an operating point, so it is the fair way to compare signals
  measured on distributions with different score scales.
* **Transfer, at a fixed operating point.** The same numeric threshold applied to
  every distribution, reporting coverage and generation precision on each. This
  is the property that actually failed: a signal can separate well within each
  corpus and still have no single number that works across both.

    python3 calibrate_rel.py --models leaf-mt-int8 --nl2bash <nl2bash>/data/bash
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _lib.paths import experiment  # noqa: E402

RETRIEVAL = experiment("nl2sh-retrieval")
sys.path.insert(0, str(RETRIEVAL))
import retrieve as R  # noqa: E402
import dense_index as D  # noqa: E402
import queries as Q  # noqa: E402
from eval_dense import tldr_from_chunks  # noqa: E402

SIGNALS = ("margin", "ratio", "relmargin", "top1")


def signals(ranked: list[tuple[str, float]]) -> dict[str, float]:
    """All four candidate signals from one utility ranking.

    A single-candidate ranking is maximally confident by construction: margin is
    top1 itself, ratio 0, relmargin 1. An empty ranking is minimally confident.
    """
    if not ranked:
        return {"margin": 0.0, "ratio": 1.0, "relmargin": 0.0, "top1": 0.0}
    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    denom = top1 if abs(top1) > 1e-9 else 1e-9
    return {"margin": top1 - top2, "ratio": top2 / denom,
            "relmargin": (top1 - top2) / denom, "top1": top1}


def auc(scores: list[float], labels: list[bool], higher_is_confident: bool) -> float | None:
    """Rank-based AUC with tie correction. None when one class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    order = np.argsort(np.array(scores, dtype=float), kind="stable")
    ranks = np.empty(len(scores), dtype=float)
    srt = np.array(scores, dtype=float)[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rpos = ranks[np.array(labels, dtype=bool)].sum()
    a = (rpos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return round(float(a if higher_is_confident else 1 - a), 3)


def measure(rows: list[dict], rank_fn, k: int, show: int) -> list[dict]:
    per = []
    for r in rows:
        ranked = rank_fn(r["nl"])
        utils = [u for u, _ in ranked]
        gold = r["utility"]
        rec = {"gold": gold, "nl": r["nl"],
               "names_utility": bool(r.get("names_utility")),
               "gold_at_top1": bool(utils) and utils[0] == gold,
               "gold_in_gen": gold in utils[:k],
               "gold_in_show": gold in utils[:show]}
        rec.update(signals(ranked))
        per.append(rec)
    return per


def sweep(per: list[dict], signal: str, thresholds: list[float],
          higher_is_confident: bool, show: int) -> list[dict]:
    n = len(per)
    out = []
    for thr in thresholds:
        # The confident side is above the threshold for a higher-is-confident
        # signal and below it for `ratio`, where a small top2/top1 means the
        # leader is unchallenged.
        above = ([p for p in per if p[signal] >= thr] if higher_is_confident
                 else [p for p in per if p[signal] <= thr])
        conf = {id(p) for p in above}
        below = [p for p in per if id(p) not in conf]
        out.append({
            "threshold": thr,
            "coverage": round(len(above) / n, 3) if n else None,
            "gen_precision_top1": round(sum(p["gold_at_top1"] for p in above) / len(above), 3) if above else None,
            "gen_precision_topk": round(sum(p["gold_in_gen"] for p in above) / len(above), 3) if above else None,
            "abstain_recall_at_show": round(sum(p["gold_in_show"] for p in below) / len(below), 3) if below else None,
        })
    return out


HIGHER = {"margin": True, "ratio": False, "relmargin": True, "top1": True}
GRIDS = {
    "margin": [0, 0.5, 1, 2, 3, 5, 8, 12],
    "ratio": [0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3],
    "relmargin": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7],
    "top1": [0, 1, 2, 3, 5, 8, 12, 20],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[],
                    help="dense models to also calibrate the hybrid arm on")
    ap.add_argument("--alpha", type=float, default=0.5, help="wsum weight on dense")
    ap.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    ap.add_argument("--nl2bash", type=Path, default=None)
    ap.add_argument("--nl2bash-n", type=int, default=300)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--leaky", action="store_true",
                    help="keep rows whose request names the gold utility")
    ap.add_argument("--out", type=Path, default=HERE / "results_calibrate_rel.json")
    a = ap.parse_args()

    chunks = R.load_chunks(a.chunks)
    index = R.Index(chunks)
    utilities = np.array([c.utility for c in chunks], dtype=object)
    tldr = tldr_from_chunks(chunks)
    sets = Q.load_all(tldr, a.nl2bash, a.nl2bash_n)
    if not a.leaky:
        sets = {k: [r for r in v if not r.get("names_utility")] for k, v in sets.items()}

    arms = {"bm25": lambda q: D.rank_utilities(index.scores(q), utilities, a.pool,
                                               positive_only=True)}
    for model in a.models:
        _, _, dense = D.load(model, a.chunks)
        arms[f"dense:{model}"] = (
            lambda q, d=dense: D.rank_utilities(d.scores(q), utilities, a.pool))
        arms[f"wsum{a.alpha}:bm25+{model}"] = (
            lambda q, d=dense: D.wsum(
                D.rank_utilities(index.scores(q), utilities, a.pool, positive_only=True),
                D.rank_utilities(d.scores(q), utilities, a.pool), a.alpha))

    out = {"config": {"k": a.k, "show": a.show, "alpha": a.alpha,
                      "leak_free": not a.leaky},
           "distributions": {name: {"n": len(rows),
                                    "constant_prior": Q.constant_prior(rows)}
                             for name, rows in sets.items()},
           "arms": {}}

    for arm_name, rank_fn in arms.items():
        arm = {"per_distribution": {}}
        for dist, rows in sets.items():
            per = measure(rows, rank_fn, a.k, a.show)
            hits = [p for p in per if p["gold_at_top1"]]
            miss = [p for p in per if not p["gold_at_top1"]]
            entry = {"n": len(per),
                     "gold_at_top1": round(len(hits) / len(per), 3),
                     "gold_in_gen": round(sum(p["gold_in_gen"] for p in per) / len(per), 3),
                     "signals": {}}
            for sig in SIGNALS:
                entry["signals"][sig] = {
                    # Two labels, because the gate has two possible jobs. `top1`
                    # is what `calibrate.py` measured. `in_gen` is what the gate
                    # actually gates on: the generator sees k=3 sources and can
                    # only produce the gold utility if one of them is it.
                    "auc_top1": auc([p[sig] for p in per],
                                    [p["gold_at_top1"] for p in per], HIGHER[sig]),
                    "auc_in_gen": auc([p[sig] for p in per],
                                      [p["gold_in_gen"] for p in per], HIGHER[sig]),
                    "median_hit": round(statistics.median([p[sig] for p in hits]), 3) if hits else None,
                    "median_miss": round(statistics.median([p[sig] for p in miss]), 3) if miss else None,
                    "sweep": sweep(per, sig, GRIDS[sig], HIGHER[sig], a.show),
                }
            entry["rows"] = per
            arm["per_distribution"][dist] = entry
        out["arms"][arm_name] = arm

    a.out.write_text(json.dumps(out, indent=1) + "\n")

    for arm_name, arm in out["arms"].items():
        dists = list(arm["per_distribution"])
        print(f"\n=== {arm_name} — AUC against 'gold utility is in the k sources' ===")
        print(f"{'signal':<12}" + "".join(f"{d:>14}" for d in dists))
        for sig in SIGNALS:
            cells = []
            for d in dists:
                v = arm["per_distribution"][d]["signals"][sig]["auc_in_gen"]
                cells.append(f"{'   n/a' if v is None else f'{v:.3f}':>14}")
            print(f"{sig:<12}" + "".join(cells))
        for label in ("gold_at_top1", "gold_in_gen"):
            print(f"{label:<12}" + "".join(
                f"{arm['per_distribution'][d][label]:>14.3f}" for d in dists))

        print(f"\n--- transfer: one threshold, every distribution "
              f"(coverage / gen_precision@k) ---")
        for sig in SIGNALS:
            print(f"  {sig}")
            for i, thr in enumerate(GRIDS[sig]):
                cells = []
                for d in dists:
                    row = arm["per_distribution"][d]["signals"][sig]["sweep"][i]
                    gp = row["gen_precision_topk"]
                    cells.append(f"{row['coverage']:.2f}/{'  - ' if gp is None else f'{gp:.2f}'}"
                                 .rjust(14))
                print(f"    {str(thr):<8}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
