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

#: Fractions of the calibration distribution that should fall on the confident
#: side. Thresholds are read off as quantiles rather than written down as
#: numbers, because a fused score, a raw BM25 margin and a cosine live on three
#: different scales and a fixed grid tests only whichever one it was written for.
#: That is the incumbent gate's failure in miniature: `margin >= 5` is a
#: perfectly good BM25 operating point and is off the end of every other scale.
COVERAGE_TARGETS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

#: The gate as shipped, plus two scale-free thresholds that need no calibration
#: sample. These are reported as written-down constants rather than as quantiles,
#: because that is the form the question is about: `nlsh --margin 5` is a literal
#: 5 in BM25 score units, and `top2/top1 <= 0.85` is a literal 0.85 on a scale
#: that is the same everywhere.
FIXED_GATES = [("margin", 5.0), ("margin", 3.0),
               ("ratio", 0.85), ("ratio", 0.8), ("ratio", 0.7)]


def thresholds_for(per: list[dict], signal: str, targets=COVERAGE_TARGETS) -> list[float]:
    """Threshold values that give each target coverage on THIS distribution."""
    vals = sorted((p[signal] for p in per), reverse=HIGHER[signal])
    if not vals:
        return [0.0] * len(targets)
    out = []
    for t in targets:
        i = min(len(vals) - 1, max(0, int(round(t * len(vals))) - 1))
        out.append(round(float(vals[i]), 6))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[],
                    help="dense models to also calibrate the hybrid arm on")
    ap.add_argument("--alpha", type=float, default=0.5, help="wsum weight on dense")
    ap.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    ap.add_argument("--granularity", default="chunk", choices=["chunk", "page"])
    ap.add_argument("--fusion", default="wsum", choices=["wsum", "rrf"])
    ap.add_argument("--nl2bash", type=Path, default=None)
    ap.add_argument("--nl2bash-n", type=int, default=300)
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--calibrate-on", default="cyber",
                    help="distribution the thresholds are read off")
    ap.add_argument("--leaky", action="store_true",
                    help="keep rows whose request names the gold utility")
    ap.add_argument("--out", type=Path, default=HERE / "results_calibrate_rel.json")
    a = ap.parse_args()

    chunks = D.GRANULARITIES[a.granularity](R.load_chunks(a.chunks))
    index = R.Index(chunks)
    utilities = np.array([c.utility for c in chunks], dtype=object)
    tldr = tldr_from_chunks(chunks)
    sets = Q.load_all(tldr, a.nl2bash, a.nl2bash_n)
    if not a.leaky:
        sets = {k: [r for r in v if not r.get("names_utility")] for k, v in sets.items()}

    arms = {"bm25": lambda q: D.rank_utilities(index.scores(q), utilities, a.pool,
                                               positive_only=True)}
    for model in a.models:
        _, _, dense = D.load(model, a.chunks, granularity=a.granularity)
        arms[f"dense:{model}"] = (
            lambda q, d=dense: D.rank_utilities(d.scores(q), utilities, a.pool))
        def fuse(bu, du, how=a.fusion, alpha=a.alpha):
            return D.wsum(bu, du, alpha) if how == "wsum" else D.rrf(bu, du)

        name = (f"wsum{a.alpha}:bm25+{model}" if a.fusion == "wsum"
                else f"rrf:bm25+{model}")
        arms[name] = (
            lambda q, d=dense, fuse=fuse: fuse(
                D.rank_utilities(index.scores(q), utilities, a.pool, positive_only=True),
                D.rank_utilities(d.scores(q), utilities, a.pool)))

    out = {"config": {"k": a.k, "show": a.show, "alpha": a.alpha,
                      "granularity": a.granularity, "fusion": a.fusion,
                      "documents": len(chunks), "leak_free": not a.leaky},
           "distributions": {name: {"n": len(rows),
                                    "constant_prior": Q.constant_prior(rows)}
                             for name, rows in sets.items()},
           "arms": {}}

    for arm_name, rank_fn in arms.items():
        arm = {"per_distribution": {}}
        # Thresholds are set on the calibration distribution once, then applied
        # unchanged everywhere else — the transfer question, not a per-corpus refit.
        calib = measure(sets[a.calibrate_on], rank_fn, a.k, a.show)
        grids = {sig: thresholds_for(calib, sig) for sig in SIGNALS}
        arm["thresholds"] = grids
        arm["calibrated_on"] = a.calibrate_on
        arm["coverage_targets"] = COVERAGE_TARGETS
        for dist, rows in sets.items():
            per = measure(rows, rank_fn, a.k, a.show)
            hits = [p for p in per if p["gold_at_top1"]]
            miss = [p for p in per if not p["gold_at_top1"]]
            entry = {"n": len(per),
                     "gold_at_top1": round(len(hits) / len(per), 3),
                     "gold_in_gen": round(sum(p["gold_in_gen"] for p in per) / len(per), 3),
                     "signals": {}}
            for sig in SIGNALS:
                grid = grids[sig]
                entry["signals"][sig] = {
                    "thresholds": grid,
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
                    "sweep": sweep(per, sig, grid, HIGHER[sig], a.show),
                }
            entry["fixed_gates"] = [
                {"signal": sig, "threshold": thr,
                 **sweep(per, sig, [thr], HIGHER[sig], a.show)[0]}
                for sig, thr in FIXED_GATES]
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

        print("\n--- the shipped gate and two written-down alternatives "
              "(coverage / gen_precision@k) ---")
        for i, (sig, thr) in enumerate(FIXED_GATES):
            cells = []
            for d in dists:
                row = arm["per_distribution"][d]["fixed_gates"][i]
                gp = row["gen_precision_topk"]
                cells.append(f"{row['coverage']:.2f}/{'  - ' if gp is None else f'{gp:.2f}'}"
                             .rjust(14))
            op = ">=" if HIGHER[sig] else "<="
            print(f"    {sig} {op} {thr:<8g}" + "".join(cells))

        print(f"\n--- transfer: thresholds set on '{arm['calibrated_on']}', applied "
              f"unchanged (coverage / gen_precision@k) ---")
        for sig in SIGNALS:
            print(f"  {sig}   (threshold at each target coverage on "
                  f"{arm['calibrated_on']})")
            for i, thr in enumerate(arm["thresholds"][sig]):
                cells = []
                for d in dists:
                    row = arm["per_distribution"][d]["signals"][sig]["sweep"][i]
                    gp = row["gen_precision_topk"]
                    cells.append(f"{row['coverage']:.2f}/{'  - ' if gp is None else f'{gp:.2f}'}"
                                 .rjust(14))
                print(f"    {COVERAGE_TARGETS[i]:<5.0%}{thr:<10.4g}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
