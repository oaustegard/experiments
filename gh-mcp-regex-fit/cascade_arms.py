#!/usr/bin/env python3
"""Can a fallback add coverage to a precise rule set without destroying abstention?

`RESULTS.md` prices the naive answer exactly: bolting an unconditional catch-all
onto the hand-written rules took abstention from 0.867 to **0.000** on all three
splits and bought +0.014 accuracy on one of them. That rule could not say no, so
it answered every off-topic request.

The fix is not "no fallback", it is "a fallback that can abstain". The hand arm
is precise where it fires (0.667 precision at 0.730 coverage on wild), so take
its answer whenever it fires, and hand the rest to a *scored* arm gated by a
threshold. Coverage rises only where the second arm is confident; off-topic rows,
which no arm should be confident about, still fall through to a refusal.

    hand.route(q) -> answer if not None
    else: label, s = scored.score(q)[0]; answer if s >= threshold else None

The scored arm is whatever is registered — `encoder_arms` is guaranteed, and
`bm25_arms` / `spacy_arms` are picked up automatically if those modules exist.
This file also re-runs `agreement.py`'s two-router gate with the encoder in the
fitted list's place: agreement between two *lexical* routers bought 0.775/0.867
precision at ~0.20 coverage, and the open question is whether a semantically
independent second opinion gates better than a correlated one.

    python3 cascade_arms.py            # curves + results_cascade.json
    python3 eval.py cascade-enc-fusion
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import arms as arms_mod
from arms import ArmBase, register
from handwritten import HandRouter

HERE = Path(__file__).resolve().parent

SPLITS = ("family A (fitted)", "family B (held-out)", "wild (hand-authored)")
PATHS = {SPLITS[0]: HERE / "data" / "family_a.jsonl",
         SPLITS[1]: HERE / "data" / "family_b.jsonl",
         SPLITS[2]: HERE / "wild.jsonl"}

# Abstention on off-topic rows is the thing the catch-all destroyed, so it is a
# constraint on the operating point rather than one number among many.
ABSTAIN_FLOOR = 0.70
# Selection happens on family A only. It is the split the decision lists were
# fitted on and the split a deployment would have labelled traffic for; using B
# or wild to choose the threshold would be choosing on the test set.
SELECT_SPLIT = SPLITS[0]


class CascadeArm(ArmBase):
    """Precise arm first, thresholded scored arm second, abstention last."""

    def __init__(self, fallback: str = "enc-fusion-open", threshold: float = 0.5,
                 primary=None) -> None:
        self.primary = primary if primary is not None else HandRouter()
        # The fallback is built at its *open* setting where one exists: this
        # class owns the threshold, and two stacked thresholds would be a knob
        # nobody could read off the curve.
        self.fallback = arms_mod.build(fallback)
        self.fallback_name, self.threshold = fallback, threshold

    def score(self, query: str) -> list[tuple[str, float]]:
        """The fallback's ranking, with the primary's answer pinned on top.

        Pinned at 1.0 rather than merged: the hand arm has no score, and giving
        it one would invent a calibration it does not have.
        """
        s = self.fallback.score(query)
        p = self.primary.route(query)
        return [(p, 1.0)] + [x for x in s if x[0] != p] if p else s

    def route(self, query: str) -> str | None:
        p = self.primary.route(query)
        if p is not None:
            return p
        top = self.fallback.score(query)
        if not top:  # a lexical arm with no term in common: abstain
            return None
        lab, s = top[0]
        return lab if s >= self.threshold else None


def _top1(arm, queries: list[str]) -> list[tuple[str | None, float]]:
    """Fallback top-1 per query, batching the encode where the arm supports it.

    A lexical arm returns an empty ranking when the request shares no term with
    any label — that is an abstention, not a crash, so it becomes (None, -inf)
    and falls below every threshold.
    """
    if hasattr(arm, "precompute"):
        arm.precompute(queries)
    return [(s[0] if (s := arm.score(q)) else (None, float("-inf"))) for q in queries]


def grid_for(arm, rows: list[dict], n: int = 40) -> list[float]:
    """Thresholds as quantiles of the arm's own top-1 scores, not a fixed band.

    The registered arms disagree about scale by three orders of magnitude —
    measured on family A, BM25-over-training-text tops out at 40.5, cosine at
    0.96 and an RRF fusion at 0.016. A shared 0..1 grid would pin half of them
    at coverage 1.0 and the rest at 0.0 while looking like a sweep.
    """
    import numpy as np
    xs = np.array([s for _, s in _top1(arm, [r["query"] for r in rows])
                   if s != float("-inf")], dtype=float)
    if not len(xs):
        return [0.0]
    qs = np.unique(np.quantile(xs, np.linspace(0.0, 1.0, n)))
    return [float("-inf")] + [float(x) for x in qs]


def curve(fallback: str, rows: list[dict], grid: list[float], primary=None) -> list[dict]:
    """The cascade's coverage/precision/accuracy/abstention at every threshold.

    Computed once from cached top-1 scores rather than by re-running `route` per
    threshold — 90 thresholds x 988 rows would be 90x the encode for identical
    numbers.
    """
    arm = CascadeArm(fallback, 0.0, primary)
    on = [r for r in rows if r.get("label")]
    off = [r for r in rows if not r.get("label")]
    hand_on = [arm.primary.route(r["query"]) for r in on]
    hand_off = [arm.primary.route(r["query"]) for r in off]
    fb_on = _top1(arm.fallback, [r["query"] for r in on])
    fb_off = _top1(arm.fallback, [r["query"] for r in off])
    out = []
    for t in grid:
        ans = 0
        hits = 0
        for r, h, (lab, s) in zip(on, hand_on, fb_on):
            got = h if h is not None else (lab if s >= t else None)
            ans += got is not None
            hits += got == r["label"]
        abst = sum(h is None and s < t for h, (_, s) in zip(hand_off, fb_off))
        out.append({
            "threshold": round(float(t), 4) if t != float("-inf") else None,
            "coverage": round(ans / max(len(on), 1), 4),
            "precision": round(hits / ans, 4) if ans else 0.0,
            "label_acc": round(hits / max(len(on), 1), 4),
            "abstain_acc": round(abst / len(off), 4) if off else None,
        })
    return out


def pick(rows: list[dict]) -> dict:
    ok = [r for r in rows if (r["abstain_acc"] or 0) >= ABSTAIN_FLOOR]
    # Ties broken toward the higher threshold: same accuracy, more abstention
    # headroom on traffic that looks like neither split.
    key = lambda r: (r["label_acc"],
                     r["threshold"] if r["threshold"] is not None else float("-inf"))
    return max(ok or rows, key=key)


def scored_arms() -> list[str]:
    """Registered arms that expose `score()` — whatever the sibling agents shipped."""
    out = []
    for name in sorted(arms_mod.REGISTRY):
        if name.startswith("cascade"):
            continue
        try:
            a = arms_mod.build(name)
        except Exception:  # an arm whose deps are absent is simply not available
            continue
        if hasattr(a, "score"):
            out.append(name)
    return out


def screen(names: list[str], rows: list[dict], n: int = 200) -> dict:
    """Top-1 accuracy and per-query latency on the selection split.

    Only for choosing which of ~25 registered arms to cascade over. Family A is
    the training family for several of them, so a 1.000 here means memorised,
    not good — the cascade curves on B and wild are where that shows.
    """
    on = [r for r in rows if r.get("label")][:n]
    out = {}
    for name in names:
        arm = arms_mod.build(name)
        lat, hit = [], 0
        for r in on:
            t0 = time.perf_counter()
            s = arm.score(r["query"])
            lat.append((time.perf_counter() - t0) * 1000)
            hit += bool(s) and s[0][0] == r["label"]
        out[name] = {"top1_acc": round(hit / len(on), 4),
                     "median_ms": round(statistics.median(lat), 4)}
    return out


def choose_fallbacks(scores: dict) -> list[str]:
    """Every encoder arm, plus the best member of each sibling family.

    `bm25_arms.py` and `spacy_arms.py` register a dozen variants between them;
    cascading over all of them would print a curve nobody reads. One per prefix,
    picked on the selection split only.
    """
    keep = [n for n in scores if n.startswith("enc-") and n.endswith("-open")]
    fams: dict[str, str] = {}
    for n in scores:
        if n.startswith("enc-"):
            continue
        fam = n.split("-")[0]
        if fam not in fams or scores[n]["top1_acc"] > scores[fams[fam]]["top1_acc"]:
            fams[fam] = n
    return keep + sorted(fams.values())


def agreement(fallback: str, rows: list[dict], threshold: float) -> dict:
    """hand ∧ X: coverage and precision where both name the same target.

    `agreement.py` measured this for hand ∧ fitted (0.775 / 0.867 precision at
    ~0.20 coverage). The two arms there share this repo's cue layer and
    catalogue; an encoder shares neither, so if the gate is really buying
    independence it should gate at least as well here.
    """
    on = [r for r in rows if r.get("label")]
    arm = CascadeArm(fallback, threshold)
    hand = [arm.primary.route(r["query"]) for r in on]
    fb = _top1(arm.fallback, [r["query"] for r in on])
    n = len(on)

    def stat(sel):
        return {"coverage": round(len(sel) / n, 4),
                "precision": round(sum(sel) / len(sel), 4) if sel else 0.0,
                "n": len(sel)}

    agree = [r["label"] == h for r, h, (lab, s) in zip(on, hand, fb)
             if h is not None and h == lab]
    agree_t = [r["label"] == h for r, h, (lab, s) in zip(on, hand, fb)
               if h is not None and h == lab and s >= threshold]
    other = [r["label"] == lab for r, (lab, s) in zip(on, fb)]
    return {"hand alone": stat([r["label"] == h for r, h in zip(on, hand) if h is not None]),
            f"{fallback} alone (top-1)": stat(other),
            "hand ∧ arm agree": stat(agree),
            f"hand ∧ arm agree, s>={threshold:.2f}": stat(agree_t)}


register("cascade-enc-fusion", lambda: CascadeArm("enc-fusion-open", 0.33))
register("cascade-enc-centroid", lambda: CascadeArm("enc-centroid-open", 0.47))


def main(argv=None) -> int:
    from eval import load_split, score

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fallbacks", nargs="*", default=None,
                    help="registered scored arms to cascade over (default: screened)")
    args = ap.parse_args(argv)

    arms_mod.load_all()
    rows = {k: load_split(v) for k, v in PATHS.items()}
    available = scored_arms()
    print(f"scored arms registered: {', '.join(available)}")
    if arms_mod.UNAVAILABLE:
        print("unavailable:", json.dumps(arms_mod.UNAVAILABLE))
    scr = screen(available, rows[SELECT_SPLIT])
    fallbacks = args.fallbacks or choose_fallbacks(scr)
    print(f"\ncascading over: {', '.join(fallbacks)}\n")
    print(f"{'scored arm':<22}{'top1 acc (A)':>14}{'ms':>9}")
    for n in available:
        print(f"{n:<22}{scr[n]['top1_acc']:>14.3f}{scr[n]['median_ms']:>9.3f}")
    print()

    out: dict = {"fallbacks": fallbacks, "abstain_floor": ABSTAIN_FLOOR,
                 "select_split": SELECT_SPLIT, "screen": scr,
                 "unavailable": dict(arms_mod.UNAVAILABLE)}

    # ── the curves ──────────────────────────────────────────────────────────
    chosen = {}
    for fb in fallbacks:
        # One grid per arm, quantiles of its own scores on the selection split,
        # then the *same* thresholds applied to B and wild — a threshold is a
        # fixed operating parameter, not something re-derived per split.
        grid = grid_for(arms_mod.build(fb), rows[SELECT_SPLIT])
        out.setdefault("grids", {})[fb] = [None if t == float("-inf") else round(t, 4)
                                           for t in grid]
        for sname in SPLITS:
            out.setdefault("curves", {}).setdefault(fb, {})[sname] = \
                curve(fb, rows[sname], grid)
        thr = pick(out["curves"][fb][SELECT_SPLIT])["threshold"]
        # A `None` threshold is the -inf end of the grid: the arm was selected to
        # answer everything the hand rules did not. Kept, not special-cased —
        # that is the catch-all, and the curve should be allowed to choose it.
        chosen[fb] = float("-inf") if thr is None else thr
    out["chosen_thresholds"] = {k: (None if v == float("-inf") else v)
                                for k, v in chosen.items()}

    for fb in fallbacks:
        print(f"cascade: hand -> {fb}   (threshold = quantiles of {fb} top-1 score)")
        hdr = f"{'thr':>9}" + "".join(f"{s[:6] + ' cov':>13}{'acc':>7}{'abst':>7}"
                                      for s in SPLITS)
        print(hdr)
        print("-" * len(hdr))
        n = len(out["curves"][fb][SELECT_SPLIT])
        for i in range(0, n, max(1, n // 12)):
            r0 = out["curves"][fb][SELECT_SPLIT][i]
            t = r0["threshold"]
            line = f"{'-inf' if t is None else format(t, '.3f'):>9}"
            for sname in SPLITS:
                r = out["curves"][fb][sname][i]
                line += (f"{r['coverage']:>13.3f}{r['label_acc']:>7.3f}"
                         f"{r['abstain_acc']:>7.3f}")
            print(line)
        sel = pick(out["curves"][fb][SELECT_SPLIT])
        print(f"selected on {SELECT_SPLIT} at abstention >= {ABSTAIN_FLOOR}: "
              f"threshold {sel['threshold']}")
        for sname in SPLITS:
            o = pick(out["curves"][fb][sname])
            out.setdefault("oracle", {}).setdefault(fb, {})[sname] = o
            print(f"  oracle-best on {sname:<22} thr {o['threshold']}  "
                  f"acc {o['label_acc']:.3f}  abst {o['abstain_acc']:.3f}  "
                  f"(diagnostic only — chosen on the split it is scored on)")
        print()

    # ── the comparable table, cold instances, eval.py's own scorer ──────────
    hdr = (f"{'arm':<26}{'split':<22}{'cov':>7}{'prec':>7}{'acc':>7}"
           f"{'tool':>7}{'meth':>7}{'abst':>7}{'args':>7}{'ms':>9}")
    print(hdr)
    print("-" * len(hdr))
    table = [("hand (abstains)", lambda: HandRouter()),
             ("hand + catch-all", lambda: HandRouter(fallback=True))]
    for fb in fallbacks:
        table.append((f"{fb} alone", lambda fb=fb: arms_mod.build(fb)))
        table.append((f"cascade -> {fb}",
                      lambda fb=fb: CascadeArm(fb, chosen[fb])))
    for tag, make in table:
        for sname in SPLITS:
            t0 = time.perf_counter()
            arm = make()
            build_ms = (time.perf_counter() - t0) * 1000
            s = score(arm, rows[sname])
            s.pop("errors")
            s["build_ms"] = round(build_ms, 1)
            out.setdefault("arms", {}).setdefault(tag, {})[sname] = s
            f = lambda k: "  -  " if s[k] is None else f"{s[k]:.3f}"
            print(f"{tag:<26}{sname:<22}{f('coverage'):>7}{f('precision'):>7}"
                  f"{f('label_acc'):>7}{f('tool_acc'):>7}"
                  f"{f('method_acc_given_tool'):>7}{f('abstain_acc'):>7}"
                  f"{f('args_acc'):>7}{s['median_latency_ms']:>9.4f}")
        print()

    # ── the agreement gate, against agreement.py's fitted-list baseline ─────
    from router import Router
    print("agreement gate (hand ∧ second opinion)\n")
    hdr = f"{'split':<22}{'gate':<40}{'cov':>8}{'prec':>9}{'n':>6}"
    print(hdr)
    print("-" * len(hdr))
    for sname in SPLITS[1:]:
        on = [r for r in rows[sname] if r.get("label")]
        fitted = Router(HERE / "rules_schema.json")
        hand = HandRouter()
        pairs = [(r["label"], hand.route(r["query"]), fitted.route(r["query"])) for r in on]
        ag = [g == h for g, h, f_ in pairs if h is not None and h == f_]
        row = {"coverage": round(len(ag) / len(on), 4),
               "precision": round(sum(ag) / len(ag), 4) if ag else 0.0, "n": len(ag)}
        out.setdefault("agreement", {}).setdefault(sname, {})["hand ∧ fitted-schema"] = row
        print(f"{sname:<22}{'hand ∧ fitted-schema':<40}{row['coverage']:>8.3f}"
              f"{row['precision']:>9.3f}{row['n']:>6}")
        for fb in fallbacks:
            a = agreement(fb, rows[sname], chosen[fb])
            out["agreement"][sname].update({f"{k} [{fb}]": v for k, v in a.items()})
            for k, v in a.items():
                print(f"{sname:<22}{k.replace('arm', fb):<40}{v['coverage']:>8.3f}"
                      f"{v['precision']:>9.3f}{v['n']:>6}")
        print()

    try:
        import encoder_arms
        out["encoder_load_ms"] = round(encoder_arms.LOAD_MS, 1)
        print(f"encoder load {encoder_arms.LOAD_MS:.0f} ms (once per process, "
              f"amortised over every query)")
    except ImportError:
        pass
    (HERE / "results_cascade.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote results_cascade.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
