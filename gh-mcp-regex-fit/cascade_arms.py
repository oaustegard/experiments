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

import json
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
        lab, s = self.fallback.score(query)[0]
        return lab if s >= self.threshold else None


def _top1(arm, queries: list[str]) -> list[tuple[str, float]]:
    """Fallback top-1 per query, batching the encode where the arm supports it."""
    if hasattr(arm, "precompute"):
        arm.precompute(queries)
    return [arm.score(q)[0] for q in queries]


def curve(fallback: str, rows: list[dict], grid, primary=None) -> list[dict]:
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
            "threshold": round(float(t), 3),
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
    return max(ok or rows, key=lambda r: (r["label_acc"], r["threshold"]))


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


def main() -> int:
    import numpy as np

    from eval import load_split, score

    arms_mod.load_all()
    rows = {k: load_split(v) for k, v in PATHS.items()}
    fallbacks = [n for n in scored_arms() if n.endswith("-open") or "enc-" not in n]
    if not fallbacks:
        raise SystemExit("no scored arm registered; run encoder_arms.py first")
    print(f"scored arms available: {', '.join(scored_arms())}")
    print(f"cascading over: {', '.join(fallbacks)}")
    if arms_mod.UNAVAILABLE:
        print("unavailable:", json.dumps(arms_mod.UNAVAILABLE))
    print()

    out: dict = {"fallbacks": fallbacks, "abstain_floor": ABSTAIN_FLOOR,
                 "select_split": SELECT_SPLIT, "unavailable": dict(arms_mod.UNAVAILABLE)}
    grid = np.round(np.arange(0.0, 0.95, 0.01), 3)

    # ── the curves ──────────────────────────────────────────────────────────
    chosen = {}
    for fb in fallbacks:
        for sname in SPLITS:
            c = curve(fb, rows[sname], grid)
            out.setdefault("curves", {}).setdefault(fb, {})[sname] = c
        chosen[fb] = pick(out["curves"][fb][SELECT_SPLIT])["threshold"]
    out["chosen_thresholds"] = chosen

    for fb in fallbacks:
        print(f"cascade: hand -> {fb}, threshold sweep")
        hdr = f"{'thr':>6}" + "".join(f"{s[:6] + ' cov':>12}{'acc':>7}{'abst':>7}" for s in SPLITS)
        print(hdr)
        print("-" * len(hdr))
        by = {s: {r["threshold"]: r for r in out["curves"][fb][s]} for s in SPLITS}
        for t in grid:
            t = round(float(t), 3)
            if abs(t * 100 - round(t * 100)) > 1e-6 or round(t * 100) % 5:
                continue  # print every 0.05, the JSON keeps every 0.01
            line = f"{t:>6.2f}"
            for s in SPLITS:
                r = by[s][t]
                line += f"{r['coverage']:>12.3f}{r['label_acc']:>7.3f}{r['abstain_acc']:>7.3f}"
            print(line)
        print(f"selected on {SELECT_SPLIT} at abstention >= {ABSTAIN_FLOOR}: "
              f"threshold {chosen[fb]:.2f}")
        for sname in SPLITS:
            o = pick(out["curves"][fb][sname])
            out.setdefault("oracle", {}).setdefault(fb, {})[sname] = o
            print(f"  oracle-best on {sname:<22} thr {o['threshold']:.2f}  "
                  f"acc {o['label_acc']:.3f}  abst {o['abstain_acc']:.3f}  "
                  f"(diagnostic only — chosen on the test split)")
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
