#!/usr/bin/env python3
"""Compiled rules in front, the live model behind — and what the front tier saves.

The deployment question the first pass could not answer: *what fraction of
requests does a microsecond layer settle, so the model never runs?* Both halves
now exist, and the live responses are cached, so this joins them per row rather
than re-inferring.

Reported per split:

* what the front tier alone gets, and at what coverage
* what the model alone gets
* the cascade: front tier where it fires, model on the remainder
* **calls saved** — the fraction of requests that never reach the model, which
  is the number that decides whether the front tier is worth compiling

    python3 cascade_live.py --split wild --front compiled-gemini-iter2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arms as arms_mod
from eval import load_split
from live_eval import SPLITS, subsample

HERE = Path(__file__).resolve().parent


def predictions(arm, rows: list[dict]) -> list[str | None]:
    return [arm.route(r["query"]) for r in rows]


def summarise(name: str, pred: list[str | None], rows: list[dict]) -> dict:
    on = [(p, r) for p, r in zip(pred, rows) if r.get("label")]
    off = [(p, r) for p, r in zip(pred, rows) if not r.get("label")]
    ans = [(p, r) for p, r in on if p is not None]
    hits = sum(p == r["label"] for p, r in on)
    return {"arm": name, "n": len(on),
            "coverage": len(ans) / len(on) if on else None,
            "precision": sum(p == r["label"] for p, r in ans) / len(ans) if ans else None,
            "label_acc": hits / len(on) if on else None,
            "abstain_acc": sum(p is None for p, _ in off) / len(off) if off else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="wild", choices=list(SPLITS))
    ap.add_argument("--front", default="compiled-gemini-iter2")
    ap.add_argument("--per-label", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, default=HERE / "results_cascade_live.json")
    a = ap.parse_args()

    arms_mod.load_all()
    from gemini_arms import GeminiLiveArm
    from gemini_client import DEFAULT_MODEL

    rows = subsample(load_split(SPLITS[a.split]), a.per_label, a.seed)
    if a.front in ("hand", "hand+fallback"):
        from handwritten import HandRouter
        front = HandRouter(fallback=a.front.endswith("fallback"))
    else:
        front = arms_mod.build(a.front)
    live = GeminiLiveArm(a.model or DEFAULT_MODEL)

    p_front = predictions(front, rows)
    p_live = predictions(live, rows)   # cache hits when live_eval ran the same rows
    p_casc = [f if f is not None else l for f, l in zip(p_front, p_live)]

    on = [r for r in rows if r.get("label")]
    fired = sum(1 for f, r in zip(p_front, rows) if r.get("label") and f is not None)
    out = {"split": a.split, "n_rows": len(rows), "front": a.front,
           "model": live.model,
           "calls_saved_routable": fired / len(on) if on else None,
           "calls_saved_all": sum(1 for f in p_front if f is not None) / len(rows),
           "arms": [summarise(a.front, p_front, rows),
                    summarise("live", p_live, rows),
                    summarise(f"cascade {a.front} -> live", p_casc, rows)]}

    for s in out["arms"]:
        ab = s["abstain_acc"]
        ab_s = "  -  " if ab is None else f"{ab:.3f}"
        print(f"{s['arm']:<40} cov {s['coverage']:.3f}  prec {s['precision']:.3f}  "
              f"acc {s['label_acc']:.3f}  abst {ab_s}")
    print(f"model calls avoided: {out['calls_saved_all']:.1%} of all requests")

    prev = json.loads(a.out.read_text()) if a.out.is_file() else {}
    prev[f"{a.split}|{a.front}"] = out
    a.out.write_text(json.dumps(prev, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
