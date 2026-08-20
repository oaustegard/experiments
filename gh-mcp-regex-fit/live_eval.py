#!/usr/bin/env python3
"""The live-model arm: what per-query inference gets, as a ceiling for the rules.

Every arm before this one is deterministic and costs microseconds. This one asks
the model on every request, which is the thing a compiled rule set is trying to
approximate — and the escalation target a cascade would hand its abstentions to.

Two mechanics worth knowing:

* **Prefetch, then score.** `eval.score` walks rows serially, so scoring a
  network arm directly would serialise 988 round trips. Prompts are prefetched
  through `generate_many` at the module's fixed concurrency of 2 (see
  `gemini_client`), and the scoring pass then reads the disk cache. That means
  the `ms` column for this arm is a cache-hit number and is meaningless;
  wall-clock per call is reported separately and is the honest latency figure.
* **Subsampling is declared, not silent.** Family A and B are 988 rows each.
  `--per-label` caps rows per target with a fixed seed, and the sample size is
  printed and written into the results file.

    python3 live_eval.py --split wild
    python3 live_eval.py --split b --per-label 2
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from eval import load_split, score
from gemini_arms import GeminiLiveArm, LIVE_PROMPT

HERE = Path(__file__).resolve().parent
SPLITS = {"a": HERE / "data" / "family_a.jsonl",
          "b": HERE / "data" / "family_b.jsonl",
          "wild": HERE / "wild.jsonl"}


def subsample(rows: list[dict], per_label: int | None, seed: int) -> list[dict]:
    if not per_label:
        return rows
    rng = random.Random(seed)
    by: dict[str | None, list[dict]] = {}
    for r in rows:
        by.setdefault(r.get("label"), []).append(r)
    out = []
    for lab in sorted(by, key=lambda x: (x is None, x)):
        pool = by[lab]
        out += rng.sample(pool, min(per_label, len(pool)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="wild", choices=list(SPLITS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--per-label", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--out", type=Path, default=HERE / "results_live.json")
    a = ap.parse_args()

    from gemini_client import DEFAULT_MODEL, generate_many
    model = a.model or DEFAULT_MODEL
    arm = GeminiLiveArm(model)
    rows = subsample(load_split(SPLITS[a.split]), a.per_label, a.seed)
    print(f"{model} live on split {a.split}: {len(rows)} rows "
          f"({sum(1 for r in rows if r.get('label'))} routable)")

    prompts = [LIVE_PROMPT.format(catalogue=arm.cat, query=r["query"]) for r in rows]
    t0 = time.perf_counter()
    got = generate_many(prompts, model=model, thinking_budget=0,
                        max_output_tokens=256, response_json=True)
    wall = time.perf_counter() - t0
    n_fail = sum(1 for g in got if g is None)
    print(f"prefetch: {wall:.1f}s wall, {wall / max(1, len(rows)) * 1000:.0f} ms/call "
          f"at concurrency 2, {n_fail} failed")

    s = score(arm, rows)          # cache hits; latency column is not meaningful
    errors = s.pop("errors")
    blob = {"model": model, "split": a.split, "n_rows": len(rows),
            "per_label": a.per_label, "seed": a.seed,
            "prefetch_wall_s": round(wall, 1),
            "ms_per_call_at_conc2": round(wall / max(1, len(rows)) * 1000, 1),
            "n_prefetch_failed": n_fail, **s}
    prev = json.loads(a.out.read_text()) if a.out.is_file() else {}
    prev[f"{model}|{a.split}"] = blob
    prev[f"{model}|{a.split}|errors"] = errors[:40]
    a.out.write_text(json.dumps(prev, indent=1) + "\n")
    for k in ("coverage", "precision", "label_acc", "tool_acc",
              "method_acc_given_tool", "abstain_acc", "args_acc"):
        v = blob.get(k)
        print(f"  {k:<22} {'-' if v is None else f'{v:.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
