#!/usr/bin/env python3
"""One table across both experiments, and paired tests where they are paired.

Every arm here answered the same 62 queries from `needle-bsky/evalset.jsonl` and
was scored by the same code, so each contrast is paired and McNemar's exact test
is the right one. What is *not* matched is stated in RESULTS.md: Needle sees
five tools per turn and Monad sees all eighteen, Needle was LoRA-tuned and Monad
fully fine-tuned, and the two ship at very different sizes.

    python3 compare.py            # table + contrasts
    python3 compare.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _lib.paths import experiment

NEEDLE = experiment("needle-bsky")

# label -> (results file, what it is)
ARMS = [
    ("needle-base", NEEDLE / "results_tuned-min.json", "Needle 2, base weights, 18 declared"),
    ("needle-lora", NEEDLE / "results_finetuned.json", "Needle 2, LoRA on 800 rows, 18 declared"),
    ("needle-2stage", NEEDLE / "results_two_stage_heuristic.json", "Needle 2, regex stage 1 + <=5 tools"),
    ("needle-oracle", NEEDLE / "results_oracle-tuned-min.json", "Needle 2, oracle five-tool catalogue"),
    ("monad-base", HERE / "results_base.json", "Monad 56M, zero-shot, 18 declared"),
    ("monad-e1", HERE / "results_tuned-e1.json", "Monad 56M, full FT 1 epoch"),
    ("monad-e2", HERE / "results_tuned-e2.json", "Monad 56M, full FT 2 epochs"),
    ("monad-e3", HERE / "results_tuned-e3.json", "Monad 56M, full FT 3 epochs"),
]

CONTRASTS = [
    ("needle-base", "monad-e3", "purpose-built base vs fine-tuned generalist"),
    ("needle-lora", "monad-e3", "same 800 rows, each model's own fine-tune path"),
    ("needle-2stage", "monad-e3", "best Needle configuration vs fine-tuned Monad"),
    ("monad-e1", "monad-e3", "does the third epoch buy anything"),
]


def mcnemar(a_rows, b_rows, key="tool_ok"):
    by_a = {r["id"]: r for r in a_rows}
    by_b = {r["id"]: r for r in b_rows}
    ids = [i for i in by_a if i in by_b]
    a_only = sum(1 for i in ids if by_a[i][key] and not by_b[i][key])
    b_only = sum(1 for i in ids if not by_a[i][key] and by_b[i][key])
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p": 1.0, "paired_on": len(ids)}
    k = min(a_only, b_only)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2**n))
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n, "p": round(p, 5), "paired_on": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    loaded = {}
    for label, path, what in ARMS:
        if path.exists():
            d = json.loads(path.read_text())
            loaded[label] = {"summary": d["summary"], "rows": d["rows"], "what": what}

    out = {
        "arms": {k: {"what": v["what"], **v["summary"]} for k, v in loaded.items()},
        "contrasts": [],
    }
    for x, y, why in CONTRASTS:
        if x in loaded and y in loaded:
            m = mcnemar(loaded[x]["rows"], loaded[y]["rows"])
            m.update({"a": x, "b": y, "why": why, "metric": "tool_ok"})
            out["contrasts"].append(m)

    if a.json:
        print(json.dumps(out, indent=1))
        return 0

    print(f"{'arm':15}{'routable':>9}{'args':>7}{'refuse':>8}{'invented':>9}{'parse':>7}{'ms':>8}  what")
    for label, _, what in ARMS:
        if label not in loaded:
            continue
        s = loaded[label]["summary"]
        parse = s.get("parse_ok_rate")
        parse_s = f"{parse:.3f}" if parse is not None else "  n/a"
        print(
            f"{label:15}{s['tool_acc_routable']:9.3f}{s['args_acc_routable']:7.3f}"
            f"{s['refusal_acc']:8.3f}{s['invented_rate']:9.3f}{parse_s:>7}"
            f"{s['median_latency_ms']:8.0f}  {what}"
        )

    print("\nPaired McNemar on tool_ok:")
    for c in out["contrasts"]:
        print(
            f"  {c['a']:14} vs {c['b']:10} {c['why']:44} "
            f"{c['a_only']:2d} / {c['b_only']:2d}  p={c['p']}  (n={c['paired_on']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
