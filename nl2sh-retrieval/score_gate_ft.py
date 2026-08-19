#!/usr/bin/env python3
"""Score the fine-tuned gate against the baselines that make its number mean something.

The raw `utility_acc` from `pleias_gate.py` is not interpretable on this sample,
for the reason the retrieval verification pass already caught once: **27 of the
40 gate rows have `find` as their gold utility**, so a model that answers "find"
every single time scores **0.675**. Any fine-tune number below that is worse than
a constant, and a number modestly above it may be nothing but the skew.

Three slices are therefore reported, and only the second and third carry a claim:

* **all 40** — comparable to the base model's 0.025, and to the constant prior.
* **non-find (n=13)** — where routing is actually being measured.
* **uncontaminated** — NL2Bash contains duplicate English strings, so 2 of the 40
  gate rows also appear in the fine-tune's training set despite the split being
  drawn by index. Those two are excluded.

    python3 score_gate_ft.py --base results_gate.json --ft results_gate_ft.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def slices(rows: list[dict], contaminated: set[int]) -> dict:
    def acc(rs):
        return round(sum(r["utility_ok"] for r in rs) / len(rs), 3) if rs else None

    nonfind = [r for r in rows if r["gold_utility"] != "find"]
    clean = [r for i, r in enumerate(rows, 1) if i not in contaminated]
    clean_nonfind = [r for r in clean if r["gold_utility"] != "find"]
    return {
        "all": {"n": len(rows), "utility_acc": acc(rows),
                "command_rate": round(sum(bool(r["command"]) for r in rows) / len(rows), 3),
                "verbatim": round(sum(r["verbatim"] for r in rows) / len(rows), 3)},
        "non_find": {"n": len(nonfind), "utility_acc": acc(nonfind)},
        "uncontaminated": {"n": len(clean), "utility_acc": acc(clean)},
        "uncontaminated_non_find": {"n": len(clean_nonfind), "utility_acc": acc(clean_nonfind)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=HERE / "results_gate.json")
    ap.add_argument("--ft", type=Path, default=HERE / "results_gate_ft.json")
    ap.add_argument("--contaminated", type=Path,
                    default=HERE / "data" / "gate_contaminated_rows.json")
    ap.add_argument("--out", type=Path, default=HERE / "results_gate_comparison.json")
    a = ap.parse_args()

    contaminated = set(json.loads(a.contaminated.read_text())) if a.contaminated.is_file() else set()
    base = json.loads(a.base.read_text())["rows"]
    ft = json.loads(a.ft.read_text())["rows"]

    n_find = sum(r["gold_utility"] == "find" for r in base)
    out = {
        "contaminated_rows_excluded": sorted(contaminated),
        "constant_prior_always_find": {
            "n": len(base), "n_find_golds": n_find,
            "utility_acc_all": round(n_find / len(base), 3),
            "utility_acc_non_find": 0.0,
            "why": "27 of 40 golds are `find`; a constant answer scores this without routing",
        },
        "base_model": slices(base, contaminated),
        "fine_tuned": slices(ft, contaminated),
    }
    a.out.write_text(json.dumps(out, indent=1) + "\n")

    print(f"{'slice':<28}{'base':>10}{'fine-tuned':>13}{'always-find':>14}")
    print("-" * 65)
    for key, prior in (("all", out["constant_prior_always_find"]["utility_acc_all"]),
                       ("non_find", 0.0),
                       ("uncontaminated", None),
                       ("uncontaminated_non_find", 0.0)):
        b, f = out["base_model"][key], out["fine_tuned"][key]
        ps = "  -  " if prior is None else f"{prior:.3f}"
        label = "%s (n=%d)" % (key, b["n"])
        print(f"{label:<28}{b['utility_acc']:>10.3f}"
              f"{f['utility_acc']:>13.3f}{ps:>14}")
    print(f"\ncommand rate  base {out['base_model']['all']['command_rate']:.3f} "
          f"-> fine-tuned {out['fine_tuned']['all']['command_rate']:.3f}")
    print(f"verbatim rate base {out['base_model']['all']['verbatim']:.3f} "
          f"-> fine-tuned {out['fine_tuned']['all']['verbatim']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
