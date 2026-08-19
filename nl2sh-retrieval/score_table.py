#!/usr/bin/env python3
"""Score the four Claude iteration rounds against the four Gemini ones, three splits each.

`gh-mcp-regex-fit/eval.py` prints this table but writes its JSON to the harness's
own `results.json`, which other passes own. This writes to the experiment
directory instead, adds the round-indexed shape the question actually asks for
(round -> split -> accuracy), and runs the paired McNemar contrasts that say
whether a round-to-round move is real — reusing `mcnemar.exact_p` rather than
re-deriving it.

Reads family A, family B and the wild set. The *reviser* only ever saw family A;
this script is the scorer, which is a different job and runs after every round is
frozen.

    python3 score_table.py --out results_claude_iteration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CLAUDE = [("cleanroom", "rules_claude-cleanroom.json"),
          ("iter1", "rules_claude-iter1.json"),
          ("iter2", "rules_claude-iter2.json"),
          ("iter3", "rules_claude-iter3.json"),
          ("iter3-speculative", "rules_claude-iter3-speculative.json")]
GEMINI = [("cleanroom", "rules_gemini-cleanroom.json"),
          ("iter1", "rules_gemini-iter1.json"),
          ("iter2", "rules_gemini-iter2.json"),
          ("iter3", "rules_gemini-iter3.json")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", type=Path, default=Path("../gh-mcp-regex-fit"))
    ap.add_argument("--out", type=Path, default=Path("results_claude_iteration.json"))
    a = ap.parse_args()
    harness = (HERE / a.harness).resolve()
    sys.path.insert(0, str(harness))
    from eval import load_split, score          # noqa: E402
    from gemini_arms import CompiledRouter      # noqa: E402
    from mcnemar import exact_p                 # noqa: E402

    splits = {"family_a": harness / "data" / "family_a.jsonl",
              "family_b": harness / "data" / "family_b.jsonl",
              "wild": harness / "wild.jsonl"}
    rows = {k: load_split(p) for k, p in splits.items()}

    out: dict = {"splits": {k: {"n_routable": sum(1 for r in v if r.get("label")),
                                "n_offtopic": sum(1 for r in v if not r.get("label"))}
                            for k, v in rows.items()},
                 "arms": {}}
    routers = {}
    for author, rounds in (("claude", CLAUDE), ("gemini", GEMINI)):
        for name, fn in rounds:
            key = f"{author}-{name}"
            r = CompiledRouter(harness / fn)
            routers[key] = r
            rec = {"rules_file": fn, "n_rules": len(r.rules),
                   "supervision": r.meta.get("supervision"), "splits": {}}
            for sname, rws in rows.items():
                s = score(r, rws)
                s.pop("errors")
                rec["splits"][sname] = s
            out["arms"][key] = rec

    # Round-to-round contrasts on the two held-out splits, plus the in-sample one.
    contrasts = []
    for author, rounds in (("claude", CLAUDE), ("gemini", GEMINI)):
        names = [f"{author}-{n}" for n, _ in rounds]
        pairs = list(zip(names, names[1:])) + [(names[0], names[-1])]
        for x, y in pairs:
            for sname in ("family_a", "family_b", "wild"):
                on = [r for r in rows[sname] if r.get("label")]
                bx = cy = 0
                for r in on:
                    ox = routers[x].route(r["query"]) == r["label"]
                    oy = routers[y].route(r["query"]) == r["label"]
                    bx += ox and not oy
                    cy += oy and not ox
                contrasts.append({"split": sname, "a": x, "b": y,
                                  "a_only": bx, "b_only": cy, "n": len(on),
                                  "p": exact_p(bx, cy)})
    out["mcnemar"] = contrasts

    (HERE / a.out if not a.out.is_absolute() else a.out).write_text(
        json.dumps(out, indent=1) + "\n")

    w = f"{'arm':<28}{'rules':>6}{'family A':>10}{'family B':>10}{'wild':>8}"
    print(w); print("-" * len(w))
    for key, rec in out["arms"].items():
        print(f"{key:<28}{rec['n_rules']:>6}"
              f"{rec['splits']['family_a']['label_acc']:>10.3f}"
              f"{rec['splits']['family_b']['label_acc']:>10.3f}"
              f"{rec['splits']['wild']['label_acc']:>8.3f}")
    print()
    print(f"{'split':<10}{'contrast':<42}{'a only':>7}{'b only':>7}{'p':>10}")
    for c in contrasts:
        if c["a_only"] or c["b_only"]:
            print(f"{c['split']:<10}{c['a']+' vs '+c['b']:<42}"
                  f"{c['a_only']:>7}{c['b_only']:>7}{c['p']:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
