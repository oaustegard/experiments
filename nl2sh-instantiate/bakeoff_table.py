#!/usr/bin/env python3
"""Score several runs over the rows they all share.

`score.py` scores one file at a time over whatever rows that file holds. A
bake-off needs the stricter thing: a 4B arm that ran 60 rows before its budget
ran out and a 270M arm that ran all 179 are not comparable until both are read
on the same rows. This restricts every run to the intersection of their row
keys, then hands each slice to `score.score` so the columns are the ones
`score.py` already defines and nothing is reimplemented.

The intersection is printed with the table. A bake-off row computed over 60
rows and one computed over 179 do not go in the same table without that number
next to them.

    python3 bakeoff_table.py results_it_generate.json results_4b_generate.json
    python3 bakeoff_table.py --all-rows results_*_generate.json --out bakeoff.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import score as S  # noqa: E402


def key(row: dict) -> tuple:
    return (row.get("nl"), row.get("cmd") or row.get("gold_cmd"))


def load(p: Path, all_rows: bool) -> tuple[dict, dict]:
    d = json.loads(p.read_text())
    rows = d["rows"] if all_rows else [r for r in d["rows"] if not r.get("names_utility")]
    return d["summary"], {key(r): r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--all-rows", action="store_true",
                    help="score every row; default is the leak-free subset")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    loaded = [(p,) + load(p, a.all_rows) for p in a.results]
    common = set.intersection(*(set(m) for _, _, m in loaded))
    if not common:
        print("no rows are common to every run", file=sys.stderr)
        return 1
    order = [k for k in loaded[0][2] if k in common]

    table, inferred_dtype = {}, []
    for p, summ, by_key in loaded:
        rows = [by_key[k] for k in order]
        # Runs made before `--dtype` existed carry no dtype key. Every one of
        # them was float32 — `run_gen.py` hardcoded it and RESULTS.md says so —
        # but the label is inferred rather than read, so the footnote names which
        # rows it was inferred for.
        precision = summ.get("quant") or summ.get("dtype")
        if precision is None:
            precision, inferred = "float32", True
        else:
            inferred = False
        label = f"{Path(summ['model']).name}/{precision}/{summ['condition']}"
        if inferred:
            inferred_dtype.append(label)
        table[label] = {
            **S.score(rows),
            "runner": summ.get("runner", "transformers"),
            "decode_tok_per_s": summ.get("decode_tok_per_s"),
            "mean_seconds": summ.get("mean_seconds"),
            "rows_in_file": summ.get("n"),
            "source": p.name,
        }

    first = loaded[0][2]
    c = Counter(first[k]["utility"] for k in order)
    prior = {"n": len(order),
             "most_common_utility": c.most_common(1)[0][0],
             "constant_utility_prior": round(c.most_common(1)[0][1] / len(order), 3),
             "distinct_utilities": len(c)}
    out = {"slice": "all rows" if a.all_rows else "leak-free",
           "common_rows": len(order),
           "rows_per_file": {p.name: len(m) for p, _, m in loaded},
           "prior": prior, "runs": table,
           "dtype_inferred_float32": inferred_dtype}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")

    cols = ["routing", "routing_nondegenerate", "utility_mentioned", "bullet_echo",
            "degenerate", "literal_reproduction", "gold_arg_recall", "exact"]
    w = max(len(k) for k in table) + 2
    print(f"{out['slice']}, common to all {len(loaded)} runs: n={len(order)} "
          f"(files hold {', '.join(str(len(m)) for _, _, m in loaded)})  "
          f"constant-utility prior {prior['constant_utility_prior']:.3f} "
          f"({prior['most_common_utility']}), {prior['distinct_utilities']} distinct\n")
    print(f"{'run':<{w}}" + "".join(f"{c[:13]:>15}" for c in cols) + f"{'tok/s':>9}")
    print("-" * (w + 15 * len(cols) + 9))
    for k, v in table.items():
        print(f"{k:<{w}}"
              + "".join(f"{'  -  ':>15}" if v[c] is None else f"{v[c]:>15.3f}" for c in cols)
              + (f"{v['decode_tok_per_s']:>9.1f}" if v["decode_tok_per_s"] else f"{'  -  ':>9}"))
    if inferred_dtype:
        print(f"\nfloat32 inferred, not recorded, for: {', '.join(inferred_dtype)} "
              "(these predate the --dtype flag; run_gen.py hardcoded float32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
