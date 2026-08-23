#!/usr/bin/env python3
"""Score a run past `utility_ok`, which is the metric issue #52 opens by distrusting.

`utility_ok` reads the leading token. Stage 1 reported 0.250 end-to-end routing
under it while a read of all 41 outputs put genuinely-runnable-and-correct nearer
0.05. Four cheap metrics sit between those two numbers and none of them needs a
sandbox:

* **routing** — `utility_ok`, carried forward so the columns line up with stage 1.
* **degenerate** — the token-repeat loop (`mv -t X /usr/ -f X /usr/ -f …`) that
  routes correctly and does nothing. Counted as a repeated adjacent token or a
  bigram appearing three times or more.
* **literal reproduction** — of the values `extract_params.py` lifted out of the
  request, how many come back verbatim in the command. This is the argument
  axis the vocabulary argument in `nl2sh-selfhist/MODELS.md` is about, measured
  directly instead of inferred from routing.
* **gold argument recall** — of the gold command's tokens after argv[0], how many
  appear in the prediction. Sensitive to flags and operands the request implies
  but does not spell out, which literal reproduction cannot see.
* **exact** — whitespace-normalised string equality with the gold. A floor, not
  a target: many requests have more than one right answer.

The constant-utility prior is printed beside every routing number, per
`nl2sh-retrieval/score_gate_ft.py`: on this eval the most common gold utility
carries a small share, but the number is only interpretable next to it.

    python3 score.py results_it_generate.json results_it_instantiate.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def tokens(cmd: str) -> list[str]:
    return cmd.split()


# A chunk starting with a letter, repeated three times or more inside one token:
# `my_my_my…`, `user.html.html.html…`. Anchoring on a letter is deliberate —
# without it the rule fires on `100.100.100.4` and `8.8.8.8`, where the repetition
# is an address rather than a loop.
INTRA_TOKEN = re.compile(r"([A-Za-z][\w.\-]{1,9}?)\1{2,}")


def degenerate(cmd: str) -> bool:
    """A token-repeat loop: routes to the right utility and does nothing.

    Two rules, because the first version missed half of them. Across tokens:
    a repeated adjacent token, or a bigram appearing three times
    (`mv -t X /usr/ -f X /usr/ -f`). Within one token: `INTRA_TOKEN`
    (`apt -f -o my_my_my_my…`), which the token-level rule cannot see because the
    whole loop is one whitespace-delimited word. Adding the second rule moved
    the fine-tuned generate arm 0.183 -> 0.183 and the instantiate arm
    0.024 -> 0.049, so it costs the arm it was found on, not the other one.
    """
    if INTRA_TOKEN.search(cmd or ""):
        return True
    t = tokens(cmd)
    if len(t) < 3:
        return False
    if any(t[i] == t[i + 1] for i in range(len(t) - 1)):
        return True
    bigrams = Counter(zip(t, t[1:]))
    return bool(bigrams) and max(bigrams.values()) >= 3


def literal_hits(row: dict) -> tuple[int, int]:
    lits = row.get("literals") or []
    cmd = row.get("command") or ""
    return sum(1 for v in lits if v in cmd), len(lits)


def gold_arg_hits(row: dict) -> tuple[int, int]:
    gold = tokens(row.get("gold_cmd") or "")[1:]
    pred = set(tokens(row.get("command") or ""))
    return sum(1 for g in gold if g in pred), len(gold)


def bullet_echo(row: dict) -> bool:
    """The output is a source-line, not a command.

    Under the instantiation prompts the model answers in the shape of the block
    it was shown — `- john — Crack password hashes: john path/to/hashes.txt` —
    which `gold_utility` reads as the utility `-`. Counting it separately keeps
    the routing column strictly comparable with stage 1 while naming how much of
    the loss is this one behaviour.
    """
    return (row.get("command") or "").lstrip().startswith(("- ", "-\t", "* "))


def utility_mentioned(row: dict) -> bool:
    """Does the gold utility appear as a token anywhere in the raw generation.

    Knowing which utility to reach for and emitting a runnable command are
    different abilities — `monad-bsky` measured routing and transcription
    separately for the same reason (METHODS.md, `copy_probe.py`). Applied
    identically to every condition, so it is a comparison and not a rescue.
    """
    raw = row.get("raw") or row.get("command") or ""
    return re.search(r"(?:^|[^A-Za-z0-9_.-])" + re.escape(row["utility"]) + r"(?:$|[^A-Za-z0-9_.-])", raw) is not None


def norm(cmd: str) -> str:
    return re.sub(r"\s+", " ", (cmd or "").strip())


def score(rows: list[dict]) -> dict:
    lit_h = lit_n = arg_h = arg_n = 0
    rows_with_lits = 0
    for r in rows:
        h, n = literal_hits(r)
        lit_h += h
        lit_n += n
        rows_with_lits += bool(n)
        h, n = gold_arg_hits(r)
        arg_h += h
        arg_n += n
    n = len(rows)
    return {
        "n": n,
        "routing": round(sum(r["utility_ok"] for r in rows) / n, 3),
        "command_rate": round(sum(bool(r["command"]) for r in rows) / n, 3),
        "degenerate": round(sum(degenerate(r.get("command") or "") for r in rows) / n, 3),
        "routing_nondegenerate": round(
            sum(r["utility_ok"] and not degenerate(r.get("command") or "") for r in rows) / n, 3),
        "exact": round(sum(norm(r.get("command")) == norm(r.get("gold_cmd")) for r in rows) / n, 3),
        "bullet_echo": round(sum(bullet_echo(r) for r in rows) / n, 3),
        "utility_mentioned": round(sum(utility_mentioned(r) for r in rows) / n, 3),
        "rows_with_literals": rows_with_lits,
        "literal_reproduction": round(lit_h / lit_n, 3) if lit_n else None,
        "gold_arg_recall": round(arg_h / arg_n, 3) if arg_n else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--all-rows", action="store_true",
                    help="score every row; default is the leak-free subset")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    table, prior = {}, None
    for p in a.results:
        d = json.loads(p.read_text())
        rows = d["rows"] if a.all_rows else [r for r in d["rows"] if not r.get("names_utility")]
        # dtype joins the key only when the run recorded one, so files made
        # before `--dtype` existed keep the labels they have always had. Without
        # it two runs of one model and condition at different precisions collide
        # and the first is silently dropped — and dtype is not cosmetic here:
        # 270M float32 vs bfloat16 on identical rows and seed differ by 0.073
        # routing, which is larger than several effects this file is used to read.
        sm = d["summary"]
        key = f"{Path(sm['model']).name}/{sm['condition']}"
        if sm.get("quant") or sm.get("dtype"):
            key += f"/{sm.get('quant') or sm['dtype']}"
        if key in table:
            raise SystemExit(
                f"two runs share the key {key!r}; refusing to drop one silently. "
                "Give them distinguishable summaries or score them separately.")
        table[key] = {**score(rows), **{
            "decode_tok_per_s": d["summary"].get("decode_tok_per_s"),
            "mean_seconds": d["summary"].get("mean_seconds"),
            "mean_prompt_tokens": d["summary"].get("mean_prompt_tokens"),
        }}
        c = Counter(r["utility"] for r in rows)
        prior = {"n": len(rows), "most_common_utility": c.most_common(1)[0][0],
                 "constant_utility_prior": round(c.most_common(1)[0][1] / len(rows), 3),
                 "distinct_utilities": len(c)}

    out = {"slice": "all rows" if a.all_rows else "leak-free", "prior": prior, "runs": table}
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n")

    cols = ["routing", "routing_nondegenerate", "utility_mentioned", "bullet_echo",
            "degenerate", "literal_reproduction", "gold_arg_recall", "exact"]
    w = max(len(k) for k in table) + 2
    print(f"{out['slice']}  n={prior['n']}  constant-utility prior "
          f"{prior['constant_utility_prior']:.3f} ({prior['most_common_utility']}), "
          f"{prior['distinct_utilities']} distinct utilities\n")
    print(f"{'run':<{w}}" + "".join(f"{c[:13]:>15}" for c in cols))
    print("-" * (w + 15 * len(cols)))
    for k, v in table.items():
        print(f"{k:<{w}}" + "".join(
            f"{'  -  ':>15}" if v[c] is None else f"{v[c]:>15.3f}" for c in cols))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
