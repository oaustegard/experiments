#!/usr/bin/env python3
"""Score `extract_params.py` against NL2Bash, with the gold set stated in full.

WHY THIS IS AWKWARD, AND HOW IT IS HANDLED
------------------------------------------
NL2Bash has no parameter annotations. It has (nl, cm) pairs. So there is no
human-labelled answer key for "which spans of this request are parameters", and
any evaluation has to *derive* one. Deriving it badly is the easy way to publish
a flattering number, so the derivation is spelled out here rather than buried:

  PRECISION -- an extracted value is correct if it occurs verbatim in the gold
  command. This is a proxy, and it is wrong in both directions. It over-credits:
  extracting `txt` scores a hit against `"*.txt"` by substring. It under-credits:
  a genuinely correct extraction the reference solution chose not to use (the
  request says "the current directory", the command says `.`) scores a miss.
  A stricter token-equality variant is reported alongside to bound the first.

  RECALL -- the denominator is built from the command, not from the request:
  take the command's *operand* tokens (drop the utility leading each pipeline
  stage, drop `-flags`, drop shell operators), and keep those that occur in the
  request. That is precisely the task's "values that appear in the gold command
  and also appear in the NL". Where an operand does not itself appear in the
  request, it is split on shell metacharacters and its pieces are tested instead
  -- so `s/1\\.2\\.3\\.4/5.6.7.8/g` contributes `5.6.7.8`, which the request does
  contain. Splitting is conditional on the whole token missing, so a path that
  *is* present contributes itself and not four inflating fragments.

  Two recall denominators are reported, and the gap between them is the finding:
    * recall_all     -- every operand present in the request, including bare
                        English words carrying no structural cue at all.
    * recall_marked  -- restricted to operands that are quoted in the request or
                        carry a non-alphabetic character or a digit.
  A conservative extractor cannot reach the first by construction. Reporting
  only the second would hide how much of the copyable content is unmarked, which
  is the number that decides whether a span-copying model needs help here.

TUNE/HOLDOUT
------------
400 pairs sampled with a fixed seed; the first 300 are the development split the
patterns were iterated against, the last 100 are touched once at the end. Both
are reported. `gh-mcp-regex-fit` measured a hand-written rule set drop from
0.984 to 0.239 across phrasing families, so an unheld-out number from this repo
should not be believed.

    python3 eval_extract.py --data path/to/nl2bash/data/bash
"""

from __future__ import annotations

import argparse
import json
import random
import re
import math
from collections import Counter, defaultdict
from pathlib import Path

from extract_params import extract

HERE = Path(__file__).resolve().parent

SEED = 20260819
N_SAMPLE = 400
N_HOLDOUT = 100

# Shell tokens that are syntax, never a parameter value.
_OPERATORS = {
    "|", "||", "&&", ";", "&", ">", ">>", "<", "<<", "2>", "2>&1", "(", ")",
    "{", "}", "{}", "\;", "\\{}", "+", "!", "[", "]", "[[", "]]", "then",
    "do", "done", "fi", "else", "elif", "if", "while", "for", "in", "esac",
}
_STAGE_BREAK = {"|", "||", "&&", ";", "&", "(", ")", "\n"}
_SUBTOKEN_SPLIT = re.compile(r"[\s/,=:;{}()\[\]|<>'\"`]+")


def shell_tokens(cmd: str):
    """Whitespace-split respecting quotes and backslashes.

    Not a shell parser -- NL2Bash contains genuinely unparseable fragments and
    `shlex` raises on them. Quote state is tracked only well enough to keep a
    quoted argument in one piece.
    """
    toks, cur, quote, esc = [], [], None, False
    for ch in cmd:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"`":
            quote = ch
            cur.append(ch)
            continue
        if ch.isspace():
            if cur:
                toks.append("".join(cur))
                cur = []
            continue
        cur.append(ch)
    if cur:
        toks.append("".join(cur))
    return toks


def _strip_quotes(tok: str) -> str:
    while len(tok) >= 2 and tok[0] == tok[-1] and tok[0] in "'\"`":
        tok = tok[1:-1]
    return tok


def gold_operands(cmd: str):
    """Operand tokens of a command: not the stage utility, not a flag."""
    ops = []
    expect_utility = True
    for tok in shell_tokens(cmd):
        bare = tok
        if bare in _STAGE_BREAK or bare in ("|", "||", "&&", ";"):
            expect_utility = True
            continue
        if expect_utility:
            expect_utility = False
            continue  # the utility name itself is not a parameter
        if bare in _OPERATORS:
            continue
        if bare.startswith("-") and len(bare) > 1:
            continue
        val = _strip_quotes(tok).strip()
        if len(val) >= 2:
            ops.append(val)
    return ops


def gold_values(nl: str, cmd: str):
    """Operand strings that are actually present in the request.

    Returns (all_values, marked_values). `marked` keeps only the ones a
    structural extractor could in principle see: quoted in the request, or
    carrying a non-alphabetic character.
    """
    all_vals, marked = set(), set()
    for op in gold_operands(cmd):
        pieces = [op] if op in nl else [
            p for p in _SUBTOKEN_SPLIT.split(op) if len(p) >= 2 and p in nl]
        for p in pieces:
            if p not in nl or len(p) < 2:
                continue
            all_vals.add(p)
            quoted = any(f"{q}{p}{q}" in nl for q in ("'", '"', "`", "“"))
            if quoted or not p.isalpha():
                marked.add(p)
    return all_vals, marked


def score_pair(nl: str, cmd: str):
    got = extract(nl)
    pred = []  # (kind, value) unique
    seen = set()
    for kind, spans in got.items():
        for sp in spans:
            key = (kind, sp["value"])
            if key not in seen:
                seen.add(key)
                pred.append(key)

    tokens_in_cmd = {_strip_quotes(t) for t in shell_tokens(cmd)} | {cmd}
    cmd_lower = cmd.lower()
    hits_sub, hits_tok, hits_ci = [], [], []
    for kind, val in pred:
        hits_sub.append(val in cmd)
        hits_tok.append(any(val == t for t in tokens_in_cmd))
        hits_ci.append(val.lower() in cmd_lower)

    pred_values = {v for _, v in pred}
    all_gold, marked_gold = gold_values(nl, cmd)

    def found(g):
        return any(g == v or g in v for v in pred_values)

    def found_exact(g):
        return g in pred_values

    return {
        "pred": pred,
        "hits_sub": hits_sub,
        "hits_tok": hits_tok,
        "hits_ci": hits_ci,
        "gold_all": all_gold,
        "gold_marked": marked_gold,
        "found_all": {g for g in all_gold if found(g)},
        "found_marked": {g for g in marked_gold if found(g)},
        "found_marked_exact": {g for g in marked_gold if found_exact(g)},
    }


def wilson(hits: int, n: int, z: float = 1.96):
    """95% Wilson score interval. The holdout is 100 pairs; a point estimate
    from it without an interval would invite exactly the over-reading this
    repo's own `gh-mcp-regex-fit` entry warns about."""
    if not n:
        return None
    phat = hits / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def aggregate(rows):
    n_pred = sum(len(r["pred"]) for r in rows)
    n_sub = sum(sum(r["hits_sub"]) for r in rows)
    n_tok = sum(sum(r["hits_tok"]) for r in rows)
    n_ci = sum(sum(r["hits_ci"]) for r in rows)
    g_all = sum(len(r["gold_all"]) for r in rows)
    f_all = sum(len(r["found_all"]) for r in rows)
    g_mk = sum(len(r["gold_marked"]) for r in rows)
    g_un = g_all - g_mk
    f_mk = sum(len(r["found_marked"]) for r in rows)
    f_mk_x = sum(len(r["found_marked_exact"]) for r in rows)
    f_un = f_all - f_mk

    per_kind = defaultdict(lambda: [0, 0])
    for r in rows:
        for (kind, _), hit in zip(r["pred"], r["hits_sub"]):
            per_kind[kind][1] += 1
            per_kind[kind][0] += int(hit)

    def rate(a, b):
        return round(a / b, 4) if b else None

    return {
        "n_pairs": len(rows),
        "n_extracted": n_pred,
        "extracted_per_request": round(n_pred / len(rows), 3) if rows else 0,
        "precision_substring": rate(n_sub, n_pred),
        "precision_substring_ci95": wilson(n_sub, n_pred),
        "precision_substring_caseless": rate(n_ci, n_pred),
        "precision_token_exact": rate(n_tok, n_pred),
        "gold_all": g_all,
        "recall_all": rate(f_all, g_all),
        "recall_all_ci95": wilson(f_all, g_all),
        "gold_marked": g_mk,
        "recall_marked": rate(f_mk, g_mk),
        "recall_marked_exact_match_only": rate(f_mk_x, g_mk),
        "gold_unmarked": g_un,
        "recall_unmarked": rate(f_un, g_un),
        "per_kind": {
            k: {"n": v[1], "precision": rate(v[0], v[1])}
            for k, v in sorted(per_kind.items(), key=lambda kv: -kv[1][1])
        },
    }


def failure_report(rows, top=15):
    fp = Counter()
    fn = Counter()
    fn_unmarked = Counter()
    for r in rows:
        for (kind, val), hit in zip(r["pred"], r["hits_sub"]):
            if not hit:
                fp[(kind, val)] += 1
        for g in r["gold_marked"] - r["found_marked"]:
            fn[g] += 1
        for g in (r["gold_all"] - r["gold_marked"]) - r["found_all"]:
            fn_unmarked[g] += 1
    return {
        "false_positive_kinds": Counter(k for (k, _), c in fp.items() for _ in range(c)).most_common(),
        "top_false_positives": [{"kind": k, "value": v, "n": c} for (k, v), c in fp.most_common(top)],
        "top_missed_marked": [{"value": v, "n": c} for v, c in fn.most_common(top)],
        "top_missed_unmarked": [{"value": v, "n": c} for v, c in fn_unmarked.most_common(top)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="nl2bash/data/bash",
                    help="directory holding all.nl and all.cm (default: %(default)s)")
    ap.add_argument("--out", default=str(HERE / "results_extract.json"))
    ap.add_argument("--show-failures", type=int, default=0,
                    help="print N dev-split rows with their misses, for tuning")
    args = ap.parse_args()

    data = Path(args.data)
    nls = (data / "all.nl").read_text(encoding="utf-8", errors="replace").splitlines()
    cms = (data / "all.cm").read_text(encoding="utf-8", errors="replace").splitlines()
    assert len(nls) == len(cms), (len(nls), len(cms))

    idx = random.Random(SEED).sample(range(len(nls)), N_SAMPLE)
    dev_idx, hold_idx = idx[: N_SAMPLE - N_HOLDOUT], idx[N_SAMPLE - N_HOLDOUT:]

    def run(indices):
        return [score_pair(nls[i], cms[i]) for i in indices]

    dev, hold = run(dev_idx), run(hold_idx)
    allr = dev + hold

    if args.show_failures:
        shown = 0
        for i in dev_idx:
            r = score_pair(nls[i], cms[i])
            bad = [(k, v) for (k, v), h in zip(r["pred"], r["hits_sub"]) if not h]
            miss = r["gold_marked"] - r["found_marked"]
            if not bad and not miss:
                continue
            print(f"[{i}] {nls[i]}\n  CMD {cms[i]}")
            if bad:
                print(f"  FP  {bad}")
            if miss:
                print(f"  FN  {sorted(miss)}")
            print()
            shown += 1
            if shown >= args.show_failures:
                break
        return 0

    # How often does the request even contain everything the command needs?
    # The analogue of gh-mcp-regex-fit's cue-presence probe.
    complete = sum(
        1 for i in idx
        if gold_operands(cms[i]) and all(
            o in nls[i] or any(p in nls[i] for p in _SUBTOKEN_SPLIT.split(o) if len(p) >= 2)
            for o in gold_operands(cms[i])))
    with_ops = sum(1 for i in idx if gold_operands(cms[i]))

    out = {
        "seed": SEED,
        "n_sample": N_SAMPLE,
        "splits": {"dev": N_SAMPLE - N_HOLDOUT, "holdout": N_HOLDOUT},
        "dev": aggregate(dev),
        "holdout": aggregate(hold),
        "all400": aggregate(allr),
        "cue_presence": {
            "pairs_with_operands": with_ops,
            "pairs_where_every_operand_appears_in_request": complete,
            "fraction": round(complete / with_ops, 4) if with_ops else None,
        },
        "failures_dev": failure_report(dev),
        "failures_holdout": failure_report(hold),
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("dev", "holdout", "all400", "cue_presence")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
