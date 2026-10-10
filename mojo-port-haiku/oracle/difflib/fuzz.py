"""Differential oracle for a Mojo port of difflib.SequenceMatcher.

The candidate is a package directory `mdifflib/` exposing `SequenceMatcher`,
a drop-in for `difflib.SequenceMatcher` (same constructor, same public
methods and attributes, including `b2j`, `bjunk`, `bpopular`).

Each case builds both matchers on the same inputs and compares
get_matching_blocks, get_opcodes, ratio, quick_ratio, real_quick_ratio,
find_longest_match over random ranges, get_grouped_opcodes, the junk and
popular sets, and set_seq1/set_seq2 reuse. Inputs: strings over small and
large alphabets, lists of lines, tuples of ints, lists of mixed hashables;
isjunk None / a predicate / IS_CHARACTER_JUNK; autojunk on and off; lengths
up to 1500 so the 200-element autojunk threshold and the popular-element
pruning both fire.

    python3 oracle/difflib/fuzz.py --pkg DIR [--split dev|test] [--n 600]

Prints one JSON line with pass/total and the first mismatches.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import json
import random
import sys
from pathlib import Path

SEEDS = {"dev": 1001, "test": 2002}


def rand_seq(rng: random.Random):
    kind = rng.choice(["str_small", "str_large", "lines", "ints", "mixed", "str_small"])
    n = rng.choice([0, 1, 2, 5, 20, 80, 199, 200, 201, 400, 900, 1500])
    if kind == "str_small":
        return "".join(rng.choice("ab c\t") for _ in range(n)), kind
    if kind == "str_large":
        return "".join(chr(rng.randint(32, 0x24F)) for _ in range(n)), kind
    if kind == "lines":
        pool = [f"line {i}\n" for i in range(rng.randint(1, 60))] + ["\n", "  # \n", "}\n"]
        return [rng.choice(pool) for _ in range(n // 4)], kind
    if kind == "ints":
        return tuple(rng.randint(0, rng.choice([2, 10, 1000])) for _ in range(n)), kind
    pool = [1, 2.0, "x", b"y", (1, 2), None, True, "z", 3]
    return [rng.choice(pool) for _ in range(n // 3)], kind


def mutate(rng: random.Random, seq):
    items = list(seq)
    for _ in range(rng.randint(0, max(1, len(items) // 5))):
        if not items or rng.random() < 0.3:
            items.insert(rng.randint(0, len(items)), items[rng.randrange(len(items))] if items else "q")
        elif rng.random() < 0.5:
            del items[rng.randrange(len(items))]
        else:
            i = rng.randrange(len(items))
            items[i] = items[rng.randrange(len(items))]
    if isinstance(seq, str):
        return "".join(items)
    return type(seq)(items)


def junk_for(rng: random.Random, kind: str):
    choice = rng.choice(["none", "none", "pred", "charjunk"])
    if choice == "none":
        return None, "None"
    if choice == "charjunk" and kind in ("str_small", "str_large"):
        return difflib.IS_CHARACTER_JUNK, "IS_CHARACTER_JUNK"
    if kind == "lines":
        return difflib.IS_LINE_JUNK, "IS_LINE_JUNK"
    target = {"str_small": " ", "str_large": "a", "ints": 0, "mixed": None}.get(kind, " ")
    return (lambda x, t=target: x == t), f"x == {target!r}"


def observe(sm, rng: random.Random, la: int, lb: int) -> dict:
    out = {
        "blocks": [tuple(m) for m in sm.get_matching_blocks()],
        "opcodes": sm.get_opcodes(),
        "ratio": sm.ratio(),
        "quick_ratio": sm.quick_ratio(),
        "real_quick_ratio": sm.real_quick_ratio(),
        "grouped": [list(g) for g in sm.get_grouped_opcodes(rng.randint(0, 4))],
        "bjunk": sorted(map(repr, sm.bjunk)),
        "bpopular": sorted(map(repr, sm.bpopular)),
    }
    lm = []
    for _ in range(4):
        alo = rng.randint(0, la)
        ahi = rng.randint(alo, la)
        blo = rng.randint(0, lb)
        bhi = rng.randint(blo, lb)
        lm.append(tuple(sm.find_longest_match(alo, ahi, blo, bhi)))
    lm.append(tuple(sm.find_longest_match()))
    out["longest"] = lm
    return out


def run_case(cand_cls, i: int, seed: int) -> tuple[bool, dict]:
    rng = random.Random(seed * 100003 + i)
    a, kind = rand_seq(rng)
    if rng.random() < 0.8:
        b = mutate(rng, a)
    else:
        # An unrelated second sequence of the same kind, so isjunk fits both.
        b, kind_b = rand_seq(rng)
        while kind_b != kind:
            b, kind_b = rand_seq(rng)
    isjunk, junk_desc = junk_for(rng, kind)
    autojunk = rng.random() < 0.7
    obs_seed = rng.random()
    results = []
    for cls in (difflib.SequenceMatcher, cand_cls):
        sm = cls(isjunk, a, b, autojunk=autojunk)
        r1 = observe(sm, random.Random(obs_seed), len(a), len(b))
        # Reuse: swap in a new seq1, then a new seq2 (the b-side index rebuild).
        a2 = mutate(random.Random(obs_seed), a)
        sm.set_seq1(a2)
        r2 = observe(sm, random.Random(obs_seed + 1), len(a2), len(b))
        sm.set_seq2(a)
        r3 = observe(sm, random.Random(obs_seed + 2), len(a2), len(a))
        results.append((r1, r2, r3))
    same = results[0] == results[1]
    detail = {}
    if not same:
        for stage, (e, g) in enumerate(zip(*results)):
            diffs = [k for k in e if e[k] != g[k]]
            if diffs:
                k = diffs[0]
                detail = {
                    "case": i, "kind": kind, "len_a": len(a), "len_b": len(b),
                    "isjunk": junk_desc, "autojunk": autojunk, "stage": ["init", "set_seq1", "set_seq2"][stage],
                    "field": k, "expected": repr(e[k])[:300], "got": repr(g[k])[:300],
                    "a": repr(a)[:200], "b": repr(b)[:200],
                }
                break
    return same, detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="directory containing mdifflib/")
    ap.add_argument("--split", default="dev", choices=list(SEEDS))
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.pkg).resolve()))
    result = {"split": args.split, "total": args.n}
    try:
        cand_cls = importlib.import_module("mdifflib").SequenceMatcher
    except Exception as e:  # noqa: BLE001
        result.update(passed=0, import_error=f"{type(e).__name__}: {e}", all_pass=False)
        print(json.dumps(result))
        return 1
    passed, examples = 0, []
    for i in range(args.n):
        try:
            ok, detail = run_case(cand_cls, i, SEEDS[args.split])
        except Exception as e:  # noqa: BLE001
            ok, detail = False, {"case": i, "error": f"{type(e).__name__}: {e}"}
        passed += ok
        if not ok and len(examples) < 6:
            examples.append(detail)
    result.update(passed=passed, examples=examples, all_pass=passed == args.n)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if passed == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
