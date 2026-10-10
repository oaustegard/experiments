"""Differential oracle for a Mojo port of yake 0.7.3.

The candidate is a package directory `myake/` exposing `KeywordExtractor`
with yake's constructor (lan, n, dedup_lim, dedup_func, window_size, top;
lemmatize stays False and features None) and `extract_keywords(text)`
returning a list of (keyword, score).

Each document runs under every config in CONFIGS. A doc/config passes when
the keyword lists match in order and every score agrees within a relative
1e-9. An order difference is forgiven only between keywords whose reference
scores are within that tolerance of each other, since a different summation
order may break a near-tie differently.

    python3 oracle/yake/check.py --pkg DIR [--split dev|test] [--configs all|default]
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import yake

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[1] / "data" / "yake_corpus"
REL = 1e-9

CONFIGS = [
    {},
    {"n": 1},
    {"n": 2, "dedup_func": "levs"},
    {"n": 3, "dedup_func": "jaro", "top": 50},
    {"window_size": 2, "dedup_lim": 0.7},
    {"n": 4, "top": 10, "dedup_lim": 1.0},
]


def close(x: float, y: float) -> bool:
    return math.isclose(x, y, rel_tol=REL, abs_tol=1e-15)


def compare(expect, got) -> str | None:
    if len(expect) != len(got):
        return f"length {len(got)} != expected {len(expect)}"
    exp_score = {k: s for k, s in expect}
    for i, ((ek, es), (gk, gs)) in enumerate(zip(expect, got)):
        if ek == gk:
            if not close(es, gs):
                return f"rank {i} {ek!r}: score {gs!r} != {es!r}"
            continue
        if gk in exp_score and close(exp_score[gk], es) and close(gs, es):
            continue  # near-tie swapped
        return f"rank {i}: got {gk!r} ({gs!r}), expected {ek!r} ({es!r})"
    if sorted(k for k, _ in expect) != sorted(k for k, _ in got):
        return "same length but different keyword set"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="directory containing myake/")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--configs", default="all", choices=["all", "default"])
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.pkg).resolve()))
    configs = CONFIGS if args.configs == "all" else CONFIGS[:1]
    docs = sorted((CORPUS / args.split).glob("*.md"))
    result = {"split": args.split, "total": len(docs) * len(configs)}
    try:
        cand_mod = importlib.import_module("myake")
    except Exception as e:  # noqa: BLE001
        result.update(passed=0, import_error=f"{type(e).__name__}: {e}", all_pass=False)
        print(json.dumps(result))
        return 1
    passed, examples = 0, []
    extra_texts = ["", "   ", "Hello.", "The the THE. A a a a.", "Ünïcødé wörds, ok? Yes! 3.14 and 42%."]
    cases = [(d.name, d.read_text(encoding="utf-8")) for d in docs] + [(f"edge{i}", t) for i, t in enumerate(extra_texts)]
    result["total"] = len(cases) * len(configs)
    for cfg in configs:
        ref = yake.KeywordExtractor(**cfg)
        try:
            cand = cand_mod.KeywordExtractor(**cfg)
        except Exception as e:  # noqa: BLE001
            examples.append({"config": cfg, "error": f"constructor: {type(e).__name__}: {e}"})
            continue
        for name, text in cases:
            expect = ref.extract_keywords(text)
            try:
                got = [(str(k), float(s)) for k, s in cand.extract_keywords(text)]
                why = compare(expect, got)
            except Exception as e:  # noqa: BLE001
                why = f"{type(e).__name__}: {e}"
            if why is None:
                passed += 1
            elif len(examples) < 8:
                examples.append({"doc": name, "config": cfg, "why": why[:400]})
    result.update(passed=passed, examples=examples, all_pass=passed == result["total"])
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
