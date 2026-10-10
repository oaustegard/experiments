"""Differential oracle for a Mojo port of snowballstemmer.

The candidate is a package directory `msnowball/` exposing `stemmer(lang)`,
which returns an object with `stemWord(word)` and `stemWords(words)`. The
reference is the pure-Python stemmer class from snowballstemmer (never the
PyStemmer-backed `snowballstemmer.stemmer()`, which delegates to C when
PyStemmer is installed).

Splits: `dev` is the even-indexed half of each Snowball vocabulary. `test` is
the odd-indexed half plus generated words (vocabulary words with suffixes
grafted on, case-mixed and punctuated tokens, random strings over the
language's alphabet) checked against the reference live.

    python3 oracle/snowball/check.py --pkg DIR [--split dev|test] [--langs english,german]

Prints one JSON line: per-language pass/total plus the first few mismatches.
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "snowball"
LANGS = ["english", "german", "russian", "french"]


def reference(lang: str):
    mod = importlib.import_module(f"snowballstemmer.{lang}_stemmer")
    cls = getattr(mod, f"{lang.capitalize()}Stemmer")
    return cls()


def vocab(lang: str) -> list[str]:
    words = (DATA / lang / "voc.txt").read_text(encoding="utf-8").split("\n")
    return [w for w in words if w]


def generated(lang: str, words: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = sorted({c for w in words for c in w})
    suffixes = sorted({w[-k:] for w in words for k in (1, 2, 3, 4) if len(w) > k})
    out = []
    for _ in range(4000):
        w = rng.choice(words)
        out.append(w + rng.choice(suffixes))
    for _ in range(2000):
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 16))))
    for _ in range(500):
        w = rng.choice(words)
        out.append(w.capitalize() if rng.random() < 0.5 else w.upper())
    out += ["", "a", "-", "x1y2", "naïve", "über", "ёлка", "l'homme", "co-operate", "a" * 60]
    return out


def split_words(lang: str, split: str) -> list[str]:
    words = vocab(lang)
    if split == "dev":
        return words[0::2]
    return words[1::2] + generated(lang, words, seed=zlib.crc32(lang.encode()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True, help="directory containing msnowball/")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--langs", default=",".join(LANGS))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.pkg).resolve()))
    result = {"split": args.split, "langs": {}}
    try:
        cand_mod = importlib.import_module("msnowball")
    except Exception as e:  # noqa: BLE001 - report any import failure
        result["import_error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(result))
        return 1

    ok_all = True
    for lang in args.langs.split(","):
        words = split_words(lang, args.split)
        ref = reference(lang)
        expect = [ref.stemWord(w) for w in words]
        entry = {"total": len(words)}
        try:
            cand = cand_mod.stemmer(lang)
            got = list(cand.stemWords(words))
            single = [cand.stemWord(w) for w in words[:200]]
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["passed"] = 0
            result["langs"][lang] = entry
            ok_all = False
            continue
        if len(got) != len(words):
            entry["error"] = f"stemWords returned {len(got)} items for {len(words)} words"
            got = (got + [None] * len(words))[: len(words)]
        bad = [(w, e, g) for w, e, g in zip(words, expect, got) if e != g]
        bad += [(w, e, g) for w, e, g in zip(words[:200], expect[:200], single) if e != g]
        entry["passed"] = len(words) - sum(1 for w, e, g in zip(words, expect, got) if e != g)
        entry["single_word_mismatches"] = sum(1 for e, g in zip(expect[:200], single) if e != g)
        entry["examples"] = [{"word": w, "expected": e, "got": g} for w, e, g in bad[:8]]
        ok_all &= not bad and "error" not in entry
        result["langs"][lang] = entry
    result["all_pass"] = ok_all
    print(json.dumps(result, ensure_ascii=False))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
