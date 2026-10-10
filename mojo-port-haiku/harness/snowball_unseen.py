"""Check a multi-language msnowball build against the pure-Python stemmers on
languages the porting agent never saw, using Snowball's own vocabularies.

    python3 harness/snowball_unseen.py --pkg runs/r1-unseen/snowball \
        --vocab /tmp/sbdata/snowball-data [--cap 30000]

Vocabularies come from github.com/snowballstem/snowball-data (38 MB, not
vendored). Each language is capped at --cap words (every k-th word) plus
3,000 generated words from oracle/snowball/check.py's generator.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "oracle/snowball"))
from check import generated  # noqa: E402
from snowballstemmer.basestemmer import BaseStemmer  # noqa: E402


def reference(lang: str):
    mod = importlib.import_module(f"snowballstemmer.{lang}_stemmer")
    cls = next(v for v in vars(mod).values() if isinstance(v, type) and issubclass(v, BaseStemmer) and v is not BaseStemmer)
    return cls()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--cap", type=int, default=30000)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.pkg).resolve()))
    ms = importlib.import_module("msnowball")
    out = {}
    for lang in ms._LANGS:
        vf = Path(args.vocab) / lang / "voc.txt"
        if not vf.exists():
            out[lang] = {"skipped": "no voc.txt"}
            continue
        words = [w for w in vf.read_text(encoding="utf-8").split("\n") if w]
        step = max(1, len(words) // args.cap)
        words = words[::step][: args.cap]
        words += generated(lang, words, seed=zlib.crc32(lang.encode()))[:3000]
        ref = reference(lang)
        t = time.perf_counter(); expect = ref.stemWords(words); py_s = time.perf_counter() - t
        cand = ms.stemmer(lang)
        cand.stemWords(words[:100])
        t = time.perf_counter(); got = cand.stemWords(words); mj_s = time.perf_counter() - t
        bad = [(w, e, g) for w, e, g in zip(words, expect, got) if e != g]
        out[lang] = {"total": len(words), "passed": len(words) - len(bad), "speedup": round(py_s / mj_s, 1),
                     "examples": [list(b) for b in bad[:3]]}
        print(lang, out[lang]["passed"], "/", len(words), "x%.1f" % (py_s / mj_s), bad[:2], flush=True)
    (Path(args.pkg) / "unseen_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
