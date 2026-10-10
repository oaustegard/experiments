"""Wall-clock benchmark of a candidate port against its Python reference,
measured from Python, so the boundary and the shim are inside the timing.

    python3 bench/bench.py --target snowball|difflib|yake --pkg DIR

Each workload runs once to warm up, then `--reps` times; the minimum is
reported. Prints one JSON line.
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def best(fn, reps: int) -> float:
    fn()
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


def corpus_texts(split: str = "test") -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted((DATA / "yake_corpus" / split).glob("*.md"))]


def bench_snowball(pkg, reps: int) -> dict:
    import Stemmer  # PyStemmer: the C build of the same algorithms, as a ceiling

    out = {}
    for lang in ["english", "german", "russian", "french"]:
        words = (DATA / "snowball" / lang / "voc.txt").read_text(encoding="utf-8").split("\n")
        words = [w for w in words if w]
        mod = importlib.import_module(f"snowballstemmer.{lang}_stemmer")
        ref = getattr(mod, f"{lang.capitalize()}Stemmer")()
        cand = pkg.stemmer(lang)
        c = Stemmer.Stemmer(lang, 0)  # cache off: the timed words repeat across reps
        sample = words[:5000]
        row = {
            "words": len(words),
            "py_batch_s": best(lambda: ref.stemWords(words), reps),
            "mojo_batch_s": best(lambda: cand.stemWords(words), reps),
            "c_batch_s": best(lambda: c.stemWords(words), reps),
            "py_single_s": best(lambda: [ref.stemWord(w) for w in sample], reps),
            "mojo_single_s": best(lambda: [cand.stemWord(w) for w in sample], reps),
            "c_single_s": best(lambda: [c.stemWord(w) for w in sample], reps),
        }
        row["batch_speedup"] = row["py_batch_s"] / row["mojo_batch_s"]
        row["single_speedup"] = row["py_single_s"] / row["mojo_single_s"]
        row["mojo_vs_c_batch"] = row["c_batch_s"] / row["mojo_batch_s"]
        out[lang] = row
    return out


def edit(rng: random.Random, seq, rate: float):
    items = list(seq)
    for _ in range(int(len(items) * rate)):
        i = rng.randrange(max(1, len(items)))
        r = rng.random()
        if r < 0.4 and items:
            del items[min(i, len(items) - 1)]
        elif r < 0.8:
            items.insert(i, rng.choice(items) if items else "x")
        elif items:
            items[min(i, len(items) - 1)] = rng.choice(items)
    return "".join(items) if isinstance(seq, str) else items


def bench_difflib(pkg, reps: int) -> dict:
    cand = pkg.SequenceMatcher
    rng = random.Random(7)
    texts = corpus_texts()
    char_pairs = []
    for t in texts[:40]:
        a = t[:3000]
        char_pairs.append((a, edit(rng, a, 0.05)))
    line_pairs = []
    for t in texts:
        a = t.splitlines(keepends=True)
        line_pairs.append((a, edit(rng, a, 0.1)))
    words = sorted({w for t in texts for w in t.split() if w.isalpha() and len(w) > 3})[:6000]
    queries = [edit(rng, w, 0.2) for w in rng.sample(words, 150)]

    def ratios(cls):
        return [cls(None, a, b).ratio() for a, b in char_pairs]

    def unified(cls):
        saved = difflib.SequenceMatcher
        difflib.SequenceMatcher = cls
        try:
            return [list(difflib.unified_diff(a, b)) for a, b in line_pairs]
        finally:
            difflib.SequenceMatcher = saved

    def close_matches(cls):
        saved = difflib.SequenceMatcher
        difflib.SequenceMatcher = cls
        try:
            return [difflib.get_close_matches(q, words) for q in queries]
        finally:
            difflib.SequenceMatcher = saved

    base = difflib.SequenceMatcher
    out = {}
    for name, fn in [("char_ratio_3k", ratios), ("unified_diff_lines", unified), ("get_close_matches", close_matches)]:
        assert fn(base) == fn(cand), f"{name}: candidate output differs"
        py = best(lambda: fn(base), reps)
        mj = best(lambda: fn(cand), reps)
        out[name] = {"py_s": py, "mojo_s": mj, "speedup": py / mj}
    return out


def bench_yake(pkg, reps: int) -> dict:
    import yake

    texts = corpus_texts()[:40]
    ref = yake.KeywordExtractor()
    cand = pkg.KeywordExtractor()
    py = best(lambda: [ref.extract_keywords(t) for t in texts], reps)
    mj = best(lambda: [cand.extract_keywords(t) for t in texts], reps)
    return {"docs": len(texts), "py_s_per_doc": py / len(texts), "mojo_s_per_doc": mj / len(texts), "speedup": py / mj}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=["snowball", "difflib", "yake"])
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.pkg).resolve()))
    name = {"snowball": "msnowball", "difflib": "mdifflib", "yake": "myake"}[args.target]
    pkg = importlib.import_module(name)
    fn = {"snowball": bench_snowball, "difflib": bench_difflib, "yake": bench_yake}[args.target]
    print(json.dumps({"target": args.target, "results": fn(pkg, args.reps)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
