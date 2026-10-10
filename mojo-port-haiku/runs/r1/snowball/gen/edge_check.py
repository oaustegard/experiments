"""Edge-case differential test: port vs pure-Python reference, beyond the oracle.

Run from the snowball directory: python3 gen/edge_check.py
"""
import importlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import msnowball  # noqa: E402

LANGS = ["english", "german", "russian", "french"]


def reference(lang):
    mod = importlib.import_module(f"snowballstemmer.{lang}_stemmer")
    return getattr(mod, f"{lang.capitalize()}Stemmer")()


EDGE = [
    "", "a", "ab", "y", "Y", "é", "😀", "a😀b", "\ud800", "x\udfffy",
    "A" * 200, "ing" * 40, "ational" * 30, "ies" * 50, "ed" * 100,
    "\x00", "\n", "  ", "l'homme", "co-operate", "naïve", "über", "ёлка",
    "Straße", "ſ", "İstanbul", "ß" * 30,
]


def random_words(lang, seed, n):
    rng = random.Random(seed)
    alpha = {
        "english": "abcdefghijklmnopqrstuvwxyzYEIY'",
        "german": "abcdefghiklmnoprstuwzäöüßAUÄ",
        "russian": "абвгдежзийклмнопрстуфхцчшщъыьэюяёАБВ",
        "french": "abcdeéèêàâîôûçëïüoeiyAÉ'-",
    }[lang]
    out = []
    for _ in range(n):
        k = rng.randint(0, 25)
        out.append("".join(rng.choice(alpha) for _ in range(k)))
    return out


def main():
    bad = 0
    total = 0
    for lang in LANGS:
        ref = reference(lang)
        cand = msnowball.stemmer(lang)
        words = EDGE + random_words(lang, seed=len(lang), n=3000)
        expect = [ref.stemWord(w) for w in words]
        got = cand.stemWords(words)
        for w, e, g in zip(words, expect, got):
            total += 1
            if e != g:
                bad += 1
                if bad <= 10:
                    print("MISMATCH", lang, repr(w), repr(e), repr(g))
        # single-word path must agree too
        for w, e in zip(words[:50], expect[:50]):
            total += 1
            if cand.stemWord(w) != e:
                bad += 1
                print("SINGLE MISMATCH", lang, repr(w))
        # empty batch
        assert cand.stemWords([]) == []
    print(f"checked {total} results, {bad} mismatches")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
