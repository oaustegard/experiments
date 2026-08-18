#!/usr/bin/env python3
"""Does a soft lexical ranker route this catalogue better than hard rules do?

The fitted decision lists collapse from 0.984 on the family they were fitted on
to 0.239 on a held-out phrasing, and `RESULTS.md` blames vocabulary: a rule keyed
on `tok:diff` cannot fire on *"what code does this PR actually change"*. BM25 is
the cheapest test of that diagnosis because it fails soft — a query carrying none
of a label's learned words is still ranked by the words it does share, so the
failure mode is a worse ranking rather than no answer at all.

Three document sources, one ranker:

* `schema` — a label's document is its own schema text (tool name, title,
  description, parameter names and descriptions, method enum name and gloss).
  **Nothing is fitted on any split**, so family A carries no privilege and the
  three splits should score within noise of each other. That is the hypothesis
  under test, and it is falsifiable: if A still beats B here, the gap is the
  generator's, not the fitter's.
* `train` — a label's document is its 12 family-A queries. Same information the
  decision lists were fitted on, without the hard firing condition.
* `both` — fusion, weighted sum of max-normalised scores or RRF.

Plus a Porter stemmer written inline (how much of the vocabulary gap is pure
morphology?) and a threshold sweep (BM25 emits a score, so abstention need not be
all-or-nothing — the catch-all in `handwritten.py` bought +0.014 accuracy for all
0.867 of the wild set's abstention).

    python3 bm25_arms.py                     # full table + sweeps, writes results_bm25.json
    python3 eval.py bm25-schema bm25-both    # single arms through the shared scorer
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections import Counter
from pathlib import Path

from arms import ArmBase, labels, register
from catalogue import load as load_catalogue
from fit import tokens as _base_tokens

HERE = Path(__file__).resolve().parent
OFF = "__offtopic__"

# --------------------------------------------------------------------------- #
# Porter stemmer, inline. A dependency for eleven suffix rules is not a trade
# worth making, and the point of the stemmed arm is to isolate morphology from
# synonymy — for that, textbook Porter is the reference implementation, not a
# better lemmatiser.
# --------------------------------------------------------------------------- #

_VOWEL = frozenset("aeiou")


def _cons(w: str, i: int) -> bool:
    c = w[i]
    if c in _VOWEL:
        return False
    if c != "y":
        return True
    return i == 0 or not _cons(w, i - 1)


def _shape(w: str) -> str:
    return "".join("c" if _cons(w, i) else "v" for i in range(len(w)))


def _m(w: str) -> int:
    return _shape(w).count("vc")


def _has_vowel(w: str) -> bool:
    return "v" in _shape(w)


def _double_cons(w: str) -> bool:
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w: str) -> bool:
    n = len(w)
    return (n >= 3 and _cons(w, n - 3) and not _cons(w, n - 2)
            and _cons(w, n - 1) and w[-1] not in "wxy")


_STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
          ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
          ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
          ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
          ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")]
_STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
          ("ical", "ic"), ("ful", ""), ("ness", "")]
_STEP4 = ["ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism",
          "ate", "iti", "ous", "ive", "ize", "al", "er", "ic", "ou"]

_STEM_CACHE: dict[str, str] = {}


def porter(word: str) -> str:
    """Porter 1980, steps 1a-5b. Cached: the same ~600 words recur every query."""
    hit = _STEM_CACHE.get(word)
    if hit is not None:
        return hit
    w = word
    if len(w) > 2:
        w = _porter_uncached(w)
    _STEM_CACHE[word] = w
    return w


def _porter_uncached(w: str) -> str:
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]

    if w.endswith("eed"):
        if _m(w[:-3]) > 0:
            w = w[:-1]
    else:
        for suf in ("ed", "ing"):
            if w.endswith(suf) and _has_vowel(w[: -len(suf)]):
                w = w[: -len(suf)]
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
                elif _double_cons(w) and not w.endswith(("l", "s", "z")):
                    w = w[:-1]
                elif _m(w) == 1 and _cvc(w):
                    w += "e"
                break

    if w.endswith("y") and _has_vowel(w[:-1]):
        w = w[:-1] + "i"

    for suf, rep in _STEP2:
        if w.endswith(suf) and _m(w[: -len(suf)]) > 0:
            w = w[: -len(suf)] + rep
            break
    for suf, rep in _STEP3:
        if w.endswith(suf) and _m(w[: -len(suf)]) > 0:
            w = w[: -len(suf)] + rep
            break
    for suf in _STEP4:
        if w.endswith(suf) and _m(w[: -len(suf)]) > 1:
            w = w[: -len(suf)]
            break
    else:
        if w.endswith("ion") and _m(w[:-3]) > 1 and w[:-3].endswith(("s", "t")):
            w = w[:-3]

    if w.endswith("e") and (_m(w[:-1]) > 1 or (_m(w[:-1]) == 1 and not _cvc(w[:-1]))):
        w = w[:-1]
    if w.endswith("ll") and _m(w) > 1:
        w = w[:-1]
    return w


def _tok(text: str, stemming: bool = False) -> list[str]:
    """`fit.tokens` verbatim, so every arm in this repo sees the same surface forms."""
    t = _base_tokens(text)
    return [porter(x) for x in t] if stemming else t


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #

# Three of the seven dispatchers gloss their methods inline ("1. get_diff - Get
# the diff of a pull request"); the other four say only "The method to execute".
# Where a gloss exists it belongs to one method, not to all nine, so it is parsed
# out rather than pasted into every sibling document.
GLOSS = re.compile(
    r"(?:\d+\.|-)\s*'?([a-z_]+)'?\s*[-–]\s*(.*?)"
    r"(?=\s(?:\d+\.|-)\s*'?[a-z_]+'?\s*[-–]|$)", re.S)


def schema_docs(cat: dict, name_boost: int = 1, stemming: bool = False) -> dict[str, list[str]]:
    """One document per routing target, from the catalogue alone. No query touches this."""
    out: dict[str, list[str]] = {}
    for name, tool in cat.items():
        shared = _tok(name.replace("_", " "), stemming) * name_boost
        shared += _tok(tool["title"], stemming) + _tok(tool["description"], stemming)
        for pname, p in tool["params"].items():
            if pname == "method":
                continue
            shared += _tok(pname.replace("_", " "), stemming)
            shared += _tok(p["description"], stemming)
            for e in p.get("enum") or []:
                shared += _tok(str(e).replace("_", " "), stemming)
        m = tool["params"].get("method")
        if m and m.get("enum"):
            desc = m["description"]
            first = GLOSS.search(desc)
            intro = _tok(desc[: first.start()] if first else desc, stemming)
            gl = dict(GLOSS.findall(desc))
            for e in m["enum"]:
                out[f"{name}::{e}"] = (
                    shared + intro
                    + _tok(str(e).replace("_", " "), stemming) * name_boost
                    + _tok(gl.get(e, ""), stemming))
        else:
            out[name] = shared
    return out


def _mask_entities(query: str) -> str:
    """Blank the spans `cues.extract` would bind as arguments.

    Family A and family B draw from disjoint entity pools by construction, so
    every owner, repo and branch name in a training document is a term that can
    only ever match the wrong split. This measures whether that noise costs
    anything.
    """
    from cues import extract
    for v in extract(query).values():
        s = str(v)
        if len(s) > 1:
            query = query.replace(s, " ")
    return query


def train_docs(rows: list[dict], stemming: bool = False, mask: bool = False,
               offtopic: bool = False) -> dict[str, list[str]]:
    """One document per target, concatenating that target's family-A queries."""
    docs: dict[str, list[str]] = {lab: [] for lab in labels()}
    off: list[str] = []
    for r in rows:
        q = _mask_entities(r["query"]) if mask else r["query"]
        t = _tok(q, stemming)
        if r.get("label"):
            docs[r["label"]] += t
        else:
            off += t
    docs = {k: v for k, v in docs.items() if v}
    if offtopic and off:
        docs[OFF] = off
    return docs


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

class BM25:
    """Okapi BM25 over 79 documents, doc-side weights precomputed at build time.

    k1=1.2 / b=0.75 are the textbook values and are deliberately left alone: the
    schema arm's entire claim is that no split was used to choose anything, and a
    k1 swept on family A would forfeit it for a fraction of a point.

    Scoring is a dict accumulation over postings, not a matrix product — with 79
    documents and ~7 query terms numpy's setup cost dominates its own kernel, and
    the regex arms this is compared against run in 0.036 ms.
    """

    def __init__(self, docs: dict[str, list[str]], k1: float = 1.2, b: float = 0.75):
        self.ids = list(docs)
        n = len(docs)
        lens = [len(t) for t in docs.values()]
        avgdl = (sum(lens) / n) if n else 1.0
        df: Counter = Counter()
        for toks in docs.values():
            df.update(set(toks))
        self.post: dict[str, list[tuple[int, float]]] = {}
        for i, toks in enumerate(docs.values()):
            dl = len(toks)
            norm = k1 * (1 - b + b * dl / avgdl)
            for t, f in Counter(toks).items():
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                self.post.setdefault(t, []).append((i, idf * f * (k1 + 1) / (f + norm)))

    def score(self, qtokens: list[str]) -> list[tuple[str, float]]:
        acc: dict[int, float] = {}
        for t in qtokens:
            for i, w in self.post.get(t, ()):
                acc[i] = acc.get(i, 0.0) + w
        return sorted(((self.ids[i], s) for i, s in acc.items()), key=lambda kv: -kv[1])


def _normalise(ranked: list[tuple[str, float]]) -> dict[str, float]:
    """Max-normalisation, not softmax: BM25 scores have no calibrated scale, and
    dividing by the query's own best score is the only rescaling that does not
    invent one."""
    if not ranked:
        return {}
    top = ranked[0][1] or 1.0
    return {k: v / top for k, v in ranked}


def _rrf(ranked: list[tuple[str, float]], k: int = 60) -> dict[str, float]:
    return {lab: 1.0 / (k + i + 1) for i, (lab, _) in enumerate(ranked)}


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #

class BM25Arm(ArmBase):
    """A ranker with an optional gate. `route` is `score`'s top row, gated."""

    def __init__(self, source: str = "schema", stemming: bool = False,
                 name_boost: int = 1, weight: float = 0.5, fusion: str = "sum",
                 gate: str = "none", threshold: float = 0.0,
                 mask: bool = False, offtopic: bool = False):
        self.source, self.stemming, self.weight, self.fusion = source, stemming, weight, fusion
        self.gate, self.threshold = gate, threshold
        cat = load_catalogue("session")
        self.schema = self.train = None
        if source in ("schema", "both"):
            self.schema = BM25(schema_docs(cat, name_boost, stemming))
        if source in ("train", "both"):
            rows = [json.loads(x) for x in
                    (HERE / "data" / "family_a.jsonl").read_text().splitlines() if x.strip()]
            self.train = BM25(train_docs(rows, stemming, mask, offtopic))

    def rank(self, query: str) -> list[tuple[str, float]]:
        """Every candidate with a nonzero score, `__offtopic__` included if built."""
        q = _tok(query, self.stemming)
        if self.source == "schema":
            return self.schema.score(q)
        if self.source == "train":
            return self.train.score(q)
        a, b = self.schema.score(q), self.train.score(q)
        if self.fusion == "rrf":
            fa, fb = _rrf(a), _rrf(b)
        else:
            fa, fb = _normalise(a), _normalise(b)
        keys = set(fa) | set(fb)
        w = self.weight
        fused = {k: w * fa.get(k, 0.0) + (1 - w) * fb.get(k, 0.0) for k in keys}
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def score(self, query: str) -> list[tuple[str, float]]:
        return [(l, s) for l, s in self.rank(query) if l != OFF]

    def route(self, query: str) -> str | None:
        ranked = self.rank(query)
        if not ranked or ranked[0][0] == OFF:
            return None
        if self.gate != "none" and gate_stat(ranked, self.gate) < self.threshold:
            return None
        return ranked[0][0]


def gate_stat(ranked: list[tuple[str, float]], gate: str) -> float:
    """Three ways to read confidence off a ranking.

    `abs` is the raw top score and therefore grows with query length, which is a
    property of the sentence rather than of the decision; `margin` and `ratio`
    are length-free. Which one actually gates better is measured below.
    """
    if not ranked:
        return 0.0
    top = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if gate == "abs":
        return top
    if gate == "margin":
        return top - second
    return (top - second) / top if top else 0.0


# Every constant below that a split had a vote in was chosen on **family A**,
# the only split it is legitimate to choose on, and each is reported against its
# ungated / unweighted sibling so the choice can be priced.
# Fusion weight: schema*0.2 + train*0.8. Argmax on family A (0.997 at w=0.2
# against 0.993 for train alone), and family B independently agrees (0.341 vs
# 0.320) — a rare case where the split you are allowed to tune on picks the same
# point as the one you are not.
BEST_W = 0.2
# Gate: raw top-1 fused score, thresholded at the family-A argmax of overall
# accuracy (routable hits + correct abstains). `margin` and `ratio` both gate
# worse at matched coverage — see the sweep in `main`.
BEST_GATE, BEST_T = "abs", 0.841

ARMS: dict[str, dict] = {
    "bm25-schema": dict(source="schema"),
    "bm25-schema-b3": dict(source="schema", name_boost=3),
    "bm25-schema-stem": dict(source="schema", stemming=True),
    "bm25-train": dict(source="train"),
    "bm25-train-mask": dict(source="train", mask=True),
    "bm25-train-stem": dict(source="train", stemming=True),
    "bm25-train+off": dict(source="train", offtopic=True),
    "bm25-both": dict(source="both", weight=BEST_W),
    "bm25-both-rrf": dict(source="both", fusion="rrf", weight=BEST_W),
    "bm25-both-stem": dict(source="both", weight=BEST_W, stemming=True),
    "bm25-both+off": dict(source="both", weight=BEST_W, offtopic=True),
    "bm25-gated": dict(source="both", weight=BEST_W, gate=BEST_GATE, threshold=BEST_T),
    "bm25-gated+off": dict(source="both", weight=BEST_W, offtopic=True,
                           gate=BEST_GATE, threshold=BEST_T),
}

for _name, _kw in ARMS.items():
    register(_name, lambda kw=_kw: BM25Arm(**kw))


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #

def quick(arm, rows: list[dict]) -> tuple[float, float]:
    """label_acc and abstain_acc only — the sweeps run this hundreds of times."""
    on = [r for r in rows if r.get("label")]
    off = [r for r in rows if not r.get("label")]
    hits = sum(arm.route(r["query"]) == r["label"] for r in on)
    ab = sum(arm.route(r["query"]) is None for r in off)
    return hits / len(on), (ab / len(off) if off else 0.0)


def sweep_thresholds(arm, rows: list[dict], gate: str, grid: list[float]) -> list[dict]:
    """Rank once, gate many times. Re-ranking per threshold would be 40x the work."""
    pre = [(r.get("label"), arm.rank(r["query"])) for r in rows]
    n_on = sum(1 for lab, _ in pre if lab) or 1
    n_off = sum(1 for lab, _ in pre if not lab) or 1
    out = []
    for t in grid:
        answered = hits = ab = 0
        for lab, ranked in pre:
            got = None
            if ranked and ranked[0][0] != OFF and gate_stat(ranked, gate) >= t:
                got = ranked[0][0]
            if lab:
                answered += got is not None
                hits += got == lab
            else:
                ab += got is None
        out.append({
            "threshold": round(t, 4),
            "coverage": round(answered / n_on, 4),
            "precision": round(hits / answered, 4) if answered else 0.0,
            "label_acc": round(hits / n_on, 4),
            "abstain_acc": round(ab / n_off, 4),
            "overall": round((hits + ab) / (n_on + n_off), 4),
        })
    return out


def recall_at_k(arm, rows: list[dict], ks=(1, 3, 5, 10)) -> dict[int, float]:
    """Where the right answer sits in the ranking, not just whether it is on top.

    Top-1 is the wrong question if the ranker is a candidate generator handing a
    shortlist to a model: 79 targets in a prompt is a real cost, 5 is not.
    """
    on = [r for r in rows if r.get("label")]
    hits = dict.fromkeys(ks, 0)
    for r in on:
        ranked = [l for l, _ in arm.score(r["query"])]
        pos = ranked.index(r["label"]) if r["label"] in ranked else 10 ** 6
        for k in ks:
            hits[k] += pos < k
    return {k: round(v / len(on), 4) for k, v in hits.items()}


def stem_effect(plain, stemmed, rows: list[dict], limit: int = 5) -> dict:
    """Which queries the stemmer fixed, which it broke, and what it collapsed.

    An accuracy delta says stemming lost; it does not say whether it lost by
    failing to merge morphology or by merging too much. Both are counted here.
    """
    fixed, broke = [], []
    for r in rows:
        if not r.get("label"):
            continue
        a = plain.route(r["query"]) == r["label"]
        b = stemmed.route(r["query"]) == r["label"]
        if b and not a:
            fixed.append({"query": r["query"], "gold": r["label"],
                          "plain_said": plain.route(r["query"])})
        elif a and not b:
            broke.append({"query": r["query"], "gold": r["label"],
                          "stem_said": stemmed.route(r["query"])})
    return {"fixed": len(fixed), "broke": len(broke),
            "fixed_examples": fixed[:limit], "broke_examples": broke[:limit]}


def morphology_gap(cat: dict, rows: list[dict], limit: int = 8) -> dict:
    """How much of the query/schema vocabulary gap is morphology alone.

    Two counts, in opposite directions: query terms that match no document
    unstemmed but do once stemmed (what stemming can recover), and distinct
    document surface forms that share a stem (what it destroys).
    """
    plain_vocab, stem_vocab = set(), {}
    for toks in schema_docs(cat).values():
        for w in toks:
            plain_vocab.add(w)
            stem_vocab.setdefault(porter(w), set()).add(w)
    recovered: Counter = Counter()
    n_on = touched = 0
    for r in rows:
        if not r.get("label"):
            continue
        n_on += 1
        got = False
        for w in set(_tok(r["query"])):
            if w not in plain_vocab and porter(w) in stem_vocab:
                recovered[(w, sorted(stem_vocab[porter(w)])[0])] += 1
                got = True
        touched += got
    collisions = {k: sorted(v) for k, v in stem_vocab.items() if len(v) > 1}
    return {
        "queries_with_a_recovered_term": round(touched / (n_on or 1), 4),
        "top_recovered": [{"query_term": a, "schema_term": b, "n": n}
                          for (a, b), n in recovered.most_common(limit)],
        "n_schema_terms": len(plain_vocab),
        "n_stems": len(stem_vocab),
        "n_colliding_stems": len(collisions),
        "example_collisions": [v for _, v in sorted(collisions.items())[:limit]],
    }


def consensus(cat_arm, rows: list[dict]) -> dict:
    """The `agreement.py` gate, both sides deterministic and both of them BM25.

    RESULTS.md priced two-router agreement at 0.867 precision for 0.203 coverage
    on wild. Schema-BM25 and train-BM25 disagree for different reasons than the
    hand and fitted arms do, so the same gate is worth re-pricing here.
    """
    on = [r for r in rows if r.get("label")]
    off = [r for r in rows if not r.get("label")]
    agree = hits = 0
    for r in on:
        a = cat_arm.schema.score(_tok(r["query"]))
        b = cat_arm.train.score(_tok(r["query"]))
        if a and b and a[0][0] == b[0][0]:
            agree += 1
            hits += a[0][0] == r["label"]
    ab = 0
    for r in off:
        a = cat_arm.schema.score(_tok(r["query"]))
        b = cat_arm.train.score(_tok(r["query"]))
        ab += not (a and b and a[0][0] == b[0][0])
    return {"coverage": round(agree / len(on), 4),
            "precision": round(hits / agree, 4) if agree else 0.0,
            "label_acc": round(hits / len(on), 4),
            "abstain_acc": round(ab / len(off), 4) if off else None}


def main() -> int:
    from eval import load_split, score

    splits = {
        "family A (fitted)": load_split(HERE / "data" / "family_a.jsonl"),
        "family B (held-out)": load_split(HERE / "data" / "family_b.jsonl"),
        "wild (hand-authored)": load_split(HERE / "wild.jsonl"),
    }
    cat = load_catalogue("session")
    out: dict = {"arms": {}}

    hdr = (f"{'arm':<18}{'split':<22}{'cov':>7}{'prec':>7}{'acc':>7}{'tool':>7}"
           f"{'meth':>7}{'abst':>7}{'args':>7}{'ms':>9}")
    print(hdr)
    print("-" * len(hdr))
    built = {}
    for name, kw in ARMS.items():
        arm = built[name] = BM25Arm(**kw)
        for sname, rows in splits.items():
            s = score(arm, rows)
            s.pop("errors")
            out["arms"].setdefault(name, {})[sname] = s
            f = lambda k: "  -  " if s[k] is None else f"{s[k]:.3f}"
            print(f"{name:<18}{sname:<22}{f('coverage'):>7}{f('precision'):>7}"
                  f"{f('label_acc'):>7}{f('tool_acc'):>7}{f('method_acc_given_tool'):>7}"
                  f"{f('abstain_acc'):>7}{f('args_acc'):>7}{s['median_latency_ms']:>9.4f}")
        print()

    # --- recall@k: is a weak top-1 still a usable shortlist? -----------------
    print("recall@k over the 79 targets (routable rows only)")
    tag = {"family A (fitted)": "A", "family B (held-out)": "B",
           "wild (hand-authored)": "wild"}
    print(f"{'arm':<18}" + "".join(f"{tag[s]+'@'+str(k):>10}"
                                   for s in splits for k in (1, 3, 5, 10)))
    out["recall_at_k"] = {}
    for name, arm in built.items():
        if name.startswith("bm25-gated"):
            continue  # the gate changes `route`, not the ranking `score` returns
        line = f"{name:<18}"
        for sname, rows in splits.items():
            r = recall_at_k(arm, rows)
            out["recall_at_k"].setdefault(name, {})[sname] = r
            line += "".join(f"{r[k]:>10.3f}" for k in (1, 3, 5, 10))
        print(line)
    print()

    # --- fusion weight, and whether normalised-sum or RRF wins ---------------
    print("fusion sweep (label_acc; w=1.0 is schema-only, w=0.0 is train-only)")
    print(f"{'w':>5}{'sum A':>9}{'sum B':>9}{'sum wild':>10}"
          f"{'rrf A':>9}{'rrf B':>9}{'rrf wild':>10}")
    fusion_rows = []
    for i in range(11):
        w = i / 10
        row = {"w": w}
        for mode in ("sum", "rrf"):
            arm = BM25Arm(source="both", fusion=mode, weight=w)
            for sname, rows in splits.items():
                row[f"{mode}|{sname}"] = round(quick(arm, rows)[0], 4)
        fusion_rows.append(row)
        g = lambda m, s: f"{row[f'{m}|{s}']:.3f}"
        print(f"{w:>5.1f}{g('sum','family A (fitted)'):>9}{g('sum','family B (held-out)'):>9}"
              f"{g('sum','wild (hand-authored)'):>10}{g('rrf','family A (fitted)'):>9}"
              f"{g('rrf','family B (held-out)'):>9}{g('rrf','wild (hand-authored)'):>10}")
    out["fusion_sweep"] = fusion_rows
    best_w = max(fusion_rows, key=lambda r: r["sum|family A (fitted)"])["w"]
    print(f"  argmax on family A: w={best_w} (pinned as BEST_W={BEST_W})")
    print()

    # --- what stemming actually did -----------------------------------------
    print("stemming diagnostic")
    out["stemming"] = {"morphology_gap": morphology_gap(cat, splits["family B (held-out)"])}
    g = out["stemming"]["morphology_gap"]
    print(f"  family-B queries with >=1 term that only matches a schema doc once stemmed:"
          f" {g['queries_with_a_recovered_term']:.3f}")
    print(f"  schema terms {g['n_schema_terms']} -> {g['n_stems']} stems, "
          f"{g['n_colliding_stems']} of which merge >1 surface form")
    print("  recovered: " + ", ".join(f"{d['query_term']}~{d['schema_term']}({d['n']})"
                                      for d in g["top_recovered"]))
    print("  collisions: " + "; ".join("/".join(v) for v in g["example_collisions"]))
    for pair in (("bm25-schema", "bm25-schema-stem"), ("bm25-train", "bm25-train-stem")):
        for sname in ("family B (held-out)", "wild (hand-authored)"):
            e = stem_effect(built[pair[0]], built[pair[1]], splits[sname])
            out["stemming"][f"{pair[1]}|{sname}"] = e
            print(f"  {pair[1]:<18}{sname:<22} fixed {e['fixed']:>3}  broke {e['broke']:>3}")
            for mark, key, said in (("fix", "fixed_examples", "plain_said"),
                                    ("brk", "broke_examples", "stem_said")):
                for d in e[key][:2]:
                    print(f"      [{mark}] {d['query'][:58]:<58} gold={d['gold']}"
                          f" other={d[said]}")
    print()

    # --- abstention: three gate statistics on the best ungated arm -----------
    base = built["bm25-both"]
    grids = {
        "abs": [round(x, 3) for x in _quantile_grid(base, splits["family A (fitted)"], "abs")],
        "margin": [round(x, 3) for x in _quantile_grid(base, splits["family A (fitted)"], "margin")],
        "ratio": [i / 40 for i in range(21)],
    }
    out["threshold_sweep"] = {}
    for gate, grid in grids.items():
        out["threshold_sweep"][gate] = {}
        print(f"threshold sweep on bm25-both (w={BEST_W}), gate={gate}")
        print(f"{'t':>8}" + "".join(f"{tag[s]+' '+k:>11}"
                                    for s in splits for k in ("cov", "acc", "abst")))
        rows_by_split = {s: sweep_thresholds(base, r, gate, grid) for s, r in splits.items()}
        for j, t in enumerate(grid):
            line = f"{t:>8.3f}"
            for s in splits:
                d = rows_by_split[s][j]
                line += f"{d['coverage']:>11.3f}{d['label_acc']:>11.3f}{d['abstain_acc']:>11.3f}"
            print(line)
        for s, rows in rows_by_split.items():
            out["threshold_sweep"][gate][s] = rows
        best = max(rows_by_split["family A (fitted)"], key=lambda d: d["overall"])
        print(f"  best on family A by overall accuracy (routable hits + correct abstains):"
              f" t={best['threshold']} cov={best['coverage']} acc={best['label_acc']}"
              f" abst={best['abstain_acc']}")
        print()

    # --- the agreement gate, both sides BM25 --------------------------------
    print("consensus gate (schema and train name the same target)")
    print(f"{'split':<22}{'cov':>8}{'prec':>8}{'acc':>8}{'abst':>8}")
    out["consensus"] = {}
    for sname, rows in splits.items():
        c = consensus(built["bm25-both"], rows)
        out["consensus"][sname] = c
        print(f"{sname:<22}{c['coverage']:>8.3f}{c['precision']:>8.3f}"
              f"{c['label_acc']:>8.3f}{c['abstain_acc']:>8.3f}")
    print()

    out["config"] = {"k1": 1.2, "b": 0.75, "weight": BEST_W, "gate": BEST_GATE,
                     "threshold": BEST_T, "n_labels": len(labels())}
    (HERE / "results_bm25.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote results_bm25.json")
    return 0


def _quantile_grid(arm, rows: list[dict], gate: str, n: int = 21) -> list[float]:
    """Thresholds at observed quantiles, so every step moves coverage."""
    vals = sorted(gate_stat(arm.rank(r["query"]), gate) for r in rows)
    return [vals[min(len(vals) - 1, int(i * len(vals) / n))] for i in range(n)]


if __name__ == "__main__":
    raise SystemExit(main())
