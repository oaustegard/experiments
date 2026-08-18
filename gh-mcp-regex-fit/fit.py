#!/usr/bin/env python3
"""Induce an ordered decision list over structural cues and lexical features.

This is the part `monad-bsky/regex_only.py` did by hand. There, twenty rules
were written after reading the eval's failures, which made 0.833 a fitted number
with no way to say how fitted. Here the rules are *searched for* under a fixed
objective, so the author chooses the hypothesis space and the data chooses the
rules.

Algorithm — greedy precision-constrained covering, the CN2/RIPPER shape:

    while some candidate clears (min_precision, min_coverage):
        pick the highest-precision candidate, ties broken by coverage
        append (literals -> majority label) to the list
        delete every row it covered, right or wrong
    rows never covered are ABSTAINS, not a fallback label

The deletion of *wrongly* covered rows is what makes it a decision list rather
than a rule set: a later rule can never repair an earlier rule's mistake, which
is the same "stage-1 error is unrecoverable" property `needle-bsky/two_stage.py`
found the hard way.

Candidates are conjunctions of at most two literals, at most one negated. They
are enumerated from the rows themselves, so every candidate covers something.

    python3 fit.py --vocab schema --min-prec 0.85 --min-cov 3
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from catalogue import load as load_catalogue
from cues import CUE_NAMES, cues

HERE = Path(__file__).resolve().parent
WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "is", "it", "me",
        "my", "i", "you", "please", "can", "do", "does", "did", "s", "t"}


def tokens(q: str) -> list[str]:
    return [w for w in WORD.findall(q.lower()) if w not in STOP and len(w) > 1]


def schema_vocab(catalogue: dict) -> set[str]:
    """Every word the catalogue itself uses. A feature space with no query in it."""
    v: set[str] = set()
    for name, tool in catalogue.items():
        v |= set(tokens(name.replace("_", " ")))
        v |= set(tokens(tool["title"]))
        v |= set(tokens(tool["description"]))
        for pname, p in tool["params"].items():
            v |= set(tokens(pname.replace("_", " ")))
            v |= set(tokens(p["description"]))
            for e in p.get("enum") or []:
                v |= set(tokens(str(e).replace("_", " ")))
    return v


def label_text(catalogue: dict) -> dict[str, set[str]]:
    """Bag of words for every routing target, taken from the schema alone.

    This is the escape hatch from the fitter's basic limitation: a rule built on
    `tok:diff` can only fire on queries that literally say "diff", and no amount
    of fitting on one phrasing invents the synonyms another phrasing uses. An
    overlap feature is computed against the *catalogue* at inference time, so it
    can carry a word the training queries never contained.
    """
    out: dict[str, set[str]] = {}
    for name, tool in catalogue.items():
        base = set(tokens(name.replace("_", " "))) | set(tokens(tool["title"])) \
            | set(tokens(tool["description"]))
        m = tool["params"].get("method")
        if m and m.get("enum"):
            for e in m["enum"]:
                out[f"{name}::{e}"] = base | set(tokens(str(e).replace("_", " ")))
        else:
            out[name] = base
    return out


def _idf(texts: dict[str, set[str]]) -> dict[str, float]:
    import math
    n = len(texts)
    df = Counter(w for words in texts.values() for w in words)
    return {w: math.log(n / c) for w, c in df.items()}


def overlap_feats(q_tokens: set[str], texts: dict[str, set[str]], top: int,
                  idf: dict[str, float] | None = None) -> set[str]:
    """The `top` labels whose schema text shares the most *distinctive* words.

    Without the IDF weight this is useless: `owner`, `repository`, `pull` and
    `request` appear in most of the 79 descriptions, so a plain overlap count
    ranks by verbosity rather than by fit.
    """
    idf = idf if idf is not None else _idf(texts)
    scored = []
    for label, words in texts.items():
        hit = q_tokens & words
        if hit:
            scored.append((sum(idf.get(w, 0.0) for w in hit) / len(words) ** 0.5, label))
    scored.sort(reverse=True)
    return {f"ov:{label}" for sc, label in scored[:top] if sc > 0}


_IDF_CACHE: dict[int, dict[str, float]] = {}


def featurise(q: str, vocab: set[str] | None, bigrams: bool = True,
              texts: dict[str, set[str]] | None = None, top: int = 3) -> set[str]:
    f = {f"cue:{k}" for k, v in cues(q).items() if v}
    ts = tokens(q)
    keep = [t for t in ts if vocab is None or t in vocab]
    f |= {f"tok:{t}" for t in keep}
    if bigrams:
        f |= {f"bi:{a}_{b}" for a, b in zip(keep, keep[1:])}
    if texts:
        idf = _IDF_CACHE.get(id(texts))
        if idf is None:
            idf = _IDF_CACHE[id(texts)] = _idf(texts)
        f |= overlap_feats(set(ts), texts, top, idf)
    return f


class Fitter:
    def __init__(self, rows, vocab, min_prec, min_cov, max_pairs_per_row,
                 bigrams=True, texts=None, top=3, score="precision"):
        self.rows = rows
        self.labels = [r["label"] for r in rows]
        self.feats = [featurise(r["query"], vocab, bigrams, texts, top) for r in rows]
        self.min_prec, self.min_cov = min_prec, min_cov
        self.score_fn = score
        self.max_pairs_per_row = max_pairs_per_row

        df = Counter(f for fs in self.feats for f in fs)
        n = len(rows)
        # A literal must be rare enough to discriminate and common enough to matter.
        self.usable = {f for f, c in df.items() if c >= min_cov}
        self.df = df
        self.n = n
        self.bits: dict[str, int] = defaultdict(int)
        for i, fs in enumerate(self.feats):
            for f in fs:
                if f in self.usable:
                    self.bits[f] |= 1 << i
        self.all_mask = (1 << n) - 1

    def _label_hist(self, mask: int) -> Counter:
        h = Counter()
        m = mask
        while m:
            b = m & -m
            h[self.labels[b.bit_length() - 1]] += 1
            m ^= b
        return h

    def _candidates(self, remaining: int) -> set[tuple]:
        """Literal sets drawn from the surviving rows, so coverage is never zero."""
        cands: set[tuple] = set()
        m = remaining
        while m:
            b = m & -m
            i = b.bit_length() - 1
            m ^= b
            fs = sorted(
                (f for f in self.feats[i] if f in self.usable),
                key=lambda f: self.df[f],
            )[: self.max_pairs_per_row]
            for a in fs:
                cands.add(((a, True),))
            for x in range(len(fs)):
                for y in range(x + 1, len(fs)):
                    cands.add(((fs[x], True), (fs[y], True)))
        # Negated literals are not enumerated here: they are proposed only as a
        # repair on a candidate that just missed min_precision (see fit()).
        return cands

    def _mask(self, lits: tuple) -> int:
        m = self.all_mask
        for f, positive in lits:
            m &= self.bits[f] if positive else ~self.bits[f] & self.all_mask
        return m

    def fit(self, verbose=False) -> list[dict]:
        remaining = self.all_mask
        rules: list[dict] = []
        while True:
            cands = self._candidates(remaining)
            best = None
            for lits in cands:
                cov = self._mask(lits) & remaining
                k = cov.bit_count()
                if k < self.min_cov:
                    continue
                hist = self._label_hist(cov)
                label, hits = hist.most_common(1)[0]
                prec = hits / k
                if prec < self.min_prec:
                    # one refinement pass: try excluding the feature most
                    # responsible for the impurity
                    wrong = cov & ~self._label_bits(label) & self.all_mask
                    blame = Counter(
                        f for f in self.usable
                        if (self.bits[f] & wrong).bit_count() > 0
                        and (self.bits[f] & cov & self._label_bits(label)).bit_count() == 0
                    )
                    if not blame or len(lits) > 1:
                        continue
                    f_bad = max(blame, key=lambda f: (self.bits[f] & wrong).bit_count())
                    lits2 = lits + ((f_bad, False),)
                    cov2 = self._mask(lits2) & remaining
                    k2 = cov2.bit_count()
                    if k2 < self.min_cov:
                        continue
                    hist2 = self._label_hist(cov2)
                    label2, hits2 = hist2.most_common(1)[0]
                    prec2 = hits2 / k2
                    if prec2 < self.min_prec:
                        continue
                    lits, cov, k, label, prec, hits = lits2, cov2, k2, label2, prec2, hits2
                # Laplace-corrected precision is the standard CN2 answer to a
                # greedy learner preferring a 3-row rule at 1.000 over a 60-row
                # rule at 0.95 — which is precisely the overfitting path here.
                sc = prec if self.score_fn == "precision" else (hits + 1) / (k + 2)
                key = (round(sc, 6), k)
                if best is None or key > best[0]:
                    best = (key, lits, label, cov, k, prec)
            if best is None:
                break
            _, lits, label, cov, k, prec = best
            rules.append({"literals": [[f, p] for f, p in lits], "label": label,
                          "n": k, "precision": round(prec, 4)})
            remaining &= ~cov & self.all_mask
            if verbose and len(rules) % 20 == 0:
                print(f"  {len(rules):3d} rules, {remaining.bit_count():4d} rows uncovered")
        return rules

    def _label_bits(self, label) -> int:
        m = 0
        for i, l in enumerate(self.labels):
            if l == label:
                m |= 1 << i
        return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", choices=["schema", "open", "cues"], default="schema")
    ap.add_argument("--min-prec", type=float, default=0.85)
    ap.add_argument("--min-cov", type=int, default=3)
    ap.add_argument("--max-feats", type=int, default=14)
    ap.add_argument("--train", type=Path, default=HERE / "data" / "family_a.jsonl")
    ap.add_argument("--no-bigrams", action="store_true")
    ap.add_argument("--overlap", type=int, default=0,
                    help="add ov:<label> features for the top-N schema matches")
    ap.add_argument("--score", choices=["precision", "laplace"], default="precision")
    ap.add_argument("--tag", default=None, help="output name suffix")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = [json.loads(x) for x in a.train.read_text().splitlines() if x.strip()]
    cat = load_catalogue("session")
    vocab = {"schema": schema_vocab(cat), "open": None, "cues": set()}[a.vocab]
    print(f"train {len(rows)} rows, vocab={a.vocab}"
          f"{f' ({len(vocab)} words)' if vocab else ''}, "
          f"min_prec={a.min_prec}, min_cov={a.min_cov}")

    texts = label_text(cat) if a.overlap else None
    f = Fitter(rows, vocab, a.min_prec, a.min_cov, a.max_feats,
               bigrams=not a.no_bigrams, texts=texts, top=a.overlap, score=a.score)
    print(f"  {len(f.usable)} usable literals of {len(f.df)} features")
    rules = f.fit(verbose=True)
    covered = sum(r["n"] for r in rules)
    print(f"  {len(rules)} rules, {covered}/{len(rows)} train rows covered "
          f"({covered / len(rows):.3f})")

    out = a.out or HERE / f"rules_{a.tag or a.vocab}.json"
    out.write_text(json.dumps({
        "vocab": a.vocab, "bigrams": not a.no_bigrams, "overlap": a.overlap,
        "score": a.score, "min_precision": a.min_prec, "min_coverage": a.min_cov,
        "max_feats_per_row": a.max_feats, "train": a.train.name,
        "n_train": len(rows), "rules": rules}, indent=1) + "\n")
    print(f"  wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
