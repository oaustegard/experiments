#!/usr/bin/env python3
"""Does one rung up the NLP stack — spaCy — close the held-out gap regex left?

`RESULTS.md` diagnosed the fitted decision list's collapse (0.984 fitted -> 0.239
held out) as a *vocabulary* failure, not a structure failure: a fitted rule
learns `tok:diff` and never fires on "what code does this PR actually change",
where a human writes `\\b(diff|patch|changeset)\\b` from knowledge of the
language. Three things spaCy has and a regex does not are candidate fixes, and
they are separable, so this file measures them one at a time:

1. **lemmatisation** — the cheapest hypothesis. If the gap were morphology
   ("cancelling" vs "cancel", "reviews" vs "review"), lemmatising both sides of
   a lexical-overlap router would close some of it. `spacy-lemma` against the
   identical scorer over raw tokens (`tok-overlap`) isolates exactly that.
2. **static word vectors** — the only component that can supply *synonymy*, and
   therefore the only one aimed at the actual diagnosis. `RESULTS.md` names the
   canonical miss: "approve" appears in no description among the 79 labels, so
   no amount of lexical matching reaches `pull_request_read::get_reviews`.
   Two variants, because label texts share a tool description and differ only by
   the method's two words — a plain mean vector should drown that difference and
   an IDF-weighted one should partly recover it.
3. **syntax** — ROOT verb and direct object, so "go ahead and merge it" reduces
   to (merge, it) rather than being matched on "go" and "ahead". The wild split
   is heavy on pronoun referents, so the pronoun-dobj rate is reported as a
   result in its own right: it bounds what any (verb, object) router can do.

Every arm scores against the *schema text* rather than against learned tokens,
which is the one intervention `RESULTS.md` found to help — it is the only source
that can carry a word the training queries never contained.

    python3 spacy_arms.py            # table across all three splits + results_spacy.json
    python3 spacy_arms.py --tune     # re-derive the abstention thresholds on family A

Requires `spacy` and `en_core_web_md`; raises ImportError when either is absent
so `arms.load_all()` skips the module rather than failing the whole evaluation.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Importing spaCy from this directory needs a workaround, and it is a finding.
#
# spaCy's own dependency chain imports a PyPI package named `catalogue`
# (thinc/spacy use it as a function registry). This experiment has a local
# `catalogue.py`. Running anything from this directory therefore shadows it and
# `import spacy` dies with
#
#     AttributeError: module 'catalogue' has no attribute 'create'
#
# — an AttributeError, not an ImportError, so it does not look like a missing
# dependency and would not be skipped by an `except ImportError` guard. The fix
# is to hide this directory from `sys.path` and stash any already-imported local
# `catalogue` while spaCy and the model load; spaCy binds the module object it
# got, so restoring afterwards leaves both importers working (verified: the
# local `catalogue.load("session")` still returns its 50 tools and `nlp()` still
# lemmatises after the swap back).
# ---------------------------------------------------------------------------


_SPACY_CATALOGUE = None  # the PyPI `catalogue`, kept out of the local one's way


class _unshadowed:
    """Hide this directory (and the local `catalogue`) for the duration of a spaCy call.

    Both modules want the top-level name `catalogue`, so exactly one can own it
    at a time. spaCy's imports bind module *objects*, so the name only has to be
    correct while spaCy code is executing; restoring afterwards — including
    *removing* the PyPI module when the local one was never imported — leaves
    `from catalogue import load` working for the rest of the harness.
    """

    def __enter__(self):
        self.saved = sys.modules.pop("catalogue", None)
        if _SPACY_CATALOGUE is not None:
            sys.modules["catalogue"] = _SPACY_CATALOGUE
        self.hidden = [p for p in sys.path if p in ("", ".", str(HERE))]
        for p in self.hidden:
            sys.path.remove(p)
        return self

    def __exit__(self, *exc):
        sys.path[:0] = self.hidden
        if self.saved is not None:
            sys.modules["catalogue"] = self.saved
        else:
            sys.modules.pop("catalogue", None)
        return False


try:
    with _unshadowed():
        import spacy as _spacy
        _SPACY_CATALOGUE = sys.modules["catalogue"]
except Exception as e:  # surface any failure as ImportError, per arms.load_all
    raise ImportError(f"spacy unavailable: {type(e).__name__}: {e}") from e

MODEL = "en_core_web_md"

# Check the model at *import* time, not at first route: `arms.load_all()` skips a
# module that raises ImportError, and a missing model should skip this file
# rather than blow up mid-evaluation when an arm is constructed.
with _unshadowed():
    if not _spacy.util.is_package(MODEL):
        raise ImportError(f"spaCy model {MODEL} not installed "
                          f"(python3 -m spacy download {MODEL})")

_NLP = None
LOAD_MS = 0.0


def nlp():
    """Load the model once. Load cost is reported separately from per-query cost."""
    global _NLP, LOAD_MS
    if _NLP is None:
        t0 = time.perf_counter()
        try:
            with _unshadowed():
                _NLP = _spacy.load(MODEL)
        except OSError as e:
            raise ImportError(f"spaCy model {MODEL} not installed: {e}") from e
        finally:
            LOAD_MS = (time.perf_counter() - t0) * 1000
    return _NLP


from arms import ArmBase, labels, register  # noqa: E402
from fit import tokens as raw_tokens  # noqa: E402  (same tokeniser the fitted arms use)

# ---------------------------------------------------------------------------
# Label texts. Same sources as fit.label_text — tool name, title, description,
# plus the method's own words — so a lemma/vector arm is compared against the
# fitted arms' view of the catalogue and not against a richer one.
# ---------------------------------------------------------------------------

_STOPISH = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "is", "it",
            "this", "that", "with", "be", "by", "or", "as", "at", "from"}


def label_strings(catalogue: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for name, tool in sorted(catalogue.items()):
        base = f"{name.replace('_', ' ')}. {tool['title']}. {tool['description']}"
        m = tool["params"].get("method")
        if m and m.get("enum"):
            for e in m["enum"]:
                out[f"{name}::{e}"] = f"{str(e).replace('_', ' ')}. {base}"
        else:
            out[name] = base
    return out


def _idf(bags: dict[str, set[str]]) -> dict[str, float]:
    n = len(bags)
    df = Counter(w for b in bags.values() for w in b)
    return {w: math.log(n / c) for w, c in df.items()}


def _overlap(qbag: set[str], bags: dict[str, set[str]],
             idf: dict[str, float]) -> list[tuple[str, float]]:
    """IDF-weighted overlap, length-normalised — fit.overlap_feats' scorer verbatim.

    Kept identical on purpose: the only difference between `tok-overlap` and
    `spacy-lemma` must be the tokens, so any delta is attributable to lemmas.
    """
    scored = []
    for label, words in bags.items():
        hit = qbag & words
        if hit:
            scored.append((label, sum(idf.get(w, 0.0) for w in hit) / len(words) ** 0.5))
    scored.sort(key=lambda x: -x[1])
    return scored


def _content(doc):
    """Content tokens: what a vector or a lemma bag should be built from."""
    return [t for t in doc
            if not t.is_punct and not t.is_space and t.lemma_.lower() not in _STOPISH
            and len(t.lemma_) > 1]


class _SchemaArm(ArmBase):
    """Shared plumbing: label texts, a per-instance doc cache, a top-1 with a floor.

    The doc cache is per instance and every query in a split is unique, so the
    call `eval.score` times is always a cache miss — the cache only spares the
    second parse that `ArmBase.call` would trigger for argument binding, and does
    not flatter the reported latency.
    """

    THRESH = 0.0

    def __init__(self, thresh: float | None = None):
        self.thresh = self.THRESH if thresh is None else thresh
        self._docs: dict[str, object] = {}
        self.labels = labels(self.catalogue)
        self.texts = label_strings(self.catalogue)
        nlp()  # warm, so the first routed query does not pay the model load

    def doc(self, q: str):
        d = self._docs.get(q)
        if d is None:
            d = self._docs[q] = nlp()(q)
        return d

    def route(self, query: str) -> str | None:
        s = self.score(query)
        return s[0][0] if s and s[0][1] > self.thresh else None


# ---------------------------------------------------------------------------
# 1. Lemmatisation — is the held-out gap morphology?
# ---------------------------------------------------------------------------


class TokenOverlapArm(_SchemaArm):
    """Control: the same scorer over raw regex tokens. No spaCy in the path."""

    THRESH = 0.90  # every THRESH here is argmax over family A only (`--tune`)

    def __init__(self, thresh: float | None = None):
        super().__init__(thresh)
        self.bags = {lab: set(raw_tokens(t)) for lab, t in self.texts.items()}
        self.idf = _idf(self.bags)

    def qbag(self, query: str) -> set[str]:
        return set(raw_tokens(query))

    def score(self, query: str) -> list[tuple[str, float]]:
        return _overlap(self.qbag(query), self.bags, self.idf)


class LemmaOverlapArm(TokenOverlapArm):
    """Both sides lemmatised. The delta against TokenOverlapArm *is* the answer."""

    THRESH = 0.90

    def __init__(self, thresh: float | None = None):
        super().__init__(thresh)
        self.bags = {lab: {t.lemma_.lower() for t in _content(nlp()(txt))}
                     for lab, txt in self.texts.items()}
        self.idf = _idf(self.bags)

    def qbag(self, query: str) -> set[str]:
        return {t.lemma_.lower() for t in _content(self.doc(query))}


# ---------------------------------------------------------------------------
# 2. Static word vectors — the arm aimed at the actual diagnosis.
# ---------------------------------------------------------------------------


def _unit(v):
    import numpy as np
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class VectorArm(_SchemaArm):
    """Cosine between the query's mean token vector and each label's.

    Prediction worth stating before the number: the nine `pull_request_read`
    methods share one ~60-word description and differ by two words, so a plain
    mean drowns the method signal. `weighted=True` scales each token by its IDF
    over the label texts, which is the cheapest way to let those two words matter.
    """

    THRESH = 0.65   # mean vectors; the IDF variant is registered at 0.55
    WEIGHTED = False

    def __init__(self, thresh: float | None = None, weighted: bool | None = None):
        super().__init__(thresh)
        import numpy as np
        self.weighted = self.WEIGHTED if weighted is None else weighted
        bags = {lab: {t.lemma_.lower() for t in _content(nlp()(txt))}
                for lab, txt in self.texts.items()}
        self.idf = _idf(bags)
        mat = []
        for lab in self.labels:
            mat.append(self._vec(nlp()(self.texts[lab])))
        self.mat = np.vstack(mat)

    def _vec(self, doc):
        import numpy as np
        toks = [t for t in _content(doc) if t.has_vector]
        if not toks:
            return np.zeros(nlp().vocab.vectors.shape[1], dtype="float32")
        if self.weighted:
            w = np.array([max(self.idf.get(t.lemma_.lower(), 3.0), 0.05) for t in toks],
                         dtype="float32")
        else:
            w = np.ones(len(toks), dtype="float32")
        v = (np.vstack([t.vector for t in toks]) * w[:, None]).sum(0) / w.sum()
        return _unit(v.astype("float32"))

    def score(self, query: str) -> list[tuple[str, float]]:
        q = self._vec(self.doc(query))
        sims = self.mat @ q
        order = sims.argsort()[::-1]
        # Full ranking, not a top-k: the interesting question about vectors is
        # where the gold label lands when it is *not* first, and a truncated
        # list makes "rank 40" and "unreachable" look identical.
        return [(self.labels[i], float(sims[i])) for i in order]


# ---------------------------------------------------------------------------
# 3. Syntax — ROOT verb and its direct object.
# ---------------------------------------------------------------------------

# "go ahead and merge it" parses with ROOT=go and merge as a conj; "can you show
# me the diff" with ROOT=show under an aux. Descending through these light and
# modal heads is the difference between routing on "go" and routing on "merge".
_LIGHT = {"go", "please", "can", "could", "would", "let", "want", "need", "try",
          "help", "make", "give", "do", "be", "have", "start", "keep"}
_PRON = {"it", "this", "that", "them", "these", "those", "one", "ones", "mine",
         "he", "she", "they", "him", "her", "us", "me", "there"}


def verb_object(doc) -> tuple[str | None, str | None, bool]:
    """(verb lemma, object lemma, object_is_pronoun) for the request's main clause."""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return None, None, False
    verb = root
    seen = 0
    # Descend at most twice: go -> merge, or show -> (xcomp) list.
    while seen < 2:
        seen += 1
        if verb.pos_ in ("VERB", "AUX") and verb.lemma_.lower() not in _LIGHT:
            break
        nxt = next((c for c in verb.children
                    if c.dep_ in ("conj", "xcomp", "ccomp", "advcl") and c.pos_ == "VERB"), None)
        if nxt is None:
            break
        verb = nxt
    if verb.pos_ not in ("VERB", "AUX"):
        # Nominal request — "any open PRs?" — no verb to route on.
        verb = None
    obj = None
    if verb is not None:
        obj = next((c for c in verb.children if c.dep_ in ("dobj", "obj")), None)
        if obj is None:
            prep = next((c for c in verb.children if c.dep_ == "prep"), None)
            if prep is not None:
                obj = next((c for c in prep.children if c.dep_ == "pobj"), None)
    if obj is None:
        obj = next((t for t in doc if t.dep_ in ("dobj", "obj")), None)
    is_pron = bool(obj is not None and (obj.pos_ == "PRON" or obj.lemma_.lower() in _PRON))
    return (verb.lemma_.lower() if verb is not None else None,
            obj.lemma_.lower() if obj is not None else None, is_pron)


class SyntaxArm(_SchemaArm):
    """Route on (verb, dobj) matched against schema text, verb weighted heavier.

    A pronoun object carries no lexical content, so on those rows this degrades
    to a verb-only router by construction — which is why the pronoun rate below
    is a bound on the whole approach rather than an incidental statistic.
    """

    THRESH = 0.80
    VERB_W, OBJ_W = 2.0, 1.0

    def __init__(self, thresh: float | None = None):
        super().__init__(thresh)
        self.bags = {lab: {t.lemma_.lower() for t in _content(nlp()(txt))}
                     for lab, txt in self.texts.items()}
        self.idf = _idf(self.bags)

    def score(self, query: str) -> list[tuple[str, float]]:
        verb, obj, is_pron = verb_object(self.doc(query))
        if verb is None and obj is None:
            return []
        scored = []
        for lab, words in self.bags.items():
            s = 0.0
            if verb and verb in words:
                s += self.VERB_W * self.idf.get(verb, 0.0)
            if obj and not is_pron and obj in words:
                s += self.OBJ_W * self.idf.get(obj, 0.0)
            if s > 0:
                scored.append((lab, s / len(words) ** 0.5))
        scored.sort(key=lambda x: -x[1])
        return scored


# ---------------------------------------------------------------------------
# 4. Fusion. Each component is normalised by its own top score for the query, so
# the weights compare *shapes* of the two rankings rather than incomparable units.
# ---------------------------------------------------------------------------


class FusionArm(_SchemaArm):
    THRESH = 0.90
    W_LEX, W_VEC, W_SYN = 1.0, 0.5, 0.3

    def __init__(self, thresh: float | None = None, w_vec: float | None = None,
                 w_syn: float | None = None):
        super().__init__(thresh)
        self.w_vec = self.W_VEC if w_vec is None else w_vec
        self.w_syn = self.W_SYN if w_syn is None else w_syn
        self.lex = LemmaOverlapArm(thresh=-1)
        self.vec = VectorArm(thresh=-1, weighted=True)
        self.syn = SyntaxArm(thresh=-1)
        # Share the doc cache: three arms parsing the same query three times
        # would triple the latency for no information.
        self.lex._docs = self.vec._docs = self.syn._docs = self._docs

    def score(self, query: str) -> list[tuple[str, float]]:
        acc: dict[str, float] = {}
        for w, arm in ((self.W_LEX, self.lex), (self.w_vec, self.vec), (self.w_syn, self.syn)):
            s = arm.score(query)
            if not s:
                continue
            top = s[0][1]
            if top <= 0:
                continue
            for lab, v in s[:20]:
                acc[lab] = acc.get(lab, 0.0) + w * (v / top)
        return sorted(acc.items(), key=lambda x: -x[1])


register("tok-overlap", lambda: TokenOverlapArm())
register("spacy-lemma", lambda: LemmaOverlapArm())
register("spacy-vec-mean", lambda: VectorArm(weighted=False))
register("spacy-vec-idf", lambda: VectorArm(thresh=0.55, weighted=True))
register("spacy-syntax", lambda: SyntaxArm())
register("spacy-fusion", lambda: FusionArm())


# ---------------------------------------------------------------------------
# Standalone: the comparison table, the diagnostics, results_spacy.json.
# ---------------------------------------------------------------------------

ARMS = ["tok-overlap", "spacy-lemma", "spacy-vec-mean", "spacy-vec-idf",
        "spacy-syntax", "spacy-fusion"]

# The published numbers this has to beat, from RESULTS.md.
BASELINES = {
    "hand-written regex": (0.696, 0.546, 0.486),
    "best fitted list": (0.984, 0.239, 0.351),
    "structural cues only": (0.049, 0.029, 0.013),
}


def _splits():
    from eval import load_split
    return {
        "family A (fitted)": load_split(HERE / "data" / "family_a.jsonl"),
        "family B (held-out)": load_split(HERE / "data" / "family_b.jsonl"),
        "wild (hand-authored)": load_split(HERE / "wild.jsonl"),
    }


def tune(splits) -> dict[str, float]:
    """Pick each arm's abstention floor on family A only, then apply it unchanged.

    Family A is the split the fitted arms were fitted on, so tuning there keeps
    B and wild honestly held out for these arms too.
    """
    from eval import score
    rows = splits["family A (fitted)"]
    best = {}
    for name in ARMS:
        from arms import build
        arm = build(name)
        cand = []
        for th in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
            arm.thresh = th
            s = score(arm, rows)
            # Accuracy over routable rows plus abstention on off-topic ones, the
            # two numbers the threshold actually trades against each other.
            cand.append((s["label_acc"] + 0.2 * (s["abstain_acc"] or 0), th, s["label_acc"]))
        cand.sort(reverse=True)
        best[name] = cand[0][1]
        print(f"  {name:<18} thresh={cand[0][1]:<5} acc_A={cand[0][2]:.3f}")
    return best


def diagnostics(splits) -> dict:
    """The three questions the table cannot answer on its own."""
    out = {}
    from arms import build

    # (a) How often is the direct object a pronoun? This bounds any (verb, obj)
    # router, and RESULTS.md already measured the same collapse structurally:
    # 0.514 of wild rows carry any structural cue against 0.925 of generated ones.
    pron = {}
    for sname, rows in splits.items():
        on = [r for r in rows if r.get("label")]
        vo = [verb_object(nlp()(r["query"])) for r in on]
        n_obj = sum(1 for _, o, _ in vo if o)
        pron[sname] = {
            "n": len(on),
            "has_verb": round(sum(1 for v, _, _ in vo if v) / len(on), 4),
            "has_object": round(n_obj / len(on), 4),
            "pronoun_object_rate_of_all": round(sum(1 for _, _, p in vo if p) / len(on), 4),
            "pronoun_object_rate_of_objects": round(
                sum(1 for _, _, p in vo if p) / n_obj, 4) if n_obj else None,
        }
    out["syntax"] = pron

    # (b) The canonical synonym miss from RESULTS.md: "approve" appears in no
    # description among the 79 labels. Do vectors reach get_reviews anyway?
    probes = [
        ("has anyone approved it", "pull_request_read::get_reviews"),
        ("what code does this PR actually change", "pull_request_read::get_diff"),
        ("is the CI green on my PR yet", "pull_request_read::get_status"),
        ("go ahead and merge it", "merge_pull_request"),
        ("cancel that run, it's stuck", "actions_run_trigger::cancel_workflow_run"),
    ]
    lem, vec = build("spacy-lemma"), build("spacy-vec-idf")
    syn = build("spacy-syntax")
    out["synonym_probes"] = []
    for q, gold in probes:
        v, o, p = verb_object(nlp()(q))
        row = {"query": q, "gold": gold, "verb_obj": [v, o, p]}
        for tag, arm in (("lemma", lem), ("vec-idf", vec), ("syntax", syn)):
            s = arm.score(q)[:3]
            row[tag] = [[lab, round(sc, 3)] for lab, sc in s]
            row[f"{tag}_rank"] = next(
                (i + 1 for i, (lab, _) in enumerate(arm.score(q)) if lab == gold), None)
        out["synonym_probes"].append(row)

    # (c) Is "approve" anywhere in the schema at all? RESULTS.md says no; confirm
    # rather than repeat it, since the whole vector hypothesis rests on it.
    texts = label_strings(lem.catalogue)
    out["approve_in_schema"] = sorted(l for l, t in texts.items() if "approv" in t.lower())

    # (d) The decisive test of the synonymy hypothesis. Split each routable row
    # by whether the lexical arm shares *any* word with any label: on the
    # zero-overlap rows a lexical router is structurally blind, so if vectors
    # supply synonymy at all, this is where it has to show up.
    tok, fus = build("tok-overlap"), build("spacy-fusion")
    out["zero_overlap"] = {}
    out["recall_at_k"] = {}
    for sname, rows in splits.items():
        on = [r for r in rows if r.get("label")]
        blind = [r for r in on if not tok.score(r["query"])]
        seen = [r for r in on if tok.score(r["query"])]
        hit = lambda arm, rs: (
            round(sum(1 for r in rs
                      if (arm.score(r["query"]) or [(None, 0)])[0][0] == r["label"]) / len(rs), 4)
            if rs else None)
        out["zero_overlap"][sname] = {
            "n_routable": len(on), "n_blind": len(blind), "n_seen": len(seen),
            "share_blind": round(len(blind) / len(on), 4),
            "vec_idf_acc_on_blind": hit(vec, blind),
            "vec_idf_acc_on_seen": hit(vec, seen),
            "tok_acc_on_seen": hit(tok, seen),
        }
        # k=79 is the "reachable at all" rate, and it is the number that separates
        # the two families of arm: a lexical arm only ranks labels it shares a
        # word with, so its recall@79 is capped below 1.0, while the vector arm
        # ranks all 79 by construction and reaches 1.0 for free. Anything the
        # vector arm buys has to appear as recall it gains between those bounds.
        rk = {}
        for tag, arm in (("tok-overlap", tok), ("spacy-lemma", lem),
                         ("spacy-vec-idf", vec), ("spacy-fusion", fus)):
            ranked = [[l for l, _ in arm.score(r["query"])] for r in on]
            rk[tag] = {f"@{k}": round(
                sum(1 for r, order in zip(on, ranked) if r["label"] in order[:k]) / len(on), 4)
                for k in (1, 3, 5, 10, 79)}
        out["recall_at_k"][sname] = rk
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="re-derive thresholds on family A")
    a = ap.parse_args()

    from arms import build
    from eval import score

    t0 = time.perf_counter()
    nlp()
    print(f"model {MODEL}: load {LOAD_MS:.0f} ms "
          f"(pipes: {', '.join(nlp().pipe_names)}, vectors {nlp().vocab.vectors.shape})")

    splits = _splits()
    if a.tune:
        print("\ntuning abstention floor on family A:")
        tune(splits)
        return 0

    hdr = (f"{'arm':<18}{'split':<22}{'cov':>7}{'prec':>7}{'acc':>7}{'tool':>7}"
           f"{'meth':>7}{'abst':>7}{'args':>7}{'ms':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    out: dict[str, dict] = {}
    for name in ARMS:
        t1 = time.perf_counter()
        arm = build(name)
        build_ms = (time.perf_counter() - t1) * 1000
        for sname, rows in splits.items():
            s = score(arm, rows)
            s.pop("errors")
            s["build_ms"] = round(build_ms, 1)
            out.setdefault(name, {})[sname] = s
            f = lambda k: "  -  " if s[k] is None else f"{s[k]:.3f}"
            print(f"{name:<18}{sname:<22}{f('coverage'):>7}{f('precision'):>7}"
                  f"{f('label_acc'):>7}{f('tool_acc'):>7}{f('method_acc_given_tool'):>7}"
                  f"{f('abstain_acc'):>7}{f('args_acc'):>7}{s['median_latency_ms']:>9.4f}")
        print()

    print(f"{'reference (RESULTS.md)':<40}{'A':>8}{'B':>8}{'wild':>8}")
    for k, (x, y, z) in BASELINES.items():
        print(f"{k:<40}{x:>8.3f}{y:>8.3f}{z:>8.3f}")

    diag = diagnostics(splits)
    print("\nsyntactic coverage (routable rows):")
    for sname, d in diag["syntax"].items():
        print(f"  {sname:<24} verb {d['has_verb']:.3f}  obj {d['has_object']:.3f}  "
              f"pronoun-obj {d['pronoun_object_rate_of_all']:.3f} "
              f"({d['pronoun_object_rate_of_objects']} of objects)")
    print("\nsynonym probes (rank of gold label, None = unreachable):")
    for p in diag["synonym_probes"]:
        print(f"  {p['query'][:44]:<46} lemma {str(p['lemma_rank']):>5}  "
              f"vec {str(p['vec-idf_rank']):>5}  syn {str(p['syntax_rank']):>5}   "
              f"(verb,obj)={p['verb_obj'][0]},{p['verb_obj'][1]}")
    print(f"\nlabels whose schema text contains 'approv': {diag['approve_in_schema'] or 'none'}")
    print("\nrows with zero lexical overlap against every label (a lexical router is blind):")
    for sname, d in diag["zero_overlap"].items():
        print(f"  {sname:<24} blind {d['share_blind']:.3f} (n={d['n_blind']:>3} of "
              f"{d['n_routable']})  vec-idf acc there {d['vec_idf_acc_on_blind']}  "
              f"| on the other {d['n_seen']}: vec {d['vec_idf_acc_on_seen']} "
              f"tok {d['tok_acc_on_seen']}")
    print("\nrecall@k (k=79 is 'reachable at all'; lexical arms rank only labels "
          "they share a word with):")
    ks = ("@1", "@3", "@5", "@10", "@79")
    print(f"  {'split':<24}{'arm':<16}" + "".join(f"{k:>8}" for k in ks))
    for sname, d in diag["recall_at_k"].items():
        for tag, r in d.items():
            print(f"  {sname:<24}{tag:<16}" + "".join(f"{r[k]:>8.3f}" for k in ks))

    lat = [out[n][s]["median_latency_ms"] for n in out for s in out[n]]
    payload = {
        "model": MODEL, "spacy_version": _spacy.__version__,
        "model_load_ms": round(LOAD_MS, 1),
        "median_latency_ms_overall": round(statistics.median(lat), 4),
        "baselines_from_results_md": BASELINES,
        "arms": out, "diagnostics": diag,
    }
    (HERE / "results_spacy.json").write_text(json.dumps(payload, indent=1) + "\n")
    print("\nwrote results_spacy.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
