"""Model-based property test for /mnt/skills/user/bm25/scripts/bm25.py.

Replays generated histories of corpus edits and queries against the real
CorpusIndex (via get_or_build_index) and an independent plain-Python
Okapi BM25 (Lucene variant, k1=1.5, b=0.75 -- the bm25s defaults the skill uses).

Two machines:
  NoCacheMachine  -- scoring/ranking correctness, rebuild every query
  CacheMachine    -- same, with the session cache on (the real CLI default)

Run:  python3 -m pytest -q test_bm25_model.py
Mutation calibration: BM25_MUTANT=drop_last python3 -m pytest ...
"""
import importlib.util
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from hypothesis import HealthCheck, event, settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

SKILL = Path(os.environ.get("BM25_PY", "/mnt/skills/user/bm25/scripts/bm25.py"))
spec = importlib.util.spec_from_file_location("bm25cli", SKILL)
bm25cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm25cli)

MUTANT = os.environ.get("BM25_MUTANT")
if MUTANT == "drop_last":  # plausible defect: off-by-one in the corpus walk
    _orig = bm25cli.discover_files
    def _drop_last(*a, **kw):
        files = list(_orig(*a, **kw))
        return iter(files[:-1])
    bm25cli.discover_files = _drop_last
elif MUTANT == "query_case":  # plausible defect: query not lowercased like the corpus
    import bm25s as _b
    def _q(self, q, k=10):
        t = _b.tokenize([q], lower=False, stopwords=None, show_progress=False)
        i, s = self.retriever.retrieve(t, k=min(k, len(self.docs)), show_progress=False)
        return [(int(a), float(b)) for a, b in zip(i[0], s[0])]
    bm25cli.CorpusIndex.query = _q

# ---------- independent reference model ----------
TOKEN = re.compile(r"(?u)\b\w\w+\b")  # documented bm25s default tokenization
K1, B = 1.5, 0.75

def tok(text):
    return TOKEN.findall(text.lower())

def model_scores(docs: dict, query: str) -> dict:
    """path -> Lucene BM25 score. Plain loops, no shared code with bm25s."""
    toks = {p: tok(t) for p, t in docs.items()}
    n = len(toks)
    if n == 0:
        return {}
    avgdl = sum(len(t) for t in toks.values()) / n
    df = Counter()
    for t in toks.values():
        df.update(set(t))
    out = {}
    for p, t in toks.items():
        tf = Counter(t)
        dl = len(t)
        s = 0.0
        for term in tok(query):
            if tf[term] == 0:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            norm = K1 * (1 - B + B * dl / avgdl) if avgdl else K1
            s += idf * tf[term] / (tf[term] + norm)
        out[p] = s
    return out

# ---------- generators: small pools so operations interact ----------
NAMES = st.sampled_from(["a.txt", "b.txt", "c.md", "d.py"])
WORDS = st.sampled_from(["alpha", "beta", "gamma", "delta", "Alpha", "x", "beta-gamma", "épée"])
TEXT = st.lists(WORDS, min_size=0, max_size=8).map(" ".join)
QUERY = st.lists(WORDS, min_size=1, max_size=3).map(" ".join)
K = st.integers(min_value=1, max_value=6)
KNOWN_EMPTY_VOCAB = os.environ.get("SKIP_KNOWN", "1") == "1"
TOL = 1e-3  # bm25s stores float32


class _Base(RuleBasedStateMachine):
    use_cache = False

    @initialize()
    def setup(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bm25pt-"))
        self.root = self.tmp / "corpus"
        self.root.mkdir()
        bm25cli.CACHE_ROOT = self.tmp / "cache"
        self.docs = {}

    def teardown(self):
        if hasattr(self, "tmp"):
            shutil.rmtree(self.tmp, ignore_errors=True)

    @rule(name=NAMES, text=TEXT)
    def write(self, name, text):
        (self.root / name).write_text(text, encoding="utf-8")
        self.docs[name] = text

    @rule(name=NAMES)
    def delete(self, name):
        if name in self.docs:
            (self.root / name).unlink()
            del self.docs[name]

    @rule(q=QUERY, k=K)
    def query(self, q, k):
        if KNOWN_EMPTY_VOCAB and not any(tok(t) for t in self.docs.values()):
            return  # finding 1 (empty vocabulary crash) -- tested in test_empty_vocab_*
        idx, _, _ = bm25cli.get_or_build_index(
            self.root, [], [], 2_000_000, use_cache=self.use_cache)
        # contents seen by the index must be the live corpus
        assert dict(zip(idx.paths, idx.docs)) == self.docs, "index contents != corpus"
        if not self.docs:
            return  # empty-corpus behaviour checked separately
        hits = idx.query(q, k=k)
        event(f"N={len(self.docs)} k<N={k < len(self.docs)}")
        nz = sum(1 for v in model_scores(self.docs, q).values() if v > 0)
        event(f"nonzero-scored docs={nz}")
        want = model_scores(self.docs, q)
        assert len(hits) == min(k, len(self.docs)), (hits, k)
        got_paths = [idx.paths[i] for i, _ in hits]
        assert len(set(got_paths)) == len(got_paths), f"duplicate hits {got_paths}"
        for p, s in zip(got_paths, (s for _, s in hits)):
            assert abs(s - want[p]) <= TOL * max(1, want[p]), (p, s, want[p], q)
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True), f"not ranked {scores}"
        # valid top-k: nothing left out beats the weakest result
        left_out = [want[p] for p in self.docs if p not in got_paths]
        if left_out and hits:
            assert max(left_out) <= min(scores) + TOL, (left_out, scores, q)


class NoCacheMachine(_Base):
    use_cache = False


class CacheMachine(_Base):
    use_cache = True


CFG = settings(max_examples=int(os.environ.get("N", "300")), stateful_step_count=25,
               deadline=None, suppress_health_check=list(HealthCheck))
TestNoCache = NoCacheMachine.TestCase
TestNoCache.settings = CFG
TestCache = CacheMachine.TestCase
TestCache.settings = CFG


def test_empty_corpus_query():
    """CLI on an empty dir: should return no hits, not crash."""
    tmp = Path(tempfile.mkdtemp())
    try:
        idx = bm25cli.CorpusIndex.build(tmp, [], [], 2_000_000)
        assert idx.query("alpha", k=5) == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_empty_vocab_tokenless_files():
    """Files exist but contain no 2+ char tokens (e.g. a one-letter README)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "a.txt").write_text("x y z")
        idx = bm25cli.CorpusIndex.build(tmp, [], [], 2_000_000)
        assert [s for _, s in idx.query("alpha", k=5)] == [0.0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
