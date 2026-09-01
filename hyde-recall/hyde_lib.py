"""HyDE over Muninn's FTS5 corpus — candidate implementation, pluggable fusion."""
from __future__ import annotations
import math, os, re, sys, time, json, pickle, threading, concurrent.futures as cf
sys.path.insert(0, '/mnt/skills/user'); sys.path.insert(0, os.path.expanduser('~'))
sys.path.append('/mnt/skills/user/invoking-gemini/scripts')
from remembering.scripts.memory import recall
from gemini_client import invoke_gemini

CACHE = os.path.expanduser("~/.cache/muninn/hyde_corpus_df.json")
_TTL = 24 * 3600
_lock = threading.Lock()
_sem = threading.Semaphore(3)          # CF AI Gateway 429s above ~3 concurrent
_stats = None

def corpus_stats(force=False):
    """Document frequencies over memory summaries. The corpus-vocabulary bottleneck
    needs these; MemoryIndex.build() is ~5s and hits Turso, so cache on disk."""
    global _stats
    with _lock:
        if _stats is not None and not force:
            return _stats
        if not force and os.path.exists(CACHE) and time.time() - os.path.getmtime(CACHE) < _TTL:
            _stats = json.load(open(CACHE)); return _stats
        from muninn_utils.memory_tfidf import MemoryIndex
        import numpy as np
        idx = MemoryIndex(); idx.build()
        df = np.asarray((idx.matrix > 0).sum(axis=0)).ravel()
        inv = {v: k for k, v in idx.vectorizer.vocabulary_.items()}
        _stats = {"df": {inv[i]: int(df[i]) for i in range(len(df))}, "n": len(idx.ids)}
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        json.dump(_stats, open(CACHE, "w"))
        return _stats

def _terms(t): return re.findall(r"[a-z][a-z0-9_\-]{2,}", t.lower())

INSTR = """You are writing a HYPOTHETICAL entry for the memory store excerpted below.

Study the excerpts for register, vocabulary and naming conventions. Then write ONE entry
(60-90 words) that would answer the question - the entry that SHOULD exist in this store.

Rules:
- Imitate the excerpts' idiom: the same kind of system names, tool names, error shapes,
  file paths, project nouns.
- Invent plausible specifics freely. Factual accuracy does not matter. Vocabulary does.
- Do not hedge, do not ask questions, do not mention that this is hypothetical.
- Output the entry text only."""

def _gem(**kw):
    for a in range(5):
        try:
            with _sem: return invoke_gemini(**kw)
        except Exception as e:
            if "429" not in str(e) and "Rate limited" not in str(e): raise
            time.sleep(1.5 * 2 ** a)
    return None

def pseudo_docs(question, exemplars, *, samples=3, model="lite"):
    ex = "\n\n".join(f"- {e}" for e in exemplars)
    p = f"{INSTR}\n\nEXCERPTS FROM THE STORE:\n{ex}\n\nQUESTION: {question}\n\nENTRY:"
    def one(_):
        return _gem(prompt=p, model=model, max_output_tokens=700,
                    thinking_level="minimal", temperature=1.0)
    with cf.ThreadPoolExecutor(max_workers=min(samples, 3)) as ex_:
        return [d for d in ex_.map(one, range(samples)) if d]

def select_terms(docs, question, *, cap=12, max_df_frac=0.05, min_df=2, stats=None):
    """A generated term survives only if the CORPUS contains it, discriminatively.
    This is what the dense encoder's bottleneck does in the original HyDE."""
    st = stats or corpus_stats(); DF, N = st["df"], st["n"]
    qt = set(_terms(question)); support = {}
    for d in docs:
        for t in set(_terms(d)):
            if t not in qt: support[t] = support.get(t, 0) + 1
    out = []
    for t, s in support.items():
        d = DF.get(t, 0)
        if d < min_df or d > max_df_frac * N: continue
        out.append((s * math.log((N + 1) / (d + 0.5)), t, s, d))
    out.sort(reverse=True)
    return [(t, s, d) for _, t, s, d in out[:cap]]

# ---- fusion strategies ----
def fuse_interleave(base, hyde, n):
    seen, out = set(), []
    for i in range(max(len(base), len(hyde))):
        for lst in (base, hyde):
            if i < len(lst) and lst[i] not in seen:
                seen.add(lst[i]); out.append(lst[i])
                if len(out) >= n: return out
    return out[:n]

def fuse_rrf(base, hyde, n, k=60, wb=1.0, wh=1.0):
    sc = {}
    for lst, w in ((base, wb), (hyde, wh)):
        for i, m in enumerate(lst): sc[m] = sc.get(m, 0.0) + w / (k + i + 1)
    return sorted(sc, key=lambda m: -sc[m])[:n]

def fuse_append(base, hyde, n):
    out = list(base); seen = set(base)
    for m in hyde:
        if m not in seen: seen.add(m); out.append(m)
    return out[:n]

def _ids(rows): return [str(r.get("id")) for r in rows]

def hyde_recall(question, *, n=10, samples=3, model="lite", cap=12,
                max_df_frac=0.05, min_df=2, n_exemplars=5, fusion="interleave",
                depth=None, stats=None, **rk):
    depth = depth or n
    base = recall(question, n=depth, **rk)
    ex = [(r.get("summary") or "")[:300] for r in list(base)[:n_exemplars]]
    docs = pseudo_docs(question, ex, samples=samples, model=model) if ex else []
    picked = select_terms(docs, question, cap=cap, max_df_frac=max_df_frac,
                          min_df=min_df, stats=stats) if docs else []
    terms = [t for t, _, _ in picked]
    hyde = recall(question + " " + " ".join(terms), n=depth, **rk) if terms else []
    lut = {str(r.get("id")): r for r in list(base) + list(hyde)}
    F = {"interleave": fuse_interleave, "rrf": fuse_rrf, "append": fuse_append}[fusion]
    ids = F(_ids(base), _ids(hyde), n)
    return dict(results=[lut[i] for i in ids], base=list(base), hyde=list(hyde),
                terms=terms, picked=picked, docs=docs)
