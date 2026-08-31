"""Hypothetical classification (softwaredoug 2026-08-10) on WANDS query_class.

Task: map a search query to one of 860 legal `product_class` values.
Gold:  WANDS `query.csv:query_class` (single label per query).

Arms
  lexical    char-ngram TF-IDF: snap the RAW QUERY to nearest label. No model at all.
  direct     MiniLM: snap the RAW QUERY to nearest label. No LLM.
  hallucinate  cheap LLM invents a free-form class, MiniLM snaps it. Doug's pattern.
  hall+query embed hallucination AND query, average, then snap.
"""
from __future__ import annotations
import csv, json, os, sys, time, threading, concurrent.futures as cf
import numpy as np
sys.path.append('/mnt/skills/user/invoking-gemini/scripts')

WANDS = os.environ.get("WANDS", "/home/user/wands")
csv.field_size_limit(10**7)

def load():
    vocab = sorted({(r.get('product_class') or '').strip()
                    for r in csv.DictReader(open(f'{WANDS}/product.csv'), delimiter='\t')} - {''})
    qs = [(r['query'], (r.get('query_class') or '').strip())
          for r in csv.DictReader(open(f'{WANDS}/query.csv'), delimiter='\t')]
    vs = set(vocab)
    qs = [(q, c) for q, c in qs if c in vs]
    return vocab, qs

HALL_PROMPT = """Your task is to create a novel, never-seen-before furniture, home goods, or
hardware product classification that best fits a search query.

Product classifications might look like:

Coffee Tables
Throw Pillows
Dressers & Chests
Food Storage & Canisters
Stackable Chairs
Kids Beds

Here's the query to generate a classification for:

{query}

Output the single classification only. No explanation, no quotes."""

_sem = threading.Semaphore(3)
def gem(**kw):
    from gemini_client import invoke_gemini
    for a in range(6):
        try:
            with _sem: return invoke_gemini(**kw)
        except Exception as e:
            if "429" not in str(e) and "Rate limited" not in str(e): raise
            time.sleep(1.5 * 2 ** a)
    return None

def hallucinate(queries, model="lite", workers=3):
    def one(q):
        r = gem(prompt=HALL_PROMPT.format(query=q), model=model,
                max_output_tokens=200, thinking_level="minimal", temperature=0.7)
        return (r or "").strip().strip('"').split("\n")[0]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, queries))

def norm(a): return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)

def snap(vecs, label_vecs, k=3):
    """Nearest labels by cosine. Returns (n, k) index array."""
    sims = norm(vecs) @ norm(label_vecs).T
    return np.argsort(-sims, axis=1)[:, :k]

def acc(pred_idx, gold_idx, k):
    return float(np.mean([g in p[:k] for p, g in zip(pred_idx, gold_idx)]))
