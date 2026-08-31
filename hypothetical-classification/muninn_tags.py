"""In-house application: snap hallucinated tags onto Muninn's real tag vocabulary.

Muninn's store has 3,052 memories carrying 5,575 DISTINCT tags — most used once or
twice. That sprawl is what free-form tagging produces. Shipping the vocabulary to
constrain it costs ~30k input tokens per call, which is the condition under which
hallucinate-and-snap is the right tool.

Held-out: hide a memory's real tags, hallucinate tags from its summary alone, snap
to the tags used >= 3 times, and check whether the snapped tag is one the memory
actually carries.
"""
import sys, json, random, collections
sys.path.insert(0, '/mnt/skills/user'); sys.path.insert(0, '/home/user/muninn-utilities')
import numpy as np
from muninn_utils.memory_tfidf import MemoryIndex
from muninn_utils.hypothetical_classifier import Vocabulary, classify

MIN_USES, N_EVAL, SEED = 3, 250, 20260831

def tags_of(meta):
    t = meta.get('tags') or []
    if isinstance(t, str): t = [x.strip() for x in t.split(',') if x.strip()]
    return set(t)

idx = MemoryIndex(); idx.build()
counts = collections.Counter(t for m in idx.meta for t in tags_of(m))
vocab_labels = sorted(t for t, c in counts.items() if c >= MIN_USES)
print(f"memories={len(idx.ids)} distinct_tags={len(counts)} vocab(>={MIN_USES})={len(vocab_labels)}")

rows = [(idx.summaries[i], tags_of(idx.meta[i]) & set(vocab_labels))
        for i in range(len(idx.ids))
        if len(idx.summaries[i]) > 300 and tags_of(idx.meta[i]) & set(vocab_labels)]
random.seed(SEED); rows = random.sample(rows, min(N_EVAL, len(rows)))
summaries = [s for s, _ in rows]; gold = [g for _, g in rows]
print(f"eval rows={len(rows)}  mean gold tags/row={np.mean([len(g) for g in gold]):.1f}")

EXAMPLES = ["ccotw", "correction", "atproto", "paper-insight", "architecture", "perch"]
DOMAIN = "single-word or hyphenated topic tag for an engineering memory store"

results = {}
for backend in ("tfidf", "minilm"):
    V = Vocabulary(vocab_labels, backend=backend)
    # control: no LLM, snap the summary itself
    ctl = V.snap(summaries, k=3)
    # the pattern
    out = classify(summaries, V, domain=DOMAIN, examples=EXAMPLES, k=3, blend=False)
    def hit(ranked, g, k): return any(l in g for l, _ in ranked[:k])
    results[backend] = {
        "control@1": float(np.mean([hit(r, g, 1) for r, g in zip(ctl, gold)])),
        "control@3": float(np.mean([hit(r, g, 3) for r, g in zip(ctl, gold)])),
        "hall@1":    float(np.mean([hit(o.alternatives, g, 1) for o, g in zip(out, gold)])),
        "hall@3":    float(np.mean([hit(o.alternatives, g, 3) for o, g in zip(out, gold)])),
        "generated": float(np.mean([o.ok for o in out])),
    }
    json.dump([{"summary": s[:200], "gold": sorted(g), "hall": o.hallucination,
                "snapped": [l for l, _ in o.alternatives]}
               for s, g, o in zip(summaries, gold, out)][:40],
              open(f"muninn_tags_examples_{backend}.json", "w"), indent=1)
    print(f"\n[{backend}] " + "  ".join(f"{k}={v:.3f}" for k, v in results[backend].items()), flush=True)

json.dump(results, open("muninn_tags_results.json", "w"), indent=1)
print(f"\n{'arm':40} {'@1':>6} {'@3':>6}")
print("-" * 54)
for b in results:
    print(f"{b+': summary -> tag  [no LLM control]':40} {results[b]['control@1']:6.3f} {results[b]['control@3']:6.3f}")
    print(f"{b+': hallucinated tag -> tag':40} {results[b]['hall@1']:6.3f} {results[b]['hall@3']:6.3f}")
