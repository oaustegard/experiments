"""Does the pattern recover on Muninn tags if the hallucination is multi-label?

muninn_tags.py asked for ONE invented tag per memory and lost to the no-LLM control
(0.240 vs 0.400 @1). Two candidate explanations, both testable:
  (a) the bottleneck: 1500 chars -> one short string discards most of the summary,
      while the control snaps the whole summary and keeps it;
  (b) the arity: gold carries 4.8 tags, so one invention is a bad draw.
Fix (b) by inventing 5, snapping each, and taking the union.
"""
import sys, json, random, collections, re
sys.path.insert(0,'/mnt/skills/user'); sys.path.insert(0,'/home/user/muninn-utilities')
import numpy as np
from muninn_utils.memory_tfidf import MemoryIndex
from muninn_utils.hypothetical_classifier import Vocabulary, _invoke, _MODEL
import concurrent.futures as cf

MIN_USES, N_EVAL, SEED, K_TAGS = 3, 250, 20260831, 5
def tags_of(m):
    t=m.get('tags') or []
    if isinstance(t,str): t=[x.strip() for x in t.split(',') if x.strip()]
    return set(t)

idx=MemoryIndex(); idx.build()
counts=collections.Counter(t for m in idx.meta for t in tags_of(m))
vocab_labels=sorted(t for t,c in counts.items() if c>=MIN_USES); VS=set(vocab_labels)
rows=[(idx.summaries[i], tags_of(idx.meta[i])&VS) for i in range(len(idx.ids))
      if len(idx.summaries[i])>300 and tags_of(idx.meta[i])&VS]
random.seed(SEED); rows=random.sample(rows,min(N_EVAL,len(rows)))
S=[s for s,_ in rows]; G=[g for _,g in rows]
print(f"vocab={len(vocab_labels)} rows={len(rows)} mean gold={np.mean([len(g) for g in G]):.1f}",flush=True)

P="""Invent {k} novel, never-seen-before topic tags for the engineering-memory entry below.
Tags are single words or hyphenated phrases, like: ccotw / correction / atproto /
paper-insight / architecture / perch / session-log.

Invent freely — do not try to recall real tags, do not explain. Output {k} tags on one
line, comma separated, nothing else.

ENTRY:
{entry}"""

def one(s):
    r=_invoke(P.format(k=K_TAGS, entry=s[:1500]), _MODEL, 300) or ""
    return [t.strip().strip('"').lower() for t in re.split(r"[,\n]", r) if t.strip()][:K_TAGS]
with cf.ThreadPoolExecutor(max_workers=3) as ex: H=list(ex.map(one,S))
json.dump(H, open("muninn_tags2_hall.json","w"))
print("hallucinated; e.g.", H[0], flush=True)

print(f"\n{'arm':44} {'@1':>6} {'@3':>6} {'@5':>6}")
print("-"*66)
for backend in ("tfidf","minilm"):
    V=Vocabulary(vocab_labels, backend=backend)
    ctl=V.snap(S,k=5)
    flat=[t for tags in H for t in tags]
    snapped=V.snap(flat,k=1) if flat else []
    per, c = [], 0
    for tags in H:
        per.append([snapped[c+i][0][0] for i in range(len(tags))]); c+=len(tags)
    def hitc(k): return float(np.mean([any(l in g for l,_ in r[:k]) for r,g in zip(ctl,G)]))
    def hith(k): return float(np.mean([any(l in g for l in p[:k]) for p,g in zip(per,G)]))
    print(f"{backend+': summary -> tag  [no LLM control]':44} {hitc(1):6.3f} {hitc(3):6.3f} {hitc(5):6.3f}")
    print(f"{backend+f': {K_TAGS} invented tags -> tag':44} {hith(1):6.3f} {hith(3):6.3f} {hith(5):6.3f}")
    # union of both
    def hitu(k):
        return float(np.mean([any(l in g for l,_ in r[:k]) or any(l in g for l in p[:k])
                              for r,p,g in zip(ctl,per,G)]))
    print(f"{backend+': control UNION invented':44} {hitu(1):6.3f} {hitu(3):6.3f} {hitu(5):6.3f}")
