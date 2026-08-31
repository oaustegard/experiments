"""Does the register-anchored prompt beat the novelty-anchored one on Gemini too?

Haiku collapses under the novelty prompt (0.100 acc@1 against a 0.500 no-LLM
control) because it obeys "novel, never-seen-before" and invents new VOCABULARY.
Gemini flash-lite tolerates the same prompt (0.575) by half-ignoring it. If the
register prompt is neutral-or-better on Gemini, it is the right default for both.
"""
import json, sys, numpy as np, concurrent.futures as cf
sys.path.insert(0,'/home/user/muninn-utilities')
import bench
from muninn_utils.hypothetical_classifier import _invoke, _MODEL, Vocabulary

REGISTER = """You are writing entries for a furniture and home-goods product taxonomy.

For each search query below, write the taxonomy category label that a retailer WOULD file
that query under. Write it the way the taxonomy writes labels: a short plural noun phrase,
the plainest possible retail wording, 1-4 words.

Do not worry about whether the label already exists in any real taxonomy - write the
obvious one. Do not invent novel or creative wording. Do not use marketing adjectives.
Do not explain.

Examples of the register:
  Coffee Tables
  Throw Pillows
  Dressers & Chests
  Wall & Accent Mirrors
  Bar Stools
  Kids Beds

Output one line per item, in the same order, formatted exactly as:
<n>. <label>

ITEMS:
{numbered}"""

import re
A = json.load(open("artifacts_lite.json"))
queries, goldn, vocab = A["queries"], A["gold"], A["vocab"]
gold = np.array([vocab.index(c) for c in goldn])

def batched(qs, batch=40):
    chunks=[qs[i:i+batch] for i in range(0,len(qs),batch)]
    def one(ch):
        numbered="\n".join(f"{i+1}. {q}" for i,q in enumerate(ch))
        r=_invoke(REGISTER.format(numbered=numbered), _MODEL, 4000) or ""
        got={}
        for line in r.splitlines():
            m=re.match(r"\s*(\d+)[.)]\s*(.+)",line)
            if m: got[int(m.group(1))]=m.group(2).strip().strip('"')
        return [got.get(i+1,"") for i in range(len(ch))]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        return [x for ch in ex.map(one,chunks) for x in ch]

reg = batched(queries)
json.dump(reg, open("register_hall.json","w"))
print(f"generated={np.mean([bool(x.strip()) for x in reg]):.3f}  e.g. {list(zip(queries[:5],reg[:5]))}", flush=True)

print(f"\n{'arm':46} {'acc@1':>6} {'acc@3':>6}")
print("-"*62)
for bk in ("minilm","tfidf"):
    V=Vocabulary(vocab, backend=bk)
    for nm, texts in ((f"{bk}: query -> label [control]", queries),
                      (f"{bk}: novelty prompt", A["hall"]),
                      (f"{bk}: REGISTER prompt", reg)):
        hits=V.snap(texts,k=3); idx=np.array([[vocab.index(l) for l,_ in r] for r in hits])
        print(f"{nm:46} {np.mean([g==p[0] for p,g in zip(idx,gold)]):6.3f} "
              f"{np.mean([g in p for p,g in zip(idx,gold)]):6.3f}")
