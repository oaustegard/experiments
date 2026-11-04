"""Does the shipped model's own answer-token probability tell its right answers from its wrong ones?

    python3 confidence_probe.py     # runs/skygest_e8__bridge.jsonl + judge cache -> results/confidence_probe.json

Teacher-forces each closed paper/fact answer (first 64 tokens) on models/skygest_e8 and reports AUROC of the mean,
minimum and first-8 token log-probability against the judge's 'correct', and accuracy when only the most confident
fraction is answered. A check on whether 'I don't know' training (R-Tuning) has a signal to learn from.
"""
import json, re, hashlib, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, '.')
torch.set_num_threads(3)
cache=json.load(open('data/judge_cache.json'))
ho={r['id']:r for f in ('heldout','seen') for r in map(json.loads,open(f'data/{f}.jsonl'))}
ev={r['id']:r for r in map(json.loads,open('data/eval_prompts.jsonl'))}
def verdict(r):
    s=ev[r['id']]['set']
    if s=='facts': return 'correct' if re.search(ev[r['id']]['check'], r['answer'], re.I) else 'wrong'
    if not r['answer'].strip(): return 'wrong'
    return cache.get(hashlib.sha256((ho[r['id']]['q']+'\x00'+r['answer']).encode()).hexdigest()[:24])
def auroc(pos,neg):
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/max(1,len(pos)*len(neg))
P=lambda q:f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n"
tok=AutoTokenizer.from_pretrained('models/skygest_e8'); m=AutoModelForCausalLM.from_pretrained('models/skygest_e8',dtype=torch.float32).eval()
rows=[json.loads(l) for l in open('runs/skygest_e8__bridge.jsonl')]
rows=[r for r in rows if ev[r['id']]['set'] in ('papers','facts') and r['closed'] and r['answer'].strip()]
out=[]
with torch.no_grad():
    for r in rows:
        v=verdict(r)
        if v is None: continue
        pre=tok(P(ev[r['id']]['q'])+r['trace']+'</think>\n\n',add_special_tokens=False)['input_ids']
        ans=tok(r['answer'],add_special_tokens=False)['input_ids'][:64]
        ids=torch.tensor([pre+ans]); lp=torch.log_softmax(m(ids).logits[0,len(pre)-1:-1].float(),-1)
        tl=lp[torch.arange(len(ans)),torch.tensor(ans)]
        out.append({'id':r['id'],'set':ev[r['id']]['set'],'v':v,'mean':float(tl.mean()),'min':float(tl.min()),'first8':float(tl[:8].mean())})
json.dump(out,open('results/confidence_probe.json','w'),indent=0)
for k in ('mean','min','first8'):
    c=[o[k] for o in out if o['v']=='correct']; w=[o[k] for o in out if o['v']!='correct']
    print(k,'n_correct',len(c),'n_other',len(w),'AUROC',round(auroc(c,w),3))
# abstain-below-threshold tradeoff on 'mean'
srt=sorted(out,key=lambda o:-o['mean'])
for frac in (1.0,0.75,0.5,0.25):
    k=int(len(srt)*frac); a=srt[:k]
    print(f'answer top {frac:.0%}: n={k} correct={sum(o["v"]=="correct" for o in a)/k:.0%}')
