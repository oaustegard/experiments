"""Our engine (TorchTess exact, tessdata_best eng, beam decode) on clean, shaded and real lines: as is,
flat-fielded, tightly cropped to the ink, and both. Run from tesseract-gpu/: python3 run/crop_check.py
Measured 2026-11-20: CER clean 1.06 -> 0.28 (tight), shade 1.48 -> 0.52 (tight+ff), real 14.16 -> 13.04 (tight+ff,
case-insensitive). Tesseract CLI psm 7 on the same lines: 0.31, 4.86, 12.87 (results/flatfield_check.json)."""
import sys, json, numpy as np
sys.path.insert(0,'run'); sys.path.insert(0,'.')
from PIL import Image
from tess_baseline import cer
from tessgpu import beam, improve, ref
from server.torch_engine import TorchTess
from flatfield_check import flatfield
import torch; torch.set_num_threads(3)
eng=TorchTess('models/best/eng.traineddata',device='cpu',mode='exact')
rows=[]
for base,keep in (('data/lines',{'shade','clean'}),('data/real',{'real'})):
    for l in open(f'{base}/labels.jsonl'):
        r=json.loads(l)
        if r['lang']=='eng' and r['cond'] in keep:
            from pathlib import Path
            p=Path(base)/r['file']; p=p if p.exists() else Path(r['file'])
            r['gray']=np.asarray(Image.open(p).convert('L')); rows.append(r)
def readall(fn):
    imgs=[improve.prepare(fn(r['gray']),{},eng.H,None) for r in rows]
    out=[]
    for c in range(0,len(imgs),200):
        for p in eng.recognize(imgs[c:c+200]):
            ids=beam.decode(p,eng.model) if len(p) else []
            out.append(''.join(eng.model.unicharset[u] for u in ids))
    return out
base=readall(lambda g:g); ff=readall(flatfield)
res={}
for r,b,f in zip(rows,base,ff):
    nc=r['cond']=='real'; g=(lambda s:s.casefold()) if nc else (lambda s:s)
    res.setdefault(r['cond'],[]).append((cer(g(r['text']),g(b)),cer(g(r['text']),g(f))))
SUMMARY = {}
for k,v in res.items():
    a=np.array(v); SUMMARY.setdefault(k, {})['none'] = round(float(a[:,0].mean()), 5); SUMMARY[k]['flatfield'] = round(float(a[:,1].mean()), 5); print(f"ours {k:6} n={len(v)} CER {100*a[:,0].mean():.2f}% -> flatfield {100*a[:,1].mean():.2f}%  empty-ish(>0.5) {int((a[:,0]>0.5).sum())}")

def tight(g, pad=4):
    f = flatfield(g)  # dark ink on light paper after flatfield
    ink = f < 128
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return g
    y0, y1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + pad + 1)
    return g[y0:y1, x0:x1]
for name, fn in (("tight", tight), ("tight+ff", lambda g: flatfield(tight(g)))):
    out = readall(fn)
    res2 = {}
    for r, b, o in zip(rows, base, out):
        nc = r['cond'] == 'real'; gg = (lambda s: s.casefold()) if nc else (lambda s: s)
        res2.setdefault(r['cond'], []).append((cer(gg(r['text']), gg(b)), cer(gg(r['text']), gg(o))))
    for k, v in res2.items():
        a = np.array(v); SUMMARY[k][name] = round(float(a[:,1].mean()), 5); print(f"{name:8} {k:6} CER {100*a[:,0].mean():.2f}% -> {100*a[:,1].mean():.2f}%")

json.dump({"note": "eng tessdata_best, TorchTess exact + beam; mean CER per condition; real scored case-insensitively",
           "cer": SUMMARY}, open("results/crop_check.json", "w"), indent=1)
