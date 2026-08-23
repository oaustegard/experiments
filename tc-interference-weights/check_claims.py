#!/usr/bin/env python3
"""Re-derive the numbers in the review directly from the paper's own figure data files.
Usage: python3 check_claims.py <mirror_dir>"""
import json,math,re,sys
import numpy as np
M=sys.argv[1] if len(sys.argv)>1 else "mirror"
F=f"{M}/shared/data/figures"
LAB={'emb_qk':'QK','emb_enc':'Tokens→Features','emb_ov_enc':'Tokens→OV→Features',
     'emb_unemb':'Tokens→Logits','dec_unemb':'Features→Logits','emb_ov_unemb':'Tokens→OV→Logits'}

def jsvar(path,name):
    s=open(path,encoding='utf-8').read()
    m=re.search(re.escape(name)+r'\s*=\s*',s); i=m.end()
    dec=json.JSONDecoder(); return dec.raw_decode(s,i)[0]

print("="*72)
print("[1] Pruning: Fisher effectiveness vs raw |virtual weight|, per family/sign")
print("    Paper: Fisher wins 'for every density and individual weight family")
print("    (except for negative Features→Logits weights)'.")
print("="*72)
d=jsvar(f"{F}/threshold_data.js","window.THRESH_PANELS")
xs=np.arange(0.05,0.96,0.05)
for p in d:
    w=np.array(p['weight']); f=np.array(p['fisher'])
    wi=np.interp(xs,w[:,0],w[:,1]); fi=np.interp(xs,f[:,0],f[:,1])
    lose=(fi>wi)&(wi>1e-4)          # ignore densities where both ΔL is numerical noise
    if lose.any():
        worst=(fi/wi)[lose].max()
        print(f"  {LAB[p['row']]:22s} {p['col']:14s} Fisher WORSE at {lose.sum():2d}/19 densities, up to {worst:.2f}x")
print("  (all other family/sign combinations: Fisher wins at every density)")

print()
print("="*72)
print("[2] ROPE: fraction of weights helpful at a practically-meaningful effect size")
print("    Paper's main text leads with 47.6% helpful / 'tens of percent' significant.")
print("="*72)
s=open(f"{F}/rope_data.js",encoding='utf-8').read()
XS=json.loads(s[s.index('['):s.index(']')+1])
pan=jsvar(f"{F}/rope_data.js","window.ROPE_PANELS")
def buckets(p,e):
    j=min(range(len(XS)),key=lambda k:abs(XS[k]-e))
    c=[p['cum'][a][j] for a in range(3)]
    return c[0],c[1]-c[0],c[2]-c[1],1-c[2]      # neg, practically-zero, uncertain, pos
tot=0; agg={}
for key,tag in (("__zero__","eps -> 0 (main text convention)"),("tot","eps = budget/N_total = 1.5e-8 nats/tok"),("fam","eps = budget/N_family")):
    acc=np.zeros(4); tot=0
    for p in pan:
        N=int(re.search(r'N=([\d,]+)',p['title']).group(1).replace(',',''))
        e=-12 if key=="__zero__" else p[key]
        acc+=np.array(buckets(p,e))*N; tot+=N
    n,z,u,po=acc/tot
    print(f"  {tag:42s} neg {n:5.1%}  zero {z:5.1%}  uncertain {u:5.1%}  POSITIVE {po:5.1%}")
print(f"  population N = {tot:,};  2.9% of that = {0.029*tot:,.0f} weights vs 2.9M transformer params")

print()
print("="*72)
print("[3] On the helpful arm, helpfulness IS Fisher effectiveness")
print("    2nd-order expansion of the paper's own helpfulness formula gives")
print("      E[dL] = w*E[s(1_{j=t}-p_j)] + fisher(w) + O((sw)^3)")
print("    i.e. helpfulness = -w*dL/dw + fisher  (classic OBD saliency).")
print("    Prediction: log-log slope 1.00, ratio h/fisher ~ 1.")
print("="*72)
d=jsvar(f"{F}/fisher_panels_data.js","window.FISHER_PANELS")
print(f"  {'family':24s} {'slope':>6s} {'r':>6s} {'median h/fisher':>16s}")
for p in d:
    xt=p['xticks']; yt=[t for t in p['yticks'] if 'exp' in t and not t.get('neg')]
    ax,bx=np.polyfit([t['f'] for t in xt],[float(t['exp'].replace('−','-')) for t in xt],1)
    ay,by=np.polyfit([t['f'] for t in yt],[float(t['exp'].replace('−','-')) for t in yt],1)
    X=[];Y=[]
    for fx,fy in p['series']['mean']:
        if fy>=0.45: continue                     # helpful (upper) branch only
        lx=ax*fx+bx
        if lx<-12: continue                       # drop the noise floor
        X.append(lx); Y.append(ay*fy+by)
    X=np.array(X); Y=np.array(Y)
    m,_=np.polyfit(X,Y,1); r=np.corrcoef(X,Y)[0,1]
    print(f"  {LAB[p['title'].split()[0]]:24s} {m:6.2f} {r:6.3f} {10**np.median(Y-X):16.2f}")
