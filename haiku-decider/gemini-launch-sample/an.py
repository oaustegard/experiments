import json,collections
it={json.loads(l)['id']:json.loads(l) for l in open('items.jsonl')}
R=[json.loads(l) for l in open('stack.jsonl')]
def ece(ps,ys,b=10):
    t=0
    for i in range(b):
        s=[(p,y) for p,y in zip(ps,ys) if i/b<p<=(i+1)/b or (i==0 and p==0)]
        if s: t+=len(s)/len(ps)*abs(sum(p for p,_ in s)/len(s)-sum(y for _,y in s)/len(s))
    return t
def mf1(pred,gold,labs):
    f=[]
    for c in labs:
        tp=sum(p==c==g for p,g in zip(pred,gold)); fp=sum(p==c!=g for p,g in zip(pred,gold)); fn=sum(g==c!=p for p,g in zip(pred,gold))
        f.append(0 if tp==0 else 2*tp/(2*tp+fp+fn))
    return sum(f)/len(f)
by=collections.defaultdict(list)
for r in R: by[it[r['id']]['bench']].append(r)
for b,rs in by.items():
    g=[r['label'] for r in rs]; labs=sorted(set(it[rs[0]['id']]['questions'][0]['choices'][i]['value'] for i in range(len(it[rs[0]['id']]['questions'][0]['choices']))))
    sp=[max(r['probs'],key=r['probs'].get) for r in rs]
    s1=[max(r['per_sample'][0],key=r['per_sample'][0].get) for r in rs]
    acc=lambda p:sum(a==b for a,b in zip(p,g))/len(g)
    un=[r['agreement']==1 for r in rs]
    ua=sum(p==y for p,y,u in zip(sp,g,un) if u)/max(1,sum(un)); sa=sum(p==y for p,y,u in zip(sp,g,un) if not u)/max(1,len(un)-sum(un))
    e1=ece([max(r['per_sample'][0].values()) for r in rs],[a==y for a,y in zip(s1,g)])
    es=ece([max(r['probs'].values()) for r in rs],[a==y for a,y in zip(sp,g)])
    print(f"{b:10s} n={len(rs)} acc 1-pass {acc(s1):.3f} stack {acc(sp):.3f} | macroF1 1p {mf1(s1,g,labs):.3f} st {mf1(sp,g,labs):.3f} | ECE 1p {e1:.3f} st {es:.3f} | unanimous {sum(un)}/{len(un)} acc {ua:.2f}, split acc {sa:.2f}")
    print('   confusion stack:',collections.Counter(zip(g,sp)).most_common(6))
