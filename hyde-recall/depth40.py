import json, sys, concurrent.futures as cf
sys.path.insert(0,'.')
import hyde_lib as H
res = {r["tid"]: r for r in json.load(open("eval_results.json"))}
cases = [(m,q) for m,q in json.load(open("evalset.json")) if m in res]

def one(case):
    tid,q = case
    try:
        base = H.recall(q, n=40); bids = H._ids(base)
        ex = [(r.get("summary") or "")[:300] for r in list(base)[:5]]
        docs = H.pseudo_docs(q, ex, samples=3)
        if not docs: return tid, {}
        pure = H._ids(H.recall(q + " " + " ".join(docs), n=40))
        picked = H.select_terms(docs, q, cap=12)
        terms = [t for t,_,_ in picked]
        tq = H._ids(H.recall(q + " " + " ".join(terms), n=40)) if terms else []
        def rk(l,k):
            for i,m in enumerate(l[:k]):
                if str(m).startswith(tid): return i+1
        return tid, {"docs":docs, "terms":terms,
                     "b":{k:rk(bids,k) for k in (10,20,40)},
                     "pure":{k:rk(pure,k) for k in (10,20,40)},
                     "terms_r":{k:rk(tq,k) for k in (10,20,40)},
                     "union40":rk(list(dict.fromkeys(bids[:20]+pure[:20])),40)}
    except Exception as e:
        return tid, {"err":str(e)[:80]}

with cf.ThreadPoolExecutor(max_workers=3) as ex:
    out = dict(ex.map(one, cases))
json.dump(out, open("depth40.json","w"), indent=1)
ok=[t for t,_ in cases if out[t] and "err" not in out[t]]
print(f"n={len(ok)}  errs={len(cases)-len(ok)}")
print(f"{'policy':32} {'R':>6}")
print("-"*40)
def rep(nm,f):
    h=sum(1 for t in ok if f(t)); print(f"{nm:32} {h/len(ok):6.3f}  ({h}/{len(ok)})")
for k in (10,20,40):
    rep(f"plain recall @{k}",        lambda t,k=k: out[t]["b"][k])
for k in (10,20,40):
    rep(f"HyDE pure doc @{k}",       lambda t,k=k: out[t]["pure"][k])
for k in (10,20,40):
    rep(f"HyDE filtered terms @{k}", lambda t,k=k: out[t]["terms_r"][k])
rep("UNION base@20 + pure@20",       lambda t: out[t]["union40"])
