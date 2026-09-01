"""Most faithful port: query = the pseudo-document itself (no term filtering)."""
import json, sys, concurrent.futures as cf
sys.path.insert(0,'.')
import hyde_lib as H
res = {r["tid"]: r for r in json.load(open("eval_results.json"))}
deep = json.load(open("deep.json"))
cases = [(m,q) for m,q in json.load(open("evalset.json")) if m in res]

def one(case):
    tid,q = case
    try:
        base = H.recall(q, n=20)
        ex = [(r.get("summary") or "")[:300] for r in list(base)[:5]]
        docs = H.pseudo_docs(q, ex, samples=3)
        if not docs: return tid, {}
        # (a) pure HyDE: the concatenated pseudo-docs ARE the query (Eq. 8 -> +query)
        pure = H._ids(H.recall(q + " " + " ".join(docs), n=20))
        # (b) unconditioned generator, pure form (no corpus exemplars)
        raw  = H.pseudo_docs(q, ["(no exemplars)"], samples=3)
        uncond = H._ids(H.recall(q + " " + " ".join(raw), n=20)) if raw else []
        def rk(l,k):
            for i,m in enumerate(l[:k]):
                if str(m).startswith(tid): return i+1
        return tid, {"pure10":rk(pure,10), "pure20":rk(pure,20),
                     "unc10":rk(uncond,10)}
    except Exception as e:
        return tid, {"err":str(e)[:60]}

with cf.ThreadPoolExecutor(max_workers=3) as ex:
    out = dict(ex.map(one, cases))
json.dump(out, open("faithful.json","w"))
n=len(cases); ok=[t for t,_ in cases if "err" not in out[t] and out[t]]
print(f"n={len(ok)}")
def rep(name,f): 
    h=sum(1 for t in ok if f(t)); print(f"{name:36} {h}/{len(ok)} = {h/len(ok):.3f}")
print("-"*54)
rep("plain recall @10",              lambda t: deep[t]["b10"])
rep("plain recall @20",              lambda t: deep[t]["b20"])
rep("plain recall @40",              lambda t: deep[t]["b40"])
rep("HyDE terms @10 (our impl)",     lambda t: res[t]["ranks"]["hyde"])
rep("HyDE pure doc-as-query @10",    lambda t: out[t].get("pure10"))
rep("HyDE pure doc-as-query @20",    lambda t: out[t].get("pure20"))
rep("HyDE pure, no exemplars @10",   lambda t: out[t].get("unc10"))
