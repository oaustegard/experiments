import json, sys, statistics as st, concurrent.futures as cf
sys.path.insert(0,'.')
import hyde_lib as H
res = {r["tid"]: r for r in json.load(open("eval_results.json"))}
cases = [(m,q) for m,q in json.load(open("evalset.json")) if m in res]

def probe(case):
    tid,q = case
    b = H._ids(H.recall(q, n=40))
    def rk(lst, k):
        for i,m in enumerate(lst[:k]):
            if str(m).startswith(tid): return i+1
    return tid, {"b10":rk(b,10), "b20":rk(b,20), "b40":rk(b,40)}

with cf.ThreadPoolExecutor(max_workers=6) as ex:
    deep = dict(ex.map(probe, cases))
json.dump(deep, open("deep.json","w"))

n=len(cases)
def rr(r): return 1.0/r if r else 0.0
def rep(name, f):
    h=sum(1 for t,_ in cases if f(t)); print(f"{name:34} {h}/{n} = {h/n:.3f}")
print(f"{'policy':34} R (target found)")
print("-"*52)
rep("plain recall @10",            lambda t: deep[t]["b10"])
rep("plain recall @20  [control]", lambda t: deep[t]["b20"])
rep("plain recall @40  [control]", lambda t: deep[t]["b40"])
rep("hyde @10",                    lambda t: res[t]["ranks"]["hyde"])
rep("UNION base@10 + hyde@10",     lambda t: deep[t]["b10"] or res[t]["ranks"]["hyde"])
# how much does the union actually cost in extra rows?
