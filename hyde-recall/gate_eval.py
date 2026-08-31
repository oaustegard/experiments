import json, sys, statistics as st, concurrent.futures as cf
sys.path.insert(0,'.'); sys.path.insert(0,'/home/user/muninn-utilities')
import hyde_lib as H
from muninn_utils.recall_sufficiency import term_coverage_judge

res = {r["tid"]: r for r in json.load(open("eval_results.json"))}
cases = [(m,q) for m,q in json.load(open("evalset.json")) if m in res]

def gate(case):
    tid, q = case
    base = H.recall(q, n=20)
    pool = [{"text": (r.get("summary") or ""), "tags": r.get("tags") or []} for r in list(base)[:5]]
    out = {}
    for th in (0.50, 0.66, 0.80, 0.90):
        out[th] = term_coverage_judge(q, pool, threshold=th).sufficient
    return tid, out

with cf.ThreadPoolExecutor(max_workers=6) as ex:
    gates = dict(ex.map(gate, cases))
json.dump({k:{str(a):b for a,b in v.items()} for k,v in gates.items()}, open("gates.json","w"))

def rr(r): return 1.0/r if r else 0.0
n = len(cases)
def report(name, pick):
    hits = sum(1 for t,_ in cases if pick(t))
    mrr  = st.mean(rr(pick(t)) for t,_ in cases)
    b    = sum(1 for t,_ in cases if rr(pick(t)) > rr(res[t]["ranks"]["base"]))
    w    = sum(1 for t,_ in cases if rr(pick(t)) < rr(res[t]["ranks"]["base"]))
    print(f"{name:26} {hits/n:6.3f} {mrr:7.4f}  {b}/{w}")

print(f"n={n}\n{'policy':26} {'R@10':>6} {'MRR':>7}  W/L")
print("-"*50)
report("base only",        lambda t: res[t]["ranks"]["base"])
report("hyde always",      lambda t: res[t]["ranks"]["hyde"])
report("rrf always",       lambda t: res[t]["ranks"]["rrf"])
for th in (0.50,0.66,0.80,0.90):
    fired = sum(1 for t,_ in cases if not gates[t][th])
    report(f"gate@{th:.2f} -> hyde ({fired} fire)",
           lambda t, th=th: res[t]["ranks"]["base"] if gates[t][th] else res[t]["ranks"]["hyde"])
report("ORACLE base-else-hyde", lambda t: res[t]["ranks"]["base"] or res[t]["ranks"]["hyde"])
