"""Measure HyDE expansion against plain recall on the synthetic eval set."""
import json, sys, os, statistics as st, concurrent.futures as cf
sys.path.insert(0, '.')
import hyde_lib as H

N, DEPTH = 10, 20

def rank(ids, tid):
    for i, m in enumerate(ids):
        if str(m).startswith(tid): return i + 1
    return None

def rr(r): return 1.0 / r if r else 0.0

def one(case, stats):
    tid, q = case
    try:
        base = H.recall(q, n=DEPTH)
        ex = [(r.get("summary") or "")[:300] for r in list(base)[:5]]
        docs = H.pseudo_docs(q, ex, samples=3) if ex else []
        picked = H.select_terms(docs, q, cap=12, stats=stats) if docs else []
        terms = [t for t, _, _ in picked]
        hyde = H.recall(q + " " + " ".join(terms), n=DEPTH) if terms else []
        b, h = H._ids(base), H._ids(hyde)
        arms = {
            "base":       b[:N],
            "hyde":       h[:N],
            "interleave": H.fuse_interleave(b, h, N),
            "append":     H.fuse_append(b, h, N),
            "rrf":        H.fuse_rrf(b, h, N),
            "rrf_wb2":    H.fuse_rrf(b, h, N, wb=2.0, wh=1.0),
        }
        return dict(tid=tid, q=q, nterms=len(terms),
                    ranks={k: rank(v, tid) for k, v in arms.items()})
    except Exception as e:
        return dict(tid=tid, q=q, err=f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    cases = json.load(open("evalset.json"))
    stats = H.corpus_stats()
    print(f"cases={len(cases)} corpus_n={stats['n']}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(lambda c: one(c, stats), cases))
    ok = [r for r in rows if "ranks" in r]
    json.dump(ok, open("eval_results.json", "w"), indent=1)
    print(f"ok={len(ok)} errs={len(rows)-len(ok)}\n", flush=True)
    ARMS = ["base", "hyde", "interleave", "append", "rrf", "rrf_wb2"]
    print(f"{'arm':12} {'R@10':>6} {'MRR':>7} {'vs base W/L':>12}")
    print("-" * 42)
    for a in ARMS:
        hits = sum(1 for r in ok if r["ranks"][a])
        mrr = st.mean(rr(r["ranks"][a]) for r in ok)
        w = sum(1 for r in ok if rr(r["ranks"][a]) > rr(r["ranks"]["base"]))
        l = sum(1 for r in ok if rr(r["ranks"][a]) < rr(r["ranks"]["base"]))
        print(f"{a:12} {hits/len(ok):6.3f} {mrr:7.4f} {str(w)+'/'+str(l):>12}")
    print()
    miss = [r for r in ok if not r["ranks"]["base"] and r["ranks"]["hyde"]]
    print(f"rescued by hyde (base missed, hyde found): {len(miss)}")
    for r in miss[:8]: print(f"  @{r['ranks']['hyde']:>2} {r['q'][:88]}")
    lost = [r for r in ok if r["ranks"]["base"] and not r["ranks"]["hyde"]]
    print(f"\nlost by hyde (base found, hyde missed): {len(lost)}")
    for r in lost[:8]: print(f"  base@{r['ranks']['base']:>2} {r['q'][:84]}")
