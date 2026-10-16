import json, sys, math
from collections import defaultdict

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h), min(1, c + h))

rows = [json.loads(l) for l in open(sys.argv[1])]
cell = defaultdict(list)
for r in rows: cell[(r["model"], r["think"], r["e"])].append(r)
print(f"{'model':7} {'effort':6} {'e':>4} {'acc':>7} {'95% CI':>13} {'parse_fail':>10} {'think>0':>8} {'med_think':>9} {'cost$':>7}")
for k in sorted(cell):
    rs = cell[k]; n = len(rs); ok = sum(r["ok"] for r in rs)
    pf = sum(r["got"] is None for r in rs)
    th = sorted((r["thinking_tokens"] or 0) for r in rs); tpos = sum(t > 0 for t in th)
    lo, hi = wilson(ok, n); cost = sum(r["cost"] or 0 for r in rs)
    print(f"{k[0]:7} {k[1]:6} {k[2]:>4} {ok:>3}/{n:<3} [{lo:.2f},{hi:.2f}] {pf:>10} {tpos:>8} {th[n//2]:>9} {cost:>7.3f}")
# accuracy split by whether the model actually thought
print("\nby actual thinking (pooled over effort):")
sp = defaultdict(lambda: [0, 0])
for r in rows:
    key = (r["model"], r["e"], "thought" if (r["thinking_tokens"] or 0) > 0 else "no-think")
    sp[key][0] += r["ok"]; sp[key][1] += 1
for k in sorted(sp): print(k, f"{sp[k][0]}/{sp[k][1]}")
bad = [r for r in rows if not r["ok"]]
print("\nwrong examples:"); [print(r["model"], r["think"], r["e"], r["expect"], r["got"], r["thinking_tokens"], repr(r["raw"][:60])) for r in bad[:15]]
print("total cost $", round(sum(r["cost"] or 0 for r in rows), 3))
