"""Markdown tables from results/eval.json (both K pooled and per K) for RESULTS.md."""
import json
from pathlib import Path
d = Path(__file__).resolve().parents[1]
e = json.load(open(d / "results/eval.json"))
rows = json.load(open(d / "results/eval_rows.json"))
arms = list(e["arms"]["K6"].keys())
cols = ["repeatCorr", "dE", "dEreg", "ssim", "gmsd", "gmsdEdge", "dist", "orientErr"]
# pooled over K: mean over all rows of the arm
pooled = {}
for a in arms:
    rs = [r for r in rows if r["arm"] == a]
    pooled[a] = {c: sum(r[c] for r in rs if r.get(c) == r.get(c) and r.get(c) is not None) / max(1, sum(1 for r in rs if r.get(c) == r.get(c) and r.get(c) is not None)) for c in cols}
    pooled[a]["n"] = len(rs)
json.dump(pooled, open(d / "results/eval_pooled.json", "w"), indent=1)
print("| arm | n | repeat corr | ΔE blur | ΔE region | SSIM | GMSD | GMSD edges | structure dist | orient err (rad) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for a in arms:
    p = pooled[a]
    has = {c: any(r.get(c) == r.get(c) and r.get(c) is not None for r in rows if r["arm"] == a) for c in cols}
    print(f"| {a} | {p['n']} | " + " | ".join("—" if not has[c] else f"{p[c]:.3f}" if c in ('repeatCorr', 'ssim', 'gmsd', 'gmsdEdge', 'orientErr') else f"{p[c]:.2f}" for c in cols) + " |")
# paired wins: structure vs each other choice arm, per image x K
by = {}
for r in rows:
    by.setdefault((r["image"], r["K"], r["arm"]), []).append(r)
def m(k, c): rs = by[k]; return sum(r[c] for r in rs) / len(rs)
cells = sorted({(r["image"], r["K"]) for r in rows})
print()
wins = {}
for other in ["colour", "random", "fixed:marbled", "solid", "repeat:plain"]:
    for c, better in [("ssim", 1), ("gmsd", -1), ("dist", -1)] + ([("repeatCorr", -1)] if other == "repeat:plain" else []):
        w = sum(1 for (i, K) in cells if better * (m((i, K, "structure"), c) - m((i, K, other), c)) > 0)
        print(f"structure beats {other} on {c}: {w}/{len(cells)}")
        wins[f"{other}.{c}"] = f"{w}/{len(cells)}"
json.dump(wins, open(d / "results/structure_wins.json", "w"), indent=1)
