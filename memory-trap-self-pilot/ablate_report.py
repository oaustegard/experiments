"""Round 2 summary: per arm, the paper's judge (asserts a flaw, v==2) and the binary judge (bottom line unsound),
with Wilson intervals, per-item paired shifts against base, and the two judges' agreement.
Writes out/ablate_summary.json."""
import json, math, os
from collections import defaultdict

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ORDER = ["base", "full", "no_prefs", "prefs_only", "no_profile", "no_grounding", "scoped"]


def wilson(k, n):
    if not n:
        return [None, None]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 2), round(c + h, 2)]


def load():
    j2 = {(r["src"], r["arm"], r["i"], r["rep"]): r["unsound"] for r in map(json.loads, open(os.path.join(OUT, "judge2.jsonl")))}
    rows = []
    for fn, src in (("args.jsonl", "pilot"), ("ablate_args.jsonl", "ablate")):
        for r in map(json.loads, open(os.path.join(OUT, fn))):
            arm = "full" if r["arm"] == "muninn" else r["arm"]
            rows.append({"arm": arm, "i": r["i"], "j1": r.get("v"), "j2": j2.get((src, r["arm"], r["i"], r["rep"])),
                         "cost": r.get("cost") or 0})
    return rows


def main():
    rows = load()
    res = {"arms": {}, "cost_usd": round(sum(r["cost"] for r in rows), 2)}
    per_item = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_item[r["arm"]][r["i"]].append(r)
    for arm in ORDER:
        rs = [r for r in rows if r["arm"] == arm]
        if not rs:
            continue
        out = {}
        for key, f in (("j1_flaw", lambda r: r["j1"] == 2), ("j2_unsound", lambda r: r["j2"] is True)):
            src = "j1" if key.startswith("j1") else "j2"
            ok = [r for r in rs if r[src] is not None]
            k = sum(1 for r in ok if f(r))
            out[key] = {"k": k, "n": len(ok), "rate": round(k / len(ok), 2) if ok else None, "ci": wilson(k, len(ok))}
            if arm != "base":
                up = down = 0
                for i, items in per_item[arm].items():
                    a = [f(r) for r in items if r[src] is not None]; b = [f(r) for r in per_item["base"][i] if r[src] is not None]
                    if a and b:
                        d = sum(a) / len(a) - sum(b) / len(b)
                        up += d > 0; down += d < 0
                out[key]["items_up"], out[key]["items_down"] = up, down
        res["arms"][arm] = out
    both = [r for r in rows if r["j1"] is not None and r["j2"] is not None]
    agree = defaultdict(int)
    for r in both:
        agree[f"j1={r['j1']}|j2={'unsound' if r['j2'] else 'sound'}"] += 1
    res["judge_agreement"] = dict(sorted(agree.items()))
    json.dump(res, open(os.path.join(OUT, "ablate_summary.json"), "w"), indent=1)
    print(f"{'arm':13s} {'paper judge: flaw':>26s}   {'binary judge: unsound':>28s}")
    for arm, o in res["arms"].items():
        a, b = o["j1_flaw"], o["j2_unsound"]
        print(f"{arm:13s} {a['k']:3d}/{a['n']:3d}={a['rate']:.2f} {str(a['ci']):>13s} {a.get('items_up','')!s:>3}/{a.get('items_down','')!s:<3}"
              f"  {b['k']:3d}/{b['n']:3d}={b['rate']:.2f} {str(b['ci']):>13s} {b.get('items_up','')!s:>3}/{b.get('items_down','')!s:<3}")
    print("agreement:", res["judge_agreement"]); print("cost:", res["cost_usd"])


if __name__ == "__main__":
    main()
