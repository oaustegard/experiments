"""Math-reader precision arms against fp32, from run/ref_pages.py runs with --math on the same
pages: how often an arm's page markdown equals fp32's, and how its math readings compare
(lines flagged, readings used, used readings equal to fp32's).

Stands in for the benchmark score, which needs omni-parse-bench's scorer.

    python3 run/math_arms.py OUT_DIR/math-fp32 OUT_DIR/math-mix8m16 OUT_DIR/math-mix8m8 [--json results/math_arms.json]
"""
import argparse
import json
from pathlib import Path

MATH_MIN_CONF = 0.9


def used(rec):
    """{(call, line index): text} for each math reading that replaced its line."""
    out = {}
    for c, m in enumerate(rec.get("math", [])):
        for j, (t, p) in zip(m["flags"], m.get("reads", [])):
            if "$" in t and p >= MATH_MIN_CONF:
                out[(c, j)] = t
    return out


ap = argparse.ArgumentParser()
ap.add_argument("base", type=Path)
ap.add_argument("arms", type=Path, nargs="+")
ap.add_argument("--json", type=Path)
a = ap.parse_args()
base = {f.stem: json.loads(f.read_text()) for f in (a.base / "pages").glob("*.json")}
res = {}
for arm in [a.base, *a.arms]:
    pages = {f.stem: json.loads(f.read_text()) for f in (arm / "pages").glob("*.json")}
    ids = sorted(set(pages) & set(base))
    same_md = flagged = n_used = same_used = base_used = 0
    by_suite: dict = {}
    for p in ids:
        r, b = pages[p], base[p]
        if "error" in r or "error" in b:
            continue
        s = p.split("-")[0]
        eq = r["raw"]["markdown"] == b["raw"]["markdown"]
        same_md += eq
        by_suite.setdefault(s, [0, 0])
        by_suite[s][0] += eq
        by_suite[s][1] += 1
        flagged += sum(len(m["flags"]) for m in r.get("math", []))
        u, ub = used(r), used(b)
        n_used += len(u)
        base_used += len(ub)
        same_used += sum(1 for k, t in u.items() if ub.get(k) == t)
    res[arm.name] = {"pages": len(ids), "same_markdown_as_fp32": same_md, "lines_flagged": flagged,
                     "readings_used": n_used, "readings_used_fp32": base_used,
                     "used_readings_equal_to_fp32": same_used,
                     "same_markdown_by_suite": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_suite.items())},
                     "ms_median": sorted(pages[p]["ms"] for p in ids)[len(ids) // 2] if ids else None}
print(json.dumps(res, indent=1))
if a.json:
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=1))
