"""The browser pipeline's page outputs (web/test/check_pages.mjs) against the Python package's
(opb_eval.py) on the same models and pages: identical markdown, identical line texts in the same
order, and the largest line-box difference.
    python3 run/compare_pages.py JS_DIR PY_RUN_DIR [--json results/parity.json]"""
import argparse
import difflib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("js", type=Path)
ap.add_argument("py", type=Path)
ap.add_argument("--json", type=Path)
a = ap.parse_args()
rows = []
for f in sorted(a.js.glob("*.json")):
    j = json.loads(f.read_text())
    pf = a.py / "pages" / f.name
    if not pf.exists() or "error" in j:
        rows.append({"page": j["page_id"], "error": j.get("error", "no python page")})
        continue
    p = json.loads(pf.read_text())["raw"]
    jl, pl = [l["text"] for l in j["lines"]], [l[1] for l in p["lines"]]
    box = max((max(abs(x - y) for x, y in zip(lj["box"], lp[0])) for lj, lp in zip(j["lines"], p["lines"])), default=0.0)
    md_same = (j.get("markdown") or "") == (p.get("markdown") or "")
    rows.append({"page": j["page_id"], "lines": len(pl), "lines_js": len(jl), "same_lines": jl == pl, "same_markdown": md_same,
                 "markdown_ratio": 1.0 if md_same else difflib.SequenceMatcher(None, j.get("markdown") or "", p.get("markdown") or "").ratio(),
                 "max_box_diff": box if len(jl) == len(pl) else None, "ms": j.get("ms"), "timings": j.get("timings")})
ok = [r for r in rows if "error" not in r]
summ = {"pages": len(rows), "errors": len(rows) - len(ok), "same_markdown": sum(r["same_markdown"] for r in ok),
        "same_lines": sum(r["same_lines"] for r in ok),
        "lines": sum(r["lines"] for r in ok), "min_markdown_ratio": min((r["markdown_ratio"] for r in ok), default=None)}
print(json.dumps(summ))
for r in rows:
    if "error" in r or not r["same_markdown"]:
        print(" ", r["page"][:70], r.get("error", "")[:200] or f"lines {r['lines_js']}/{r['lines']} same_lines {r['same_lines']} md ratio {r['markdown_ratio']:.4f} box {r['max_box_diff']}")
if a.json:
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps({"summary": summ, "pages": rows}, indent=1))
