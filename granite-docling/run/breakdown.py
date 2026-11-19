"""Where docling's gap to Indra comes from: pages docling ended on its own versus pages that ran to the
token cap (repetition loops), and the score per benchmark suite with each suite's loop count.

    python3 run/breakdown.py OUT BASE RUN GEN_DIR [GEN_DIR ...] --json results/breakdown.json

OUT/BASE and OUT/RUN are scored runs (pages/*.json with `lines` and `suite`); GEN_DIRs hold the
kaggle_gen.py page JSONs whose `finish` is "length" for a page that hit the cap. Scores and intervals
come from compare.py on each page set (1,000 bootstrap draws over documents; 200 per suite).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def compare(out, base, run, pages, boot):
    with tempfile.TemporaryDirectory() as d:
        pf, jf = Path(d) / "pages.txt", Path(d) / "out.json"
        pf.write_text("\n".join(pages))
        subprocess.run([sys.executable, str(HERE / "compare.py"), str(out), base, run, "--pages", str(pf),
                        "--boot", str(boot), "--json", str(jf)], check=True, capture_output=True)
        return json.loads(jf.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("base")
    ap.add_argument("run")
    ap.add_argument("gen", type=Path, nargs="+")
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    capped = {}
    for g in a.gen:
        for f in g.glob("*.json"):
            r = json.loads(f.read_text())
            capped[r["page_id"]] = r["finish"] == "length"
    suites = defaultdict(list)
    for f in (a.out / a.run / "pages").glob("*.json"):
        r = json.loads(f.read_text())
        suites[r["suite"]].append(r["page_id"])
    pages = sorted(p for ps in suites.values() for p in ps)
    res = {"pages": len(pages), "capped_pages": sum(capped.get(p, False) for p in pages)}
    for name, keep in (("ended", [p for p in pages if not capped.get(p, False)]),
                       ("capped", [p for p in pages if capped.get(p, False)])):
        c = compare(a.out, a.base, a.run, keep, 1000)
        res[name] = {"pages": len(keep), a.base: c["runs"][a.base], a.run: c["runs"][a.run]}
    res["suites"] = {}
    for s, ps in sorted(suites.items()):
        c = compare(a.out, a.base, a.run, ps, 200)
        res["suites"][s] = {"pages": len(ps), "capped": sum(capped.get(p, False) for p in ps),
                            a.base: round(100 * c["runs"][a.base]["headline"], 2),
                            a.run: round(100 * c["runs"][a.run]["headline"], 2)}
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("pages", "capped_pages")}))
    for k in ("ended", "capped"):
        r = res[k]
        print(k, r["pages"], "pages:", a.base, round(100 * r[a.base]["headline"], 2), a.run,
              round(100 * r[a.run]["headline"], 2), r[a.run].get("ci95"))


if __name__ == "__main__":
    main()
