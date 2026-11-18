"""What a pack costs on the wire: each file gzipped at level 6 (what GitHub Pages serves) and,
for reference, brotli at quality 11, summed into the core download (page code, detector,
Latin recognizer, order, orientation and script models, vocabulary, Unicode table) and the
lazy modules (a script group's recognizer, vocabulary and n-gram table; the Latin n-gram table).
    python3 run/pack_bytes.py web/models/PACK [--code web] [--json out.json]"""
import argparse
import gzip
import json
from pathlib import Path

import brotli

ap = argparse.ArgumentParser()
ap.add_argument("pack", type=Path)
ap.add_argument("--code", type=Path, help="directory of the page's JS, HTML and CSS")
ap.add_argument("--json", type=Path)
a = ap.parse_args()


def cost(p: Path) -> dict:
    b = p.read_bytes()
    return {"raw": len(b), "gz": len(gzip.compress(b, 6)), "br": len(brotli.compress(b, quality=11))}


man = json.loads((a.pack / "manifest.json").read_text())
groups = {"core": ["manifest.json", "uexc.json", "vocab.json"] + [man["models"][m]["file"] for m in ("det", "rec", "order", "orient", "script") if m in man["models"]]}
for m in man["models"]:
    if m.startswith("rec_"):
        g = m[4:]
        groups[g] = [man["models"][m]["file"], f"vocab_{g}.json"] + ([man["lm"][g]] if g in man.get("lm", {}) else [])
if "base" in man.get("lm", {}):
    groups["lm-base"] = [man["lm"]["base"]]
out = {}
for g, files in groups.items():
    c = [cost(a.pack / f) for f in files]
    out[g] = {k: sum(x[k] for x in c) for k in ("raw", "gz", "br")} | {"files": files}
if a.code:
    files = sorted(p for p in a.code.iterdir() if p.suffix in (".js", ".html", ".css"))
    c = [cost(p) for p in files]
    out["code"] = {k: sum(x[k] for x in c) for k in ("raw", "gz", "br")} | {"files": [p.name for p in files]}
for g, v in out.items():
    print(f"{g:8s} raw {v['raw'] / 1e3:8.1f} KB  gzip {v['gz'] / 1e3:8.1f} KB  brotli {v['br'] / 1e3:8.1f} KB")
if a.json:
    a.json.write_text(json.dumps(out, indent=1))
