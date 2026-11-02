"""Round 4: lossy float precision as a size knob ("pay for digits", from the image-to-wgsl
session's codec design). Every scene shader (26 test SVGs, the animated sunrise at t=1.3,
four Shadertoy GLSL fixtures) is minified exactly and with float literals rounded to N
significant digits; each is rendered at 256 px and compared with the exact render.

Uses the maintained compiler, the compiling-wgsl skill (claude-skills checkout beside this
repo, or COMPILING_WGSL=<path to its scripts/>).

    python3 tests/digits_sweep.py [--json results/digits_sweep.json]
"""
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SKILL = Path(os.environ.get("COMPILING_WGSL", HERE.parents[1] / "claude-skills" / "compiling-wgsl" / "scripts"))
sys.path.insert(0, str(SKILL))
from forge.gpu import image, run_jobs            # noqa: E402
from forge.glsl import glsl_to_wgsl              # noqa: E402
from forge.ast import Lit, walk                  # noqa: E402
from forge.minify import minify, shorten_literal  # noqa: E402
from forge.printer import to_wgsl                # noqa: E402
from forge.types import lit_type                 # noqa: E402
from forge.wgsl import parse                     # noqa: E402


def minify_digits(src, digits):
    """Minify, then round every float literal to `digits` significant digits (the rejected
    lossy option, kept here so the measurement can be rerun)."""
    out = minify(src)
    if digits is None:
        return out
    m = parse(out)
    for n in walk(m):
        if isinstance(n, Lit):
            ty = lit_type(n.text)
            if ty in ("AFloat", "f32"):
                v = float(n.text.rstrip("f"))
                if v:
                    cand = repr(float(f"{v:.{digits}g}")) + ("f" if ty == "f32" else "")
                    if lit_type(cand) == ty:
                        n.text = shorten_literal(cand)
    return to_wgsl(m)
from forge.svg import read_svg                   # noqa: E402

DIGITS = [2, 3, 4, 5, 6, None]
W = 256


def scenes():
    for f in sorted((HERE / "tests" / "svg").glob("*.svg")) + sorted((HERE / "tests" / "svg" / "real").glob("*.svg")):
        sc = read_svg(f.read_text())
        h = round(W * sc.viewbox[3] / sc.viewbox[2])
        yield f.stem, sc.compile(), h, 0.0, "svg"
    sys.path.insert(0, str(HERE / "examples"))
    import sunrise
    yield "sunrise", sunrise.build().compile(), round(W * 200 / 320), 1.3, "python"
    for f in sorted((SKILL / "selftest").glob("st_*.glsl")):
        yield f.stem, glsl_to_wgsl(f.read_text()).wgsl, round(W * 9 / 16), 1.7, "glsl"


def main():
    jobs, meta = [], []
    for name, src, h, t, kind in scenes():
        for d in DIGITS:
            out = minify_digits(src, d)
            jobs.append({"id": f"{name}|{d}", "kind": "render", "wgsl": out, "width": W, "height": h, "time": t})
            meta.append((name, kind, d, len(out), len(gzip.compress(out.encode(), 9)), h))
    res = run_jobs(jobs)
    rows = {}
    for name, kind, d, raw, gz, h in meta:
        rows.setdefault(name, {"kind": kind})[str(d)] = {"raw": raw, "gz": gz, "ok": res[f"{name}|{d}"]["ok"]}
    summary = {}
    for name, r in rows.items():
        ref = image(res[f"{name}|None"], W, [m[5] for m in meta if m[0] == name][0]).astype(float)
        for d in DIGITS[:-1]:
            x = res[f"{name}|{d}"]
            if not x["ok"]:
                r[str(d)].update(psnr=None, identical=False)
                continue
            img = image(x, W, ref.shape[0]).astype(float)
            diff = np.abs(img - ref)
            mse = float((diff ** 2).mean())
            r[str(d)].update(identical=bool(mse == 0), max_diff=int(diff.max()),
                             psnr=round(10 * np.log10(255 ** 2 / mse), 2) if mse else 99.0)
    for d in DIGITS[:-1]:
        ok = [r for r in rows.values() if r[str(d)].get("psnr") is not None]
        summary[str(d)] = {
            "raw_ratio": round(sum(r[str(d)]["raw"] for r in rows.values()) / sum(r["None"]["raw"] for r in rows.values()), 4),
            "gz_ratio": round(sum(r[str(d)]["gz"] for r in rows.values()) / sum(r["None"]["gz"] for r in rows.values()), 4),
            "identical": sum(r[str(d)]["identical"] for r in ok), "n": len(rows),
            "min_psnr": min(r[str(d)]["psnr"] for r in ok) if ok else None,
            "max_diff": max(r[str(d)]["max_diff"] for r in ok) if ok else None,
            "failed_to_render": len(rows) - len(ok)}
        s = summary[str(d)]
        print(f"digits {d}: raw x{s['raw_ratio']:.3f}  gz x{s['gz_ratio']:.3f}  identical {s['identical']}/{s['n']}  "
              f"min PSNR {s['min_psnr']}  max pixel diff {s['max_diff']}  render failures {s['failed_to_render']}")
    worst = {d: sorted(((r[str(d)].get("psnr") or 0, n) for n, r in rows.items()))[:3] for d in ("3", "4")}
    print("worst at 3 and 4 digits:", worst)
    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.write_text(json.dumps({"width": W, "summary": summary, "cases": rows}, indent=1))


if __name__ == "__main__":
    main()
