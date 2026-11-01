"""SVG front-end against Chromium's own SVG rasteriser.

Each SVG under tests/svg/ (and tests/svg/real/) is drawn by Chromium (Skia) as the
reference and by the compiled WGSL shader on WebGPU, both at the same pixel size, both
composited on white. The minified shader is rendered too and must match the unminified
one bit for bit.

Metrics per image: mean absolute error per channel (0-255), the share of pixels whose
worst channel differs by more than 32, and PSNR. Side-by-side images (reference |
shader | |difference| x4) go to results/svg/.

    python3 tests/diff_svg.py [--size 192] [--json results/diff_svg.json]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from forge.gpu import image, over_white, run_jobs, save_png   # noqa: E402
from forge.minify import minify                              # noqa: E402
from forge.svg import read_svg                               # noqa: E402


def main():
    size = int(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else 192
    files = sorted((HERE / "tests" / "svg").glob("*.svg")) + sorted((HERE / "tests" / "svg" / "real").glob("*.svg"))
    jobs, meta = [], []
    for f in files:
        text = f.read_text()
        t0 = time.time()
        scene = read_svg(text)
        src = scene.compile()
        compile_s = time.time() - t0
        mini = minify(src)
        vw, vh = scene.viewbox[2], scene.viewbox[3]
        w = size
        h = max(1, round(size * vh / vw))
        name = f.stem if f.parent.name == "svg" else "real/" + f.stem
        jobs.append({"id": name + "|ref", "kind": "svg", "svg": text, "width": w, "height": h})
        jobs.append({"id": name + "|wgsl", "kind": "render", "wgsl": src, "width": w, "height": h})
        jobs.append({"id": name + "|min", "kind": "render", "wgsl": mini, "width": w, "height": h})
        meta.append((name, w, h, len(text), len(src), len(mini), compile_s, scene.warnings))
    res = run_jobs(jobs)
    out_dir = HERE / "results" / "svg"
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    report = {}
    for name, w, h, n_svg, n_src, n_min, cs, warns in meta:
        r_ref, r_w, r_m = res[name + "|ref"], res[name + "|wgsl"], res[name + "|min"]
        if not (r_ref["ok"] and r_w["ok"] and r_m["ok"]):
            report[name] = {"error": [r.get("error") for r in (r_ref, r_w, r_m) if not r["ok"]]}
            print(f"{name:<24} ERROR {report[name]['error']}")
            continue
        ref = over_white(image(r_ref, w, h), premultiplied=False)
        got = over_white(image(r_w, w, h), premultiplied=True)
        same_min = r_w["rgba"] == r_m["rgba"]
        d = np.abs(ref - got)
        mae = float(d.mean())
        bad = float((d.max(axis=2) > 32).mean())
        mse = float((d ** 2).mean())
        psnr = float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else 99.0
        side = np.concatenate([ref, got, np.clip(d * 4, 0, 255)], axis=1)
        save_png(out_dir / f"{name}.png", side)
        report[name] = {"mae": round(mae, 3), "bad_px": round(bad, 5), "psnr": round(psnr, 2),
                        "minified_identical": same_min, "svg_bytes": n_svg, "wgsl_bytes": n_src,
                        "min_bytes": n_min, "compile_s": round(cs, 3),
                        "gpu_compile_ms": round(r_w.get("compile_ms", 0), 1), "warnings": warns}
        print(f"{name:<24} mae {mae:6.2f}  bad {bad*100:6.2f}%  psnr {psnr:5.1f}  min==src {same_min}  "
              f"svg {n_svg:6d}B  wgsl {n_src:6d}B  min {n_min:6d}B  {'; '.join(warns)}")
    ok = [v for v in report.values() if "mae" in v]
    summary = {"n": len(report), "rendered": len(ok),
               "all_minified_identical": all(v["minified_identical"] for v in ok),
               "median_psnr": float(np.median([v["psnr"] for v in ok])) if ok else None,
               "max_bad_px": max(v["bad_px"] for v in ok) if ok else None,
               "max_mae": max(v["mae"] for v in ok) if ok else None,
               "median_min_over_svg": float(np.median([v["min_bytes"] / v["svg_bytes"] for v in ok])) if ok else None,
               "median_min_bytes": float(np.median([v["min_bytes"] for v in ok])) if ok else None}
    print(summary)
    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"size": size, "summary": summary, "cases": report}, indent=1))


if __name__ == "__main__":
    main()
