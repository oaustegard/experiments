"""Python front-end checks.

1. Animated scene vs its own SVG export: examples/sunrise.py is compiled once; the shader
   is rendered at several times t, and the scene's ``to_svg(t)`` (time evaluated in
   Python) is rasterised by Chromium at the same t. Same metrics as diff_svg.py.
2. SDF-only scene (examples/blobs.py): no SVG equivalent, so it is checked for rendering
   at all and for the minified shader drawing bit-identical frames.

    python3 tests/diff_python.py [--json results/diff_python.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "examples"))
from forge.gpu import image, over_white, run_jobs, save_png   # noqa: E402
from forge.minify import minify                              # noqa: E402
import blobs                                                 # noqa: E402
import sunrise                                               # noqa: E402

TIMES = [0.0, 1.3, 4.7]
W, H = 480, 300


def main():
    sc = sunrise.build()
    src = sc.compile()
    mini = minify(src)
    bsrc = blobs.build().compile()
    bmini = minify(bsrc)
    jobs = []
    for t in TIMES:
        jobs.append({"id": f"sun|{t}|ref", "kind": "svg", "svg": sc.to_svg(t, W, H), "width": W, "height": H})
        jobs.append({"id": f"sun|{t}|w", "kind": "render", "wgsl": src, "width": W, "height": H, "time": t})
        jobs.append({"id": f"sun|{t}|m", "kind": "render", "wgsl": mini, "width": W, "height": H, "time": t})
        jobs.append({"id": f"blob|{t}|w", "kind": "render", "wgsl": bsrc, "width": W, "height": H, "time": t})
        jobs.append({"id": f"blob|{t}|m", "kind": "render", "wgsl": bmini, "width": W, "height": H, "time": t})
    res = run_jobs(jobs)
    out_dir = HERE / "results" / "python"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"sunrise": {}, "blobs": {}, "bytes": {
        "sunrise_wgsl": len(src), "sunrise_min": len(mini), "blobs_wgsl": len(bsrc), "blobs_min": len(bmini)}}
    for t in TIMES:
        ref = over_white(image(res[f"sun|{t}|ref"], W, H), False)
        got = over_white(image(res[f"sun|{t}|w"], W, H), True)
        d = np.abs(ref - got)
        mse = float((d ** 2).mean())
        report["sunrise"][str(t)] = {
            "mae": round(float(d.mean()), 3), "bad_px": round(float((d.max(axis=2) > 32).mean()), 5),
            "psnr": round(10 * np.log10(255 ** 2 / mse), 2) if mse else 99.0,
            "minified_identical": res[f"sun|{t}|w"]["rgba"] == res[f"sun|{t}|m"]["rgba"]}
        save_png(out_dir / f"sunrise_t{t}.png", np.concatenate([ref, got, np.clip(d * 4, 0, 255)], axis=1))
        b = over_white(image(res[f"blob|{t}|w"], W, H), True)
        save_png(out_dir / f"blobs_t{t}.png", b)
        report["blobs"][str(t)] = {"minified_identical": res[f"blob|{t}|w"]["rgba"] == res[f"blob|{t}|m"]["rgba"]}
        print(f"t={t:<4} sunrise {report['sunrise'][str(t)]}  blobs {report['blobs'][str(t)]}")
    print(report["bytes"])
    if "--json" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--json") + 1])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
