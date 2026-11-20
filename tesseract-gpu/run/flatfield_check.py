"""Does flat-fielding a line before Tesseract fix the shaded-line failures, and does it hurt elsewhere?

Flat-field: background = grey closing (max then min filter, window ~1.5x line height) smoothed by a
box blur; out = gray / background * 255, then a 1..99 percentile stretch. Dark text on light paper is
assumed; inverted lines are flipped first (improve's needs_invert test via ref).

Reads with the Tesseract 5.3.4 CLI (--oem 1 --psm 7, tessdata_best) so the numbers sit beside
data/*/tess_best.jsonl, which hold the same lines unprocessed.

    python3 run/flatfield_check.py        # -> results/flatfield_check.json
"""
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "run"))
sys.path.insert(0, str(ROOT))
from tess_baseline import cer  # noqa: E402
from tessgpu import ref  # noqa: E402


def flatfield(gray):
    g = gray.astype(np.float32)
    if ref.needs_invert(gray):
        g = 255 - g
    k = max(9, int(1.5 * g.shape[0]) | 1)
    bg = ndimage.grey_closing(g, size=(k, k))
    bg = ndimage.uniform_filter(bg, size=k)
    out = g / np.maximum(bg, 1) * 255
    lo, hi = np.percentile(out, [1, 99])
    if hi > lo:
        out = (out - lo) * 255 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def read(path, lang, tmp):
    gray = np.asarray(Image.open(path).convert("L"))
    p = Path(tmp) / (Path(path).stem + "_ff.png")
    Image.fromarray(flatfield(gray)).save(p)
    r = subprocess.run(["tesseract", str(p), "stdout", "--oem", "1", "--psm", "7",
                        "--tessdata-dir", str(ROOT / "models/best"), "-l", lang],
                       capture_output=True, text=True, env={"OMP_THREAD_LIMIT": "1", "PATH": "/usr/bin:/bin"})
    return r.stdout.strip() if r.returncode == 0 else None


def main():
    cases = []
    for base, keep in ((ROOT / "data/lines", {"shade", "clean", "blur", "lowres"}), (ROOT / "data/real", {"real"})):
        for l in open(base / "tess_best.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["lang"] == "eng" and r["cond"] in keep and r.get("pred") is not None:
                p = base / r["file"] if (base / r["file"]).exists() else ROOT / r["file"]
                cases.append((r, str(p)))
    with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(3) as ex:
        preds = list(ex.map(lambda c: read(c[1], c[0]["lang"], tmp), cases))
    groups = {}
    for (r, _), pred in zip(cases, preds):
        if pred is None:
            continue
        nocase = r["cond"] == "real"
        f = (lambda s: s.casefold()) if nocase else (lambda s: s)
        before, after = cer(f(r["text"]), f(r["pred"])), cer(f(r["text"]), f(pred))
        groups.setdefault(r["cond"], []).append((before, after))
    rows = []
    for cond, v in groups.items():
        b, a = np.array(v).T
        rows.append(dict(cond=cond, n=len(v), cer_before=round(float(b.mean()), 5), cer_after=round(float(a.mean()), 5),
                         fixed=int(((b > 0) & (a < b)).sum()), broken=int(((a > b)).sum())))
        print(f"{cond:7} n={len(v):4} CER {100*b.mean():6.2f}% -> {100*a.mean():6.2f}%  better {rows[-1]['fixed']:3}  worse {rows[-1]['broken']:3}")
    (ROOT / "results/flatfield_check.json").write_text(json.dumps(
        {"note": "eng, tessdata_best, Tesseract 5.3.4 CLI psm 7; real set scored case-insensitively", "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
