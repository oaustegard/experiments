"""Round 4b: path-data grid step ("quantum") as the size knob. Path coordinates are stored as
integers in units of Q = 2^-k nominal pixels (default k=6, 1/64 px). A coarser grid means
fewer digits per delta. Each SVG is compiled at several k and scored against Chromium at
512 px (PSNR, ink ratio), with bytes of the minified shader and of its path data, raw and gzip.

    python3 tests/quantum_sweep.py [--json results/quantum_sweep.json]
"""
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
SKILL = Path(os.environ.get("COMPILING_WGSL", HERE.parents[1] / "claude-skills" / "compiling-wgsl" / "scripts"))
sys.path.insert(0, str(SKILL))
import forge.codegen as cg                       # noqa: E402
from forge.gpu import image, over_white, run_jobs  # noqa: E402
from forge.minify import minify                  # noqa: E402
from forge.svg import read_svg                   # noqa: E402

KS = [2, 3, 4, 5, 6, 7]
W = 512
FILES = [HERE / "tests/svg/paths.svg", HERE / "tests/svg/real/tw_1f981.svg", HERE / "tests/svg/real/si_rust.svg",
         HERE / "tests/svg/real/fe_settings.svg", HERE.parent / "svg-palette-cycle/cat-cycle.svg"]
_orig_init = cg.Gen.__init__


def compile_at(text, k):
    def init(self, scene):
        _orig_init(self, scene)
        self.quantum = 2.0 ** math.floor(math.log2(self.nominal_px / 2 ** k))
    cg.Gen.__init__ = init
    try:
        sc = read_svg(text)
        src = sc.compile()
        return sc, src, sc.data_buffer
    finally:
        cg.Gen.__init__ = _orig_init


def main():
    import base64
    jobs, meta = [], []
    for f in FILES:
        text = f.read_text()
        sc0 = read_svg(text)
        h = round(W * sc0.viewbox[3] / sc0.viewbox[2])
        jobs.append({"id": f"{f.stem}|ref", "kind": "svg", "svg": text, "width": W, "height": h})
        for k in KS:
            sc, src, data = compile_at(text, k)
            mini = minify(src)
            jobs.append({"id": f"{f.stem}|{k}", "kind": "render", "wgsl": src, "width": W, "height": h,
                         "storage": base64.b64encode(data).decode() if data else None})
            payload = mini.encode() + (data or b"")
            meta.append((f.stem, k, h, len(mini), len(gzip.compress(payload, 9)), len(data or b"")))
    res = run_jobs(jobs)
    out = {}
    for name, k, h, raw, gz, dbytes in meta:
        ref = over_white(image(res[f"{name}|ref"], W, h), False)
        got = over_white(image(res[f"{name}|{k}"], W, h), True)
        d = np.abs(ref - got)
        mse = float((d ** 2).mean())
        ink = float((255 - got).sum() / max((255 - ref).sum(), 1e-9))
        out.setdefault(name, {})[k] = {"min_wgsl": raw, "gzip_total": gz, "data_bytes": dbytes,
                                       "psnr": round(10 * np.log10(255 ** 2 / mse), 2), "ink": round(ink, 4)}
        print(f"{name:<12} grid 1/{2**k:<4} wgsl {raw:7d} B  data {dbytes:7d} B  gzip(all) {gz:7d} B  "
              f"PSNR {out[name][k]['psnr']:5.1f}  ink {ink:.4f}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
