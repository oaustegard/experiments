"""Build the single-file tool page: tracer internals + scene.js + page.html + sample.

    python3 web/bundle.py   ->  web/index.html

The tracer internals are sliced from svg-in-browser/web/trace.js (everything
from the WGSL shader up to its layered entry point), so a fix there flows in
on the next bundle. The page must stay one self-contained file, which is what
austegard.com tool pages require.
"""

import base64
import io
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
TRACE = HERE.parents[1] / "svg-in-browser/web/trace.js"


def main():
    trace = TRACE.read_text()
    core = trace[trace.index("const KMEANS_WGSL"):trace.index("const DEFAULT_LEVELS")]
    scene = (HERE / "scene.js").read_text().replace("export async function", "async function")
    im = Image.open(HERE.parent / "src/starry_night.jpg").convert("RGB")
    im.thumbnail((900, 900))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    sample = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    page = (HERE / "page.html").read_text()
    page = page.replace("/*CODE*/", "// --- tracer internals, from svg-in-browser/web/trace.js ---\n" + core + "\n" + scene)
    page = page.replace("/*SAMPLE*/", sample)
    out = HERE / "index.html"
    out.write_text(page)
    print(f"{out}: {len(page):,} B")


if __name__ == "__main__":
    sys.exit(main())
