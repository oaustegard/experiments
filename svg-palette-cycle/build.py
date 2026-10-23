"""Warhol 2x2 palette cycle from one image-to-svg geometry.

Geometry is emitted once inside a <symbol>; each tone band's fill is a
registered CSS custom property (@property, syntax <color>), so four <use>
instances animate the same keyframes with quarter-cycle offsets and always
show four different palettes.

Usage: python3 build.py [source.jpg] [out.svg]
"""
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, "/home/user/claude-skills/image-to-svg/scripts")
from pipeline import image_to_svg  # noqa: E402

SOURCE_URL = "https://storage.googleapis.com/mediapipe-assets/cat.jpg"
SIZE = 400          # each quadrant's viewBox width
CYCLE = 12          # seconds for one quadrant to pass through all palettes
HOLD = 0.7          # fraction of each palette's slot spent holding (rest is the cross-fade)

# Darkest → lightest, one entry per tone band. The background rect takes the lightest.
PALETTES = {
    "hot":    ["#1a0a1a", "#7a0f3a", "#e8173e", "#ff5c8a", "#ff9f43", "#ffe14c"],
    "jungle": ["#0a1a0a", "#145c22", "#1b9a3b", "#5ce65c", "#b6f04a", "#f4ff9a"],
    "ocean":  ["#0a0a2e", "#10288a", "#1f5fe0", "#4c99ff", "#7fd0ff", "#d6f2ff"],
    "grape":  ["#2a0a3a", "#5e1680", "#8b24aa", "#cc66ff", "#ff8ad8", "#ffd6f5"],
}
# Placeholder grays mark the bands in the skill's output; luminance-sorted like the palettes.
BANDS = ["#000000", "#333333", "#666666", "#999999", "#cccccc", "#ffffff"]


def upscaled_source(src, factor=2):
    """Upscaling before tracing keeps detail that fixed-size morphology would merge (memory 3763e2a0)."""
    img = cv2.imread(str(src))
    out = Path(tempfile.mkdtemp()) / "up.png"
    cv2.imwrite(str(out), cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC))
    return out


def keyframes(order):
    """One @keyframes walking through palettes in `order`, holding each, cross-fading between."""
    n = len(order)
    slot = 100 / n
    frames = []
    for i, name in enumerate(order):
        start, end = i * slot, i * slot + slot * HOLD
        decl = ";".join(f"--b{b}:{c}" for b, c in enumerate(PALETTES[name]))
        at = f"{start:.3g}%,{end:.3g}%" if i else f"0%,{end:.3g}%"
        frames.append(f"{at}{{{decl}}}")
    decl = ";".join(f"--b{b}:{c}" for b, c in enumerate(PALETTES[order[0]]))
    frames.append(f"100%{{{decl}}}")
    return "@keyframes cycle{" + "".join(frames) + "}"


def build(src, out):
    svg, _ = image_to_svg(str(upscaled_source(src)), mode="illustration", K=18,
                          smooth="kuwahara:3", palette=BANDS, bg_clusters=0,
                          svg_width=SIZE)
    scope = re.search(r'id="(s\w+)"', svg).group(1)
    class_color = dict(re.findall(rf"#{scope} \.(c\d+)\{{fill:(#\w+);", svg))
    short = {f"#{b[1]}{b[3]}{b[5]}": i for i, b in enumerate(BANDS)}
    band_of = {c: short[col] for c, col in class_color.items()}
    height = re.search(r'viewBox="0 0 \d+ (\d+)"', svg).group(1)
    body = re.search(r"(<g stroke-width.*?</g>)", svg, re.S).group(1)
    body = re.sub(r"\n\s*", "", body)

    names = list(PALETTES)
    lines = []
    for b in range(len(BANDS)):
        lines.append(f"@property --b{b}{{syntax:'<color>';inherits:true;initial-value:{BANDS[b]}}}")
    lines.append("".join(f".{c}{{fill:var(--b{b});stroke:var(--b{b})}}" for c, b in band_of.items()))
    lines.append(".bg{fill:var(--b{len(BANDS) - 1})}")  # var() is not allowed in presentation attributes
    lines.append(keyframes(names))
    # Static per-quadrant palettes are the reduced-motion frame; the animation overrides them.
    for q, name in enumerate(names):
        decl = ";".join(f"--b{b}:{c}" for b, c in enumerate(PALETTES[name]))
        lines.append(f".q{q}{{{decl};animation:cycle {CYCLE}s cubic-bezier(.65,0,.35,1) {-q * CYCLE / 4:g}s infinite}}")
    lines.append("@media (prefers-reduced-motion:reduce){use{animation:none}}")

    quads = "".join(
        f'<use href="#art" class="q{q}" x="{(q % 2) * SIZE}" y="{(q // 2) * int(height)}" width="{SIZE}" height="{height}"/>'
        for q in range(4))
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2 * SIZE} {2 * int(height)}" role="img">'
           f'<title>Palette cycle</title><style><![CDATA[{"".join(lines)}]]></style>'
           f'<symbol id="art" viewBox="0 0 {SIZE} {height}">'
           f'<rect class="bg" width="{SIZE}" height="{height}"/>{body}</symbol>{quads}</svg>')
    Path(out).write_text(doc)
    return doc


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).with_name("cat-cycle.svg"))
    if src is None:
        src = Path(tempfile.mkdtemp()) / "cat.jpg"
        urllib.request.urlretrieve(SOURCE_URL, src)
    doc = build(src, out)
    print(f"{out}: {len(doc):,} bytes")
