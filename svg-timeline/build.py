"""Starry Night paints itself: a traced painting given a timeline.

    python3 build.py [source.jpg] [out.svg]

image-to-svg traces the source into ~4,700 flat-color shapes. The scene then
gives each shape a life on a 48 s loop: a charcoal underdrawing sketches the
big forms, paint lands shape by shape (large forms first, sweeping left to
right), the village goes to deep night with the stars glowing, then to dawn,
then the paint falls away off the sketch and the sketch erases itself.
Everything compiles through timeline.py to CSS keyframes; no JavaScript.
"""

import colorsys
import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/mnt/skills/user/image-to-svg/scripts")
import pipeline as P  # noqa: E402

from timeline import Motion, Sheet, num  # noqa: E402

HERE = Path(__file__).parent
D = 48.0
W = 1000

BEAT = {
    "sketch": 0.3, "sketched": 3.2,
    "paint": 2.0, "painted": 17.0,
    "dusk": 22.0, "night": 24.5, "nightEnd": 29.0,
    "dawn": 31.5, "dawnEnd": 34.5, "day": 37.0,
    "fall": 38.5, "fallen": 45.0,
    "erase": 45.2, "erased": 47.2,
}
CANVAS = "#ebe3d0"
CHARCOAL = "#4a3f35"


# --- colors ---------------------------------------------------------------

def rgb(h):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def hexc(c):
    return "#" + "".join(f"{round(max(0, min(1, v)) * 255):02x}" for v in c)


def mix(a, b, t):
    return [x + (y - x) * t for x, y in zip(a, b)]


def lum(c):
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def is_light(c):
    """Stars, moon, lit windows: bright and warm."""
    h, l, s = colorsys.rgb_to_hls(*c)
    return lum(c) > 0.62 and c[0] >= c[2] and (0.08 < h < 0.2 or s < 0.25)


def night(c):
    if is_light(c):
        return mix(c, rgb("#fff1a8"), 0.35)
    h, l, s = colorsys.rgb_to_hls(*c)
    dh = ((0.63 - h + 0.5) % 1) - 0.5
    return list(colorsys.hls_to_rgb((h + 0.55 * dh) % 1, l * 0.42, s * 0.85))


DAWN_MAP = [(0, "#2b1838"), (0.35, "#7a3f6b"), (0.6, "#e0866e"), (0.85, "#ffd39a"), (1, "#fff4dc")]


def dawn(c):
    L = lum(c)
    for (a, ca), (b, cb) in zip(DAWN_MAP, DAWN_MAP[1:]):
        if L <= b:
            g = mix(rgb(ca), rgb(cb), (L - a) / (b - a))
            break
    return mix(c, g, 0.55)


# --- tracing ---------------------------------------------------------------

def trace(src):
    _, flow = P.image_to_svg(str(src), mode="painting", svg_width=W, smooth="kuwahara:3", precision=0)
    out = flow.value(P.extract_contours)
    return out["shapes"], out["svg_h"], flow.value(P.detect_background)["bg_hex"]


def points(d):
    """First subpath of an 'M x,y L x,y … Z' path as an (N, 2) array."""
    first = d.split("Z")[0]
    return np.array([[float(v) for v in p.split(",")] for p in re.findall(r"-?[\d.]+,-?[\d.]+", first)])


def underdrawing(src, w, h):
    """Charcoal outlines: region borders of a blurred 5-color posterization,
    simplified, then Chaikin-smoothed. Adjacent regions both draw their
    border, which doubles the line the way a sketch does."""
    im = cv2.resize(cv2.imread(str(src)), (w, h), interpolation=cv2.INTER_AREA)
    b = cv2.GaussianBlur(cv2.bilateralFilter(im, 9, 60, 9), (15, 15), 0)
    cv2.setRNGSeed(3)
    _, lab, _ = cv2.kmeans(b.reshape(-1, 3).astype(np.float32), 5, None,
                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1), 3, cv2.KMEANS_PP_CENTERS)
    lab = cv2.medianBlur(lab.reshape(h, w).astype(np.uint8), 9)
    lines = []
    for c in range(5):
        cs, _ = cv2.findContours((lab == c).astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for ct in cs:
            if cv2.arcLength(ct, True) < 120:
                continue
            pts = cv2.approxPolyDP(ct, 2.5, True).reshape(-1, 2).astype(float)
            for _ in range(2):
                nxt = np.roll(pts, -1, axis=0)
                pts = np.stack([0.75 * pts + 0.25 * nxt, 0.25 * pts + 0.75 * nxt], 1).reshape(-1, 2)
            # Drop the stretches that run along the canvas edge.
            inside = (pts[:, 0] > 3) & (pts[:, 0] < w - 3) & (pts[:, 1] > 3) & (pts[:, 1] < h - 3)
            if inside.all():
                runs, closed = [pts], "Z"
            else:
                k = int(np.argmin(inside))  # start the walk on an edge point
                pts, inside = np.roll(pts, -k, 0), np.roll(inside, -k)
                runs, cur, closed = [], [], ""
                for p, ok in zip(pts, inside):
                    if ok:
                        cur.append(p)
                    elif cur:
                        runs.append(np.array(cur))
                        cur = []
                if cur:
                    runs.append(np.array(cur))
            for run in runs:
                if len(run) < 2 or np.linalg.norm(np.diff(run, axis=0), axis=1).sum() < 40:
                    continue
                d = "M" + "L".join(f"{round(x)} {round(y)}" for x, y in run) + closed
                lines.append((run[:, 0].mean(), d))
    return sorted(lines)


# --- scene -----------------------------------------------------------------

def tenth(t):
    return round(t * 10) / 10


def fifth(t):
    return round(t * 5) / 5


def merged(items):
    """One path per color (evenodd shapes stay separate: merging could flip
    which overlaps count as holes)."""
    by_color, out = {}, []
    for c, s in items:
        if s.get("fill_rule"):
            out.append(f'<path class="c{c}" fill-rule="evenodd" d="{P._relative_path(s["path"], 0)}"/>')
        else:
            by_color.setdefault(c, []).append(s["path"])
    for c, ds in by_color.items():
        out.append(f'<path class="c{c}" d="{P._relative_path("".join(ds), 0)}"/>')
    return out


def build(src, out):
    shapes, H, bg_hex = trace(src)
    colors = list(dict.fromkeys(P._short_hex(s["color"]) for s in shapes))
    cidx = {c: i for i, c in enumerate(colors)}
    base = {c: rgb(c) for c in colors}

    sheet = Sheet(D)
    rnd = random.Random(7)

    # Color: every palette entry is an @property <color> on the root.
    def palette_at(fn):
        return {f"--c{i}": hexc(fn(base[c])) for i, c in enumerate(colors)}

    orig = palette_at(lambda c: c)
    grade = (Motion(**orig).at(BEAT["dusk"]).to(BEAT["night"] - BEAT["dusk"], "inOut", **palette_at(night))
             .at(BEAT["nightEnd"]).to(BEAT["dawn"] - BEAT["nightEnd"], "inOut", **palette_at(dawn))
             .at(BEAT["dawnEnd"]).to(BEAT["day"] - BEAT["dawnEnd"], "inOut", **orig))
    canvas_grade = (Motion(fill=CANVAS).at(BEAT["dusk"]).to(2.5, "inOut", fill="#1d2233")
                    .at(BEAT["nightEnd"]).to(2.5, "inOut", fill="#e9c7b0")
                    .at(BEAT["dawnEnd"]).to(2.5, "inOut", fill=CANVAS))

    # Order: big forms first, then a left-to-right, top-to-bottom sweep.
    areas = np.log([s["area"] + 1 for s in shapes])
    amin, amax = areas.min(), areas.max()
    paint_span = BEAT["painted"] - 0.7 - BEAT["paint"]
    fall_span = BEAT["fallen"] - 0.8 - BEAT["fall"]
    slots, lights = {}, {}
    for i, s in enumerate(shapes):
        pts = points(s["path"])
        cx, cy = pts.mean(axis=0)
        size = 1 - (areas[i] - amin) / (amax - amin)
        sweep = 0.75 * cx / W + 0.25 * cy / H
        order = min(1, max(0, 0.55 * size + 0.45 * sweep + rnd.uniform(-0.03, 0.03)))
        start = fifth(BEAT["paint"] + order * paint_span)
        c = cidx[P._short_hex(s["color"])]
        slots.setdefault(start, []).append((c, s))
        if is_light(base[colors[c]]):
            lights.setdefault(rnd.randrange(6), []).append((c, s))

    # One group per time slot; inside it, one path per color. The trace is a
    # near-partition of the canvas (holes are cut with evenodd), so shapes
    # can leave strict largest-first order without visibly overlapping.
    fills = []
    for start in sorted(slots):
        drop = fifth(BEAT["fall"] + (start - BEAT["paint"]) / paint_span * fall_span)
        dab = (Motion(opacity=0, transform="translateY(-10px)").at(start)
               .to(0.9, "out", opacity=1, transform="none")
               .at(drop).to(0.8, "in", opacity=0, transform="translateY(40px)"))
        fills.append(f'<g class="{sheet.cls(dab)}">')
        fills += merged(slots[start])
        fills.append("</g>")

    # Underdrawing: outlines of the largest forms, drawn on with dashes.
    sketch_sheet = Sheet(D, prefix="s")
    sketches = []
    for cx, d in underdrawing(src, W, H):
        start = tenth(BEAT["sketch"] + (cx / W) * (BEAT["sketched"] - 1.2 - BEAT["sketch"]) + rnd.uniform(0, 0.3))
        gone = tenth(BEAT["erase"] + (cx / W) * (BEAT["erased"] - 0.6 - BEAT["erase"]))
        line = (Motion(**{"stroke-dashoffset": 1}).at(start).to(1.0, "inOut", **{"stroke-dashoffset": 0})
                .at(gone).to(0.6, "in", **{"stroke-dashoffset": -1}))
        sketches.append(f'<path class="{sketch_sheet.cls(line)}" d="{d}" pathLength="1"/>')

    # Night glow: blurred copies of the lights, swelling and twinkling.
    glow_sheet = Sheet(D, prefix="g")
    envelope = (Motion(opacity=0).at(BEAT["dusk"] + 1).to(2, "inOut", opacity=1)
                .at(BEAT["nightEnd"] - 0.5).to(2.5, "inOut", opacity=0))
    twinkles = []
    for k in sorted(lights):
        period = (2.4, 3.2, 4.0)[k % 3]
        tw = Motion(opacity=.3).to(period / 2, "sine", opacity=1).to(period / 2, "sine", opacity=.3)
        cls = glow_sheet.cls(tw, duration=period, delay=-round(period * k / 6, 2))
        twinkles.append(f'<g class="{cls}"><g filter="url(#bl)">')
        twinkles += merged(lights[k])
        twinkles.append("</g></g>")

    vars_css = "\n".join(f"@property --c{i}{{syntax:'<color>';inherits:true;initial-value:{c}}}"
                         for i, c in enumerate(colors))
    color_css = "".join(f".c{i}{{fill:var(--c{i});stroke:var(--c{i})}}" for i in range(len(colors)))
    root_cls = sheet.cls(grade)
    canvas_cls = sheet.cls(canvas_grade)
    env_cls = glow_sheet.cls(envelope)
    # Ground: the trace's background color shows through the gaps the trace
    # leaves, so it washes in once most paint is down and leaves first.
    bg = rgb(bg_hex)
    ground_c = min(range(len(colors)), key=lambda i: sum((a - b) ** 2 for a, b in zip(base[colors[i]], bg)))
    ground_cls = sheet.cls(Motion(opacity=0).at(11).to(4, "inOut", opacity=1).at(BEAT["fall"]).to(1.5, "in", opacity=0))
    style = f"""<style><![CDATA[
{vars_css}
{color_css}
#v,:where(#v *){{animation-duration:{num(D)}s;animation-iteration-count:infinite;animation-fill-mode:both;animation-timing-function:linear}}
#k path{{fill:none;stroke:{CHARCOAL};stroke-width:1.4;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1 1;opacity:.7}}
{sheet.css()}
{sketch_sheet.css()}
{glow_sheet.css()}
@media (prefers-reduced-motion:reduce){{#v,#v *{{animation:none!important}}#k,#glow{{display:none}}}}
]]></style>"""

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" id="v" class="{root_cls}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">
<title id="t">Starry Night, painting itself</title>
<desc id="d">A charcoal sketch of Van Gogh's Starry Night is painted over shape by shape, falls to deep night with the stars glowing, warms into dawn, then the paint falls away and the sketch erases itself.</desc>
<defs><filter id="bl" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="7"/></filter></defs>
{style}
<rect class="{canvas_cls}" width="{W}" height="{H}" fill="{CANVAS}"/>
<rect id="ground" class="c{ground_c} {ground_cls}" width="{W}" height="{H}"/>
<g id="k">
{chr(10).join(sketches)}
</g>
<g id="f" stroke-width="1" stroke-linejoin="round">
{chr(10).join(fills)}
</g>
<g id="glow" class="{env_cls}" style="mix-blend-mode:screen">
{chr(10).join(twinkles)}
</g>
</svg>
"""
    Path(out).write_text(svg)
    print(f"{out}: {len(svg):,} B, {sum(f.startswith("<path") for f in fills)} paths, {len(colors)} colors, {len(sketches)} sketch lines, "
          f"{sum(map(len, lights.values()))} lights, {len(sheet.frames) + len(sketch_sheet.frames) + len(glow_sheet.frames)} keyframes")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else HERE / "src/starry_night.jpg",
          sys.argv[2] if len(sys.argv) > 2 else HERE / "starry-timeline.svg")
