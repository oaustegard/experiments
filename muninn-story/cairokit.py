"""cairokit: the pieces pycairo leaves out, each verified on pycairo 1.29 / cairo 1.18.

    pip install --break-system-packages pycairo numpy uharfbuzz fonttools skia-pathops
"""
import math, subprocess
import cairo
import numpy as np


# ------------------------------------------------------------------ probe --
def pango():
    """Return (Pango, PangoCairo) modules, or None. Pango = paragraph layout,
    wrapping, bidi, font fallback, markup. Often absent in containers (PyGObject
    built for another Python, or no typelibs); then use Shaper below."""
    try:
        import gi
        gi.require_version("Pango", "1.0"); gi.require_version("PangoCairo", "1.0")
        from gi.repository import Pango, PangoCairo
        return Pango, PangoCairo
    except Exception:
        return None


# ---------------------------------------------------------------- surfaces --
def new_image(w, h, bg=(0.07, 0.07, 0.09), fmt=cairo.FORMAT_ARGB32):
    """ImageSurface + Context, background painted unless bg is None."""
    s = cairo.ImageSurface(fmt, w, h)
    c = cairo.Context(s)
    if bg is not None:
        c.set_source_rgb(*bg); c.paint()
    return s, c


def label(c, text, x, y, size=14, fg=(0.9, 0.9, 0.85), box=(0, 0, 0, 0.55)):
    """Latin-only caption on a translucent box (toy text API: fine for ASCII labels)."""
    c.save()
    c.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    c.set_font_size(size)
    e = c.text_extents(text)
    if box:
        c.set_source_rgba(*box); c.rectangle(x - 6, y - size - 4, e.x_advance + 12, size + 12); c.fill()
    c.set_source_rgb(*fg); c.move_to(x, y); c.show_text(text)
    c.restore()


# ----------------------------------------------------- numpy pixel bridge --
def pixels(s):
    """Live (h, w, 4) uint8 view of an ARGB32 surface. Channel order is B,G,R,A,
    alpha-PREMULTIPLIED. Call s.mark_dirty() after writing through the view."""
    s.flush()
    a = np.ndarray((s.get_height(), s.get_stride() // 4, 4), np.uint8, s.get_data())
    return a[:, : s.get_width(), :]


def from_array(bgra):
    """(h, w, 4) array (BGRA, premultiplied, 0-255) -> new ARGB32 ImageSurface."""
    h, w, _ = bgra.shape
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    pixels(s)[:] = np.clip(bgra, 0, 255).astype(np.uint8)
    s.mark_dirty()
    return s


def from_rgb_float(rgb):
    """(h, w, 3) float RGB in 0..1 -> opaque ARGB32 surface (for numpy 'shaders')."""
    h, w, _ = rgb.shape
    out = np.empty((h, w, 4), np.float32)
    out[..., 0], out[..., 1], out[..., 2] = rgb[..., 2], rgb[..., 1], rgb[..., 0]
    out[..., :3] *= 255; out[..., 3] = 255
    return from_array(out)


def _box(a, r, axis):
    pad = [(0, 0)] * a.ndim; pad[axis] = (r + 1, r)
    cs = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
    n = cs.shape[axis]
    return (np.take(cs, range(2 * r + 1, n), axis=axis) - np.take(cs, range(0, n - 2 * r - 1), axis=axis)) / (2 * r + 1)


def gauss_blur(a, sigma):
    """Three box passes per axis ~ gaussian. Works on premultiplied data, which
    is exactly what you want: no dark fringes."""
    r = max(1, int(round(math.sqrt(4 * sigma * sigma + 1) / 2)))
    a = a.astype(np.float32)
    for ax in (0, 1):
        for _ in range(3):
            a = _box(a, r, ax)
    return a


def blurred(s, sigma, gain=1.0):
    """New surface: gaussian-blurred copy of s (drop shadows, glows, soft masks)."""
    return from_array(gauss_blur(pixels(s), sigma) * gain)


def glow(sharp, radii=((6, 2.2), (22, 2.5)), bg=(0.03, 0.02, 0.06)):
    """Neon composite: blurred halos (sigma, gain) under the sharp layer, ADD-ed.
    ~240 ms at 800x600 for two radii."""
    px = pixels(sharp)
    acc = sum(gauss_blur(px, sg) * g for sg, g in radii)
    out, c = new_image(sharp.get_width(), sharp.get_height(), bg)
    c.set_source_surface(from_array(acc)); c.paint()
    c.set_operator(cairo.OPERATOR_ADD); c.set_source_surface(sharp); c.paint()
    c.set_operator(cairo.OPERATOR_OVER)
    return out


# ------------------------------------------- shaped text (HarfBuzz + outlines) --
class Shaper:
    """Correct text for any script. cairo's show_text() does no shaping: Arabic
    comes out disjoint and left-to-right, Indic conjuncts fall apart, no kerning.
    This shapes with HarfBuzz and emits glyph OUTLINES as cairo paths, so the
    result also works on PDF/SVG surfaces and can be bent along curves."""

    def __init__(self, font_path, features=None):
        import uharfbuzz as hb
        from fontTools.ttLib import TTFont
        self.hb = hb
        self.font = hb.Font(hb.Face(hb.Blob.from_file_path(font_path)))
        self.upem = self.font.face.upem
        tt = TTFont(font_path)
        self.glyphs, self.order = tt.getGlyphSet(), tt.getGlyphOrder()
        self.features = features or {"kern": True, "liga": True}

    def shape(self, text, direction=None, script=None, language=None):
        """[(gid, x_advance, x_offset, y_offset)] in font units, visual order."""
        buf = self.hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
        if direction: buf.direction = direction
        if script: buf.script = script
        if language: buf.language = language
        self.hb.shape(self.font, buf, self.features)
        return [(i.codepoint, p.x_advance, p.x_offset, p.y_offset)
                for i, p in zip(buf.glyph_infos, buf.glyph_positions)]

    def glyph_path(self, c, gid):
        """Append one glyph outline in font units, y-up. Caller sets the CTM."""
        self.glyphs[self.order[gid]].draw(_CairoPen(self.glyphs, c))

    def width(self, text, size):
        return sum(a for _, a, _, _ in self.shape(text)) * size / self.upem

    def text_path(self, c, text, x, y, size):
        """Append the shaped run at baseline (x, y). Returns advance in px.
        Then c.fill() / c.stroke() / c.clip() as usual."""
        k = size / self.upem; pen = 0
        for gid, adv, xo, yo in self.shape(text):
            c.save()
            c.translate(x + (pen + xo) * k, y - yo * k); c.scale(k, -k)
            self.glyph_path(c, gid)
            c.restore()               # the path survives restore; CTM was applied on append
            pen += adv
        return pen * k

    def text_on_curve(self, c, text, walker, size, start=0.0):
        """Append glyphs along a curve. walker(dist) -> (x, y, angle) or None past
        the end (see polyline_walker). Each glyph is centred on its arc position."""
        k = size / self.upem; pen = start; n = 0
        for gid, adv, xo, yo in self.shape(text):
            a = adv * k
            p = walker(pen + a / 2)
            if p is None:
                break
            x, y, ang = p
            c.save(); c.translate(x, y); c.rotate(ang)
            c.translate(-a / 2 + xo * k, -yo * k); c.scale(k, -k)
            self.glyph_path(c, gid); c.restore()
            pen += a; n += 1
        return n


try:
    from fontTools.pens.basePen import BasePen

    class _CairoPen(BasePen):
        def __init__(self, gs, c):
            super().__init__(gs); self.c = c
        def _moveTo(self, p): self.c.move_to(*p)
        def _lineTo(self, p): self.c.line_to(*p)
        def _curveToOne(self, a, b, p): self.c.curve_to(*a, *b, *p)
        def _qCurveToOne(self, q, p):          # quadratic -> cubic
            x0, y0 = self.c.get_current_point()
            self.c.curve_to(x0 + 2 / 3 * (q[0] - x0), y0 + 2 / 3 * (q[1] - y0),
                            p[0] + 2 / 3 * (q[0] - p[0]), p[1] + 2 / 3 * (q[1] - p[1]), *p)
        def _closePath(self): self.c.close_path()
except ImportError:          # fontTools missing: Shaper will fail at construction
    pass


def polyline_walker(points):
    """Arc-length parametrisation of a sampled curve. Sample densely (1-2 px),
    in the SAME space you draw in (after any squash/scale), or glyph spacing drifts."""
    P = np.asarray(points, float)
    seg = np.diff(P, axis=0); L = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0], np.cumsum(L)])

    def at(d):
        if d < 0 or d > cum[-1]:
            return None
        i = min(np.searchsorted(cum, d, side="right") - 1, len(seg) - 1)
        t = (d - cum[i]) / (L[i] or 1)
        x, y = P[i] + seg[i] * t
        return x, y, math.atan2(seg[i, 1], seg[i, 0])
    at.length = cum[-1]
    return at


# ---------------------------------------------- path geometry (skia-pathops) --
def to_pathops(c):
    """Current cairo path -> pathops.Path (fill rule taken from the context)."""
    import pathops
    p = pathops.Path()
    p.fillType = (pathops.FillType.EVEN_ODD if c.get_fill_rule() == cairo.FILL_RULE_EVEN_ODD
                  else pathops.FillType.WINDING)
    for kind, pts in c.copy_path():
        if kind == cairo.PATH_MOVE_TO: p.moveTo(*pts)
        elif kind == cairo.PATH_LINE_TO: p.lineTo(*pts)
        elif kind == cairo.PATH_CURVE_TO: p.cubicTo(*pts)
        elif kind == cairo.PATH_CLOSE_PATH: p.close()
    return p


def append_pathops(c, p):
    """Append a pathops.Path to the cairo context's current path."""
    import pathops
    for verb, pts in p:
        if verb == pathops.PathVerb.MOVE: c.move_to(*pts[0])
        elif verb == pathops.PathVerb.LINE: c.line_to(*pts[0])
        elif verb == pathops.PathVerb.CUBIC: c.curve_to(*pts[0], *pts[1], *pts[2])
        elif verb == pathops.PathVerb.QUAD:
            x0, y0 = c.get_current_point(); (qx, qy), (x, y) = pts
            c.curve_to(x0 + 2/3*(qx-x0), y0 + 2/3*(qy-y0), x + 2/3*(qx-x), y + 2/3*(qy-y), x, y)
        elif verb == pathops.PathVerb.CLOSE: c.close_path()


def boolean(c, build_a, build_b, op="union"):
    """Real path booleans (cairo only has clipping). build_x(c) appends a path.
    op: union | intersection | difference | xor. Result is appended to c's path."""
    import pathops
    c.new_path(); build_a(c); a = to_pathops(c)
    c.new_path(); build_b(c); b = to_pathops(c)
    c.new_path()
    ops = {"union": pathops.PathOp.UNION, "intersection": pathops.PathOp.INTERSECTION,
           "difference": pathops.PathOp.DIFFERENCE, "xor": pathops.PathOp.XOR}
    append_pathops(c, pathops.op(a, b, ops[op]))


def stroke_to_path(c, width, cap="round", join="round", miter=4.0):
    """Replace the current path with the outline of its stroke (cairo has no
    stroke-to-path). Useful for offset outlines, hatching masks, SVG export."""
    import pathops
    p = to_pathops(c)
    caps = {"butt": pathops.LineCap.BUTT_CAP, "round": pathops.LineCap.ROUND_CAP,
            "square": pathops.LineCap.SQUARE_CAP}
    joins = {"miter": pathops.LineJoin.MITER_JOIN, "round": pathops.LineJoin.ROUND_JOIN,
             "bevel": pathops.LineJoin.BEVEL_JOIN}
    p.stroke(width, caps[cap], joins[join], miter)
    p.convertConicsToQuads()          # round caps/joins emit conics; simplify() rejects them
    p.simplify()
    c.new_path(); append_pathops(c, p)


# ------------------------------------------------- 2.5D: painter + Gouraud --
def painter_order(depth):
    """Indices far-to-near. depth: (N,) camera-space z, larger = farther."""
    return np.argsort(-np.asarray(depth))


def gouraud_mesh(quads, colors, order=None):
    """One MeshPattern with a straight-edged Coons patch per quad and a colour
    per corner = Gouraud shading. Patches paint in insertion order, so pass the
    painter order. quads (N,4,2) screen xy; colors (N,4,3|4) 0..1.
    10k quads: ~130 ms, faster than 10k flat fills. PDF keeps it vector
    (ShadingType 6/7); the SVG backend rasterizes it."""
    m = cairo.MeshPattern()
    rgba = colors.shape[-1] == 4
    for q in (range(len(quads)) if order is None else order):
        m.begin_patch(); m.move_to(*quads[q, 0])
        for k in (1, 2, 3): m.line_to(*quads[q, k])
        for k in range(4):
            (m.set_corner_color_rgba if rgba else m.set_corner_color_rgb)(k, *colors[q, k])
        m.end_patch()
    return m


def project(P, f=9.0, scale=100.0, cx=0.0, cy=0.0):
    """Perspective: camera at z=-f looking +z. P (...,3) -> (xy (...,2), z (...))."""
    z = P[..., 2] + f
    k = scale * f / z
    return np.stack([cx + P[..., 0] * k, cy + P[..., 1] * k], -1), z


# ----------------------------------------------------------------- output --
def frames_to_mp4(pattern, out, fps=30, crf=20):
    """pattern like 'frames/f%04d.png'. Keep frame w and h even for yuv420p."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), out], check=True)


def contact_sheet(files, out, cols=2, scale=0.5, gap=8, bg=(0.02, 0.02, 0.03)):
    """Tile PNGs into one image (cheap to view in a single Read)."""
    imgs = [cairo.ImageSurface.create_from_png(f) for f in files]
    w = int(max(i.get_width() for i in imgs) * scale); h = int(max(i.get_height() for i in imgs) * scale)
    rows = math.ceil(len(imgs) / cols)
    s, c = new_image(cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap, bg)
    for k, img in enumerate(imgs):
        c.save(); c.translate(gap + (k % cols) * (w + gap), gap + (k // cols) * (h + gap))
        c.scale(scale, scale); c.set_source_surface(img); c.get_source().set_filter(cairo.FILTER_BEST)
        c.paint(); c.restore()
    s.write_to_png(out)
    return out
