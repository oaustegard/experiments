"""Mona Lisa, typing — a pycairo + numpy homage to Leonardo (c. 1503-1519).

Every pixel is drawn in code: cairo rasterises shapes, splines, strokes and text; numpy
composites glazes, blurs (sfumato), noise and a small z-buffered sphere-swept renderer for
the hands and sleeves; the laptop is real 3D geometry through a pinhole camera matched to
the panel. No pixels come from the original painting.

    pip install pycairo numpy scipy
    python3 mona_lisa_typing.py            # -> mona_lisa_typing.png (960 x 1431), ~2 min on one core
"""


# ======================================================================
# core
# ======================================================================
"""Shared helpers: cairo rasterises every shape; numpy composites, blurs, textures."""
import math
import cairo
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, map_coordinates

W, H = 960, 1431
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


# ------------------------------------------------------------ paths --------
def spline(c, pts, closed=True, t=0.5):
    """Catmull-Rom through pts -> cubic Beziers appended to c."""
    P = [np.asarray(p, float) for p in pts]
    n = len(P)
    if n < 2:
        return
    c.move_to(*P[0])
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        p0 = P[(i - 1) % n] if (closed or i > 0) else P[i]
        p1 = P[i]
        p2 = P[(i + 1) % n]
        p3 = P[(i + 2) % n] if (closed or i + 2 < n) else P[(i + 1) % n]
        b1 = p1 + (p2 - p0) * t / 3
        b2 = p2 - (p3 - p1) * t / 3
        c.curve_to(*b1, *b2, *p2)
    if closed:
        c.close_path()


def poly(c, pts, closed=True):
    c.move_to(*pts[0])
    for p in pts[1:]:
        c.line_to(*p)
    if closed:
        c.close_path()


def mask(fn, blur=0.0, w=W, h=H):
    """Rasterise whatever fn(c) fills/strokes into a float coverage mask."""
    s = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
    c = cairo.Context(s)
    c.set_source_rgba(0, 0, 0, 1)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    fn(c)
    s.flush()
    a = np.ndarray((h, s.get_stride()), np.uint8, s.get_data())[:, :w].astype(np.float32) / 255.0
    if blur:
        a = gaussian_filter(a, blur)
    return a


def fill_spline(pts, blur=0.0, t=0.5):
    def f(c):
        spline(c, pts, True, t)
        c.fill()
    return mask(f, blur)


def fill_poly(pts, blur=0.0):
    def f(c):
        poly(c, pts)
        c.fill()
    return mask(f, blur)


def stroke_spline(pts, width, blur=0.0, closed=False, alpha=1.0):
    def f(c):
        c.set_source_rgba(0, 0, 0, alpha)
        spline(c, pts, closed)
        c.set_line_width(width)
        c.stroke()
    return mask(f, blur)


def ellipse(cx, cy, rx, ry, rot=0.0, blur=0.0):
    def f(c):
        c.save()
        c.translate(cx, cy)
        c.rotate(rot)
        c.scale(rx, ry)
        c.arc(0, 0, 1, 0, 2 * math.pi)
        c.restore()
        c.fill()
    return mask(f, blur)


# ------------------------------------------------------------ colour -------
def rgb(*v):
    return np.array(v, np.float32) / 255.0


def lay(img, m, col, a=1.0):
    """Paint col (3,) or (H,W,3) through coverage m (H,W) at opacity a."""
    k = np.clip(m * a, 0, 1)[..., None]
    img *= (1 - k)
    img += k * col


def mul(img, m, col, a=1.0):
    """Multiply-glaze: darken towards img*col through m."""
    k = np.clip(m * a, 0, 1)[..., None]
    img *= (1 - k) + k * np.asarray(col, np.float32)


def ramp(t, stops):
    """Piecewise-linear gradient map. t (H,W) in 0..1, stops [(pos, rgb)]."""
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,), np.float32)
    pos = [s[0] for s in stops]
    cols = [np.asarray(s[1], np.float32) for s in stops]
    for ch in range(3):
        out[..., ch] = np.interp(t, pos, [c[ch] for c in cols])
    return out


def vgrad(y0, y1, c0, c1):
    t = np.clip((YY - y0) / (y1 - y0), 0, 1)[..., None]
    return np.asarray(c0, np.float32) * (1 - t) + np.asarray(c1, np.float32) * t


# ------------------------------------------------------------ noise --------
def fbm(scale=64, octaves=5, seed=0, persist=0.55, h=H, w=W, aniso=(1.0, 1.0)):
    """Value-noise fBm in ~[0,1]. aniso stretches (y, x) feature size."""
    r = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, tot, s = 1.0, 0.0, float(scale)
    for _ in range(octaves):
        gh = max(2, int(h / (s * aniso[0])) + 3)
        gw = max(2, int(w / (s * aniso[1])) + 3)
        g = r.random((gh, gw)).astype(np.float32)
        z = zoom(g, (h / (gh - 2), w / (gw - 2)), order=3)[:h, :w]
        acc += amp * z
        tot += amp
        amp *= persist
        s /= 2
        if s < 1.5:
            break
    acc /= tot
    lo, hi = np.percentile(acc, 1), np.percentile(acc, 99)
    return np.clip((acc - lo) / (hi - lo + 1e-6), 0, 1)


def noise1d(n, scale, seed, octaves=5, persist=0.55):
    r = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float32)
    acc = np.zeros(n, np.float32)
    amp, tot, s = 1.0, 0.0, float(scale)
    for _ in range(octaves):
        k = int(n / s) + 4
        g = r.random(k).astype(np.float32) * 2 - 1
        acc += amp * np.interp(x / s, np.arange(k), g)
        tot += amp
        amp *= persist
        s /= 2
        if s < 1:
            break
    acc = acc / tot
    return acc / (np.abs(acc).max() + 1e-6)


def warp(field, dx, dy):
    return map_coordinates(field, [np.clip(YY + dy, 0, H - 1), np.clip(XX + dx, 0, W - 1)], order=1)


# ------------------------------------------------------------ output -------
def save(img, path):
    rgbf = np.clip(np.nan_to_num(img), 0, 1)
    h, w, _ = rgbf.shape
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    s.flush()
    a = np.ndarray((h, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :w, :]
    a[..., 0] = (rgbf[..., 2] * 255 + 0.5).astype(np.uint8)
    a[..., 1] = (rgbf[..., 1] * 255 + 0.5).astype(np.uint8)
    a[..., 2] = (rgbf[..., 0] * 255 + 0.5).astype(np.uint8)
    a[..., 3] = 255
    s.mark_dirty()
    s.write_to_png(path)


# ======================================================================
# laptop_geo
# ======================================================================
"""Laptop as real 3D geometry, projected with a pinhole camera matched to the painting.

Camera at the viewer's eye (the horizon sits near her eyes, as Leonardo placed it).
f is chosen so her face (~14.5 cm wide at ~1.3 m) spans the ~210 px it does in the painting.
World: x right, y up, z away from the viewer (metres).
"""

CX, CY, F = 480.0, 390.0, 1880.0


def project(P):
    P = np.asarray(P, float)
    x = CX + F * P[..., 0] / P[..., 2]
    y = CY - F * P[..., 1] / P[..., 2]
    return np.stack([x, y], -1)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class Laptop:
    """Local frame: u metres along the hinge (+ = viewer's right), v metres from the hinge toward
    her, h metres above the deck top. Keyboard legends are laid out from HER side."""

    def __init__(self, x0=-0.02, y0=-0.50, z_hinge=1.0, yaw_deg=14.0, opening_deg=157.0,
                 w=0.30, d=0.205, t_base=0.013, lid_len=0.19, t_lid=0.006):
        self.w, self.d, self.tb, self.L, self.tl = w, d, t_base, lid_len, t_lid
        psi = np.radians(yaw_deg)
        self.dv = np.array([np.sin(psi), 0, np.cos(psi)])
        self.du = np.array([np.cos(psi), 0, -np.sin(psi)])
        self.up = np.array([0.0, 1.0, 0.0])
        self.H0 = np.array([x0, y0 + t_base, z_hinge])
        a = np.radians(opening_deg - 90.0)
        self.dl = np.cos(a) * self.up - np.sin(a) * self.dv       # hinge -> lid top edge
        n = np.cross(self.dl, self.du)
        self.nl = n if np.dot(n, self.dv) > 0 else -n             # screen normal, toward her

    def P(self, u, v, h=0.0):
        return self.H0 + u * self.du + v * self.dv + h * self.up

    def lidP(self, u, s, off=0.0):
        """s metres from the hinge along the lid; off along the screen normal."""
        return self.H0 + u * self.du + s * self.dl + off * self.nl

    def screen_centre(self):
        return self.lidP(0.0, self.L * 0.52, 0.001)

    @staticmethod
    def p(P):
        return project(P)


# ======================================================================
# hair
# ======================================================================
rng = np.random.default_rng(42)


def resample(pts, n):
    P = np.asarray(pts, float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(t, cum, P[:, 0]), np.interp(t, cum, P[:, 1])], -1)


L_OUT = [(466, 136), (420, 144), (380, 166), (338, 210), (314, 262), (300, 330), (293, 400), (292, 470), (298, 545),
         (302, 610), (292, 660), (274, 700), (256, 734)]
L_IN = [(455, 200), (420, 203), (380, 214), (354, 236), (347, 290), (345, 340), (350, 392), (364, 440), (390, 480),
        (420, 520), (430, 560), (420, 592), (384, 616), (352, 650), (330, 700), (312, 740)]
R_OUT = [(470, 136), (530, 142), (590, 176), (628, 220), (652, 272), (664, 330), (676, 395), (690, 460), (704, 525),
         (718, 590), (740, 640), (764, 688), (784, 728)]
R_IN = [(457, 200), (500, 204), (536, 220), (550, 262), (551, 320), (548, 372), (541, 420), (528, 462), (540, 505),
        (556, 556), (574, 612), (590, 660), (608, 702), (630, 740), (652, 776)]


def strands(c, outer, inner, n, col_fn, width_fn, jitter, seed, wave=0.0):
    r = np.random.default_rng(seed)
    O = resample(outer, 120)
    I = resample(inner, 120)
    for i in range(n):
        k = r.random() ** 0.9
        base = I * (1 - k) + O * k
        j = r.normal(0, jitter, 2)
        ph = r.random() * 6.28
        pts = base + j
        if wave:
            tt = np.linspace(0, 1, len(pts))
            wy = np.clip((pts[:, 1] - 330) / 250, 0, 1)
            pts = pts + np.stack([np.sin(pts[:, 1] / 22 + ph) * wave * 2.2 * wy, np.zeros_like(tt)], -1)
        a0 = r.random() * 0.15
        a1 = min(1.0, a0 + 0.35 + r.random() * 0.6)
        seg = pts[int(a0 * len(pts)):int(a1 * len(pts))]
        if len(seg) < 4:
            continue
        col = col_fn(k, r)
        c.set_source_rgba(*col)
        c.set_line_width(width_fn(r))
        spline(c, seg[::3], closed=False)
        c.stroke()


def rgba_layer(draw):
    """Draw with colour onto a transparent ARGB surface; return (rgb, alpha) floats."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    c = cairo.Context(s)
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    c.set_line_join(cairo.LINE_JOIN_ROUND)
    draw(c)
    s.flush()
    a = np.ndarray((H, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :W, :].astype(np.float32) / 255
    alpha = a[..., 3]
    col = np.stack([a[..., 2], a[..., 1], a[..., 0]], -1) / np.maximum(alpha[..., None], 1e-4)
    return col, alpha


def comp(img, col, alpha, blur=0.0, a=1.0):
    if blur:
        pre = col * alpha[..., None]
        pre = np.stack([gaussian_filter(pre[..., i], blur) for i in range(3)], -1)
        alpha = gaussian_filter(alpha, blur)
        col = pre / np.maximum(alpha[..., None], 1e-4)
    lay(img, alpha, col, a)


def ringlet(c, x0, y0, length, amp, turns, lean, w, col_dark, col_light, r):
    t = np.linspace(0, 1, 90)
    ph = r.random() * 6.28
    xs = x0 + lean * t * length + amp * np.sin(t * turns * 6.283 + ph) * (0.6 + 0.4 * t)
    ys = y0 + t * length
    pts = np.stack([xs, ys], -1)
    c.set_source_rgba(*col_dark)
    c.set_line_width(w)
    spline(c, pts[::3], closed=False)
    c.stroke()
    # highlight catches the outward-facing half of each turn
    for k in range(int(turns) + 1):
        a = (k + 0.18) / turns
        b = (k + 0.5) / turns
        if a >= 1:
            break
        i0, i1 = int(a * 89), int(min(b, 1) * 89)
        if i1 - i0 < 3:
            continue
        c.set_source_rgba(*col_light)
        c.set_line_width(w * 0.35)
        spline(c, pts[i0:i1:2], closed=False)
        c.stroke()


def paint_hair(img, face_m):
    hair_m = fill_spline(L_OUT[::-1] + R_OUT[1:] + [(640, 700), (360, 700)], blur=3.2)

    # base colour: warm dark brown, lit crown fading to near-black sides
    base = ramp(np.clip((YY - 140) / 520, 0, 1) + 0.25 * np.clip((XX - 470) / 200, 0, 1),
                [(0, rgb(74, 40, 20)), (0.25, rgb(50, 24, 16)), (0.6, rgb(36, 16, 14)), (1.0, rgb(20, 9, 12))])
    band = np.maximum(fill_spline(L_OUT + L_IN[::-1], blur=3.5, t=0.4), fill_spline(R_OUT + R_IN[::-1], blur=3.5, t=0.4))
    vfade = np.clip((770 - YY) / 110, 0, 1) ** 0.9
    band = band * vfade
    cover = np.maximum(hair_m * (1 - face_m) * np.clip((740 - YY) / 60, 0, 1), band)
    lay(img, cover, base)
    # soft shadow the hair casts onto the skin just inside its edge
    lay(img, np.clip(gaussian_filter(band, 5) - band, 0, 1) * face_m, rgb(80, 50, 26), 0.6)
    # crown sheen following the skull
    sheen = stroke_spline([(360, 196), (400, 160), (458, 146), (520, 152), (580, 182), (616, 222)], 16, blur=10)
    lay(img, sheen * cover, rgb(150, 104, 50), 0.6)
    lay(img, stroke_spline([(352, 236), (372, 196), (410, 166), (452, 152)], 14, blur=9) * cover, rgb(150, 106, 50), 0.5)
    lay(img, ellipse(322, 440, 26, 150, blur=24) * cover, rgb(84, 38, 26), 0.4)     # the lit, reddish left fall of hair
    lay(img, stroke_spline([(520, 150), (580, 180), (618, 222), (642, 270)], 10, blur=8) * hair_m, rgb(100, 62, 32), 0.35)

    # long strands from the part down both sides
    def cL(k, r):
        v = 0.4 + 1.1 * r.random() ** 3
        return (0.6 * v, 0.32 * v, 0.15 * v, (0.08 + 0.3 * r.random() * v) * (0.45 + 0.55 * k))
    def cR(k, r):
        v = 0.35 + 0.9 * r.random() ** 3
        return (0.42 * v, 0.25 * v, 0.14 * v, (0.06 + 0.2 * r.random() * v) * (0.45 + 0.55 * k))
    def cDark(k, r):
        return (0.08, 0.04, 0.03, 0.25 + 0.3 * r.random())
    col, al = rgba_layer(lambda c: (strands(c, L_OUT, L_IN, 170, cDark, lambda r: 1.5 + 2 * r.random(), 3, 1),
                                    strands(c, R_OUT, R_IN, 170, cDark, lambda r: 1.5 + 2 * r.random(), 3, 2)))
    comp(img, col, al * vfade, blur=1.0)
    col, al = rgba_layer(lambda c: (strands(c, L_OUT, L_IN, 140, cL, lambda r: 0.8 + 1.6 * r.random(), 2.5, 3, wave=3.0),
                                    strands(c, R_OUT, R_IN, 120, cR, lambda r: 0.8 + 1.4 * r.random(), 2.5, 4, wave=2.6)))
    comp(img, col, al * vfade, blur=1.3)

    # corkscrew ringlets below the ears, reddish on the lit left
    r = np.random.default_rng(9)
    def locks_left(c):
        for i in range(46):
            y0 = 330 + r.random() * 360
            x0 = np.interp(y0, [300, 450, 600], [296, 294, 300]) + r.random() * np.interp(y0, [300, 450, 600], [50, 70, 100])
            ln = min(30 + r.random() * 90, 740 - y0)
            v = 0.5 + 0.6 * r.random()
            ringlet(c, x0, y0, ln, 2 + r.random() * 4.5, ln / (10 + r.random() * 14), -0.2 + r.random() * 0.3,
                    2 + r.random() * 3.5, (0.12, 0.05, 0.04, 0.3), (0.66 * v, 0.44 * v, 0.22 * v, 0.55), r)
    def locks_right(c):
        for i in range(30):
            y0 = 400 + r.random() * 200
            x0 = 556 + (y0 - 400) * 0.25 + r.random() * 90
            ln = min(60 + r.random() * 110, 680 - y0)
            ringlet(c, x0, y0, ln, 2.5 + r.random() * 3.5, ln / (14 + r.random() * 10), 0.12 + r.random() * 0.15,
                    2.5 + r.random() * 3, (0.06, 0.03, 0.04, 0.35), (0.36, 0.22, 0.12, 0.45), r)
    col, al = rgba_layer(locks_left)
    comp(img, col, al * cover, blur=1.6, a=0.8)
    col, al = rgba_layer(locks_right)
    comp(img, col, al * cover, blur=1.6, a=0.8)

    # the outer silhouette is broken by loose locks rather than blurred
    def outer_locks(c):
        for pts, side in ((L_OUT, -1), (R_OUT, 1)):
            P = resample(pts, 80)
            for i in range(6, 78):
                x, y = P[i]
                if rs.random() < 0.5:
                    continue
                d = side * (2 + rs.random() * 6)
                c.set_source_rgba(0.14, 0.07, 0.05, 0.3 + 0.4 * rs.random())
                c.set_line_width(1.0 + rs.random() * 2.0)
                c.move_to(x - d, y - 10)
                c.curve_to(x + d, y - 4, x + d * 1.2, y + 6, x + d * 0.3, y + 16)
                c.stroke()
    rs = np.random.default_rng(78)
    col, al = rgba_layer(outer_locks)
    comp(img, col, al * np.clip((720 - YY) / 80, 0, 1), blur=0.9)
    # stray locks that break the clean line where hair meets skin
    rs = np.random.default_rng(77)
    def stray(c):
        for side, pts in ((1, L_IN), (-1, R_IN)):
            P = resample(pts, 60)
            for i in range(12, 58, 3):
                x, y = P[i]
                dx = side * (2 + rs.random() * 7)
                c.set_source_rgba(0.1, 0.05, 0.03, 0.35 + 0.3 * rs.random())
                c.set_line_width(1.2 + rs.random() * 2.2)
                c.move_to(x - side * 6, y - 14)
                c.curve_to(x + dx, y - 6, x + dx, y + 8, x - side * 2, y + 18)
                c.stroke()
    col, al = rgba_layer(stray)
    comp(img, col, al, blur=1.0)
    # the veil's shadow at the top of the forehead, and warm hair roots at the hairline
    lay(img, np.clip(1 - (YY - 204) / 30, 0, 1) * face_m * (YY > 190), rgb(150, 104, 52), 0.45)
    # the hair part and the fine veil edge across the forehead
    lay(img, stroke_spline([(458, 142), (457, 170), (456, 199)], 2.0, blur=1.2), rgb(150, 104, 58), 0.5)
    veil_edge = [(348, 250), (370, 224), (410, 210), (456, 206), (505, 210), (540, 224), (552, 240)]
    lay(img, stroke_spline(veil_edge, 1.3, blur=0.6), rgb(90, 60, 34), 0.45)
    lay(img, stroke_spline([(p[0], p[1] - 3) for p in veil_edge], 6, blur=3), rgb(130, 90, 44), 0.3)
    lay(img, stroke_spline([(p[0], p[1] + 2) for p in veil_edge], 1.0, blur=0.6), rgb(236, 206, 130), 0.2)
    return hair_m


def paint_veil(img):
    """Transparent gauze: a faint dark halo round the head, stronger down the left side."""
    halo = fill_spline(L_OUT[::-1] + R_OUT[1:] + [(640, 700), (360, 700)], blur=2)
    ring = np.clip(gaussian_filter(halo, 7) * 1.6 - halo, 0, 1)
    lay(img, ring * (YY < 700), rgb(46, 40, 28), 0.22)
    v = fill_spline([(318, 240), (300, 300), (284, 380), (282, 470), (288, 560), (300, 640), (320, 700),
                     (330, 690), (310, 600), (298, 520), (296, 440), (300, 350), (318, 270), (338, 222)], blur=1.5)
    lay(img, gaussian_filter(v, 3), rgb(40, 34, 22), 0.22)
    lay(img, stroke_spline([(318, 240), (300, 300), (284, 380), (282, 470), (288, 560), (300, 640), (320, 700)], 1.0, 0.7),
        rgb(30, 24, 16), 0.35)


# ======================================================================
# landscape2
# ======================================================================
"""Leonardo's imaginary landscape, painted with brush touches rather than gradients."""

XS = np.arange(W + 1, dtype=np.float32)


def ridged(n, scale, seed, octaves=4):
    acc = np.zeros(n, np.float32)
    amp, tot, s = 1.0, 0.0, scale
    for o in range(octaves):
        v = 1 - np.abs(noise1d(n, s, seed + o, octaves=2))
        acc += amp * v ** 2
        tot += amp
        amp *= 0.5
        s /= 2.2
    return acc / tot


def region_from_top(top, bottom, x0=0, x1=W, blur=1.0):
    sel = (XS >= x0) & (XS <= x1)
    xs, ts = XS[sel][::2], top[sel][::2]
    bt = bottom if np.ndim(bottom) else np.full_like(XS, bottom)
    bs = bt[sel][::2]
    pts = list(zip(xs, ts)) + list(zip(xs[::-1], bs[::-1]))
    return fill_poly(pts, blur=blur)


def strokes(img, m, n, colour, length, width, angle, jitter, alpha, seed, bbox=None, curve=0.0, blur=0.7):
    """Scatter n brush touches inside mask m. colour(x, y, r) -> rgb 0..1."""
    r = np.random.default_rng(seed)
    ys, xs = np.nonzero(m > 0.5)
    if len(xs) == 0:
        return
    idx = r.integers(0, len(xs), n)

    def draw(c):
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        for i in idx:
            x, y = xs[i] + r.random() - 0.5, ys[i] + r.random() - 0.5
            L = length * (0.5 + r.random())
            a = angle + r.normal(0, jitter)
            dx, dy = math.cos(a) * L / 2, math.sin(a) * L / 2
            col = colour(x, y, r)
            c.set_source_rgba(col[0], col[1], col[2], alpha * (0.5 + 0.5 * r.random()))
            c.set_line_width(width * (0.6 + 0.8 * r.random()))
            c.move_to(x - dx, y - dy)
            if curve:
                c.curve_to(x - dx * 0.3 + curve * dy, y - dy * 0.3 - curve * dx, x + dx * 0.3 + curve * dy,
                           y + dy * 0.3 - curve * dx, x + dx, y + dy)
            else:
                c.line_to(x + dx, y + dy)
            c.stroke()
    col, al = rgba_layer(draw)
    comp(img, col, al * m, blur=blur)


def palette(stops, top_fn=None, depth=60.0, var=0.12):
    """Colour by depth below a ridge top (lit crest -> dark base) with random variation."""
    def f(x, y, r):
        if top_fn is not None:
            t = 1 - min(1.0, max(0.0, (y - top_fn(x)) / depth))
        else:
            t = r.random()
        t = min(1, max(0, t + r.normal(0, var)))
        k = t * (len(stops) - 1)
        i = min(int(k), len(stops) - 2)
        c = stops[i] * (1 - (k - i)) + stops[i + 1] * (k - i)
        return c * (0.95 + 0.1 * r.random())
    return f


def form(img, top, bottom, x0, x1, stops, depth, n, length, width, angle, jitter, alpha, seed, var=0.1, blur=0.8, curve=0.0,
         fade=0.0, slope_light=0.0):
    m = region_from_top(top, bottom, x0, x1)
    if fade:
        m = m * np.clip((bottom - YY) / fade, 0, 1)
    d = YY - np.interp(XX, XS, top)
    t = np.clip(1 - d / depth, 0, 1) * 0.85 + 0.15 * fbm(18, 3, seed + 500)
    if slope_light:
        sl = np.gradient(gaussian_filter(top, 2.5))
        face = np.clip(-np.interp(XX, XS, sl) * 0.8, -1, 1)          # rising to the right = faces the light
        t = t + slope_light * face * np.exp(-np.clip(d, 0, None) / (depth * 1.2))
    lay(img, m, ramp(t, [(i / (len(stops) - 1), c) for i, c in enumerate(stops)]))
    strokes(img, m, n, palette(stops, lambda x: np.interp(x, XS, top), depth, var), length, width, angle, jitter,
            alpha, seed, blur=blur, curve=curve)
    return m


def paint_landscape(img):
    # ---------- sky: teal-green overhead, pale warm haze at the horizon ----------
    img[:] = vgrad(-80, 330, rgb(80, 104, 76), rgb(176, 178, 124))
    cl = fbm(140, 5, 3, aniso=(0.4, 2.0))
    lay(img, np.clip((cl - 0.5) * 2, 0, 1) * np.clip(1 - YY / 320, 0, 1), rgb(150, 160, 116), 0.28)
    sky_m = (YY < 360).astype(np.float32)
    strokes(img, sky_m, 2500, palette([rgb(110, 128, 90), rgb(150, 158, 112), rgb(178, 178, 126)],
                                      lambda x: -80, 400, 0.08), 26, 5, 0.0, 0.25, 0.12, 1, blur=1.5)
    haze = rgb(176, 178, 126)

    # ---------- far left: pale distant crags, then the dark tall wooded mass ----------
    top_far = 300 + 0.03 * XS - 36 * ridged(len(XS), 50, 21) - 26 * np.exp(-((XS - 200) / 60) ** 2)
    form(img, top_far, 500, 0, 420, [rgb(96, 112, 92), rgb(132, 144, 110), rgb(172, 176, 128)], 80,
         1800, 16, 3.4, math.pi / 2, 0.15, 0.35, 2)
    top_dark = 345 + 0.02 * XS - 46 * ridged(len(XS), 40, 22) * np.clip(1 - XS / 330, 0.25, 1) - 16 * noise1d(len(XS), 10, 23)
    form(img, top_dark, 500, 0, 420, [rgb(34, 48, 40), rgb(54, 72, 56), rgb(84, 102, 76), rgb(126, 136, 98)], 110,
         4000, 13, 3.4, math.pi / 2, 0.2, 0.45, 3, var=0.12)
    lay(img, region_from_top(top_dark, 500, 0, 420) * np.clip((YY - 300) / 200, 0, 1), haze, 0.16)

    # ---------- far right: bluish crag mass, dissolving into bright mist ----------
    top_r = 250 + 0.14 * np.clip(XS - 650, 0, None) - 34 * ridged(len(XS), 40, 31) - 30 * np.exp(-((XS - 720) / 40) ** 2)
    top_r = np.where(XS > 820, top_r + (XS - 820) * 0.35, top_r)
    form(img, top_r, 420, 560, W, [rgb(72, 94, 86), rgb(104, 122, 102), rgb(146, 154, 116), rgb(186, 186, 136)], 90,
         3000, 14, 3.2, math.pi / 2, 0.18, 0.4, 4)
    m_r = region_from_top(top_r, 400, 560, W)
    strokes(img, m_r * np.clip((380 - YY) / 60, 0, 1), 500, palette([rgb(70, 88, 78), rgb(96, 112, 94)], var=0.08),
            22, 2.6, math.pi / 2, 0.12, 0.35, 17)
    mist = np.clip((XX - 790) / 170, 0, 1) * np.clip((YY - 240) / 60, 0, 1) * np.clip((410 - YY) / 40, 0, 1)
    lay(img, mist, rgb(192, 190, 140), 0.7)

    # ---------- right lake + the dark-teal middle distance ----------
    lake_r = region_from_top(np.full_like(XS, 366.0), np.full_like(XS, 412.0), 520, W, blur=1.5)
    lay(img, lake_r, vgrad(366, 410, rgb(188, 186, 136), rgb(150, 156, 116)))
    strokes(img, lake_r, 300, palette([rgb(150, 156, 116), rgb(186, 184, 136)], var=0.05), 36, 2.0, 0.0, 0.03, 0.2, 5, blur=1.4)
    top_mr = 398 + 0.02 * (XS - 560) - 14 * ridged(len(XS), 60, 41)
    form(img, top_mr, 650, 540, W, [rgb(38, 54, 46), rgb(58, 78, 64), rgb(88, 106, 84), rgb(128, 138, 100)], 160,
         2600, 16, 3.0, 0.0, 0.25, 0.18, 6, var=0.12, blur=1.4)
    m_mr = region_from_top(top_mr, 650, 540, W)
    strokes(img, m_mr * (YY > 470), 260, palette([rgb(96, 96, 60), rgb(130, 122, 78), rgb(150, 142, 94)], var=0.08),
            22, 4.0, 0.0, 0.25, 0.2, 16, blur=1.6)                        # ochre fields in the valley
    lay(img, region_from_top(top_mr, 600, 540, W) * np.clip(1 - (YY - 395) / 60, 0, 1), haze, 0.3)

    # ---------- left: middle shore + the lake ----------
    top_ml = 442 + 0.03 * XS - 10 * ridged(len(XS), 50, 42)
    form(img, top_ml, 510, 0, 420, [rgb(40, 54, 46), rgb(64, 82, 64), rgb(104, 116, 88)], 60,
         1800, 22, 3.0, 0.03, 0.15, 0.4, 7)
    lake_l = region_from_top(np.full_like(XS, 486.0) + 0.02 * XS, np.full_like(XS, 690.0), 0, 420, blur=1.5)
    lay(img, lake_l, vgrad(486, 580, rgb(118, 130, 112), rgb(64, 78, 72)))
    strokes(img, lake_l, 500, palette([rgb(70, 84, 76), rgb(98, 112, 100), rgb(126, 136, 114)], var=0.06), 40, 2.2, 0.0, 0.05, 0.2, 8, blur=1.5)
    lay(img, stroke_spline([(-10, 490), (140, 486), (300, 494), (420, 499)], 2.5, blur=1.4), rgb(176, 176, 128), 0.45)

    # ---------- right warm terrain, then the valley ribbon over it ----------
    top_rl = 618 + 0.02 * (XS - 600) - 14 * ridged(len(XS), 40, 51) - 10 * noise1d(len(XS), 30, 52)
    form(img, top_rl, 900, 560, W, [rgb(70, 32, 24), rgb(104, 54, 36), rgb(138, 84, 52), rgb(162, 116, 70)], 170,
         3000, 26, 3.4, 0.08, 0.3, 0.35, 11, var=0.12)
    rt = region_from_top(top_rl, 900, 560, W)
    lay(img, rt * np.clip(fbm(40, 4, 76) - 0.35, 0, 1) * 1.6, rgb(142, 104, 58), 0.35)
    lay(img, rt * np.clip(0.55 - fbm(55, 4, 77), 0, 1) * 1.8, rgb(58, 28, 22), 0.4)
    rv = [(560, 560), (640, 574), (720, 570), (790, 586), (860, 598), (920, 592), (975, 606)]
    band = stroke_spline(rv, 30, blur=6)
    lay(img, band, rgb(168, 156, 96), 0.8)
    strokes(img, band, 900, palette([rgb(150, 140, 86), rgb(186, 176, 116), rgb(206, 196, 138)], var=0.06), 20, 3.0, 0.1, 0.2, 0.4, 9)
    lay(img, stroke_spline(rv, 4, blur=1.2), rgb(210, 202, 146), 0.45)

    def bridge(c):
        c.move_to(750, 610); c.line_to(846, 604); c.line_to(846, 617)
        for i in range(4):
            xa = 846 - i * 23
            c.line_to(xa - 4, 617)
            c.curve_to(xa - 6, 632, xa - 17, 632, xa - 19, 618)
        c.line_to(750, 623); c.close_path(); c.fill()
    lay(img, mask(bridge, 1.0), rgb(118, 100, 70), 0.6)
    lay(img, stroke_spline([(750, 610), (846, 604)], 1.6, 0.8), rgb(170, 156, 110), 0.4)

    # right ochre outcrop at the edge
    top_ro = 700 - 110 * np.clip((XS - 850) / 90, 0, 1) ** 0.6 - sum(h * np.exp(-((XS - c) / wd) ** 2)
                                                                      for c, h, wd in [(900, 26, 14), (935, 40, 16), (962, 30, 14)])
    form(img, top_ro, 700, 840, W, [rgb(56, 46, 32), rgb(96, 80, 50), rgb(134, 118, 76), rgb(160, 146, 98)], 70,
         700, 8, 4.0, math.pi / 2, 0.3, 0.25, 10, fade=60, slope_light=0.3)

    # ---------- left: ochre rocks rising from the near shore ----------
    env = np.clip(np.sin(np.clip(XS - 50, 0, 320) / 320 * np.pi), 0, 1) ** 0.5
    towers = env * (52 * ridged(len(XS), 90, 63) + 12 * ridged(len(XS), 26, 64) + 4 * ridged(len(XS), 7, 65))
    top_rk = 656 - 24 * env - towers
    form(img, top_rk, 720, 0, 384, [rgb(38, 34, 24), rgb(72, 62, 36), rgb(120, 102, 58), rgb(176, 152, 90)], 70,
         1400, 6, 4.0, math.pi / 2, 0.7, 0.2, 12, slope_light=0.3)
    rk = region_from_top(top_rk, 720, 0, 384)
    for x in np.arange(60, 380, 17):                                     # vertical fissures
        y0 = float(np.interp(x, XS, top_rk)) + 10
        lay(img, stroke_spline([(x, y0), (x + 3, y0 + 30), (x - 2, y0 + 60)], 3.0, blur=2.2) * rk, rgb(52, 42, 26), 0.12)
    strata = 0.5 + 0.5 * np.sin(YY / 7.0 + 3 * fbm(30, 3, 66))
    lay(img, rk * strata * 0.5, rgb(60, 50, 30), 0.35)
    lay(img, rk * np.clip((YY - 660) / 50, 0, 1), rgb(70, 52, 34), 0.6)             # bases dissolve into the ground

    # ---------- left foreground: red earth, the winding road ----------
    top_lg = 676 + XS * 0.04 - 8 * ridged(len(XS), 40, 71)
    form(img, top_lg, 900, 0, 364, [rgb(70, 30, 20), rgb(110, 54, 32), rgb(146, 88, 48), rgb(170, 118, 66)], 150,
         3000, 26, 3.6, 0.05, 0.35, 0.35, 13, var=0.12)
    fg = region_from_top(top_lg, 900, 0, 364)
    lay(img, fg * np.clip(fbm(40, 4, 74) - 0.35, 0, 1) * 1.6, rgb(150, 110, 56), 0.35)       # ochre patches
    lay(img, fg * np.clip(0.55 - fbm(55, 4, 75), 0, 1) * 1.8, rgb(62, 30, 22), 0.4)          # shadowed hollows
    lay(img, stroke_spline([(20, 700), (80, 740), (140, 800), (170, 860)], 30, blur=16) * fg, rgb(70, 40, 26), 0.4)
    road = [(310, 694), (220, 704), (140, 716), (74, 736), (54, 756), (92, 776), (190, 790), (222, 804), (150, 822), (40, 836)]
    lay(img, stroke_spline(road, 8, blur=2.4), rgb(160, 102, 56), 0.55)
    lay(img, stroke_spline(road, 3, blur=1.4), rgb(192, 140, 84), 0.25)
    # the dark cliff at the far left edge, blended into the forest mass
    lc = fill_spline([(-10, 320), (24, 330), (56, 420), (66, 520), (60, 610), (-10, 620)], blur=18)
    lay(img, lc, rgb(36, 46, 38), 0.5)

    # aerial perspective: everything softened by the air
    for ch in range(3):
        img[..., ch] = gaussian_filter(img[..., ch], 0.75)

    # ---------- loggia parapet + column base ----------
    par = fill_poly([(0, 856), (W, 850), (W, 904), (0, 906)], blur=1.0)
    pcol = ramp(fbm(6, 4, 81, aniso=(0.3, 3.0)) * 0.3 + 0.7 * np.clip(1 - (YY - 856) / 50, 0, 1),
                [(0, rgb(60, 38, 34)), (0.6, rgb(104, 76, 56)), (1, rgb(138, 106, 76))])
    pcol *= (1 - 0.3 * np.clip((XX - 500) / 460, 0, 1))[..., None]
    lay(img, par, pcol)
    lay(img, fill_poly([(0, 900), (W, 898), (W, H), (0, H)], blur=2.0), rgb(36, 22, 26))
    # the base of a loggia column at the left edge, barely there
    cb = fill_poly([(-5, 780), (26, 780), (30, 790), (34, 850), (-5, 852)], blur=2.5)
    lay(img, cb, ramp(np.clip(XX / 34, 0, 1), [(0, rgb(52, 38, 30)), (1, rgb(96, 74, 54))]), 0.8)
    lay(img, fill_poly([(-5, 776), (32, 776), (34, 786), (-5, 786)], blur=1.5), rgb(120, 94, 66), 0.6)
    return img


# ======================================================================
# figure
# ======================================================================
# ---------------------------------------------------------------- shapes --
HEAD_OUTER = [(468, 134), (530, 142), (590, 176), (628, 220), (652, 272), (662, 330), (668, 395), (674, 460),
              (684, 525), (698, 590), (716, 648), (740, 690), (560, 720), (340, 720), (318, 668), (308, 610),
              (298, 545), (292, 470), (293, 400), (300, 330), (314, 262), (338, 210), (380, 166), (420, 144)]

FACE = [(455, 199), (512, 206), (545, 228), (556, 270), (557, 320), (553, 372), (546, 420), (532, 460),
        (512, 486), (486, 500), (455, 507), (428, 504), (406, 492), (388, 472), (368, 440), (355, 392),
        (348, 332), (348, 278), (356, 232), (392, 208)]

# neck + chest skin, from under the jaw to the neckline
NECK_CHEST = [(400, 470), (430, 480), (470, 482), (520, 470), (548, 516), (566, 568), (594, 616), (628, 650),
              (648, 690), (636, 722), (604, 748), (556, 766), (498, 776), (440, 772), (384, 758), (338, 736),
              (310, 712), (318, 676), (346, 640), (366, 604), (380, 560)]

NECKLINE = [(304, 716), (336, 742), (384, 764), (440, 778), (498, 782), (556, 772), (606, 752), (640, 726), (656, 694)]

BODY = [(330, 640), (280, 668), (230, 700), (180, 752), (140, 812), (110, 880), (86, 960), (66, 1040), (50, 1130),
        (40, 1250), (30, 1440), (960, 1440), (960, 1170), (948, 1080), (926, 980), (900, 890), (866, 810),
        (820, 742), (770, 690), (720, 650), (640, 640)]


def skin_base():
    face = fill_spline(FACE, blur=2.4)
    face = np.maximum(face * np.clip((500 - YY) / 20 + 0.0, 0, 1), fill_spline(FACE, blur=5) * (YY > 470))
    neck = fill_spline(NECK_CHEST, blur=1.2)
    return face, neck


# ======================================================================
# face
# ======================================================================
def sm(pts, w, blur, closed=False, alpha=1.0):
    return stroke_spline(pts, w, blur=blur, closed=closed, alpha=alpha)


def g2(cx, cy, sx, sy, amp, rot=0.0):
    x, y = XX - cx, YY - cy
    if rot:
        c, s = np.cos(rot), np.sin(rot)
        x, y = c * x + s * y, -s * x + c * y
    return amp * np.exp(-0.5 * ((x / sx) ** 2 + (y / sy) ** 2))


def ridge_line(pts, sigma, amp_fn):
    """Height ridge along a polyline; amp_fn(t) for t in 0..1 along it."""
    P = np.asarray(pts, np.float32)
    z = np.zeros((H, W), np.float32)
    n = 40
    for i in range(n + 1):
        t = i / n
        k = t * (len(P) - 1)
        j = min(int(k), len(P) - 2)
        p = P[j] + (P[j + 1] - P[j]) * (k - j)
        s = sigma(t) if callable(sigma) else sigma
        z = np.maximum(z, amp_fn(t) * np.exp(-0.5 * (((XX - p[0]) / s) ** 2 + ((YY - p[1]) / (s * 0.9)) ** 2)))
    return z


LIGHT = np.array([-0.40, -0.50, 0.77], np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def shade_height(z, wrap=0.35):
    gy, gx = np.gradient(z)
    n = np.stack([-gx, -gy, np.ones_like(z)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    d = n @ LIGHT
    return np.clip((d + wrap) / (1 + wrap), 0, 1), n


SKIN = [(0.00, rgb(96, 60, 26)), (0.30, rgb(140, 98, 44)), (0.52, rgb(186, 142, 66)),
        (0.70, rgb(216, 176, 90)), (0.85, rgb(236, 200, 112)), (1.0, rgb(250, 222, 136))]


def face_height():
    # head: ellipsoid, peak nudged left because she turns toward her right
    u = (XX - 450) / 136
    v = (YY - 372) / 228
    z = 128 * np.sqrt(np.clip(1 - u * u - v * v, 0, 1)) - 0.10 * (XX - 450)
    z += g2(440, 250, 80, 50, 8)                                   # forehead dome
    z += g2(438, 306, 70, 7, 4)                                    # brow ridge
    z -= g2(390, 330, 24, 14, 13) + g2(485, 328, 24, 14, 12)       # eye sockets
    z += g2(391, 332, 12, 8, 7) + g2(484, 330, 12, 8, 6)           # eyeballs under the lids
    z += g2(392, 402, 36, 36, 13) + g2(502, 396, 30, 34, 9)        # full cheeks
    # nose: soft bridge that swells toward a broad tip
    z += ridge_line([(438, 322), (436, 352), (433, 380), (431, 396)],
                    lambda t: 8 + 4 * t, lambda t: 2.5 + 13 * t ** 1.6)
    z += g2(431, 404, 12, 9, 9)                                    # tip bulb
    z -= g2(437, 327, 9, 8, 4)                                     # nasion
    z += g2(413, 412, 8, 7, 6) + g2(449, 410, 8, 7, 6)             # alae
    z -= ridge_line([(410, 422), (398, 434), (390, 444)], 5, lambda t: 2.0)   # smile folds
    z -= ridge_line([(452, 420), (464, 432), (472, 440)], 5, lambda t: 1.6)
    z += g2(430, 441, 30, 5, 1.5) + g2(432, 454, 22, 6, 3)         # lips, barely
    z -= g2(432, 466, 18, 5, 3)                                    # under lower lip
    z += g2(440, 490, 26, 20, 9)                                   # chin
    return gaussian_filter(z, 1.2)


def paint_face(img, face_m, neck_m):
    F, N = face_m, neck_m

    # ---------- neck + chest (behind the face), glazed ----------
    lay(img, N, ramp(np.clip((XX - 350) / 280, 0, 1), [(0, rgb(186, 146, 70)), (0.55, rgb(160, 120, 54)), (1, rgb(116, 80, 36))]))
    lay(img, fill_spline([(396, 488), (440, 500), (500, 494), (548, 490), (566, 540), (520, 556), (460, 552), (410, 540)], blur=10) * N,
        rgb(82, 52, 26), 0.92)                                                         # deep shadow under the chin
    lay(img, ellipse(474, 572, 44, 26, blur=14) * N, rgb(176, 136, 66), 0.6)          # neck catches light below it
    lay(img, ellipse(424, 566, 26, 46, blur=12) * N, rgb(90, 60, 30), 0.7)            # hair shadow on the left of the neck
    lay(img, ellipse(372, 640, 20, 60, blur=14) * N, rgb(130, 92, 40), 0.5)           # hair shadow, left
    lay(img, ellipse(606, 660, 22, 60, blur=14) * N, rgb(120, 84, 36), 0.55)          # hair shadow, right
    lay(img, ellipse(420, 596, 34, 26, blur=16) * N, rgb(186, 146, 72), 0.45)         # lit front of the neck
    lay(img, ellipse(566, 590, 34, 70, blur=20) * N, rgb(110, 76, 34), 0.6)           # neck turning away
    lay(img, ellipse(468, 712, 150, 76, blur=34) * N, rgb(238, 202, 112), 0.95)       # lit breast
    lay(img, ellipse(452, 698, 76, 38, blur=24) * N, rgb(252, 222, 138), 0.55)
    lay(img, ellipse(488, 650, 70, 40, blur=24) * N, rgb(250, 218, 132), 0.5)        # upper chest catches the light
    lay(img, ellipse(460, 690, 150, 70, blur=30) * N, rgb(250, 222, 140), 0.45)      # the whole breast is luminous
    img += (N * np.clip((YY - 600) / 180, 0, 1))[..., None] * np.array([0.0, 0.02, 0.05], np.float32)   # screen light
    lay(img, ellipse(552, 585, 26, 56, 0.25, blur=14) * N, rgb(104, 70, 32), 0.55)    # right side of the neck
    lay(img, sm([(500, 520), (522, 560), (548, 604)], 10, 7) * N, rgb(120, 82, 38), 0.4)  # the neck's muscle, shaded
    lay(img, ellipse(468, 578, 30, 28, blur=14) * N, rgb(186, 146, 72), 0.2)         # front of the neck
    lay(img, fill_spline([(420, 540), (560, 540), (580, 600), (470, 614), (410, 600)], blur=14) * N, rgb(150, 110, 52), 0.3)
    lay(img, ellipse(598, 690, 40, 50, blur=20) * N, rgb(170, 128, 58), 0.45)
    lay(img, ellipse(338, 700, 26, 40, blur=16) * N, rgb(176, 134, 60), 0.4)
    lay(img, sm([(398, 632), (428, 640), (452, 634)], 7, 6) * N, rgb(212, 172, 88), 0.35)   # collarbones
    lay(img, sm([(488, 632), (522, 638), (560, 628)], 7, 6) * N, rgb(206, 166, 82), 0.3)
    lay(img, ellipse(470, 626, 12, 9, blur=7) * N, rgb(170, 128, 58), 0.35)           # throat pit

    # ---------- face ----------
    d, fn = shade_height(face_height())
    t = d * 0.93
    lay(img, F, ramp(t, SKIN))
    L_scr = np.array([-0.15, 0.75, 0.64], np.float32); L_scr /= np.linalg.norm(L_scr)
    up = np.clip(fn @ L_scr, 0, 1) ** 2
    img += (up * F)[..., None] * np.array([0.02, 0.045, 0.075], np.float32)
    # occlusion where hair meets skin: the face sinks into shadow at its edges
    edge = (1 - gaussian_filter(F, 7)) * np.clip((np.abs(XX - 452) - 50) / 40, 0, 1) * np.clip((500 - YY) / 40, 0, 1)
    lay(img, np.clip(edge * 2.2, 0, 1) * F, rgb(120, 80, 36), 0.55)
    lay(img, ellipse(442, 332, 8, 15, blur=5) * F, rgb(118, 78, 36), 0.6)            # shadowed root of the nose
    lay(img, ellipse(468, 392, 18, 30, blur=12) * F, rgb(228, 190, 106), 0.55)        # cheek beside the nose, lit
    lay(img, ellipse(360, 270, 16, 60, blur=14) * F, rgb(150, 108, 50), 0.45)         # veil shadow, left forehead
    lay(img, ellipse(530, 470, 26, 40, blur=16) * F, rgb(96, 62, 30), 0.5)            # lower right turns into shadow
    lay(img, ellipse(530, 250, 22, 50, blur=16) * F, rgb(130, 90, 42), 0.45)          # right temple
    lay(img, ellipse(430, 434, 26, 7, blur=4) * F, rgb(160, 114, 54), 0.4)           # upper lip in half-shadow
    # warm flush on cheeks and around the eyes (half-tone colour)
    lay(img, g2(392, 395, 26, 26, 1) * F, rgb(222, 164, 94), 0.18)
    lay(img, g2(500, 395, 22, 26, 1) * F, rgb(196, 138, 72), 0.15)

    # ---------- eyes ----------
    def eye(outer, inner, top, bot, iris_c, flip, lit, r_iris=8.0, outer_shade=0.0):
        """Measured against the reference: dark crease, bright lid skin, dark lid line, bright sclera."""
        (ox, oy), (ix, iy) = outer, inner
        cx = top[0]
        crease = [(ox + 2 * flip, oy - 9), (cx - 1 * flip, top[1] - 9.5), (ix + 1 * flip, iy - 10)]
        lay(img, sm(crease, 4.2, 2.0) * F, rgb(98, 62, 30), 0.7)
        lid_skin = [(ox + 5 * flip, oy - 5), (cx, top[1] - 5), (ix - 3 * flip, iy - 5.5)]
        lay(img, sm(lid_skin, 4.0, 2.0) * F, rgb(230, 192, 108), 0.3)
        lid_up = [outer, (ox + (cx - ox) * 0.45, top[1] + 1.0), top, (ix + (cx - ix) * 0.45, top[1] + 1.4), inner]
        lid_lo = [inner, ((ix + bot[0]) / 2, bot[1] - 0.5), bot, ((ox + bot[0]) / 2, bot[1] - 1.0), outer]
        opening = fill_spline(lid_up + lid_lo[1:-1], blur=0.8, t=0.5)
        scl = ramp(np.clip((XX - ix) / (ox - ix), 0, 1), [(0, rgb(168, 132, 86)), (0.5, rgb(192, 156, 106)),
                                                           (1, rgb(178, 142, 94) * (1 - outer_shade))])
        lay(img, opening, scl * lit, 0.95)
        lay(img, ellipse(iris_c[0], iris_c[1], r_iris, r_iris * 1.03, blur=0.8) * opening, rgb(96, 58, 30), 0.98)
        lay(img, ellipse(iris_c[0] + 0.5, iris_c[1] + 2.5, r_iris * 0.6, r_iris * 0.45, blur=1.6) * opening, rgb(132, 86, 44), 0.45)
        ring = np.clip(ellipse(iris_c[0], iris_c[1], r_iris, r_iris, blur=0.7) - ellipse(iris_c[0], iris_c[1], r_iris - 1.8, r_iris - 1.8, blur=0.7), 0, 1)
        lay(img, ring * opening, rgb(52, 30, 16), 0.6)
        lay(img, ellipse(iris_c[0], iris_c[1] - 0.3, 3.2, 3.3, blur=0.8) * opening, rgb(26, 14, 8), 0.92)
        lay(img, ellipse(iris_c[0] - 2.6 * flip, iris_c[1] - 2.4, 1.2, 1.1, blur=0.8) * opening, rgb(170, 140, 104), 0.2)
        lay(img, opening * np.clip(1 - (YY - top[1]) / 3.5, 0, 1), rgb(60, 36, 18), 0.7)     # lid's shadow on the eye
        lay(img, sm([(p[0], p[1] - 1.0) for p in lid_up[:4]], 4.2, 1.2) * F, rgb(52, 32, 16), 0.9)
        lay(img, sm([(p[0], p[1] - 0.6) for p in lid_up[2:]], 2.6, 1.1) * F, rgb(66, 40, 20), 0.8)    # thinner toward the nose
        lay(img, sm([(p[0], p[1] - 3.5) for p in lid_up[1:4]], 5, 2.5) * F, rgb(120, 80, 38), 0.4)     # the lid's heavy fold
        lay(img, sm([(p[0], p[1] + 1.6) for p in lid_lo[1:-1]], 2.0, 1.2) * F, rgb(222, 184, 102), 0.3)    # lower rim lit
        lay(img, sm(lid_lo, 1.2, 0.8) * F, rgb(120, 80, 42), 0.45)
        lay(img, sm([(p[0], p[1] + 6) for p in lid_lo[1:-1]], 2.4, 1.8) * F, rgb(160, 112, 54), 0.3)       # under-eye fold
        lay(img, ellipse(ix + 1.8 * flip, iy - 0.5, 2.6, 2.0, blur=1.0) * F, rgb(150, 90, 60), 0.55)       # caruncle
        lay(img, sm([(ox + 5 * flip, oy - 1), (ox - 3 * flip, oy + 1)], 2.6, 1.0) * F, rgb(60, 36, 18), 0.6)  # outer lash corner
        if outer_shade:
            lay(img, ellipse(ox - 6 * flip, oy - 2, 9, 7, blur=3) * F, rgb(80, 50, 26), 0.5 * outer_shade)

    # ---------- nose ----------
    lay(img, sm([(413, 340), (407, 370), (403, 396), (406, 410)], 3.2, 2.6) * F, rgb(150, 104, 50), 0.45)   # left contour
    lay(img, sm([(441, 334), (443, 364), (447, 394)], 16, 6) * F, rgb(150, 104, 48), 0.5)               # right plane
    lay(img, sm([(411, 403), (408, 411), (413, 419)], 3, 2.6) * F, rgb(126, 84, 40), 0.5)                # left ala
    lay(img, sm([(452, 402), (456, 410), (451, 418)], 3.2, 2.6) * F, rgb(112, 72, 34), 0.55)             # right ala
    lay(img, ellipse(432, 418, 13, 4.5, blur=3) * F, rgb(110, 70, 34), 0.45)                                # under the tip
    lay(img, ellipse(419, 419, 4.4, 2.6, 0.25, blur=1.1) * F, rgb(64, 36, 18), 0.75)                           # nostrils
    lay(img, ellipse(445, 418, 4.4, 2.4, -0.25, blur=1.1) * F, rgb(64, 36, 18), 0.75)
    lay(img, ellipse(428, 399, 6, 5, blur=3) * F, rgb(248, 216, 132), 0.45)                                # tip light
    lay(img, sm([(427, 338), (425, 366), (424, 390)], 5, 3.5) * F, rgb(236, 200, 116), 0.35)               # bridge light

    # ---------- mouth ----------
    lay(img, ellipse(430, 440, 30, 4.5, blur=3.2) * F, rgb(158, 104, 58), 0.4)       # upper lip
    left = [(390, 440), (399, 445.5), (414, 448), (430, 448.5), (444, 447.5)]
    right = [(444, 447.5), (456, 445.8), (464, 443.8), (470, 441.5)]
    lay(img, sm(left[:2], 3.2, 1.4) * F, rgb(72, 40, 22), 0.85)
    lay(img, sm(left[1:], 2.6, 1.6) * F, rgb(90, 54, 28), 0.6)
    lay(img, sm(right, 2.4, 1.8) * F, rgb(96, 60, 32), 0.55)
    lay(img, sm(left, 7, 3.4) * F, rgb(122, 78, 42), 0.35)
    lay(img, ellipse(386, 436, 5, 9, 0.5, blur=3.2) * F, rgb(140, 96, 46), 0.35)      # the dimple, our left
    lay(img, ellipse(474, 436, 5, 8, -0.5, blur=3.4) * F, rgb(132, 88, 42), 0.35)
    lay(img, ellipse(436, 447, 10, 2.6, blur=2.0) * F, rgb(70, 40, 22), 0.45)         # parting of the lips
    lay(img, ellipse(434, 454, 18, 4.5, blur=3) * F, rgb(206, 156, 88), 0.35)         # lower lip catches light
    lay(img, fill_spline([(396, 441), (412, 437), (430, 438), (446, 437), (464, 440), (446, 445), (430, 446), (412, 445)], blur=2.2) * F,
        rgb(158, 102, 58), 0.28)                                                       # upper lip body
    lay(img, fill_spline([(404, 449), (420, 456), (438, 458), (456, 454), (464, 448), (440, 450), (420, 450)], blur=2.5) * F,
        rgb(186, 128, 72), 0.4)                                                        # lower lip body, faintly rosy
    lay(img, ellipse(434, 463, 18, 4, blur=3) * F, rgb(118, 78, 38), 0.6)            # shadow beneath the lower lip
    # ---------- value corrections measured against the reference (heat map) ----------
    lay(img, ellipse(388, 344, 30, 8, blur=6) * F, rgb(162, 116, 56), 0.5)            # under the left eye
    lay(img, ellipse(360, 330, 10, 12, blur=6) * F, rgb(140, 98, 46), 0.45)           # left eye outer corner
    lay(img, ellipse(488, 342, 30, 8, blur=6) * F, rgb(158, 112, 54), 0.45)           # under the right eye
    lay(img, ellipse(516, 328, 16, 18, blur=7) * F, rgb(96, 62, 30), 0.6)             # right temple beside the eye
    lay(img, ellipse(430, 449, 34, 9, blur=5) * F, rgb(168, 116, 60), 0.3)            # lips region
    lay(img, ellipse(482, 488, 38, 16, 0.3, blur=10) * F, rgb(128, 86, 40), 0.45)     # chin, right side
    lay(img, ellipse(452, 500, 40, 6, blur=5) * F, rgb(130, 88, 40), 0.4)             # underside of the chin
    lay(img, ellipse(492, 392, 24, 40, blur=16) * F, rgb(220, 180, 98), 0.45)         # right cheek, lit
    lay(img, ellipse(515, 222, 30, 18, blur=10) * F, rgb(140, 98, 46), 0.5)           # veil shadow, top right
    lay(img, ellipse(380, 450, 16, 20, blur=10) * F, rgb(206, 164, 86), 0.3)          # left lower cheek lighter
    lay(img, ellipse(434, 432, 30, 9, blur=6) * F, rgb(150, 100, 50), 0.45)           # upper lip in the nose's shadow
    lay(img, ellipse(442, 456, 26, 5, blur=4) * F, rgb(160, 110, 54), 0.35)           # lower lip, less lit
    lay(img, ellipse(438, 318, 12, 16, blur=7) * F, rgb(170, 122, 58), 0.45)          # top of the nose between the eyes
    lay(img, ellipse(480, 340, 22, 8, blur=6) * F, rgb(150, 104, 50), 0.35)           # right eye, lower lid
    lay(img, ellipse(512, 470, 28, 36, blur=12) * F, rgb(118, 78, 36), 0.4)           # lower right of the face
    lay(img, ellipse(510, 222, 34, 18, blur=10) * F, rgb(120, 82, 40), 0.45)          # top right, under the veil
    lay(img, ellipse(520, 270, 26, 50, blur=14) * F, rgb(150, 106, 50), 0.4)          # right side of the forehead turns
    lay(img, ellipse(392, 486, 16, 16, blur=9) * F, rgb(150, 104, 50), 0.35)          # left jaw, narrowing the chin
    lay(img, ellipse(372, 446, 18, 22, blur=10) * F, rgb(206, 164, 86), 0.35)         # left lower cheek lighter
    lay(img, ellipse(500, 385, 20, 34, blur=14) * F, rgb(214, 172, 90), 0.3)          # right cheek
    # ---------- Leonardo's halftones ----------
    lay(img, sm([(410, 418), (400, 430), (392, 441)], 7, 4.5) * F, rgb(150, 104, 50), 0.38)   # nasolabial fold, left
    lay(img, sm([(404, 420), (392, 432), (384, 443)], 6, 5) * F, rgb(226, 186, 104), 0.25)    # its lit cheek side
    lay(img, sm([(453, 418), (463, 430), (470, 440)], 6, 4.5) * F, rgb(150, 104, 50), 0.28)   # right fold
    lay(img, ellipse(376, 418, 16, 22, 0.3, blur=12) * F, rgb(176, 132, 64), 0.3)            # under the left cheekbone
    lay(img, ellipse(428, 486, 16, 12, blur=8) * F, rgb(228, 188, 104), 0.4)                  # the round of the chin, lit
    lay(img, ellipse(470, 486, 16, 14, blur=9) * F, rgb(140, 96, 46), 0.35)                   # chin turning away
    lay(img, sm([(366, 327), (388, 316.5), (412, 328)], 4.2, 2.6) * F, rgb(104, 66, 32), 0.45)  # heavier crease, left
    lay(img, sm([(454, 327), (484, 315.5), (514, 322)], 4.6, 2.6) * F, rgb(96, 60, 30), 0.55)   # heavier crease, right
    lay(img, sm([(372, 348), (390, 350.5), (408, 346)], 3.5, 2.5) * F, rgb(226, 186, 104), 0.35)  # puffy lower lid, left
    lay(img, sm([(466, 346), (486, 349), (504, 344)], 3.5, 2.5) * F, rgb(214, 172, 92), 0.3)     # puffy lower lid, right
    # ---------- accents that survive the sfumato ----------
    lay(img, sm([(440, 342), (443, 368), (447, 392), (451, 404)], 9, 3.5) * F, rgb(140, 96, 44), 0.45)   # nose shadow plane edge
    lay(img, ellipse(429, 401, 7, 5.5, blur=2.4) * F, rgb(248, 216, 132), 0.55)                          # rounded tip catches light
    lay(img, fill_spline([(413, 414), (424, 410), (438, 410), (450, 413), (446, 421), (432, 423), (418, 421)], blur=2.2) * F,
        rgb(96, 60, 30), 0.4)                                                                           # underside of the nose
    lay(img, sm([(412, 404), (408, 411), (413, 419), (422, 421)], 1.8, 0.8) * F, rgb(96, 60, 30), 0.55)  # crisp wing line
    lay(img, sm([(409, 402), (405, 410), (409, 418), (417, 421)], 2.4, 1.6) * F, rgb(110, 70, 34), 0.55)  # left wing
    lay(img, sm([(398, 445), (410, 447.5)], 1.8, 0.8) * F, rgb(72, 40, 20), 0.5)                         # mouth corner, left
    lay(img, sm([(416, 448), (432, 448.5), (446, 447.5)], 1.6, 1.1) * F, rgb(90, 54, 28), 0.35)          # the centre, soft
    lay(img, fill_spline([(486, 440), (530, 440), (528, 470), (506, 494), (478, 506), (470, 470)], blur=12) * F,
        rgb(122, 82, 38), 0.4)                                                              # lower right face in shadow
    lay(img, fill_spline([(362, 430), (384, 436), (398, 470), (412, 496), (392, 494), (370, 468)], blur=8) * F,
        rgb(140, 96, 46), 0.35)                                                             # left jaw recedes
    lay(img, ellipse(503, 331, 10, 7, blur=4) * F, rgb(70, 42, 20), 0.35)                   # far eye's outer corner sinks
    # ---------- third heat-map round ----------
    lay(img, fill_spline([(430, 300), (452, 300), (456, 360), (458, 404), (440, 408), (432, 360)], blur=8) * F,
        rgb(160, 112, 52), 0.35)                                                            # nose bridge, shadow side
    lay(img, ellipse(515, 218, 36, 22, blur=10) * F, rgb(110, 74, 36), 0.4)                # top right under the veil
    lay(img, ellipse(492, 352, 28, 8, blur=6) * F, rgb(160, 112, 52), 0.35)                # under the right eye
    lay(img, ellipse(436, 436, 30, 12, blur=6) * F, rgb(150, 100, 50), 0.3)                # upper lip, mouth
    lay(img, ellipse(446, 494, 28, 8, blur=6) * F, rgb(140, 96, 44), 0.35)                 # under the chin's round
    lay(img, ellipse(384, 404, 30, 52, blur=18) * F, rgb(232, 192, 108), 0.35)             # left cheek, luminous
    lay(img, ellipse(500, 420, 28, 40, blur=16) * F, rgb(206, 164, 86), 0.35)              # right cheek halftone
    # amplify the modelling already present: local contrast at the scale of the features
    base = np.stack([gaussian_filter(img[..., i], 14) for i in range(3)], -1)
    Fi = gaussian_filter(F, 3)[..., None] * F[..., None]
    img[:] = img + 0.5 * (img - base) * Fi
    lay(img, ellipse(530, 400, 22, 84, blur=14) * F, rgb(118, 78, 36), 0.35)          # the far cheek turns into shadow
    lay(img, ellipse(434, 480, 22, 14, blur=8) * F, rgb(226, 186, 102), 0.35)          # the lit round of the chin
    lay(img, fill_spline([(410, 490), (440, 500), (480, 496), (512, 482), (500, 500), (470, 510), (436, 510), (412, 502)], blur=6) * F,
        rgb(110, 72, 34), 0.45)                                                          # under the chin
    img *= (1 - 0.05 * F)[..., None]
    lay(img, ellipse(388, 322, 36, 20, blur=10) * F, rgb(156, 108, 52), 0.3)               # the whole left socket
    lay(img, ellipse(486, 320, 36, 20, blur=10) * F, rgb(146, 100, 48), 0.35)              # the whole right socket
    lay(img, sm([(446, 336), (449, 362), (453, 390), (456, 406)], 11, 4.5) * F, rgb(142, 98, 46), 0.4)  # nose, shadow plane
    lay(img, ellipse(388, 312, 30, 8, blur=5) * F, rgb(150, 104, 50), 0.4)                 # socket above the left eye
    lay(img, ellipse(446, 318, 14, 16, blur=6) * F, rgb(140, 96, 44), 0.4)                 # root of the nose, right
    lay(img, ellipse(516, 326, 12, 14, blur=6) * F, rgb(96, 62, 30), 0.45)                 # beyond the right eye
    lay(img, ellipse(386, 345, 20, 5, blur=4) * F, rgb(170, 124, 60), 0.35)                # under the left eye
    lay(img, ellipse(496, 350, 26, 8, blur=6) * F, rgb(206, 166, 88), 0.35)                # under the right eye, lit
    eye(outer=(364, 334), inner=(411, 336), top=(389, 327.5), bot=(388, 339.5), iris_c=(389, 332.5), flip=1, lit=1.0, r_iris=8.2)
    eye(outer=(509, 329), inner=(457, 334), top=(483, 325), bot=(484, 337.5), iris_c=(483.5, 330.5), flip=-1, lit=0.95,
        r_iris=7.9, outer_shade=0.6)
    return np.clip(F + N, 0, 1)


# ======================================================================
# dress
# ======================================================================
NECK_BAND = [(300, 728), (336, 752), (384, 772), (440, 786), (498, 790), (556, 780), (604, 760), (640, 734), (660, 704)]


def paint_dress(img, body_pts):
    body = fill_spline(body_pts, blur=1.4, t=0.4)
    # deep olive-black bodice, warmer brown-red toward the right mantle, darkest at the edges
    base = ramp(np.clip((XX - 120) / 800, 0, 1), [(0, rgb(20, 16, 17)), (0.35, rgb(32, 26, 21)),
                                                 (0.7, rgb(27, 19, 20)), (1, rgb(38, 21, 20))])
    base *= (0.85 + 0.3 * fbm(40, 4, 501))[..., None]
    lay(img, body, base)
    # light falls on the upper chest area of the bodice, from the left
    lay(img, ellipse(420, 860, 170, 110, blur=60) * body, rgb(62, 52, 36), 0.6)
    lay(img, ellipse(850, 880, 100, 160, 0.3, blur=50) * body, rgb(84, 50, 44), 0.55)   # reddish mantle, right

    r = np.random.default_rng(7)
    band = resample(NECK_BAND, 200)

    # gathered pleats fanning down from the neckline
    def pleats(c):
        for i in range(200):
            k = r.random()
            x0, y0 = band[int(k * 170)]
            if x0 > 560:
                continue
            spread = (x0 - 450) / 300
            ln = 60 + r.random() * 160
            bow = r.normal(0, 10)
            pts = [(x0, y0 + 14)]
            for j in range(1, 7):
                t = j / 6
                pts.append((x0 + spread * 90 * t + bow * np.sin(t * np.pi), y0 + 14 + ln * t))
            v = 0.5 + 0.9 * r.random() ** 2
            c.set_source_rgba(0.62 * v, 0.5 * v, 0.26 * v, 0.06 + 0.2 * r.random())
            c.set_line_width(1.2 + 2.4 * r.random())
            spline(c, pts, closed=False)
            c.stroke()
    col, al = rgba_layer(pleats)
    comp(img, col, al * body * np.clip(1 - (YY - 800) / 160, 0, 1), blur=1.6)
    # dark troughs just under the band
    lay(img, stroke_spline([(x, y + 20) for x, y in NECK_BAND[:7]], 10, blur=6), rgb(20, 16, 14), 0.4)

    # gauze drape sweeping from her left shoulder down across the bodice (fine strands)
    def gauze(c):
        for i in range(90):
            a = r.random()
            x0 = 600 + a * 150 + r.normal(0, 8)
            y0 = 700 + a * 70 + r.normal(0, 6)
            x3 = 300 + a * 260 + r.normal(0, 20)
            y3 = 1010 + a * 60 + r.normal(0, 20)
            x1, y1 = x0 - 40, y0 + 90
            x2, y2 = x3 + 90, y3 - 120
            v = 0.55 + 0.6 * r.random()
            c.set_source_rgba(0.56 * v, 0.46 * v, 0.24 * v, 0.03 + 0.07 * r.random())
            c.set_line_width(0.7 + 1.4 * r.random())
            c.move_to(x0, y0)
            c.curve_to(x1, y1, x2, y2, x3, y3)
            c.stroke()
    col, al = rgba_layer(gauze)
    comp(img, col, al * body, blur=1.4)
    drape = fill_spline([(560, 700), (700, 690), (790, 760), (700, 860), (600, 950), (480, 1040), (380, 1050),
                         (470, 930), (560, 820)], blur=25)
    lay(img, drape * body, rgb(70, 58, 36), 0.25)

    return body


def paint_shawl(img, body_pts):
    body = fill_spline(body_pts, blur=1.4, t=0.4)
    r = np.random.default_rng(71)
    # the shawl over her left shoulder: a broad band of lit fabric running from the hair over the shoulder
    band_c = [(632, 700), (680, 690), (730, 698), (780, 722), (830, 760), (874, 812), (906, 868)]
    band_m = stroke_spline(band_c, 64, blur=10) * body * np.clip((XX - 630) / 90, 0, 1)
    lay(img, band_m, rgb(84, 62, 38), 0.85)
    along = np.clip((XX - 620) / 300, 0, 1)
    lit = ramp(along, [(0, rgb(172, 136, 76)), (0.45, rgb(144, 110, 62)), (1, rgb(84, 60, 36))])
    top_edge = stroke_spline([(p[0] - 6, p[1] - 14) for p in band_c], 26, blur=10) * body
    lay(img, top_edge * np.clip((XX - 640) / 90, 0, 1), lit, 0.75)
    def band_strands(c):
        for i in range(160):
            off = r.normal(0, 18)
            v = 0.5 + 0.9 * r.random() ** 2
            c.set_source_rgba(0.74 * v, 0.58 * v, 0.32 * v, 0.05 + 0.2 * r.random())
            c.set_line_width(0.8 + 2.2 * r.random())
            P = np.array(band_c, float)
            nrm = np.array([0.35, -0.94])
            k0, k1 = r.random() * 0.4, 0.6 + r.random() * 0.4
            pts = P[int(k0 * 6):int(k1 * 6) + 1] + nrm * off
            if len(pts) < 2:
                continue
            pts = pts + nrm * r.normal(0, 4, (len(pts), 1))
            spline(c, [tuple(p) for p in pts], closed=False)
            c.stroke()
    col, al = rgba_layer(band_strands)
    comp(img, col, al * body * np.clip((XX - 640) / 90, 0, 1), blur=1.2)
    lay(img, stroke_spline([(p[0] + 8, p[1] + 26) for p in band_c], 16, blur=8) * body, rgb(28, 20, 16), 0.5)  # shadow under the band

    # its end hangs down in a soft fold in front of the shoulder
    fold = [(706, 900), (700, 870), (708, 842), (726, 820), (752, 806), (780, 800), (796, 808), (784, 826),
            (764, 842), (746, 862), (732, 884), (724, 904)]
    fold = [(700, 904), (694, 870), (700, 838), (720, 812), (750, 796), (784, 790), (806, 800), (800, 830),
            (780, 856), (760, 884), (740, 910)]
    fm = fill_spline(fold, blur=6) * 0.75
    shade_f = ramp(np.clip((XX - 690) / 140 + (YY - 790) / 160, 0, 1), [(0, rgb(150, 118, 66)), (0.5, rgb(108, 82, 48)), (1, rgb(62, 46, 30))])
    lay(img, fm, shade_f)
    lay(img, ellipse(740, 830, 34, 22, -0.6, blur=10), rgb(170, 136, 80), 0.45)                  # the lit fold
    lay(img, stroke_spline([(718, 896), (728, 868), (744, 846), (770, 828)], 3, blur=2.2), rgb(62, 44, 28), 0.4)
    lay(img, stroke_spline([(740, 900), (752, 874), (772, 850)], 2.4, blur=2.2), rgb(70, 50, 30), 0.35)
    for pts in ([(752, 760), (770, 784), (782, 800)], [(726, 762), (740, 786), (752, 806)]):
        lay(img, stroke_spline(pts, 10, blur=5), rgb(104, 80, 48), 0.35)
    lay(img, fill_spline([(724, 902), (744, 900), (764, 912), (738, 924)], blur=6), rgb(24, 16, 14), 0.45)


def paint_embroidery(img):
    """Knotwork border: a braided chain along the neckline with hanging knot motifs beneath."""
    band = resample(NECK_BAND, 400)
    tang = np.gradient(band, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], -1)          # points down/outward from the skin
    fade = np.clip((560 - band[:, 0]) / 60, 0, 1)
    lay(img, stroke_spline([tuple(p + nrm[i] * 7) for i, p in enumerate(band[::20])], 18, blur=1.2), rgb(74, 52, 28), 0.9)

    def chain(c):
        # braided chain: two interlaced sinusoids
        for ph in (0, np.pi):
            pts = [band[i] + nrm[i] * (4 + 3.2 * np.sin(i * 0.9 + ph)) for i in range(0, 330)]
            c.set_line_width(1.3)
            for j in range(0, len(pts) - 1, 2):
                c.set_source_rgba(0.78, 0.6, 0.3, 0.8 * fade[j])
                c.move_to(*pts[j]); c.line_to(*pts[j + 1]); c.stroke()
        # hanging knot motifs: little looped figure-eights
        for i in range(6, 324, 9):
            p = band[i] + nrm[i] * 10
            ang = np.arctan2(nrm[i][1], nrm[i][0])
            c.save()
            c.translate(*p); c.rotate(ang - np.pi / 2)
            c.set_source_rgba(0.7, 0.52, 0.26, 0.5 * fade[i])
            c.set_line_width(1.0)
            c.move_to(-3, 0)
            c.curve_to(-5, 4, 5, 4, 3, 0)
            c.curve_to(5, 5, -2, 8, 0, 8)
            c.curve_to(2, 8, -5, 5, -3, 0)
            c.stroke()
            c.restore()
    col, al = rgba_layer(chain)
    comp(img, col, al, blur=0.45)
    lay(img, stroke_spline([tuple(p - nrm[i] * 1) for i, p in enumerate(band[:330:10])], 1.4, blur=0.7), rgb(150, 112, 56), 0.5)


def paint_lap(img):
    """Her lap under the laptop: heavy dark cloth with a few broad, soft folds."""
    base = fill_poly([(0, 1060), (W, 1060), (W, H), (0, H)], blur=20)
    lay(img, base * np.clip((YY - 1040) / 80, 0, 1), rgb(30, 22, 22), 0.6)
    r = np.random.default_rng(17)
    folds = [
        ([(20, 1140), (90, 1220), (150, 1330), (190, 1431)], 26, rgb(66, 52, 40), 0.45),
        ([(60, 1110), (140, 1180), (210, 1280), (260, 1431)], 16, rgb(58, 46, 36), 0.35),
        ([(10, 1260), (60, 1330), (90, 1431)], 18, rgb(20, 14, 16), 0.5),
        ([(940, 1120), (880, 1220), (840, 1320), (820, 1431)], 26, rgb(72, 46, 40), 0.4),
        ([(900, 1100), (830, 1180), (770, 1300), (740, 1431)], 14, rgb(60, 40, 36), 0.35),
        ([(960, 1260), (920, 1340), (900, 1431)], 20, rgb(22, 14, 16), 0.5),
        ([(300, 1400), (480, 1420), (660, 1400)], 30, rgb(52, 40, 34), 0.35),
    ]
    for pts, w, colr, a in folds:
        lay(img, stroke_spline(pts, w, blur=w * 0.7), colr, a)


# ======================================================================
# laptop
# ======================================================================
PITCH, CAP = 0.0185, 0.0152
# rows from the hinge (function row) toward her (space row), legends from HER left to right
ROWS = [
    (0.0120, 0.0080, [("esc", 1.0)] + [(f"F{i}", 1.0) for i in range(1, 13)] + [("", 1.0)]),
    (0.0290, CAP, [("`", 1)] + [(c, 1) for c in "1234567890-="] + [("del", 1.5)]),
    (0.0475, CAP, [("tab", 1.5)] + [(c, 1) for c in "QWERTYUIOP[]\\"]),
    (0.0660, CAP, [("caps", 1.75)] + [(c, 1) for c in "ASDFGHJKL;'"] + [("return", 1.75)]),
    (0.0845, CAP, [("shift", 2.25)] + [(c, 1) for c in "ZXCVBNM,./"] + [("shift", 2.25)]),
    (0.1030, CAP, [("fn", 1), ("ctrl", 1), ("opt", 1), ("cmd", 1.25), ("", 5.0), ("cmd", 1.25), ("opt", 1),
                   ("<", 1), ("^", 1), (">", 1)]),
]
KB_W = 14.5 * PITCH


def key_centres():
    """{legend: (u, v)} in laptop metres, u from the viewer's frame (her left = +u)."""
    out = {}
    for v, hgt, keys in ROWS:
        x = 0.0
        for leg, wu in keys:
            kx = x + wu * PITCH / 2
            u = KB_W / 2 - kx
            out.setdefault(leg, (u, v))
            x += wu * PITCH
    return out


def local_path(c, L, pts_uv, h=0.0):
    P = project(np.array([L.P(u, v, h) for u, v in pts_uv]))
    c.move_to(*P[0])
    for q in P[1:]:
        c.line_to(*q)
    c.close_path()


def rrect_uv(u0, v0, u1, v1, r, n=6):
    pts = []
    for (cu, cv, a0) in [(u1 - r, v0 + r, -90), (u1 - r, v1 - r, 0), (u0 + r, v1 - r, 90), (u0 + r, v0 + r, 180)]:
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cu + r * math.cos(a), cv + r * math.sin(a)))
    return pts


def paint_laptop_base(img, L):
    w, d = L.w, L.d
    # the deck's right side face (turned toward us by the yaw)
    side = mask(lambda c: (c.move_to(*project(L.P(w / 2, 0, 0))), c.line_to(*project(L.P(w / 2, d, 0))),
                           c.line_to(*project(L.P(w / 2, d, -L.tb))), c.line_to(*project(L.P(w / 2, 0, -L.tb))),
                           c.close_path(), c.fill()), blur=0.5)
    lay(img, side, rgb(60, 54, 46))
    # hinge-side face of the base
    front = mask(lambda c: (c.move_to(*project(L.P(-w / 2, 0, 0))), c.line_to(*project(L.P(w / 2, 0, 0))),
                            c.line_to(*project(L.P(w / 2, 0, -L.tb))), c.line_to(*project(L.P(-w / 2, 0, -L.tb))),
                            c.close_path(), c.fill()), blur=0.5)
    lay(img, front, rgb(50, 44, 38))

    # deck top: pewter aluminium, darker toward her (reflects the dark dress)
    deck = mask(lambda c: (local_path(c, L, rrect_uv(-w / 2, 0, w / 2, d, 0.010), 0), c.fill()), blur=0.6)
    far = project(L.P(0, d))[1]
    near = project(L.P(0, 0))[1]
    t = np.clip((YY - far) / (near - far), 0, 1)
    col = ramp(t, [(0, rgb(50, 45, 39)), (0.5, rgb(82, 75, 64)), (1, rgb(104, 96, 82))])
    col *= (0.96 + 0.08 * fbm(3, 2, 301)[..., None])                          # brushed finish
    lx = np.clip((XX - project(L.P(-w / 2, d / 2))[0]) / 520, 0, 1)
    col *= (1.06 - 0.14 * lx)[..., None]                                       # light from the left
    lay(img, deck, col)
    lay(img, stroke_spline([tuple(project(L.P(u, d - 0.001))) for u in np.linspace(-w / 2 + 0.01, w / 2 - 0.01, 8)],
                           1.2, blur=0.6), rgb(150, 138, 112), 0.4)            # far rim catches light

    # keyboard well
    well = mask(lambda c: (local_path(c, L, rrect_uv(-KB_W / 2 - 0.003, 0.0045, KB_W / 2 + 0.003, 0.1135, 0.004), 0),
                           c.fill()), blur=0.5)
    lay(img, well, rgb(40, 36, 32), 0.85)

    # keys: dark caps, a lighter near lip (the side facing us), legends upside down to us
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    cl = cairo.Context(s)
    cl.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    # rasterise all caps / lips in single passes (fast)
    def draw_caps(c):
        for v, hgt, keys in ROWS:
            x = 0.0
            for leg, wu in keys:
                kx0, kx1 = x + 0.0016, x + wu * PITCH - 0.0016
                u0, u1 = KB_W / 2 - kx1, KB_W / 2 - kx0
                local_path(c, L, rrect_uv(u0, v - hgt / 2, u1, v + hgt / 2, 0.0018, 3), 0.0014)
                x += wu * PITCH
        c.fill()
    def draw_lips(c):
        for v, hgt, keys in ROWS:
            x = 0.0
            for leg, wu in keys:
                kx0, kx1 = x + 0.0016, x + wu * PITCH - 0.0016
                u0, u1 = KB_W / 2 - kx1, KB_W / 2 - kx0
                v0 = v - hgt / 2
                c.move_to(*project(L.P(u0, v0, 0.0014)))
                c.line_to(*project(L.P(u1, v0, 0.0014)))
                c.line_to(*project(L.P(u1, v0, 0.0)))
                c.line_to(*project(L.P(u0, v0, 0.0)))
                c.close_path()
                x += wu * PITCH
        c.fill()
    lips = mask(draw_lips, blur=0.3)
    caps = mask(draw_caps, blur=0.35)
    lay(img, lips, rgb(56, 50, 44))
    kc = ramp(np.clip((YY - far) / (near - far), 0, 1), [(0, rgb(20, 17, 16)), (1, rgb(34, 30, 28))])
    lay(img, caps, kc)

    # legends: an affine frame per key, mapping HER reading direction onto the screen
    cl.set_source_rgba(0.72, 0.68, 0.6, 0.75)
    for v, hgt, keys in ROWS:
        x = 0.0
        for leg, wu in keys:
            if leg:
                kx = x + wu * PITCH / 2
                u = KB_W / 2 - kx
                P0 = L.P(u, v, 0.0015)
                p0 = project(P0)
                e = 0.001
                ex = (project(P0 - e * L.du) - p0) / e      # her rightward
                ey = (project(P0 + e * L.dv) - p0) / e      # her 'down' on the page = toward her
                cl.save()
                cl.transform(cairo.Matrix(ex[0], ex[1], ey[0], ey[1], p0[0], p0[1]))
                size = 0.0048 if len(leg) == 1 else 0.0028
                cl.set_font_size(size)
                ext = cl.text_extents(leg)
                cl.move_to(-ext.width / 2 - ext.x_bearing, ext.height / 2)
                cl.show_text(leg)
                cl.restore()
            x += wu * PITCH
    s.flush()
    a = np.ndarray((H, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :W, :].astype(np.float32) / 255
    lay(img, gaussian_filter(a[..., 3], 0.4), rgb(150, 140, 120), 0.55)

    # trackpad
    tp = mask(lambda c: (local_path(c, L, rrect_uv(-0.055, 0.124, 0.055, 0.190, 0.006), 0), c.fill()), blur=0.5)
    lay(img, tp, rgb(84, 78, 68), 0.55)
    lay(img, mask(lambda c: (local_path(c, L, rrect_uv(-0.055, 0.124, 0.055, 0.190, 0.006), 0),
                             c.set_line_width(1.0), c.stroke()), blur=0.4), rgb(70, 64, 56), 0.6)
    return deck


def paint_laptop_lid(img, L):
    w = L.w
    corners = [L.lidP(-w / 2, 0), L.lidP(w / 2, 0), L.lidP(w / 2, L.L), L.lidP(-w / 2, L.L)]
    cam = -np.mean(corners, axis=0)
    sees_screen = np.dot(L.nl, cam) > 0
    # lid thickness: the top edge band
    top_band = [L.lidP(-w / 2, L.L, 0), L.lidP(w / 2, L.L, 0), L.lidP(w / 2, L.L, -L.tl), L.lidP(-w / 2, L.L, -L.tl)]
    body = [L.lidP(-w / 2, 0, -L.tl), L.lidP(w / 2, 0, -L.tl), L.lidP(w / 2, L.L, -L.tl), L.lidP(-w / 2, L.L, -L.tl),
            L.lidP(-w / 2, L.L, 0), L.lidP(w / 2, L.L, 0), L.lidP(w / 2, 0, 0), L.lidP(-w / 2, 0, 0)]
    hull = project(np.array(body))
    from scipy.spatial import ConvexHull
    hh = hull[ConvexHull(hull).vertices]
    lidm = fill_poly([tuple(p) for p in hh], blur=0.5)
    lay(img, lidm, rgb(64, 58, 50))
    tb = fill_poly([tuple(p) for p in project(np.array(top_band))], blur=0.4)
    lay(img, tb, rgb(120, 112, 96), 0.7)
    if sees_screen:
        paint_screen(img, L)
    return lidm, sees_screen


def screen_texture(tw=1200, th=760):
    """What she sees: a document in progress, drawn upright in her frame with cairo."""
    s = cairo.ImageSurface(cairo.FORMAT_RGB24, tw, th)
    c = cairo.Context(s)
    c.set_source_rgb(0.86, 0.88, 0.9); c.paint()
    c.set_source_rgb(0.72, 0.75, 0.8); c.rectangle(0, 0, tw, 46); c.fill()          # title bar
    for i, col in enumerate([(0.85, 0.35, 0.3), (0.9, 0.7, 0.3), (0.4, 0.7, 0.4)]):
        c.set_source_rgb(*col); c.arc(28 + i * 30, 23, 9, 0, 6.283); c.fill()
    c.set_source_rgb(0.97, 0.97, 0.98); c.rectangle(170, 80, tw - 340, th - 80); c.fill()   # the page
    r = np.random.default_rng(5)
    y = 130
    c.set_source_rgb(0.25, 0.27, 0.3)
    c.rectangle(230, y, 420, 26); c.fill()                                             # a heading
    y += 70
    while y < th - 40:
        ln = tw - 460 - (r.random() * 300 if r.random() < 0.3 else r.random() * 60)
        c.set_source_rgb(0.42, 0.44, 0.48)
        c.rectangle(230, y, ln, 12); c.fill()
        y += 30 if r.random() > 0.15 else 60
    c.set_source_rgb(0.2, 0.4, 0.85); c.rectangle(230 + 380, y - 64, 4, 22); c.fill()       # the cursor
    s.flush()
    a = np.ndarray((th, s.get_stride() // 4, 4), np.uint8, s.get_data())[:, :tw, :].astype(np.float32) / 255
    return np.stack([a[..., 2], a[..., 1], a[..., 0]], -1)


def paint_screen(img, L):
    """Ray-cast every pixel of the glass onto the lid plane and sample the document."""
    w = L.w
    bez = 0.008
    scr_all = [L.lidP(-w / 2, 0), L.lidP(w / 2, 0), L.lidP(w / 2, L.L), L.lidP(-w / 2, L.L)]
    lay(img, fill_poly([tuple(p) for p in project(np.array(scr_all))], blur=0.4), rgb(22, 20, 22))
    scr = [L.lidP(-w / 2 + bez, bez), L.lidP(w / 2 - bez, bez), L.lidP(w / 2 - bez, L.L - bez), L.lidP(-w / 2 + bez, L.L - bez)]
    sm_ = fill_poly([tuple(p) for p in project(np.array(scr))], blur=0.5)
    ys, xs = np.nonzero(sm_ > 0.01)
    FOC = F
    d = np.stack([(xs + 0.5 - CX) / FOC, -(ys + 0.5 - CY) / FOC, np.ones(len(xs))], -1)
    n = L.nl
    P0 = L.lidP(0, 0)
    tt = (P0 @ n) / (d @ n)
    Pw = d * tt[:, None]
    rel = Pw - P0
    u = rel @ L.du
    sl = rel @ L.dl
    tex = screen_texture()
    th, tw = tex.shape[:2]
    tx = np.clip(((w / 2 - bez) - u) / (w - 2 * bez) * (tw - 1), 0, tw - 1)       # her left edge = +u
    ty = np.clip(((L.L - bez) - sl) / (L.L - 2 * bez) * (th - 1), 0, th - 1)     # her top edge = far end of the lid
    from scipy.ndimage import map_coordinates
    colr = np.stack([map_coordinates(gaussian_filter(tex[..., i], 1.2), [ty, tx], order=1) for i in range(3)], -1)
    # seen at a grazing angle the glass mostly reflects: dim and tint the image, add a sheen
    view = -Pw / np.linalg.norm(Pw, axis=1, keepdims=True)
    cosv = np.clip(view @ n, 0, 1)
    colr = colr * (0.5 + 0.4 * cosv[:, None]) * np.array([0.78, 0.84, 0.82])
    out = img.copy()
    out[ys, xs] = colr
    k = sm_[..., None]
    img[:] = img * (1 - k) + out * k
    # a faint bloom of cool light around the glass
    glow = gaussian_filter(sm_, 10)
    img += glow[..., None] * np.array([0.035, 0.05, 0.06], np.float32)
    return sm_


# ======================================================================
# hands
# ======================================================================
"""Typing hands + forearms + sleeves as sphere-swept 3D bones, z-buffered, lit, supersampled."""
FOC = F

SS = 3                                   # supersampling factor
BX0, BY0, BX1, BY1 = 0, 620, 960, 1431   # region we render
SW, SH = (BX1 - BX0) * SS, (BY1 - BY0) * SS

MAT_SKIN, MAT_NAIL, MAT_SLEEVE, MAT_CUFF, MAT_PALM = 1, 2, 3, 4, 5


class Scene:
    def __init__(self):
        self.C, self.R, self.mat, self.seg, self.t, self.axis = [], [], [], [], [], []
        self.nseg = 0

    def capsule(self, a, b, ra, rb, mat, spacing=0.1):
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = np.linalg.norm(b - a)
        n = max(2, int(L / (min(ra, rb) * spacing)) + 1)
        sid = self.nseg
        self.nseg += 1
        ax = (b - a) / max(L, 1e-9)
        for i in range(n):
            t = i / (n - 1)
            self.C.append(a + (b - a) * t)
            self.R.append(ra + (rb - ra) * t)
            self.mat.append(mat)
            self.seg.append(sid)
            self.t.append(t)
            self.axis.append(ax)
        return sid

    def sphere(self, c, r, mat):
        sid = self.nseg
        self.nseg += 1
        self.C.append(np.asarray(c, float)); self.R.append(r); self.mat.append(mat)
        self.seg.append(sid); self.t.append(0.5); self.axis.append(np.array([0, 1.0, 0]))
        return sid

    def raster(self):
        C = np.array(self.C); R = np.array(self.R)
        z = np.full((SH, SW), np.inf, np.float32)
        nrm = np.zeros((SH, SW, 3), np.float32)
        pos = np.zeros((SH, SW, 3), np.float32)
        mat = np.zeros((SH, SW), np.int16)
        seg = np.full((SH, SW), -1, np.int32)
        tt = np.zeros((SH, SW), np.float32)
        p = project(C)
        px = (p[:, 0] - BX0) * SS
        py = (p[:, 1] - BY0) * SS
        rp = FOC * R / C[:, 2] * SS
        for i in range(len(C)):
            x0, x1 = int(max(0, px[i] - rp[i] - 1)), int(min(SW, px[i] + rp[i] + 2))
            y0, y1 = int(max(0, py[i] - rp[i] - 1)), int(min(SH, py[i] + rp[i] + 2))
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            dx = (xx + 0.5 - px[i]) / rp[i]
            dy = (yy + 0.5 - py[i]) / rp[i]
            d2 = dx * dx + dy * dy
            ins = d2 < 1
            nz = np.sqrt(np.clip(1 - d2, 0, 1))
            depth = C[i, 2] - R[i] * nz
            zb = z[y0:y1, x0:x1]
            upd = ins & (depth < zb)
            if not upd.any():
                continue
            zb[upd] = depth[upd]
            nb = nrm[y0:y1, x0:x1]
            nb[upd] = np.stack([dx[upd], -dy[upd], -nz[upd]], -1)
            pb = pos[y0:y1, x0:x1]
            pb[upd] = C[i] + R[i] * np.stack([dx[upd], -dy[upd], -nz[upd]], -1)
            mat[y0:y1, x0:x1][upd] = self.mat[i]
            seg[y0:y1, x0:x1][upd] = self.seg[i]
            tt[y0:y1, x0:x1][upd] = self.t[i]
        self.axes = np.array(self.axis)
        segax = np.zeros((self.nseg, 3))
        for s_, a_ in zip(self.seg, self.axis):
            segax[s_] = a_
        return dict(z=z, n=nrm, pos=pos, mat=mat, seg=seg, t=tt, segax=segax)


# ------------------------------------------------------------------ anatomy --
FINGERS = {  # lengths (P1, P2, P3) and radii (base, tip), metres - a slender hand
    "index": ((0.037, 0.021, 0.017), (0.0084, 0.0062)),
    "middle": ((0.041, 0.024, 0.018), (0.0086, 0.0064)),
    "ring": ((0.038, 0.023, 0.017), (0.0082, 0.0061)),
    "pinky": ((0.029, 0.017, 0.015), (0.0070, 0.0053)),
}
SPLAY = {"index": -4.0, "middle": 0.0, "ring": 4.0, "pinky": 9.0}


def finger_chain(mcp, fwd, up, lengths, angles):
    """Joint positions from the MCP, bending down by the given angles (deg from horizontal)."""
    pts = [np.asarray(mcp, float)]
    for L_, a in zip(lengths, angles):
        a = np.radians(a)
        pts.append(pts[-1] + L_ * (np.cos(a) * fwd - np.sin(a) * up))
    return pts


def build_hand(S, L, side, keys, lifted=None, splay=0.0):
    """side=+1 for her left hand (viewer's right), -1 for her right hand.
    keys: {finger: legend}. The resting fingertips land on those keys."""
    K = key_centres()
    fwd = -L.dv                      # toward the hinge (away from her)
    up = L.up
    across = L.du                    # viewer's right
    rest_angles = (14, 60, 86)
    tips, mcps = {}, {}
    for f, leg in keys.items():
        u, v = K[leg]
        lens, (rb, rt) = FINGERS[f]
        tip = L.P(u, v, 0.0014 + rt * 0.85)
        chain = finger_chain(np.zeros(3), fwd, up, lens, rest_angles)
        off = chain[-1]
        mcps[f] = tip - off
        tips[f] = tip
    # palm frame from the knuckles; knuckle heights follow a gentle arch
    order = ["index", "middle", "ring", "pinky"]
    for i, f in enumerate(order):
        mcps[f] = mcps[f] + up * (0.004 * (1 - abs(i - 1.2) / 2.0) - 0.0022 * i) + (-fwd) * (0.004 * i)
    mid = (mcps["middle"] + mcps["ring"]) / 2
    wrist = mid - fwd * 0.078 - up * 0.022
    radial = -side * across          # thumb side points toward the keyboard centre
    # fingers
    for f in order:
        lens, (rb, rt) = FINGERS[f]
        ang = rest_angles
        if lifted and f in lifted:
            ang = lifted[f]
        a = np.radians(SPLAY[f] * side * -1)
        fwd_f = np.cos(a) * fwd + np.sin(a) * across
        pts = finger_chain(mcps[f], fwd_f, up, lens, ang)
        r_pip, r_dip = rb * 0.84, rb * 0.76
        S.sphere(pts[0], rb * 1.06, MAT_SKIN)                               # knuckle, soft
        S.capsule(pts[0], pts[0] + 0.55 * (pts[1] - pts[0]), rb * 1.0, rb * 0.8, MAT_SKIN)
        S.capsule(pts[0] + 0.55 * (pts[1] - pts[0]), pts[1], rb * 0.8, r_pip, MAT_SKIN)
        S.capsule(pts[1], pts[1] + 0.5 * (pts[2] - pts[1]), r_pip, r_dip * 0.95, MAT_SKIN)
        S.capsule(pts[1] + 0.5 * (pts[2] - pts[1]), pts[2], r_dip * 0.95, r_dip, MAT_SKIN)
        tipc = pts[3] - 0.3 * (pts[3] - pts[2])
        S.capsule(pts[2], tipc, r_dip, rt, MAT_NAIL)                        # distal (nail flagged)
        S.sphere(tipc, rt * 0.97, MAT_SKIN)
    # palm: a bilinear sheet of spheres between the wrist line and the knuckle line
    w_rad = wrist + radial * 0.018
    w_uln = wrist - radial * 0.016
    k_rad = mcps["index"] - radial * 0.004
    k_uln = mcps["pinky"] + radial * 0.004
    for a in np.linspace(0, 1, 22):
        for b in np.linspace(0, 1, 24):
            p = (1 - b) * ((1 - a) * w_rad + a * w_uln) + b * ((1 - a) * k_rad + a * k_uln)
            dome = 0.007 * np.sin(np.pi * (0.15 + 0.7 * a)) * np.sin(np.pi * min(1, 0.2 + b))
            rr_ = 0.0165 - 0.0022 * b
            S.sphere(p + up * (dome - 0.002 * b - 0.005 * a - 0.0045), rr_, MAT_PALM)
    # thumb: from the wrist's radial side to the space bar
    sp_u, sp_v = K[""] if "" in K else (0.0, 0.103)
    cmc = wrist + radial * 0.02 + fwd * 0.012 - up * 0.004
    t_tip_uv = (float(np.dot(mcps["index"] - L.H0, L.du)) + side * -0.020, 0.100)
    t_tip = L.P(t_tip_uv[0], t_tip_uv[1], 0.0075)
    t_mcp = cmc + (t_tip - cmc) * 0.45 + up * 0.012 + radial * 0.004
    t_ip = cmc + (t_tip - cmc) * 0.78 + up * 0.007
    S.capsule(cmc, t_mcp, 0.012, 0.0092, MAT_SKIN)
    S.capsule(t_mcp, t_ip, 0.0092, 0.0082, MAT_SKIN)
    S.capsule(t_ip, t_tip, 0.0082, 0.0072, MAT_NAIL)
    S.sphere(t_tip, 0.0072, MAT_SKIN)
    return wrist, radial


def sleeve_tube(S, a0, a1, r0, r1, seed, n=260):
    ax = unit(a1 - a0)
    Ls = np.linalg.norm(a1 - a0)
    r = np.random.default_rng(seed)
    ph = r.random(3) * 6.28
    sid = S.nseg; S.nseg += 1
    for i in range(n):
        t = i / (n - 1)
        bulge = 1 + 0.07 * np.sin(t * Ls / 0.028 * 6.28 + ph[0]) + 0.05 * np.sin(t * Ls / 0.061 * 6.28 + ph[1])
        S.C.append(a0 + (a1 - a0) * t); S.R.append((r0 + (r1 - r0) * t ** 0.8) * bulge)
        S.mat.append(MAT_SLEEVE); S.seg.append(sid); S.t.append(t); S.axis.append(ax)


def build_arm(S, wrist, radial, elbow, shoulder=None):
    ax = unit(elbow - wrist)
    # wrist is wider than tall: two parallel capsules
    for o in (-0.0065, 0.0065):
        S.capsule(wrist + radial * o, wrist + radial * o + ax * 0.06, 0.0138, 0.0165, MAT_SKIN)
    S.capsule(wrist + ax * 0.03, elbow, 0.021, 0.032, MAT_SKIN)
    # sleeve: begins a little behind the wrist, billows toward the elbow
    cuff = wrist + ax * 0.045
    S.capsule(cuff, cuff + ax * 0.012, 0.030, 0.034, MAT_CUFF)
    a0, a1 = cuff + ax * 0.01, elbow + ax * 0.04
    Ls = np.linalg.norm(a1 - a0)
    r = np.random.default_rng(int(abs(elbow[0]) * 1000))
    n = 260
    ph = r.random(3) * 6.28
    sid = S.nseg; S.nseg += 1
    for i in range(n):
        t = i / (n - 1)
        bulge = 1 + 0.07 * np.sin(t * Ls / 0.028 * 6.28 + ph[0]) + 0.05 * np.sin(t * Ls / 0.061 * 6.28 + ph[1])
        S.C.append(a0 + (a1 - a0) * t); S.R.append((0.035 + 0.016 * t ** 0.8) * bulge)
        S.mat.append(MAT_SLEEVE); S.seg.append(sid); S.t.append(t); S.axis.append(ax)
    if shoulder is not None:
        # the elbow: a rounded knot of cloth, then the upper arm rising into the shadow of the mantle
        S.sphere(elbow + ax * 0.03, 0.058, MAT_SLEEVE)
        sleeve_tube(S, elbow + ax * 0.03, shoulder, 0.05, 0.042, int(abs(elbow[0]) * 1000) + 1, n=220)


def build_scene(L):
    S = Scene()
    # her left hand (viewer's right): A S D F
    wl, rl = build_hand(S, L, +1, {"index": "F", "middle": "D", "ring": "S", "pinky": "A"},
                        lifted={"ring": (6, 48, 76), "pinky": (8, 30, 50)})
    # her right hand (viewer's left): J K L ; - the index finger is up, mid-keystroke
    wr, rr = build_hand(S, L, -1, {"index": "J", "middle": "K", "ring": "L", "pinky": ";"},
                        lifted={"index": (-10, 24, 50), "pinky": (10, 40, 64)})
    elbow_l = np.array([0.215, -0.405, 1.30])
    elbow_r = np.array([-0.235, -0.39, 1.33])
    build_arm(S, wl, rl, elbow_l, shoulder=np.array([0.156, -0.223, 1.40]))
    build_arm(S, wr, rr, elbow_r, shoulder=np.array([-0.134, -0.223, 1.40]))
    return S


SKIN_H = [(0.00, rgb(64, 34, 16)), (0.28, rgb(104, 64, 28)), (0.50, rgb(144, 98, 44)),
          (0.70, rgb(174, 128, 60)), (0.86, rgb(196, 152, 78)), (1.0, rgb(214, 176, 100))]
SATIN = [(0.0, rgb(24, 12, 12)), (0.3, rgb(52, 28, 18)), (0.55, rgb(88, 52, 28)),
         (0.78, rgb(132, 90, 48)), (1.0, rgb(176, 136, 80))]
KEY = unit([-0.52, 0.58, -0.63])          # toward Leonardo's light, world coords
COOL = np.array([0.62, 0.78, 0.95], np.float32)


def masked_blur(a, m, s):
    num = gaussian_filter(a * m, s)
    den = gaussian_filter(m.astype(np.float32), s)
    return num / np.maximum(den, 1e-4)


def shade(G, L):
    mat, n, P = G["mat"], G["n"].copy(), G["pos"]
    palm = (mat == MAT_PALM)
    skin = (mat == MAT_SKIN) | (mat == MAT_NAIL) | palm
    cloth = (mat == MAT_SLEEVE) | (mat == MAT_CUFF)
    # smooth away the seams between swept spheres, per material; the palm sheet needs more
    for grp, sg in ((palm, 5.0 * SS), (skin, 1.6 * SS), (cloth, 1.6 * SS)):
        g = grp.astype(np.float32)
        for c in range(3):
            n[..., c] = np.where(grp, masked_blur(n[..., c], g, sg), n[..., c])
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)

    # sleeve folds: irregular satin ridges wrapping the forearm, bump-mapped in screen space
    seg, t = G["seg"], G["t"]
    ax = G["segax"][np.clip(seg, 0, None)]
    yy, xx = np.mgrid[0:SH, 0:SW].astype(np.float32)
    along = (P * ax).sum(-1)
    upv = np.array([0, 1.0, 0], np.float32)
    e1 = np.cross(ax, upv); e1 /= np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), 1e-6)
    e2 = np.cross(ax, e1)
    th = np.arctan2((n * e2).sum(-1), (n * e1).sum(-1))
    rr = np.random.default_rng(8)
    hgt = np.zeros((SH, SW), np.float32)
    wob = gaussian_filter(rr.standard_normal((SH, SW)).astype(np.float32), 10 * SS)
    wob /= wob.std() + 1e-6
    lo, hi = np.percentile(along[cloth], [1, 99]) if cloth.any() else (0, 1)
    s = lo
    while s < hi:
        w = rr.uniform(0.0025, 0.0055)
        amp = rr.uniform(0.5, 1.0)
        b1, ph1, m1 = rr.uniform(0.002, 0.009), rr.uniform(0, 6.28), rr.integers(1, 3)
        centre = s + b1 * np.sin(th * m1 + ph1) + 0.003 * wob
        fade = 0.5 + 0.5 * np.sin(th * rr.integers(1, 3) + rr.uniform(0, 6.28))
        x = (along - centre) / w
        hgt += amp * (0.4 + 0.6 * fade) * np.where(x < 0, np.exp(-x * x), np.exp(-x * x * 0.3))
        s += rr.uniform(0.008, 0.02)
    hgt = np.where(cloth, hgt, 0)
    gy_, gx_ = np.gradient(gaussian_filter(hgt, 0.8 * SS))
    k = np.where(cloth, 1.0, 0.0) * 2.2 * SS
    n = n + np.stack([-gx_ * k, gy_ * k, np.zeros_like(gx_)], -1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    trough = np.clip(hgt / 1.2, 0, 1)

    view = unit([0, 0, -1.0])
    dk = n @ KEY
    wrap = np.clip((dk + 0.18) / 1.18, 0, 1)
    term = np.exp(-((dk - 0.05) / 0.22) ** 2)            # the warm band where light turns to shadow

    # screen: a cool area light facing her, reaching the fronts of the fingers
    S_c = L.screen_centre()
    l = S_c[None, None, :] - P
    dist = np.linalg.norm(l, axis=-1) + 1e-6
    lh = l / dist[..., None]
    emit = np.clip(((P - S_c[None, None, :]) @ L.nl) / dist, 0, 1) ** 0.6
    scr = 1.4 * np.clip((n * lh).sum(-1), 0, 1) * emit / (1 + (dist / 0.16) ** 2)

    # ambient occlusion from the depth buffer (crevices between fingers, under the palm)
    z = np.where(mat > 0, G["z"], np.nan)
    zf = np.where(np.isnan(z), 0, z)
    wv = (~np.isnan(z)).astype(np.float32)
    zb = gaussian_filter(zf, 4 * SS) / np.maximum(gaussian_filter(wv, 4 * SS), 1e-4)
    ao = np.clip(1 - np.clip((np.nan_to_num(z, nan=0) - zb) / 0.012, 0, 1) * 0.7, 0.3, 1)

    col = np.zeros((SH, SW, 3), np.float32)
    hvec = unit(KEY + view)
    # skin
    cs = ramp(wrap * 0.97 * ao, SKIN_H)
    cs += term[..., None] * np.array([0.08, 0.005, -0.03], np.float32)
    # blood near the surface: knuckles and fingertips a little rosier; faint mottling everywhere
    rosy = np.where(mat == MAT_NAIL, 1.0, 0.0) * (0.4 + 0.6 * t)
    rosy = gaussian_filter(rosy.astype(np.float32), 2 * SS)
    cs += rosy[..., None] * np.array([0.05, -0.01, -0.02], np.float32)
    mott = gaussian_filter(np.random.default_rng(12).standard_normal((SH, SW)).astype(np.float32), 3 * SS)
    mott /= mott.std() + 1e-6
    cs *= (1 + 0.025 * mott)[..., None]
    spec = np.clip(n @ hvec, 0, 1) ** 10
    cs += spec[..., None] * np.array([0.03, 0.025, 0.02], np.float32)
    # nails: the dorsal face of the distal phalanx, facing the hinge and us
    fwd = -L.dv
    dors = fwd[None, None, :] - (ax * (ax @ fwd)[..., None])
    dors /= np.maximum(np.linalg.norm(dors, axis=-1, keepdims=True), 1e-6)
    G0 = G["n"]
    nd = (G0 * dors).sum(-1)
    # oval: across the finger |sin| of the angle from the dorsal direction, along it the capsule parameter
    side = np.sqrt(np.clip(1 - nd ** 2, 0, 1))
    oval = (side / 0.62) ** 2 + ((t - 0.72) / 0.3) ** 2
    nail_hard = (mat == MAT_NAIL) & (nd > 0) & (oval < 1)
    nail = gaussian_filter(nail_hard.astype(np.float32), 0.7 * SS)
    rim = np.clip(gaussian_filter(nail_hard.astype(np.float32), 1.0 * SS) - nail_hard, 0, 1)
    cn = cs * np.array([1.02, 0.9, 0.86], np.float32) + 0.03 + (np.clip(n @ hvec, 0, 1) ** 24)[..., None] * 0.18
    cs = cs * (1 - nail[..., None]) + cn * nail[..., None]
    cs *= (1 - 0.3 * np.clip(rim * 2, 0, 1))[..., None]
    # sleeves: satin with a sheen that follows the folds
    cc = ramp(np.clip(wrap * 0.85 * ao + 0.14 * np.clip(n @ hvec, 0, 1) ** 6, 0, 1), SATIN)
    cc *= (0.7 + 0.3 * trough)[..., None]
    cuff = (mat == MAT_CUFF)
    cc = np.where(cuff[..., None], cc * 0.85, cc)
    col = np.where(skin[..., None], cs, col)
    col = np.where(cloth[..., None], cc, col)
    # cool screen light on top (additive, like light)
    col += (scr[..., None] * COOL[None, None, :] * np.where(skin, 0.9, 0.35)[..., None])
    return col, (mat > 0).astype(np.float32), (skin | cloth)


def downsample(a):
    h, w = a.shape[0] // SS, a.shape[1] // SS
    return a[:h * SS, :w * SS].reshape(h, SS, w, SS, *a.shape[2:]).mean(axis=(1, 3))


def hand_shadow(L, S):
    """Soft shadow of the hands on the deck along Leonardo's light."""
    C = np.array(S.C); R = np.array(S.R)
    h = (C - L.H0) @ L.up
    keep = (h > -0.005) & (h < 0.09) & (np.array(S.mat) != MAT_SLEEVE)
    C, R, h = C[keep], R[keep], h[keep]
    Pp = C - KEY[None, :] * (h / KEY[1])[:, None]
    p = project(Pp)
    rp = FOC * R / Pp[:, 2]
    def f(c):
        for (x, y), r in zip(p, rp):
            c.new_sub_path(); c.arc(x, y, r * 1.05, 0, 6.2832)
        c.fill()
    return mask(f, blur=6)


def render_hands(img, L, deck_mask):
    S = build_scene(L)
    sh = hand_shadow(L, S) * deck_mask
    lay(img, sh, rgb(20, 16, 14), 0.55)
    G = S.raster()
    col, cov, _ = shade(G, L)
    # upper sleeves dissolve into the dark of the dress toward the shoulders (they pass under the mantle)
    Y = G["pos"][..., 1]
    cloth = (G["mat"] == MAT_SLEEVE)
    col = np.where(cloth[..., None], col * (0.08 + 0.92 * np.clip((-0.33 - Y) / 0.07, 0, 1) ** 1.5)[..., None], col)
    cov = cov * np.where(cloth, np.clip((-0.24 - Y) / 0.06, 0, 1), 1.0)
    # never outside her silhouette: the body mask, supersampled
    from scipy.ndimage import zoom as _zoom
    bm = fill_spline(BODY, blur=2.0, t=0.4)[BY0:BY1, BX0:BX1]
    bm = _zoom(bm, SS, order=1)[:SH, :SW]
    cov = cov * np.where(cloth, bm, 1.0)
    colp = downsample(col * cov[..., None])
    a = downsample(cov)
    colp = colp / np.maximum(a[..., None], 1e-4)
    colp = np.stack([gaussian_filter(colp[..., i], 0.55) for i in range(3)], -1)
    region = img[BY0:BY1, BX0:BX1]
    lay(region, a, colp)

    # fine crinkle pleats: short kinked strokes following each forearm, pale ochre and dark umber
    import cairo as _c
    sl = downsample(((G["mat"] == MAT_SLEEVE) | (G["mat"] == MAT_CUFF)).astype(np.float32) * cov)
    slm = np.zeros((H, W), np.float32)
    slm[BY0:BY1, BX0:BX1] = sl[:BY1 - BY0, :BX1 - BX0]
    rr = np.random.default_rng(23)
    ys, xs = np.nonzero(slm > 0.6)
    axes = G["segax"][np.clip(G["seg"], 0, None)]
    axd = downsample(axes)
    def crinkle(c):
        for i in rr.integers(0, len(xs), 900):
            x, y = xs[i], ys[i]
            a3 = axd[min(y - BY0, axd.shape[0] - 1), min(x - BX0, axd.shape[1] - 1)]
            ang = np.arctan2(-a3[1], a3[0]) + np.pi / 2 + rr.normal(0, 0.35)     # across the arm = along the folds
            Lk = 8 + 18 * rr.random()
            dx, dy = np.cos(ang) * Lk / 2, np.sin(ang) * Lk / 2
            kink = rr.normal(0, 2.5)
            light = rr.random() < 0.5
            v = 0.6 + 0.5 * rr.random()
            c.set_source_rgba(*((0.62 * v, 0.46 * v, 0.24 * v) if light else (0.1, 0.06, 0.04)), 0.12 + 0.2 * rr.random())
            c.set_line_width(1 + 1.8 * rr.random())
            c.move_to(x - dx, y - dy)
            c.line_to(x + kink * np.sin(ang), y - kink * np.cos(ang))
            c.line_to(x + dx, y + dy)
            c.stroke()
    col2, al2 = rgba_layer(crinkle)
    lum = img @ np.array([0.3, 0.55, 0.15], np.float32)
    litm = np.clip((gaussian_filter(lum, 4) - 0.16) / 0.18, 0, 1)
    comp(img, col2, al2 * slm * litm, blur=0.9)

    # contact shadows where the fingertips meet the keys
    K = None
    tips = [np.array(p) for p, m_ in zip(S.C, S.mat) if m_ == MAT_SKIN]
    Cs = np.array(S.C); Ms = np.array(S.mat); Rs = np.array(S.R)
    hgt = (Cs - L.H0) @ L.up
    low = (hgt < 0.012) & ((Ms == MAT_SKIN) | (Ms == MAT_NAIL))
    pp = project(Cs[low] - L.up[None, :] * (hgt[low] - 0.0015)[:, None])
    def contact(c):
        for (x, y) in pp:
            c.new_sub_path(); c.arc(x, y + 2, 5, 0, 6.2832)
        c.fill()
    lay(img, mask(contact, blur=3) * deck_mask, rgb(10, 8, 8), 0.45)
    full = np.zeros((H, W), np.float32)
    full[BY0:BY1, BX0:BX1] = a[:BY1 - BY0, :BX1 - BX0]
    return S, G, full


# ======================================================================
# age
# ======================================================================
"""Five centuries on a poplar panel: craquelure, yellowed varnish, grain, vignette."""
from scipy.spatial import Voronoi


def crack_network(seed, spacing, jitter=0.45, aniso=1.0, wobble=0.8, width=0.7, keep=1.0):
    """Voronoi cell boundaries on a jittered grid, drawn as wobbly cairo lines. Returns coverage."""
    r = np.random.default_rng(seed)
    gy = np.arange(-spacing * aniso, H + spacing * aniso, spacing * aniso)
    gx = np.arange(-spacing, W + spacing, spacing)
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([X.ravel() + r.uniform(-jitter, jitter, X.size) * spacing,
                    Y.ravel() + r.uniform(-jitter, jitter, X.size) * spacing * aniso], -1)
    vor = Voronoi(pts)
    V = vor.vertices

    def draw(c):
        c.set_line_cap(cairo.LINE_CAP_ROUND)
        for (a, b) in vor.ridge_vertices:
            if a < 0 or b < 0 or r.random() > keep:
                continue
            p, q = V[a], V[b]
            if not (-20 < p[0] < W + 20 and -20 < p[1] < H + 20):
                continue
            d = np.hypot(*(q - p))
            if d > spacing * 3:
                continue
            n = max(2, int(d / 3))
            t = np.linspace(0, 1, n)
            seg = p[None, :] + (q - p)[None, :] * t[:, None]
            nrm = np.array([-(q - p)[1], (q - p)[0]]) / (d + 1e-6)
            seg += nrm[None, :] * (r.normal(0, wobble, n) * np.sin(np.pi * t))[:, None]
            c.set_source_rgba(0, 0, 0, 0.35 + 0.65 * r.random())
            c.set_line_width(width * (0.6 + 0.8 * r.random()))
            c.move_to(*seg[0])
            for s in seg[1:]:
                c.line_to(*s)
            c.stroke()
    return mask(draw)


def age(img, strength=1.0):
    out = img.copy()
    lum = out @ np.array([0.3, 0.55, 0.15], np.float32)

    # craquelure: a coarse network plus a fine one, stronger where the paint is light
    coarse = crack_network(11, 17, aniso=1.8, width=0.75, keep=0.85, wobble=0.9)
    fine = crack_network(12, 6.5, aniso=1.7, width=0.5, keep=0.85, wobble=0.4)
    cracks = np.clip(coarse * 0.6 + fine * 0.6, 0, 1)
    # vary crack visibility across the panel
    vis = 0.45 + 0.55 * fbm(120, 3, 13)
    cracks *= vis * strength
    light = np.clip((lum - 0.18) / 0.5, 0, 1)
    dark_line = cracks * (0.32 + 0.8 * light)
    out *= (1 - dark_line[..., None] * np.array([0.55, 0.6, 0.7], np.float32))
    # in the dark passages the cracks read as faint pale lines (lifted varnish)
    pale = cracks * (1 - light) * 0.10
    out += pale[..., None] * np.array([0.5, 0.45, 0.35], np.float32)
    # each cell slightly cupped: a whisper of light just beside the crack
    halo = np.clip(gaussian_filter(cracks, 1.0) - cracks, 0, 1)
    out += halo[..., None] * light[..., None] * 0.09

    # paint texture: fine brushy noise + faint vertical poplar grain
    grain = fbm(3, 3, 21, aniso=(6.0, 0.3)) - 0.5
    tex = fbm(1.5, 2, 22) - 0.5
    brush = fbm(5, 3, 23, aniso=(0.5, 2.5)) - 0.5                     # horizontal-ish brush drag
    out *= (1 + 0.07 * grain[..., None] + 0.05 * tex[..., None] + 0.04 * brush[..., None])

    # yellowed varnish: warms and sinks the blues in the lights; the darks keep a cool, purplish depth
    lum2 = np.clip(out @ np.array([0.3, 0.55, 0.15], np.float32), 0, 1)[..., None]
    tint = (1 - lum2) * np.array([1.0, 1.0, 1.0], np.float32) + lum2 * np.array([1.0, 0.975, 0.88], np.float32)
    out = out * tint
    dark = np.clip(1 - lum2 / 0.3, 0, 1)
    out = out + dark * np.array([0.012, 0.0, 0.05], np.float32)
    out = out * 0.97 + np.array([0.018, 0.014, 0.006], np.float32)

    # a gentle S-curve and a touch more colour in the middle tones
    sc = out * out * (3 - 2 * out)
    out = out * 0.78 + sc * 0.22
    lum3 = (out @ np.array([0.3, 0.55, 0.15], np.float32))[..., None]
    mid = 1 - np.abs(lum3 - 0.45) / 0.45
    out = lum3 + (out - lum3) * (1 + 0.12 * np.clip(mid, 0, 1))

    # darkened edges of the panel
    cx, cy = W * 0.5, H * 0.46
    rr = np.sqrt(((XX - cx) / (W * 0.62)) ** 2 + ((YY - cy) / (H * 0.62)) ** 2)
    vig = np.clip(1 - 0.35 * np.clip(rr - 0.55, 0, 1) ** 1.4, 0, 1)
    out *= vig[..., None]

    # a few specks and losses in the dark lower right (the original has them too)
    r = np.random.default_rng(31)
    def specks(c):
        for _ in range(8):
            x, y = r.uniform(760, 950), r.uniform(900, 1200)
            c.arc(x, y, r.uniform(0.3, 0.9), 0, 6.283)
            c.fill()
    sp = mask(specks, blur=0.4) * np.clip((0.35 - lum) / 0.2, 0, 1)
    lay(out, sp, rgb(150, 140, 120), 0.5)
    return np.clip(out, 0, 1)


# ======================================================================
# main
# ======================================================================
import sys
import time


def render(stage="all"):
    t0 = time.time()
    img = np.zeros((H, W, 3), np.float32)
    paint_landscape(img)
    paint_veil(img)
    paint_dress(img, BODY)
    face, neck = skin_base()
    skin = paint_face(img, face, neck)
    paint_hair(img, skin)
    paint_shawl(img, BODY)
    paint_embroidery(img)
    paint_lap(img)
    L = Laptop()
    # contact shadow of the laptop on her lap
    sh = fill_poly([tuple(project(L.P(u, v, -L.tb))) for u, v in [(-L.w / 2 - 0.01, -0.01), (L.w / 2 + 0.01, -0.01),
                                                                      (L.w / 2 + 0.01, L.d + 0.01), (-L.w / 2 - 0.01, L.d + 0.01)]], blur=12)
    lay(img, sh, rgb(12, 8, 10), 0.6)
    pre_lap = img.copy()
    deck = paint_laptop_base(img, L)
    _, _, handcov = render_hands(img, L, deck)
    lidm, _ = paint_laptop_lid(img, L)
    # the machine is part of the painting: warm it like old bronze, and soften it more than the rest
    _pj = project
    lapm = np.clip(fill_poly([tuple(_pj(L.P(u, v, 0))) for u, v in [(-L.w / 2, 0), (L.w / 2, 0), (L.w / 2, L.d), (-L.w / 2, L.d)]], blur=2)
                   + lidm, 0, 1)
    lapm = lapm * (1 - np.clip(gaussian_filter(handcov, 1.0) * 1.5, 0, 1))
    warm = img * np.array([1.0, 0.9, 0.7], np.float32) * 0.92
    soft = np.stack([gaussian_filter(warm[..., i], 0.9) for i in range(3)], -1)
    img[:] = img * (1 - lapm[..., None] * 0.85) + soft * (lapm[..., None] * 0.85)
    # sfumato: nothing in Leonardo has a hard edge - soften the whole panel, keep a trace of the crisp
    soft = np.stack([gaussian_filter(img[..., i], 0.8) for i in range(3)], -1)
    img[:] = img * 0.6 + soft * 0.4
    print(f"render {time.time() - t0:.1f}s")
    return img


if __name__ == "__main__":
    img = render()
    save(age(img), "mona_lisa_typing.png")
