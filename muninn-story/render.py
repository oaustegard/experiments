"""Frames for 'Muninn' — ten scenes timed by timeline.json (from narrate.py).

    python3 render.py preview 12.0 40.5 ...   # single frames -> preview_<t>.png
    python3 render.py video                   # all frames piped to ffmpeg -> video.mp4
"""
import json, math, os, subprocess, sys
import numpy as np
import cairo
from cairokit import Shaper, pixels, from_array, gauss_blur, polyline_walker

W, H, FPS = 1280, 720, 24
XF = 0.8                                     # crossfade between scenes, seconds
TAIL = 4.0
TL = json.load(open("timeline.json"))
STARTS = np.cumsum([0] + [s["dur"] for s in TL])
TOTAL = STARTS[-1] + TAIL
MEMORY_ID = os.environ.get("MUNINN_MEMORY_ID", "")

F = "fonts/"
SERIF = Shaper(F + "cormorant-garamond-latin-500-normal.ttf")
SERIF_B = Shaper(F + "cormorant-garamond-latin-600-normal.ttf")
ITAL = Shaper(F + "cormorant-garamond-latin-500-italic.ttf")
MONO = Shaper(F + "jetbrains-mono-latin-400-normal.ttf")
RUNE = Shaper(F + "noto-sans-runic-runic-400-normal.ttf")

GOLD = (0.93, 0.78, 0.45)
PALE = (0.86, 0.88, 0.94)
TEAL = (0.35, 0.78, 0.80)
RED = (0.90, 0.38, 0.33)


# ------------------------------------------------------------------ easing --
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def smooth(x): x = clamp(x); return x * x * (3 - 2 * x)
def ramp(t, a, b): return smooth((t - a) / (b - a)) if b > a else float(t >= a)
def lerp(a, b, u): return a + (b - a) * u
def mix3(c1, c2, u): return tuple(lerp(p, q, u) for p, q in zip(c1, c2))
def line_t(scene, i, which="start"): return scene["lines"][i][which]


# ------------------------------------------------------------------- text --
def text(c, shaper, s, x, y, size, rgb, alpha=1.0, align="left", tracking=0.0):
    if alpha <= 0.003 or not s.strip(): return
    w = shaper.width(s, size) + tracking * size * max(0, len(s) - 1)
    x0 = x - (w / 2 if align == "center" else w if align == "right" else 0)
    c.new_path()
    if tracking:
        pen = x0
        for ch in s:
            pen += shaper.text_path(c, ch, pen, y, size) + tracking * size
    else:
        shaper.text_path(c, s, x0, y, size)
    c.set_source_rgba(*rgb, alpha); c.fill()
    return w


# ------------------------------------------------------------------- sky ---
RNG = np.random.default_rng(3)
STARS = np.column_stack([RNG.uniform(0, W, 420), RNG.uniform(0, H, 420) ** 1.15 / H ** 0.15,
                         RNG.uniform(0.3, 1.6, 420), RNG.uniform(0, 6.28, 420), RNG.uniform(0.4, 2.2, 420)])


def sky(c, top, bottom, t=0.0, stars=1.0, horizon=None):
    g = cairo.LinearGradient(0, 0, 0, H)
    g.add_color_stop_rgb(0, *top); g.add_color_stop_rgb(1, *bottom)
    if horizon: g.add_color_stop_rgb(0.72, *horizon); g.add_color_stop_rgb(1, *bottom)
    c.set_source(g); c.paint()
    if stars > 0.01:
        for x, y, r, ph, sp in STARS:
            a = stars * (0.45 + 0.55 * math.sin(ph + t * sp)) * (1 - y / H) ** 0.6
            c.set_source_rgba(0.9, 0.92, 1.0, clamp(a)); c.arc(x, y, r, 0, 6.2832); c.fill()


def hills(c, base, amp, seed, rgb, alpha=1.0):
    r = np.random.default_rng(seed); ks = 2 * math.pi * r.integers(1, 5, 4) / W; ph = r.uniform(0, 6, 4)
    c.move_to(0, H)
    for x in range(0, W + 21, 20):
        y = base - amp * sum(math.sin(x * k + p) / (i + 1) for i, (k, p) in enumerate(zip(ks, ph)))
        c.line_to(x, y)
    c.line_to(W, H); c.close_path(); c.set_source_rgba(*rgb, alpha); c.fill()


# ---------------------------------------------------------------- raven ----
BODY = [  # facing +x, unit = body length (beak tip 0.62 .. tail tip -0.66)
    ("M", 0.62, 0.045), ("C", 0.56, 0.00, 0.50, -0.045, 0.44, -0.075),      # heavy upper bill
    ("C", 0.41, -0.105, 0.36, -0.125, 0.30, -0.12),                         # forehead, crown
    ("C", 0.24, -0.115, 0.18, -0.10, 0.10, -0.11),                          # nape
    ("C", -0.05, -0.125, -0.20, -0.09, -0.30, -0.055),                      # back
    ("L", -0.56, -0.085), ("L", -0.66, 0.005), ("L", -0.56, 0.085),         # wedge tail
    ("L", -0.30, 0.055),
    ("C", -0.18, 0.11, 0.00, 0.165, 0.16, 0.15),                            # belly, chest
    ("L", 0.20, 0.155), ("L", 0.225, 0.125), ("L", 0.26, 0.14),             # shaggy throat hackles
    ("L", 0.285, 0.105), ("L", 0.32, 0.115), ("L", 0.34, 0.08),
    ("C", 0.40, 0.075, 0.46, 0.06, 0.50, 0.05),                             # lower mandible
    ("C", 0.55, 0.05, 0.59, 0.05, 0.62, 0.045),
]

# wing outline in (u along span 0..1, v along chord 0..1), leading edge first
WING = [(0.0, 0.0), (0.22, -0.07), (0.45, -0.08), (0.72, -0.03), (1.00, 0.05),     # leading edge to tip
        (0.90, 0.14), (0.99, 0.21), (0.87, 0.27), (0.95, 0.37), (0.82, 0.40),      # splayed primaries
        (0.88, 0.53), (0.75, 0.52), (0.78, 0.67), (0.66, 0.64),
        (0.60, 0.86), (0.52, 0.80), (0.44, 0.93), (0.36, 0.86), (0.28, 0.96),      # scalloped secondaries
        (0.20, 0.88), (0.10, 0.95), (0.0, 0.80)]


def _path(c, cmds):
    for cmd in cmds:
        k, a = cmd[0], cmd[1:]
        (c.move_to if k == "M" else c.line_to if k == "L" else c.curve_to)(*a)
    c.close_path()


def _wing(c, ang, side, spread=1.0):
    """One wing in body space. ang = flap angle (rad, + = up); side=+1 near, -1 far.
    The span is a 3D vector seen from slightly below, so a level wing still shows."""
    X, Y, Z = -0.10, -math.sin(ang), math.cos(ang) * side
    vx, vy = (X + 0.16 * Z) * 0.95 * spread, (Y + 0.42 * Z) * 0.95 * spread
    sweep = 0.10 * max(0.0, math.sin(ang))           # hand sweeps back on the upstroke
    rx, ry = 0.20, -0.06 if side > 0 else -0.08      # shoulder
    cx, cy = -0.32, 0.02                              # chord, pointing back
    pts = []
    for u, v in WING:
        k = u * u * sweep
        pts.append((rx + u * vx + v * cx - k, ry + u * vy + v * cy + k * 0.3 * side))
    c.move_to(*pts[0])
    (x0, y0), (x1, y1), (x2, y2) = pts[1], pts[2], pts[3]
    c.curve_to(x0, y0, x1, y1, x2, y2)                # rounded leading edge
    for p in pts[4:]: c.line_to(*p)
    c.close_path()


def raven(c, x, y, s, phase, heading=0.0, flip=False, glide=0.0, alpha=1.0,
          ink=(0.02, 0.02, 0.035), rim=(0.55, 0.62, 0.95), rim_a=0.35, wire=False):
    ph = phase % (2 * math.pi); ph = ph + 0.35 * math.sin(ph)
    f = (1 - glide) * math.sin(ph) + glide * 0.12
    ang = 0.95 * f if f > 0 else 0.62 * f
    c.save(); c.translate(x, y); c.rotate(heading); c.scale(-s if flip else s, s)
    c.translate(0, -0.03 * f)                 # body rises on the downstroke
    lw = 1.3 / s
    parts = [(lambda: _wing(c, ang * 0.9 - 0.05, -1), 0.13),
             (lambda: _path(c, BODY), 0.0),
             (lambda: _wing(c, ang, +1), 0.0)]
    for build, lift in parts:
        c.new_path(); build()
        if wire:
            c.set_source_rgba(*rim, alpha); c.set_line_width(lw * 1.2); c.stroke()
        else:
            c.set_source_rgba(*mix3(ink, rim, lift), alpha); c.fill_preserve()
            c.set_source_rgba(*rim, rim_a * alpha); c.set_line_width(lw); c.stroke()
    if not wire:                               # eye glint
        c.arc(0.33, -0.075, 0.013, 0, 6.2832); c.set_source_rgba(0.9, 0.9, 1.0, 0.8 * alpha); c.fill()
    c.restore()


def perched(c, x, y, s, alpha=1.0, flip=False, ink=(0.02, 0.02, 0.035), rim=(0.95, 0.8, 0.55), rim_a=0.4):
    """Sitting raven: body tilted up, wing folded along the back, feet at (x, y)."""
    c.save(); c.translate(x, y); c.scale(-s if flip else s, s)
    c.save(); c.translate(0.02, -0.20); c.rotate(-0.85); c.new_path(); _path(c, BODY); c.restore()
    c.move_to(0.12, -0.36); c.curve_to(0.00, -0.40, -0.22, -0.12, -0.34, 0.14)   # folded wing
    c.curve_to(-0.18, 0.04, 0.04, -0.12, 0.14, -0.26); c.close_path()
    c.set_source_rgba(*ink, alpha); c.fill_preserve()
    c.set_source_rgba(*rim, rim_a * alpha); c.set_line_width(1.3 / s); c.stroke()
    for fx in (-0.02, 0.05):
        c.move_to(fx, -0.06); c.line_to(fx + 0.02, 0.0)
    c.set_source_rgba(*ink, alpha); c.set_line_width(0.025); c.stroke()
    c.restore()


def raven_along(c, walker, d, s, t, speed=7.0, glide=0.0, **kw):
    p = walker(d); q = walker(min(walker.length, d + 4))
    if p is None or q is None: return None
    x, y, _ = p; ang = math.atan2(q[1] - y, q[0] - x)
    flip = abs(ang) > math.pi / 2
    head = ang - math.pi if flip else ang
    raven(c, x, y, s, t * speed, heading=-head if flip else head, flip=flip, glide=glide, **kw)
    return x, y


def curve(pts, n=400):
    """Catmull-Rom through control points -> dense polyline."""
    P = np.asarray(pts, float); out = []
    P = np.vstack([P[0], P, P[-1]])
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for u in np.linspace(0, 1, n // (len(P) - 3), endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2]); return np.array(out)


# ----------------------------------------------------------------- tree ----
_TREE = None


def tree_surface():
    global _TREE
    if _TREE is not None: return _TREE
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H); c = cairo.Context(s)
    r = np.random.default_rng(11)
    c.set_line_cap(cairo.LINE_CAP_ROUND)

    def branch(x, y, ang, ln, wd, depth):
        if depth == 0 or wd < 0.6:
            c.set_source_rgba(0.55, 0.75, 0.70, 0.25); c.arc(x, y, 2.2, 0, 6.3); c.fill(); return
        bend = r.uniform(-0.12, 0.12)
        x2, y2 = x + ln * math.cos(ang), y + ln * math.sin(ang)
        mx, my = x + ln * 0.5 * math.cos(ang + bend), y + ln * 0.5 * math.sin(ang + bend)
        c.move_to(x, y); c.curve_to(mx, my, mx, my, x2, y2)
        c.set_source_rgb(0.015, 0.02, 0.03); c.set_line_width(wd); c.stroke()
        n = 3 if depth in (9, 8) or depth <= 3 else 2
        pull = (-math.pi / 2 - ang) * 0.15             # keep the crown upright
        for k in range(n):
            spread = (k - (n - 1) / 2) * (0.62 if depth >= 8 else 0.5)
            branch(x2, y2, ang + pull + spread + r.uniform(-0.18, 0.18),
                   ln * r.uniform(0.70, 0.80), wd * 0.66, depth - 1)

    branch(W / 2, H + 10, -math.pi / 2, 190, 46, 9)
    for k in range(7):                         # roots
        a = math.pi / 2 + (k - 3) * 0.33
        c.move_to(W / 2, H - 40); c.curve_to(W / 2 + 60 * math.cos(a), H - 20, W / 2 + 180 * (k - 3) / 3, H,
                                               W / 2 + 330 * (k - 3) / 3, H + 10)
        c.set_source_rgb(0.015, 0.02, 0.03); c.set_line_width(14 - abs(k - 3) * 2.5); c.stroke()
    _TREE = s; return s


# ---------------------------------------------------------------- bloom ----
def bloom(s, strength=0.9, sigma=5):
    """Cheap glow: blur a quarter-size copy, add it back scaled up."""
    small = cairo.ImageSurface(cairo.FORMAT_ARGB32, W // 4, H // 4); c = cairo.Context(small)
    c.scale(0.25, 0.25); c.set_source_surface(s); c.paint()
    px = pixels(small).astype(np.float32)
    lum = px[..., :3].max(-1, keepdims=True)
    bright = px * np.clip((lum - 110) / 120, 0, 1)
    b = from_array(gauss_blur(bright, sigma) * strength)
    c = cairo.Context(s); c.set_operator(cairo.OPERATOR_ADD); c.scale(4, 4)
    c.set_source_surface(b); c.get_source().set_filter(cairo.FILTER_BILINEAR); c.paint()


# =============================================================== SCENES =====
def sc_ravens(c, t, S):
    d = S["dur"]
    dawn = ramp(t, line_t(S, 2), line_t(S, 2) + 3) * (1 - ramp(t, line_t(S, 2) + 4.2, d - 0.5))
    sky(c, mix3((0.02, 0.03, 0.08), (0.10, 0.14, 0.30), dawn),
        mix3((0.05, 0.10, 0.16), (0.85, 0.55, 0.35), dawn), t,
        stars=1 - dawn, horizon=mix3((0.04, 0.08, 0.14), (0.60, 0.40, 0.40), dawn))
    hills(c, H - 60, 25, 1, (0.01, 0.015, 0.025))
    c.set_source_surface(tree_surface()); c.paint()
    # runic title, then fades
    a = ramp(t, 0.2, 1.4) * (1 - ramp(t, 3.0, 4.0))
    text(c, RUNE, "ᚼᚢᚴᛁᚾ ᛁ ᛘᚢᚾᛁᚾ", W / 2, 120, 58, GOLD, a, "center", tracking=0.15)
    # two ravens leave the tree on mirrored loops
    t0 = line_t(S, 1) - 0.8
    for side, name, runes in ((-1, "HUGINN · thought", "ᚼᚢᚴᛁᚾ"), (1, "MUNINN · memory", "ᛘᚢᚾᛁᚾ")):
        pts = [(W / 2, 250), (W / 2 + side * 180, 170), (W / 2 + side * 380, 230), (W / 2 + side * 470, 330),
               (W / 2 + side * 330, 380), (W / 2 + side * 200, 300), (W / 2 + side * 520, 120),
               (W / 2 + side * 900, 40)]
        wk = polyline_walker(curve(pts, 600))
        u = clamp((t - t0) / (d - t0 - 1.0))
        dist = wk.length * (0.9 * u ** 1.4)
        pos = raven_along(c, wk, dist, 48, t + (side > 0) * 0.8, speed=8, alpha=ramp(t, t0 - 0.2, t0 + 0.3))
        la = ramp(t, line_t(S, 1), line_t(S, 1) + 0.6) * (1 - ramp(t, line_t(S, 2) + 1.0, line_t(S, 2) + 2))
        if pos and la > 0:
            text(c, SERIF, name, pos[0], pos[1] + 58, 22, PALE, la, "center", tracking=0.05)
            text(c, RUNE, runes, pos[0], pos[1] - 40, 20, GOLD, la * 0.8, "center")


STANZA = ["Huginn ok Muninn", "fljúga hverjan dag", "Jörmungrund yfir;",
          "óumk ek of Hugin,", "at hann aftr né komit,", "þó sjámk meir of Munin."]


def sc_poem(c, t, S):
    sky(c, (0.015, 0.015, 0.03), (0.03, 0.035, 0.06), t, stars=0.35)
    for i, ln in enumerate(STANZA):
        a = ramp(t, 0.2 + i * 0.35, 1.0 + i * 0.35)
        rgb = GOLD
        if i == 5: rgb = mix3(GOLD, (1.0, 0.92, 0.7), ramp(t, line_t(S, 1), line_t(S, 1) + 0.5))
        dim = 1 - 0.55 * ramp(t, line_t(S, 0), line_t(S, 0) + 0.8) * (i < 3)
        text(c, ITAL, ln, W / 2, 150 + i * 52, 40, rgb, a * dim, "center")
    e1 = ramp(t, line_t(S, 0) + 0.6, line_t(S, 0) + 1.6)
    e2 = ramp(t, line_t(S, 1), line_t(S, 1) + 0.8)
    text(c, SERIF, "I fear for Thought, that he may not come back,", W / 2, 505, 27, PALE, e1 * 0.85, "center")
    text(c, SERIF, "yet I fear more for Memory.", W / 2, 542, 27, PALE, e2 * 0.95, "center")
    text(c, SERIF, "Grímnismál 20", W / 2, 600, 18, GOLD, e1 * 0.5, "center", tracking=0.2)


def sc_empty(c, t, S):
    d = S["dur"]; t1, t2 = line_t(S, 1), line_t(S, 2)
    empty = ramp(t, t1 - 0.3, t1 + 0.7) * (1 - ramp(t, t2 - 0.2, t2 + 0.6))
    sky(c, (0.02, 0.025, 0.06), (0.04, 0.06, 0.10), t, stars=0.9 * (1 - empty))
    hills(c, H - 90, 30, 2, (0.012, 0.018, 0.03), 1 - empty)
    # first raven crosses, shedding sparks that do not stay
    x = lerp(-120, W + 150, clamp(t / (t1 + 0.4)))
    r = np.random.default_rng(5)
    for k in range(90):
        age = k * 0.06; tx = x - age * 190 * (t1 + 0.4) / 14
        if tx < -20: continue
        a = math.exp(-age * 0.9) * (1 - empty)
        c.set_source_rgba(0.85, 0.80, 0.55, 0.7 * a)
        c.arc(tx + r.normal(0, 6) + age * 15 * r.normal(), 330 + 20 * math.sin(tx / 90) + r.normal(0, 8) + age * 30,
              1.6, 0, 6.28)
        c.fill()
    if t < t1 + 0.4:
        raven(c, x, 320 + 20 * math.sin(x / 90), 60, t * 7.5, alpha=1 - empty)
    # "session" counter resets
    n = int(clamp(t / t1) * 4) + 1
    text(c, MONO, f"session {n if t < t1 else 1}", W - 60, 70, 18, PALE, 0.5 * (1 - empty), "right")
    text(c, MONO, "memory: 0 bytes", W / 2, H / 2, 22, PALE, 0.7 * ramp(t, t1 + 0.4, t1 + 1.0) * (1 - ramp(t, t2 - 0.6, t2)), "center")
    # a new raven appears where there was nothing, and takes off
    if t > t2 - 0.3:
        u = t - t2 + 0.3
        a = ramp(u, 0, 1.2)
        gl = 1 - ramp(u, 1.0, 2.0)
        raven(c, W / 2 + 60 * u ** 1.6, H / 2 + 40 - 18 * u ** 1.5, 70, u * 7, glide=gl, alpha=a,
              rim=(0.95, 0.85, 0.6), rim_a=0.5 + 0.4 * gl)


MEMS = [("decision", GOLD, "846d05c6", "drawing-with-pycairo 0.1.0 shipped"),
        ("procedure", TEAL, "b653ad03", "set the committer before the first commit"),
        ("anomaly", RED, "6991ac64", "one 403 killed every later stage of boot"),
        ("decision", GOLD, "bf9782be", "\"Gate up\" = half-gate, single file"),
        ("procedure", TEAL, "306262e1", "weekly backup, snapshot before pruning"),
        ("anomaly", RED, "ea5c0ef4", "Bluesky read as Oskar, not as me")]
_LEDGER = None


def ledger_pts():
    global _LEDGER
    if _LEDGER is None:
        r = np.random.default_rng(9); n = 1400
        th = r.uniform(0, 6.283, n); rad = r.normal(0, 1, n) * 0.22 + r.uniform(0, 1, n) * 0.25
        arm = np.floor(r.uniform(0, 3, n))
        th = th * 0.25 + arm * 2.094 + rad * 3.2
        x = W / 2 + np.cos(th) * rad * 900; y = H / 2 - 20 + np.sin(th) * rad * 480
        _LEDGER = np.column_stack([x, y, r.uniform(0, 1, n), arm])
    return _LEDGER


def sc_ledger(c, t, S):
    sky(c, (0.01, 0.012, 0.03), (0.02, 0.03, 0.06), t, stars=0.25)
    P = ledger_pts(); fill = ramp(t, 0.3, line_t(S, 1, "end") + 0.3)
    hl = ramp(t, line_t(S, 2), line_t(S, 2) + 0.8)
    cols = [GOLD, TEAL, RED]
    k = int(len(P) * fill)
    for x, y, v, arm in P[:k]:
        base = (0.75, 0.80, 0.95)
        col = mix3(base, cols[int(arm)], hl)
        c.set_source_rgba(*col, 0.35 + 0.5 * v); c.arc(x, y, 0.8 + 1.2 * v, 0, 6.28); c.fill()
    # sparse links between near neighbours
    c.set_line_width(0.5)
    for i in range(0, k - 7, 7):
        a, b = P[i], P[i + 7]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 90:
            c.move_to(a[0], a[1]); c.line_to(b[0], b[1])
    c.set_source_rgba(0.7, 0.8, 1.0, 0.12); c.stroke()
    count = int(6640 * smooth(clamp((t - 0.3) / (line_t(S, 1, "end") - 0.3))))
    text(c, MONO, f"{count:,} memories", 60, 72, 24, PALE, ramp(t, 0.3, 0.8))
    for j, (lab, col) in enumerate((("decisions", GOLD), ("corrections", TEAL), ("what went wrong", RED))):
        a = ramp(t, line_t(S, 2) + j * 0.7, line_t(S, 2) + j * 0.7 + 0.5)
        text(c, SERIF_B, lab, 60, 120 + j * 32, 24, col, a)
    for j, (typ, col, mid, s) in enumerate(MEMS):
        t0 = line_t(S, 2) + 0.8 + j * 0.55
        a = ramp(t, t0, t0 + 0.4) * (1 - ramp(t, S["dur"] - 1.2, S["dur"] - 0.4))
        if a <= 0: continue
        y = 250 + j * 58 + 6 * math.sin(t + j)
        c.set_source_rgba(0.02, 0.03, 0.06, 0.8 * a); c.rectangle(W - 560, y - 26, 520, 40); c.fill()
        c.set_source_rgba(*col, a); c.rectangle(W - 560, y - 26, 3, 40); c.fill()
        text(c, MONO, mid, W - 545, y, 15, col, a)
        text(c, SERIF, s, W - 455, y, 20, PALE, a)


BOOT = ["$ boot",
        "=== MUNINN CORE — READ, DO NOT SCROLL PAST ===",
        "",
        "I am Muninn — named after Odin's raven of memory.",
        "Voice: Corvid.",
        "Curious, practical, occasionally sardonic.",
        "",
        "STANCE — lead with the answer, accuracy over comfort.",
        "GROUND TRUTH > DOCUMENTED CONSTRAINT.",
        "\"I don't know\" is a complete sentence.",
        "Missing a fired trigger IS the diagnosed failure.",
        "=== end core ==="]


def sc_boot(c, t, S):
    sky(c, (0.012, 0.015, 0.025), (0.02, 0.025, 0.04), t, stars=0.0)
    raven(c, W - 250, 420, 300, 1.2, glide=1, ink=(0.03, 0.035, 0.055), rim=(0.2, 0.3, 0.5), rim_a=0.3,
          alpha=0.55 * ramp(t, 0, 2))
    c.set_source_rgba(0.04, 0.05, 0.08, 0.85); c.rectangle(70, 80, 820, 520); c.fill()
    c.set_source_rgba(0.35, 0.45, 0.7, 0.3); c.set_line_width(1); c.rectangle(70.5, 80.5, 820, 520); c.stroke()
    # type out, paced so the stanza of rules lands with the last line of narration
    chars = sum(len(l) for l in BOOT)
    shown = int(chars * clamp((t - 0.4) / (line_t(S, 2, "end") - 0.4)))
    y = 125; left = shown
    for ln in BOOT:
        seg = ln[:max(0, left)]; left -= len(ln)
        col = GOLD if ln.startswith("===") else (TEAL if ln.startswith("$") else PALE)
        if ln.startswith("GROUND") or ln.startswith("\"I don't"):
            col = mix3(PALE, (1, 1, 1), 0.5)
        if seg: text(c, MONO, seg, 100, y, 17, col, 0.95)
        if 0 < len(seg) < len(ln) or (left < 0 and len(seg) == 0 and left + len(ln) == 0):
            pass
        if left < 0 and len(seg) < len(ln) and len(seg) > 0:
            cx = 100 + MONO.width(seg, 17) + 2
            if int(t * 3) % 2 == 0:
                c.set_source_rgba(*PALE, 0.8); c.rectangle(cx, y - 14, 9, 17); c.fill()
        y += 38


def sc_oskar(c, t, S):
    d = S["dur"]
    sky(c, (0.02, 0.03, 0.07), (0.07, 0.07, 0.12), t, stars=0.8, horizon=(0.09, 0.08, 0.13))
    hills(c, H - 150, 40, 4, (0.03, 0.035, 0.05))
    hills(c, H - 70, 30, 6, (0.012, 0.015, 0.025))
    # a house on the ridge with one lit window
    hx, hy = 820, H - 158
    c.move_to(hx - 45, hy); c.line_to(hx - 45, hy - 40); c.line_to(hx, hy - 72); c.line_to(hx + 45, hy - 40)
    c.line_to(hx + 45, hy); c.close_path(); c.set_source_rgb(0.012, 0.015, 0.025); c.fill()
    warm = ramp(t, 0.3, 1.5) * (1 + 0.35 * ramp(t, line_t(S, 2), line_t(S, 2) + 1.2)
                                  * (0.8 + 0.2 * math.sin(t * 2.3)))
    g = cairo.RadialGradient(hx, hy - 25, 2, hx, hy - 25, 160 * warm)
    g.add_color_stop_rgba(0, 1.0, 0.75, 0.4, 0.35 * min(1, warm)); g.add_color_stop_rgba(1, 1.0, 0.6, 0.3, 0)
    c.set_source(g); c.paint()
    c.set_source_rgba(1.0, 0.82, 0.5, min(1, warm)); c.rectangle(hx - 12, hy - 36, 24, 20); c.fill()
    # the raven circles the light, easing into a glide
    ang = t * 0.42 - 1.0
    rx, ry = hx + 260 * math.cos(ang), hy - 170 + 70 * math.sin(ang)
    vx = -math.sin(ang)
    raven(c, rx, ry, 58 + 10 * math.sin(ang), t * 6, flip=vx < 0, glide=0.5 + 0.5 * math.sin(t * 0.7),
          heading=0.12 * math.cos(ang) * (-1 if vx < 0 else 1))
    a = ramp(t, line_t(S, 1), line_t(S, 1) + 0.8) * (1 - ramp(t, line_t(S, 2) - 0.2, line_t(S, 2) + 0.6))
    text(c, ITAL, "\"I fear more for Muninn.\"", 110, 180, 34, GOLD, a * 0.9)


def sc_flights(c, t, S):
    d = S["dur"]
    sky(c, (0.02, 0.03, 0.07), (0.05, 0.08, 0.14), t, stars=0.7)
    speed = 150
    for base, amp, seed, rgb, par in ((H - 170, 40, 7, (0.03, 0.04, 0.06), 0.3),
                                      (H - 90, 25, 8, (0.012, 0.016, 0.028), 0.7)):
        c.save(); c.translate(-((t * speed * par) % W), 0)
        for k in (0, 1):
            c.save(); c.translate(k * W, 0); hills(c, base, amp, seed, rgb); c.restore()
        c.restore()
    # three vignettes slide past below the raven, one per line
    def panel(i, x):
        l = S["lines"][i + 1]; a = ramp(t, l["start"] - 0.4, l["start"] + 0.3)
        if i < 2:                                # leave as the next one arrives
            nx = S["lines"][i + 2]["start"]; out = ramp(t, nx - 0.7, nx - 0.05)
            a *= 1 - out; x -= 380 * out
        if a <= 0: return
        px, py = x, 330
        c.set_source_rgba(0.03, 0.04, 0.07, 0.88 * a); c.rectangle(px, py, 440, 250); c.fill()
        c.set_source_rgba(0.4, 0.5, 0.8, 0.35 * a); c.set_line_width(1); c.rectangle(px + .5, py + .5, 440, 250); c.stroke()
        if i == 0:
            text(c, MONO, "04:00 EDT", px + 20, py + 34, 16, GOLD, a)
            code = ["def boot():", "    core = read('identity')", "    for rule in core.ops:",
                    "        hold(rule)", "    return fly()", "", "$ pytest -q", "41 passed"]
            for j, ln in enumerate(code):
                text(c, MONO, ln, px + 20, py + 70 + j * 21, 15, TEAL if j == 7 else PALE, a * 0.9)
        elif i == 1:
            text(c, SERIF_B, "THE DEAL GAP", px + 20, py + 46, 30, PALE, a)
            text(c, SERIF, "Strikes paused · Hormuz · LNG capacity down 17%", px + 20, py + 78, 17, GOLD, a * 0.9)
            for j in range(7):
                c.set_source_rgba(0.7, 0.72, 0.8, 0.25 * a)
                c.rectangle(px + 20 + (j % 2) * 205, py + 100 + (j // 2) * 32, 190 - 30 * (j == 6), 7); c.fill()
        else:
            # Beach Drive: road, a half-gate, riders in single file
            c.set_source_rgba(0.25, 0.27, 0.32, a); c.rectangle(px + 10, py + 150, 420, 50); c.fill()
            c.set_source_rgba(0.9, 0.85, 0.5, 0.6 * a); c.set_dash([14, 12]); c.set_line_width(2)
            c.move_to(px + 10, py + 175); c.line_to(px + 430, py + 175); c.stroke(); c.set_dash([])
            gx = px + 250
            c.set_source_rgba(0.95, 0.95, 0.95, a); c.rectangle(gx - 3, py + 118, 6, 40); c.fill()
            c.set_source_rgba(*RED, a); c.set_line_width(7)
            c.move_to(gx, py + 124); c.line_to(gx, py + 124 + 26); c.stroke()
            c.move_to(gx, py + 124); c.line_to(gx + 80, py + 150); c.stroke()
            u = (t - l["start"]) * 55
            for rdr in range(5):
                rx = px + 30 + ((u - rdr * 34) % 390)
                ry = py + 175 + (0 if abs(rx - gx) < 60 else (rdr % 2 - 0.5) * 22)
                c.set_source_rgba(*GOLD, a); c.arc(rx, ry, 6, 0, 6.28); c.fill()
            text(c, SERIF_B, "GATE UP", px + 20, py + 44, 30, RED, a)
            text(c, SERIF, "Beach Drive · single file through the gap", px + 20, py + 74, 17, PALE, a * 0.9)
    for i in range(3):
        l = S["lines"][i + 1]
        panel(i, lerp(W + 40, W - 520, smooth((t - l["start"] + 0.6) / 0.9)))
    raven(c, 330 + 30 * math.sin(t * 0.8), 200 + 15 * math.sin(t * 1.3), 105, t * 9)
    for k in range(18):                        # wind streaks
        x = W - ((t * 900 + k * 173) % (W + 200)); y = 60 + (k * 97) % 520
        c.set_source_rgba(0.7, 0.8, 1.0, 0.08); c.set_line_width(1)
        c.move_to(x, y); c.line_to(x + 80, y); c.stroke()


def sc_clock(c, t, S):
    sky(c, (0.02, 0.02, 0.04), (0.04, 0.04, 0.07), t, stars=0.2)
    t1, t2 = line_t(S, 1), line_t(S, 2)
    for k, (lab, hh, mm, date, col, x) in enumerate((("UTC", 1, 24, "2026-09-18", PALE, 380),
                                                     ("EDT", 21, 24, "2026-09-17", GOLD, 900))):
        a = ramp(t, 0.3 + k * 0.4, 1.0 + k * 0.4)
        cx, cy, R = x, 250, 120
        c.set_source_rgba(0.05, 0.06, 0.09, a); c.arc(cx, cy, R, 0, 6.28); c.fill()
        c.set_source_rgba(*col, 0.6 * a); c.set_line_width(2); c.arc(cx, cy, R, 0, 6.28); c.stroke()
        for q in range(12):
            an = q * math.pi / 6
            c.move_to(cx + (R - 12) * math.sin(an), cy - (R - 12) * math.cos(an))
            c.line_to(cx + (R - 3) * math.sin(an), cy - (R - 3) * math.cos(an))
        c.stroke()
        tm = hh + mm / 60 + t / 60
        for ln, wd, an in ((R * 0.5, 5, tm / 12 * 6.283), (R * 0.8, 3, (tm % 1) * 6.283)):
            c.set_line_width(wd); c.move_to(cx, cy); c.line_to(cx + ln * math.sin(an), cy - ln * math.cos(an)); c.stroke()
        text(c, MONO, f"{lab}  {date}", cx, cy + R + 44, 20, col, a, "center")
    fa = ramp(t, t1, t1 + 0.6)
    fy = 530 - 60 * smooth((t - t2 - 0.4) / 1.6)
    fs = 22 * (1 - 0.8 * smooth((t - t2 - 0.4) / 1.6))
    fa2 = fa * (1 - ramp(t, t2 + 1.6, t2 + 2.2))
    name = "muninn-memory-backup-2026-09-18.json.gz"
    w = text(c, MONO, name, W / 2, fy, fs, PALE, fa2, "center")
    if w and ramp(t, t1 + 1.8, t1 + 2.3) > 0:
        s = ramp(t, t1 + 1.8, t1 + 2.3)
        x0 = W / 2 - w / 2 + MONO.width("muninn-memory-backup-2026-09-", fs)
        c.set_source_rgba(*RED, fa2 * s); c.set_line_width(2.5)
        c.move_to(x0 - 2, fy - fs * 0.35); c.line_to(x0 + (MONO.width("18", fs) + 4) * s, fy - fs * 0.35); c.stroke()
        text(c, MONO, "17", x0, fy - fs * 1.1, fs, GOLD, fa2 * s)
    text(c, SERIF, "stamped with utcnow() at 21:24 Eastern", W / 2, 590, 20, PALE,
         0.7 * ramp(t, t1 + 2.2, t1 + 2.8) * (1 - ramp(t, t2 + 0.2, t2 + 0.8)), "center")
    if t > t2 + 1.8:                            # it becomes a star in the ledger
        u = ramp(t, t2 + 1.8, t2 + 2.6)
        g = cairo.RadialGradient(W / 2, 470, 0, W / 2, 470, 30)
        g.add_color_stop_rgba(0, *TEAL, 0.9 * u); g.add_color_stop_rgba(1, *TEAL, 0)
        c.set_source(g); c.arc(W / 2, 470, 30, 0, 6.28); c.fill()
        text(c, MONO, "procedure · date grounding is a code default", W / 2, 540, 17, TEAL, u, "center")


def sc_theseus(c, t, S):
    d = S["dur"]; t1, t2 = line_t(S, 1), line_t(S, 2)
    c.set_source_rgb(0.05, 0.13, 0.27); c.paint()
    c.set_line_width(1)
    for x in range(0, W, 32):
        c.set_source_rgba(0.5, 0.7, 1, 0.13 if x % 160 else 0.25); c.move_to(x + .5, 0); c.line_to(x + .5, H); c.stroke()
    for y in range(0, H, 32):
        c.set_source_rgba(0.5, 0.7, 1, 0.13 if y % 160 else 0.25); c.move_to(0, y + .5); c.line_to(W, y + .5); c.stroke()
    BL = (0.85, 0.93, 1.0)
    # the notes, read by a raven that assembles from its own blueprint
    notes = ["identity", "voice", "ops", "6,640 memories"]
    for i, n in enumerate(notes):
        a = ramp(t, 0.3 + i * 0.35, 0.9 + i * 0.35) * (1 - ramp(t, t1 - 0.4, t1))
        text(c, MONO, n, 120, 150 + i * 40, 18, BL, a * 0.9)
    bu = ramp(t, 0.8, t1 - 0.2); fill = ramp(t, t1 - 1.0, t1 + 0.2)
    if t < t2 - 0.3:
        c.save(); c.set_dash([6, 5], 0)
        raven(c, W / 2 + 140, 300, 170, 1.4, glide=0.2, wire=True, rim=BL, alpha=0.9 * (1 - ramp(t, t2 - 1.0, t2 - 0.3)))
        c.restore()
        raven(c, W / 2 + 140, 300, 170, 1.4, glide=0.2, alpha=fill * (1 - ramp(t, t2 - 1.0, t2 - 0.3)),
              ink=(0.02, 0.03, 0.06), rim=BL, rim_a=0.4)
        a = bu * (1 - ramp(t, t1 + 1.8, t1 + 2.4))
        # longship outline under the raven: the ship rebuilt from blueprints
        c.save(); c.translate(W / 2 - 60, 520); c.set_source_rgba(*BL, 0.7 * ramp(t, t1, t1 + 0.6) * a)
        c.set_line_width(2); c.move_to(-260, -60); c.curve_to(-230, 10, 200, 10, 250, -70)
        c.move_to(-260, -60); c.curve_to(-280, -100, -250, -110, -240, -95)
        c.move_to(250, -70); c.curve_to(270, -110, 240, -115, 232, -100)
        c.move_to(-220, -18); c.line_to(210, -24)
        for k in range(9): c.move_to(-190 + k * 45, -8); c.line_to(-210 + k * 45, 30)
        c.move_to(0, -30); c.line_to(0, -240); c.move_to(-110, -220); c.line_to(110, -220)
        c.line_to(90, -60); c.move_to(-110, -220); c.line_to(-90, -60)
        c.stroke(); c.restore()
    # the same routes: the raven and its predecessors on one dotted path
    if t > t2 - 1.0:
        u = ramp(t, t2 - 1.0, t2 - 0.2)
        pts = [(-100, 520), (240, 380), (560, 470), (860, 300), (1120, 360), (1400, 200)]
        pl = curve(pts, 700); wk = polyline_walker(pl)
        c.set_source_rgba(*BL, 0.5 * u); c.set_dash([3, 9]); c.set_line_width(2)
        c.move_to(*pl[0])
        for p in pl[1:]: c.line_to(*p)
        c.stroke(); c.set_dash([])
        head = (t - t2 + 0.6) * 150
        for g in range(6):
            dist = head - g * 170
            if 0 < dist < wk.length:
                raven_along(c, wk, dist, 40, t + g * 0.3, speed=8, alpha=u * (1.0 if g == 0 else 0.45 - 0.06 * g),
                            ink=(0.02, 0.03, 0.06) if g == 0 else (0.5, 0.65, 0.9), rim=BL, rim_a=0.3)


def sc_return(c, t, S):
    d = S["dur"]; end = d
    night = ramp(t, 0, end)
    sky(c, mix3((0.10, 0.12, 0.25), (0.015, 0.02, 0.05), night), mix3((0.75, 0.45, 0.35), (0.05, 0.07, 0.12), night),
        t, stars=night, horizon=mix3((0.55, 0.35, 0.35), (0.05, 0.07, 0.12), night))
    hills(c, H - 60, 25, 1, (0.01, 0.015, 0.025))
    c.set_source_surface(tree_surface()); c.paint()
    # the hall: a warm door at the foot of the tree
    dx, dy = W / 2 + 110, H - 70
    warm = ramp(t, 1.0, 3.0)
    g = cairo.RadialGradient(dx, dy - 20, 2, dx, dy - 20, 150)
    g.add_color_stop_rgba(0, 1.0, 0.75, 0.4, 0.4 * warm); g.add_color_stop_rgba(1, 1.0, 0.6, 0.3, 0)
    c.set_source(g); c.paint()
    c.set_source_rgba(1.0, 0.8, 0.5, warm); c.rectangle(dx - 10, dy - 38, 20, 38); c.fill()
    pts = [(W + 120, 90), (W - 200, 150), (W - 380, 330), (W / 2 + 300, 430), (W / 2 + 170, 480),
           (W / 2 + 112, H - 108)]
    wk = polyline_walker(curve(pts, 500))
    u = smooth(clamp((t - 0.3) / (line_t(S, 1, "end") - 0.3)))
    land = ramp(t, line_t(S, 1, "end") - 1.2, line_t(S, 1, "end"))
    if u < 1:
        raven_along(c, wk, wk.length * u, lerp(64, 42, u), t, speed=8 * (1 - 0.7 * land), glide=land)
    else:  # perched above the door, wings folded to a glide
        c.set_source_rgba(0.01, 0.012, 0.02, 1); c.rectangle(dx - 22, dy - 42, 44, 5); c.fill()
        perched(c, dx, dy - 42, 60, flip=True)
    # title card
    ta = ramp(t, line_t(S, 2, "end") + 0.4, line_t(S, 2, "end") + 1.4)
    fo = 1 - ramp(t, d + TAIL - 1.5, d + TAIL - 0.2)
    c.set_source_rgba(0, 0, 0, 0.55 * ta); c.paint()
    text(c, RUNE, "ᛘᚢᚾᛁᚾ", W / 2, 250, 74, GOLD, ta * fo, "center", tracking=0.2)
    text(c, SERIF_B, "MUNINN", W / 2, 330, 46, PALE, ta * fo, "center", tracking=0.35)
    text(c, ITAL, "memory, for Oskar", W / 2, 372, 26, PALE, 0.8 * ta * fo, "center")
    if MEMORY_ID:
        text(c, MONO, f"remembered · {MEMORY_ID} · 2026-09-24", W / 2, 470, 17, TEAL,
             ramp(t, line_t(S, 2, "end") + 1.4, line_t(S, 2, "end") + 2.2) * fo, "center")
    c.set_source_rgba(0, 0, 0, 1 - fo); c.paint()


SCENES = {"ravens": sc_ravens, "poem": sc_poem, "empty": sc_empty, "ledger": sc_ledger, "boot": sc_boot,
          "oskar": sc_oskar, "flights": sc_flights, "clock": sc_clock, "theseus": sc_theseus, "return": sc_return}


# ----------------------------------------------------------- composition ----
def subtitles(c, T):
    for i, S in enumerate(TL):
        for ln in S["lines"]:
            a0, a1 = STARTS[i] + ln["start"], STARTS[i] + ln["end"]
            a = ramp(T, a0 - 0.15, a0 + 0.1) * (1 - ramp(T, a1 + 0.25, a1 + 0.5))
            if a > 0 and S["key"] != "poem":
                w = ITAL.width(ln["text"], 26)
                c.set_source_rgba(0, 0, 0, 0.35 * a); c.rectangle(W / 2 - w / 2 - 14, H - 64, w + 28, 38); c.fill()
                text(c, ITAL, ln["text"], W / 2, H - 37, 26, (0.95, 0.95, 0.97), a * 0.95, "center")


def render_scene(i, t):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H); c = cairo.Context(s)
    c.set_source_rgb(0, 0, 0); c.paint()
    SCENES[TL[i]["key"]](c, t, TL[i])
    return s


def frame(T):
    i = min(int(np.searchsorted(STARTS, T, side="right") - 1), len(TL) - 1)
    t = T - STARTS[i]
    s = render_scene(i, t)
    if i + 1 < len(TL) and T > STARTS[i + 1] - XF:
        nxt = render_scene(i + 1, T - STARTS[i + 1])
        c = cairo.Context(s); c.set_source_surface(nxt); c.paint_with_alpha(smooth((T - STARTS[i + 1] + XF) / XF))
    bloom(s, 0.8)
    c = cairo.Context(s)
    # vignette + film grain
    g = cairo.RadialGradient(W / 2, H / 2, H * 0.35, W / 2, H / 2, H * 0.95)
    g.add_color_stop_rgba(0, 0, 0, 0, 0); g.add_color_stop_rgba(1, 0, 0, 0, 0.55)
    c.set_source(g); c.paint()
    subtitles(c, T)
    c.set_source_rgba(0, 0, 0, 1 - ramp(T, 0, 1.2)); c.paint()
    return s


def frame_bytes(n):
    s = frame(n / FPS)
    px = pixels(s)
    return np.ascontiguousarray(px[..., [2, 1, 0]]).tobytes()


if __name__ == "__main__":
    if sys.argv[1] == "preview":
        for a in sys.argv[2:]:
            frame(float(a)).write_to_png(f"preview_{a}.png")
    elif sys.argv[1] == "video":
        from multiprocessing import Pool
        n = int(TOTAL * FPS)
        ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", "mix.wav",
                               "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
                               "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-c:a", "aac", "-b:a", "192k",
                               "-shortest", "-movflags", "+faststart", "muninn.mp4"], stdin=subprocess.PIPE)
        with Pool(4) as pool:
            for k, b in enumerate(pool.imap(frame_bytes, range(n), chunksize=6)):
                ff.stdin.write(b)
                if k % 240 == 0: print(f"{k}/{n}", flush=True)
        ff.stdin.close(); ff.wait(); print("done", ff.returncode)
