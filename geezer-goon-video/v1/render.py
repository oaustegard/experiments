"""Geezer Goon — animated music video renderer (pycairo -> ffmpeg).

usage: python3 render.py still T [T ...]        -> still_<T>.png
       python3 render.py chunk F0 F1 out.mp4    -> frames [F0,F1) encoded
"""
import cairo, math, json, re, sys, subprocess
import numpy as np

W, H, FPS = 1920, 1080, 30
A = json.load(open('analysis.json'))
DUR = A['dur']; NF = int(DUR * FPS)
RMS = np.array(A['rms']); ONS = np.array(A['onset']); BEATS = np.array(A['beats'])

# ---------------------------------------------------------------- lyrics
def parse_srt(path):
    txt = open(path, encoding='utf-8').read().strip()
    out = []
    for blk in re.split(r'\n\s*\n', txt):
        ls = blk.strip().split('\n')
        m = re.match(r'(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)', ls[1])
        g = [int(x) for x in m.groups()]
        t0 = g[0]*3600 + g[1]*60 + g[2] + g[3]/1000
        t1 = g[4]*3600 + g[5]*60 + g[6] + g[7]/1000
        out.append((t0, t1, ' '.join(ls[2:]).strip()))
    return out

FIX = {'Kareem': 'Karim', 'lede': 'lead', 'Gro’venor': 'Grosvenor'}
LINES, SECTIONS = [], []
for t0, t1, s in parse_srt('subs.srt'):
    for a, b in FIX.items():
        s = s.replace(a, b)
    if s.startswith('['):
        SECTIONS.append((t0, s.strip('[]').split(' —')[0].upper()))
    else:
        LINES.append([t0, t1, s])

def line_words(ln):
    t0, t1, s = ln
    words = s.split()
    win = min((t1 - t0) * 0.9, 0.072 * len(s) + 0.3)
    wts = np.array([len(w) + 2.0 for w in words]); wts /= wts.sum()
    starts = t0 + np.concatenate([[0], np.cumsum(wts)[:-1]]) * win
    return words, starts, win

# ---------------------------------------------------------------- helpers
def smooth(x):
    x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)

def keyed(keys, t):
    """piecewise smoothstep interpolation over (t, v) keys"""
    if t <= keys[0][0]: return keys[0][1]
    for (ta, va), (tb, vb) in zip(keys, keys[1:]):
        if t <= tb:
            return va + (vb - va) * smooth((t - ta) / (tb - ta)) if tb > ta else vb
    return keys[-1][1]

def rnd(i, salt=0):
    v = math.sin(i * 12.9898 + salt * 78.233) * 43758.5453
    return v - math.floor(v)

def lerp(a, b, k): return a + (b - a) * k
def lerpc(c1, c2, k): return tuple(lerp(a, b, k) for a, b in zip(c1, c2))
def hexc(h): h = h.lstrip('#'); return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))

def ease_pop(x):  # 0..1 -> overshoot pop
    x = min(1, max(0, x)); return 1 + 2.2 * (x - 1) ** 3 + 1.2 * (x - 1) ** 2 if x < 1 else 1.0

def rrect(ctx, x, y, w, h, r):
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi/2, 0); ctx.arc(x + w - r, y + h - r, r, 0, math.pi/2)
    ctx.arc(x + r, y + h - r, r, math.pi/2, math.pi); ctx.arc(x + r, y + r, r, math.pi, 1.5*math.pi)
    ctx.close_path()

# ---------------------------------------------------------------- timeline
SPEED = [(0, 9), (19, 9), (24, 19), (46, 20), (47.5, 15), (55, 15), (56.5, 22), (81, 22),
         (83.5, 25), (88, 16), (91, 20), (95, 17), (101, 17), (102.5, 14), (110, 19),
         (113, 11), (124, 11), (128, 19), (131, 22), (154, 22), (158, 23.5), (163, 25),
         (167, 26.5), (184, 27), (187, 31), (193, 33), (196, 27), (199, 22), (224, 21),
         (236, 16), (244, 11), (249.5, 7)]
PX_PER_MPH = 40.0
tt = np.arange(NF + 2) / FPS
spd = np.array([keyed(SPEED, t) for t in tt])
CAM = np.concatenate([[0], np.cumsum(spd[:-1] * PX_PER_MPH / FPS)])
def cam(t): return float(np.interp(t, tt, CAM))
MILES = 25.2 * CAM / CAM[-1]
ELEV_K = [(0, 280), (46, 272), (56, 250), (81, 222), (95, 205), (101, 232), (111, 212),
          (125, 305), (131, 292), (154, 258), (170, 270), (184, 302), (198, 286), (249.5, 280)]
elev = np.array([keyed(ELEV_K, t) for t in tt])
climb = np.concatenate([[0], np.cumsum(np.clip(np.diff(elev), 0, None))]); climb *= 600 / climb[-1]

def ride_clock(t):
    m = 0 if t < 19 else min(77, (t - 19) / (236 - 19) * 77)
    tot = 6 * 60 + 15 + m
    return f"{int(tot//60)}:{int(tot%60):02d}"

def section_at(t):
    cur = SECTIONS[0]
    for s in SECTIONS:
        if s[0] <= t + 1e-6: cur = s
    return cur

def pulse(t):
    i = np.searchsorted(BEATS, t) - 1
    if i < 0: return 0.0
    return math.exp(-(t - BEATS[i]) / 0.13)

def energy(t):
    i = min(NF - 1, max(0, int(t * FPS))); return float(RMS[i])

def hill_angle(t):  # world tilt in degrees (negative = road rises to the right)
    return keyed([(110.5, 0), (113, -5.5), (124.3, -5.5), (127, 1.5), (129.5, 0)], t)

# ---------------------------------------------------------------- time of day
SKY = [  # p, top, horizon, light, fog
    (0.00, '#5b7fbf', '#ffc98a', 1.00),
    (0.35, '#4f5fa8', '#ffae7a', 0.92),
    (0.60, '#3a3a86', '#f07f73', 0.72),
    (0.78, '#1f2356', '#9b5a7e', 0.48),
    (0.90, '#0e1233', '#3f2f58', 0.36),
    (1.00, '#05071a', '#1b1733', 0.30)]
def tod(t):
    p = t / DUR
    for a, b in zip(SKY, SKY[1:]):
        if p <= b[0]:
            k = (p - a[0]) / (b[0] - a[0])
            return (lerpc(hexc(a[1]), hexc(b[1]), k), lerpc(hexc(a[2]), hexc(b[2]), k), lerp(a[3], b[3], k))
    s = SKY[-1]; return hexc(s[1]), hexc(s[2]), s[3]

class Pal:
    def __init__(self, t):
        self.top, self.hor, self.light = tod(t)
        self.night = smooth((t / DUR - 0.62) / 0.3)
    def c(self, col, depth=0.0):
        """shade a base colour by daylight and push toward horizon haze with depth"""
        if isinstance(col, str): col = hexc(col)
        L = self.light
        base = tuple(v * L for v in col)
        base = lerpc(base, tuple(v * 0.35 for v in self.top), self.night * 0.25)
        return lerpc(base, lerpc(self.hor, self.top, 0.35), depth * (0.55 - 0.25 * self.night))

# ---------------------------------------------------------------- riders
C = hexc
RIDERS = [  # name, lane, jersey, accent, skin, helmet, scale, bike, extras
    dict(id='me', lane=0, jersey='#1d6fb8', acc='#ffd23f', skin='#e8b894', helm='#f4f4f4', s=1.0, bike='gravel'),
    dict(id='angelo', lane=0, jersey='#2a9d8f', acc='#e9c46a', skin='#c98f63', helm='#222222', s=1.0, bike='single', pack=True),
    dict(id='gary', lane=0, jersey='#f4a261', acc='#264653', skin='#f0c4a0', helm='#e63946', s=1.0, bike='road'),
    dict(id='phil', lane=0, jersey='#e63946', acc='#ffffff', skin='#e0ac86', helm='#ffffff', s=1.0, bike='road'),
    dict(id='hulk', lane=0, jersey='#3d3d4a', acc='#ffb000', skin='#d9a07a', helm='#ffb000', s=1.13, bike='road', letter='H'),
    dict(id='karim', lane=0, jersey='#ffd23f', acc='#1b1b1b', skin='#b98056', helm='#1b1b1b', s=1.0, bike='road'),
]
FAR_J = ['#8e44ad', '#16a085', '#d35400', '#2c3e50', '#c0392b', '#f1c40f', '#27ae60', '#e84393', '#0984e3']
for i, j in enumerate(FAR_J):
    RIDERS.append(dict(id=f'far{i}', lane=1, jersey=j, acc='#ffffff', skin=['#e8b894', '#c98f63', '#8d5a3b'][i % 3],
                       helm=['#ffffff', '#222222', '#e63946'][i % 3], s=0.96, bike='road'))

BASEX = dict(me=0.10, angelo=0.25, gary=0.40, phil=0.55, hulk=0.70, karim=0.85)
XK = {
    'phil':   [(0, .55), (39.6, .55), (41.6, .95), (74, .95), (78, .55), (184, .55), (187.5, .70), (195, .64), (198, .55), (237, .55), (246, 1.3)],
    'karim':  [(0, .85), (39.6, .85), (41.6, .79), (74, .79), (78, .85), (186.3, .85), (188, 1.00), (195, .96), (198, .85), (237, .85), (245, 1.35)],
    'hulk':   [(0, .70), (39.6, .70), (41.6, .64), (74, .64), (78, .70), (184, .70), (188, .83), (195, .79), (198, .70), (237, .70), (246, 1.35)],
    'angelo': [(0, .25), (121.5, .25), (126.5, .97), (135, .97), (140, .25), (184, .25), (188, .36), (195, .33), (198, .25), (237, .25), (247, 1.3)],
    'gary':   [(0, .40), (184, .40), (188, .52), (195, .50), (198, .40), (237, .40), (247, 1.3)],
    'me':     [(0, .10), (164, .10), (168, .05), (180, .05), (184, .10), (188, .22), (190, .30), (195, .35), (198, .10), (237, .10), (248, 1.3)],
}
for i in range(len(FAR_J)):
    b = 0.03 + 0.118 * i
    XK[f'far{i}'] = [(0, b), (237, b), (247.5, b + 1.15)]

def rider_x(r, t):
    return keyed(XK[r['id']], t) * W

TAGS = [  # t0, t1, rider, label, sublabel
    (14.7, 22.5, 'karim', 'KARIM', 'ON THE FRONT'),
    (32.9, 39.5, 'hulk', 'THE HULK', "BACK & STRONG"),
    (39.6, 42.7, 'phil', 'PHIL', 'TAKING A PULL'),
    (42.7, 46.2, 'gary', 'GARY', 'NOT MOVING'),
    (114.4, 127.0, 'angelo', 'ANGELO', 'ONE GEAR'),
    (149.6, 156.0, 'me', 'ME', 'HANGING ON'),
    (192.1, 198.5, 'me', 'ME', 'FAT TIRES'),
]

def ik(ax, ay, bx, by, L1, L2, sign):
    dx, dy = bx - ax, by - ay
    d = max(1e-6, min(math.hypot(dx, dy), (L1 + L2) * 0.999))
    a = math.acos(max(-1, min(1, (L1*L1 + d*d - L2*L2) / (2 * L1 * d))))
    base = math.atan2(dy, dx) + sign * a
    return ax + L1 * math.cos(base), ay + L1 * math.sin(base)

def seg(ctx, pts, width, col, cap=cairo.LINE_CAP_ROUND):
    ctx.set_source_rgb(*col); ctx.set_line_width(width); ctx.set_line_cap(cap)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.move_to(*pts[0])
    for p in pts[1:]: ctx.line_to(*p)
    ctx.stroke()

def glow(ctx, x, y, r, col, a):
    g = cairo.RadialGradient(x, y, 0, x, y, r)
    g.add_color_stop_rgba(0, *col, a); g.add_color_stop_rgba(1, *col, 0)
    ctx.set_source(g); ctx.arc(x, y, r, 0, 2*math.pi); ctx.fill()

def draw_rider(ctx, P, r, gx, gy, s, crank, wang, depth, t, lights):
    """gx, gy = ground contact point under bottom bracket; s = px per metre"""
    ctx.save(); ctx.translate(gx, gy); ctx.scale(s, s)
    R = 0.34; ay = -R
    RA, FA, BB = (-0.42, ay), (0.58, ay), (0.0, ay + 0.07)
    ST, HT, HB = (-0.17, ay - 0.55), (0.47, ay - 0.52), (0.51, ay - 0.36)
    HIP, SH = (-0.19, ay - 0.62), (0.27, ay - 0.98)
    HAND = (0.53, ay - 0.50)
    k = lambda c: P.c(c, depth)
    jer, acc, skin, shorts = k(r['jersey']), k(r['acc']), k(r['skin']), k('#17171c')
    frame = k(r['jersey'] if r['bike'] != 'gravel' else '#556b2f')
    tyre_w = 0.075 if r['bike'] == 'gravel' else 0.042
    CR = 0.17
    # far leg
    pa = crank + math.pi
    px, py = BB[0] + CR * math.cos(pa), BB[1] + CR * math.sin(pa)
    kx, ky = ik(*HIP, px, py, 0.45, 0.46, -1)
    seg(ctx, [HIP, (kx, ky), (px, py)], 0.10, lerpc(shorts, (0, 0, 0), 0.3))
    seg(ctx, [(px - 0.06, py), (px + 0.09, py)], 0.06, lerpc(shorts, (0, 0, 0), 0.3))
    # wheels
    for cx, cy in (RA, FA):
        ctx.set_source_rgb(*k('#111114')); ctx.set_line_width(tyre_w)
        ctx.arc(cx, cy, R - tyre_w / 2, 0, 2*math.pi); ctx.stroke()
        ctx.set_source_rgba(*k('#b8bcc6'), 0.9); ctx.set_line_width(0.012)
        ctx.arc(cx, cy, R - tyre_w - 0.012, 0, 2*math.pi); ctx.stroke()
        ctx.set_source_rgba(*k('#d0d4dc'), 0.55); ctx.set_line_width(0.006)
        for n in range(8):
            a = wang + n * math.pi / 4
            ctx.move_to(cx, cy); ctx.line_to(cx + (R - tyre_w) * math.cos(a), cy + (R - tyre_w) * math.sin(a))
        ctx.stroke()
        ctx.set_source_rgb(*k('#888888')); ctx.arc(cx, cy, 0.025, 0, 2*math.pi); ctx.fill()
    # frame
    fw = 0.035
    seg(ctx, [RA, BB, ST, RA], fw, frame)
    seg(ctx, [ST, HT, (HT[0] + 0.03, ay - 0.40), BB], fw, frame)
    seg(ctx, [(HT[0] + 0.03, ay - 0.40), FA], 0.028, k('#2a2a2a'))
    seg(ctx, [(HT[0], HT[1]), (HT[0] + 0.02, HT[1] - 0.05), (0.57, ay - 0.57)], 0.025, k('#2a2a2a'))
    if r['bike'] == 'gravel':  # flared drops
        seg(ctx, [(0.57, ay - 0.57), (0.63, ay - 0.52), (0.60, ay - 0.43), (0.55, ay - 0.44)], 0.025, k('#2a2a2a'))
    else:
        seg(ctx, [(0.57, ay - 0.57), (0.64, ay - 0.53), (0.62, ay - 0.45), (0.56, ay - 0.45)], 0.025, k('#2a2a2a'))
    seg(ctx, [(ST[0] + 0.02, ST[1]), (ST[0] - 0.02, ST[1] - 0.08)], 0.028, k('#2a2a2a'))
    seg(ctx, [(ST[0] - 0.13, ST[1] - 0.09), (ST[0] + 0.07, ST[1] - 0.085)], 0.04, k('#111111'))  # saddle
    ctx.set_source_rgb(*k('#9aa0aa')); ctx.set_line_width(0.012)
    ctx.arc(BB[0], BB[1], 0.10, 0, 2*math.pi); ctx.stroke()
    if r['bike'] == 'single':
        ctx.set_source_rgba(*k('#9aa0aa'), 0.8); ctx.set_line_width(0.008)
        ctx.move_to(BB[0], BB[1] - 0.10); ctx.line_to(RA[0], RA[1] - 0.05)
        ctx.move_to(BB[0], BB[1] + 0.10); ctx.line_to(RA[0], RA[1] + 0.05); ctx.stroke()
    # near leg
    px, py = BB[0] + CR * math.cos(crank), BB[1] + CR * math.sin(crank)
    seg(ctx, [BB, (px, py)], 0.03, k('#555a60'))
    kx, ky = ik(*HIP, px, py, 0.45, 0.46, -1)
    seg(ctx, [HIP, (kx, ky)], 0.13, shorts)
    seg(ctx, [(kx, ky), (px, py)], 0.10, shorts)
    seg(ctx, [(lerp(kx, px, 0.45), lerp(ky, py, 0.45)), (px, py)], 0.085, skin)
    seg(ctx, [(px - 0.07, py - 0.01), (px + 0.10, py)], 0.065, k('#f4f4f4'))
    # torso
    ctx.set_source_rgb(*jer)
    ctx.move_to(HIP[0] - 0.08, HIP[1] + 0.05)
    ctx.curve_to(HIP[0] - 0.05, HIP[1] - 0.25, SH[0] - 0.25, SH[1] - 0.05, SH[0] + 0.02, SH[1] - 0.06)
    ctx.line_to(SH[0] + 0.07, SH[1] + 0.08)
    ctx.curve_to(SH[0] - 0.15, SH[1] + 0.12, HIP[0] + 0.12, HIP[1] - 0.05, HIP[0] + 0.10, HIP[1] + 0.06)
    ctx.close_path(); ctx.fill()
    seg(ctx, [(HIP[0] - 0.02, HIP[1] - 0.12), (SH[0] - 0.02, SH[1] + 0.00)], 0.035, acc)  # stripe
    if r.get('letter'):
        ctx.save(); ctx.translate(0.02, ay - 0.80); ctx.rotate(-0.55)
        ctx.select_font_face('Anton'); ctx.set_font_size(0.17); ctx.set_source_rgb(*acc)
        ctx.move_to(-0.04, 0.06); ctx.show_text(r['letter']); ctx.restore()
    if r.get('pack'):
        ctx.save(); ctx.translate(-0.02, ay - 0.86); ctx.rotate(-0.62)
        rrect(ctx, -0.17, -0.14, 0.34, 0.20, 0.06); ctx.set_source_rgb(*k('#3b3b44')); ctx.fill()
        rrect(ctx, -0.10, -0.12, 0.2, 0.05, 0.02); ctx.set_source_rgb(*k('#5a5a66')); ctx.fill()
        ctx.restore()
    # arm
    ex, ey = ik(SH[0], SH[1], *HAND, 0.30, 0.30, 1)
    seg(ctx, [SH, (ex, ey)], 0.085, jer)
    seg(ctx, [(ex, ey), HAND], 0.07, skin)
    # head
    hx, hy = SH[0] + 0.14, SH[1] - 0.10
    ctx.set_source_rgb(*skin); ctx.arc(hx, hy, 0.095, 0, 2*math.pi); ctx.fill()
    ctx.set_source_rgb(*k(r['helm']))
    ctx.move_to(hx - 0.13, hy - 0.005)
    ctx.curve_to(hx - 0.12, hy - 0.16, hx + 0.08, hy - 0.17, hx + 0.12, hy - 0.03)
    ctx.line_to(hx - 0.13, hy - 0.005); ctx.fill()
    seg(ctx, [(hx + 0.02, hy + 0.005), (hx + 0.10, hy + 0.01)], 0.035, k('#101010'))  # shades
    if lights > 0:
        blink = 1.0 if (int(t * 3 + r['s'] * 7 + gx * 0.01) % 2 == 0) else 0.25
        ctx.restore()
        lx, ly = gx + (ST[0] - 0.03) * s, gy + (ST[1] + 0.07) * s
        glow(ctx, lx, ly, 0.35 * s, (1, 0.1, 0.1), 0.7 * lights * blink)
        ctx.set_source_rgba(1, 0.25, 0.25, lights * blink); ctx.arc(lx, ly, 0.03 * s, 0, 2*math.pi); ctx.fill()
        fx, fy = gx + 0.60 * s, gy + (ay - 0.55) * s
        g = cairo.LinearGradient(fx, fy, fx + 2.0 * s, fy)
        g.add_color_stop_rgba(0, 1, 0.97, 0.85, 0.16 * lights); g.add_color_stop_rgba(1, 1, 0.97, 0.85, 0)
        ctx.set_source(g); ctx.move_to(fx, fy); ctx.line_to(fx + 2.0 * s, fy - 0.1 * s)
        ctx.line_to(fx + 2.0 * s, fy + 0.55 * s); ctx.close_path(); ctx.fill()
        glow(ctx, fx, fy, 0.18 * s, (1, 1, 0.9), 0.8 * lights)
        return
    ctx.restore()

# ---------------------------------------------------------------- world objects
def at_screen(x_frac, t, f):
    """world x for an object on parallax layer f to be at screen x_frac*W at time t"""
    return f * cam(t) + x_frac * W

SIGNS = [  # time, text, x_frac at time
    (27.0, 'JONES BRIDGE RD', .62), (58.5, 'KENSINGTON PKWY', .62), (80.5, 'BEACH DR', .7),
    (82.5, 'EAST WEST HWY', .55), (95.5, 'RIDGE RD', .62), (98.0, 'ROSS DR', .62),
    (111.5, 'MORMON HILL', .55), (134.5, 'KENSINGTON PKWY', .62), (154.5, 'BEACH DR', .45),
    (157.0, 'FRANKLIN ST', .62), (163.5, 'GROSVENOR LN', .62), (167.0, 'KNOWLES AVE', .62),
    (184.5, 'WEXFORD DR', .55), (185.8, 'GARRETT PARK RD', .88), (218, 'KENSINGTON PKWY', .62)]
SIGN_F = 0.92
LIGHTS = [  # time centred, x_frac, state keys (t, 'r'|'y'|'g')
    (28.0, .72, [(0, 'r'), (29.6, 'g')]),
    (83.0, .62, [(0, 'g'), (82.2, 'y'), (84.6, 'r')]),
    (196.2, .66, [(0, 'y'), (195.4, 'r')]),
]
GATES = [46.3, 101.7, 224.3]
GRADE = (112.3, .66)
TEMPLE = (113.0, .62)

def light_state(keys, t):
    s = keys[0][1]
    for kt, ks in keys:
        if t >= kt: s = ks
    return s

def draw_sign(ctx, P, x, base, text):
    ctx.select_font_face('Bebas Neue'); ctx.set_font_size(34)
    w = ctx.text_extents(text).x_advance + 30
    seg(ctx, [(x, base), (x, base - 250)], 7, P.c('#8a8f96'), cairo.LINE_CAP_BUTT)
    rrect(ctx, x - w / 2, base - 262, w, 46, 6); ctx.set_source_rgb(*P.c('#1f6f43')); ctx.fill_preserve()
    ctx.set_source_rgb(*P.c('#f2f2f2')); ctx.set_line_width(2.5); ctx.stroke()
    ctx.move_to(x - w / 2 + 15, base - 226); ctx.show_text(text)

def draw_light(ctx, P, x, base, state, t):
    seg(ctx, [(x, base), (x, base - 300), (x - 120, base - 300)], 9, P.c('#3a3d42'), cairo.LINE_CAP_BUTT)
    hx, hy = x - 140, base - 300
    rrect(ctx, hx - 26, hy - 10, 52, 138, 8); ctx.set_source_rgb(*P.c('#222428')); ctx.fill()
    for i, (nm, col) in enumerate((('r', (1, .15, .12)), ('y', (1, .72, .05)), ('g', (.1, 1, .45)))):
        cy = hy + 20 + i * 40
        on = state == nm
        if on:
            flick = 0.85 + 0.15 * math.sin(t * 40) if nm == 'y' else 1
            glow(ctx, hx, cy, 70, col, 0.55 * flick)
        ctx.set_source_rgb(*(col if on else tuple(v * 0.22 for v in col)))
        ctx.arc(hx, cy, 14, 0, 2*math.pi); ctx.fill()

def draw_gate(ctx, P, x, base, t, tg):
    """Beach Drive gate seen from the side: a fixed half-gate from the far edge; the gap is mid-road"""
    seg(ctx, [(x, base), (x, base - 110)], 14, P.c('#d8d8d8'), cairo.LINE_CAP_BUTT)
    # arm reaching toward the centre line (foreshortened, lying across the far half of the road)
    ctx.save(); ctx.translate(x, base - 96)
    n = 6
    for i in range(n):
        ctx.set_source_rgb(*P.c('#e63946' if i % 2 == 0 else '#f7f7f7'))
        a0, a1 = i / n, (i + 1) / n
        ctx.move_to(a0 * 70, a0 * 72 - 9); ctx.line_to(a1 * 70, a1 * 72 - 9)
        ctx.line_to(a1 * 70, a1 * 72 + 9); ctx.line_to(a0 * 70, a0 * 72 + 9); ctx.close_path(); ctx.fill()
    ctx.restore()
    seg(ctx, [(x + 70, base - 24), (x + 70, base + 8)], 8, P.c('#d8d8d8'), cairo.LINE_CAP_BUTT)
    rrect(ctx, x - 74, base - 200, 148, 64, 6); ctx.set_source_rgb(*P.c('#f7f7f7')); ctx.fill()
    seg(ctx, [(x, base - 136), (x, base - 110)], 6, P.c('#d8d8d8'), cairo.LINE_CAP_BUTT)
    ctx.set_source_rgb(*P.c('#b3202d')); ctx.select_font_face('Bebas Neue'); ctx.set_font_size(24)
    ctx.move_to(x - 62, base - 175); ctx.show_text('ROAD CLOSED')
    ctx.move_to(x - 62, base - 149); ctx.show_text('TO MOTOR VEHICLES')

GATE_WIN = [(46.0, 55.8), (101.4, 111.0), (224.0, 232.5)]
MAP_COLS = [RJ for RJ in ['#1d6fb8', '#2a9d8f', '#f4a261', '#e63946', '#3d3d4a', '#ffd23f', '#8e44ad', '#16a085',
                            '#d35400', '#c0392b', '#f1c40f', '#27ae60', '#e84393', '#0984e3', '#ffffff']]

def draw_gate_map(ctx, t):
    """top-down cutaway: two half-gates close the road from each side, the pack threads the middle gap"""
    win = next((w for w in GATE_WIN if w[0] <= t < w[1] + 0.3), None)
    if not win: return
    k = ease_pop((t - win[0]) / 0.35) * (1 - smooth((t - win[1]) / 0.3))
    if k <= 0.01: return
    pw, ph = 860, 250
    ctx.save(); ctx.translate(W / 2, 40 + ph / 2); ctx.scale(k, k); ctx.translate(-pw / 2, -ph / 2)
    rrect(ctx, 0, 0, pw, ph, 20); ctx.set_source_rgba(0.04, 0.05, 0.08, 0.86); ctx.fill()
    ctx.select_font_face('Bebas Neue'); ctx.set_font_size(26); ctx.set_source_rgba(1, 1, 1, 0.7)
    ctx.move_to(24, 36); ctx.show_text('BEACH DRIVE  ·  FROM ABOVE')
    lab = 'SINGLE FILE THROUGH THE GAP'
    e = ctx.text_extents(lab); ctx.set_source_rgb(*hexc('#ffd23f')); ctx.move_to(pw - 24 - e.x_advance, 36); ctx.show_text(lab)
    rrect(ctx, 0, 0, pw, ph, 20); ctx.clip()
    ry0, ry1 = 60, 220; cy = (ry0 + ry1) / 2
    ctx.set_source_rgb(0.18, 0.30, 0.18); ctx.rectangle(0, 48, pw, ph); ctx.fill()
    ctx.set_source_rgb(0.26, 0.27, 0.30); ctx.rectangle(0, ry0, pw, ry1 - ry0); ctx.fill()
    ctx.set_source_rgb(0.9, 0.9, 0.85); ctx.rectangle(0, ry0 + 5, pw, 3); ctx.rectangle(0, ry1 - 8, pw, 3); ctx.fill()
    ctx.set_source_rgb(0.95, 0.76, 0.19)
    for xx in range(-40, pw + 40, 60): ctx.rectangle(xx - (t * 40) % 60, cy - 2, 30, 4)
    ctx.fill()
    gx, gap = pw * 0.52, 26
    for y0, y1 in ((ry0 - 6, cy - gap), (ry1 + 6, cy + gap)):
        n = 6
        for i in range(n):
            ctx.set_source_rgb(*hexc('#e63946' if i % 2 == 0 else '#f7f7f7'))
            a, b = lerp(y0, y1, i / n), lerp(y0, y1, (i + 1) / n)
            ctx.rectangle(gx - 7, min(a, b), 14, abs(b - a)); ctx.fill()
        ctx.set_source_rgb(0.85, 0.85, 0.85); ctx.arc(gx, y0, 10, 0, 2 * math.pi); ctx.fill()
    # 38 riders flowing left -> right
    L = pw + 200
    for i in range(38):
        base = (i * 23.0 + rnd(i, 71) * 14 + t * 150) % L - 100
        lat = (rnd(i, 72) - 0.5) * 100
        f = smooth(1 - abs(base - gx) / 260)
        y = cy + lat * (1 - f) * 0.9 + (rnd(i, 73) - 0.5) * 6
        ctx.set_source_rgb(*hexc(MAP_COLS[i % len(MAP_COLS)]))
        ctx.save(); ctx.translate(base, y)
        ctx.scale(1.6, 1); ctx.arc(0, 0, 6.5, 0, 2 * math.pi); ctx.restore(); ctx.fill()
    if t - win[0] < 2.0:
        a = 1 - smooth((t - win[0] - 1.4) / 0.6)
        ctx.select_font_face('Anton'); ctx.set_font_size(34)
        e = ctx.text_extents('GATE UP!')
        outlined(ctx, 'GATE UP!', gx + 24, ry0 + 34, hexc('#e63946'), stroke=(1, 1, 1), lw=6, a=a)
    ctx.restore()

def draw_walker(ctx, P, x, base, t, col, h=1.0, stroller=False, dog=False):
    s = 95 * h
    ph = t * 5.5
    ctx.save(); ctx.translate(x, base); ctx.scale(s, s)
    for sgn in (1, -1):
        a = 0.35 * math.sin(ph) * sgn
        seg(ctx, [(0, -0.9), (0.45 * math.sin(a), -0.9 + 0.9 * math.cos(a))], 0.12, P.c('#2b2d42'))
    seg(ctx, [(0, -0.9), (0.02, -1.5)], 0.30, P.c(col))
    seg(ctx, [(0, -1.45), (0.18 * math.sin(ph + 1), -1.0)], 0.09, P.c(col))
    ctx.set_source_rgb(*P.c('#d9a07a')); ctx.arc(0.03, -1.72, 0.14, 0, 2*math.pi); ctx.fill()
    if stroller:
        seg(ctx, [(0.1, -1.1), (0.45, -1.05), (0.55, -0.35)], 0.05, P.c('#333333'))
        ctx.set_source_rgb(*P.c('#5aa9e6'))
        ctx.move_to(0.5, -0.9); ctx.curve_to(0.55, -1.35, 1.2, -1.3, 1.15, -0.6); ctx.line_to(0.5, -0.6)
        ctx.close_path(); ctx.fill()
        for wx in (0.6, 1.05):
            ctx.set_source_rgb(*P.c('#222222')); ctx.arc(wx, -0.12, 0.12, 0, 2*math.pi); ctx.fill()
    ctx.restore()

def draw_dog(ctx, P, x, base, t, owner_hand):
    ph = t * 9
    ctx.save(); ctx.translate(x, base); s = 55; ctx.scale(s, s)
    ctx.set_source_rgb(*P.c('#8b5a2b'))
    ctx.save(); ctx.scale(1, 0.55); ctx.arc(0, -1.3, 0.55, 0, 2*math.pi); ctx.restore(); ctx.fill()
    ctx.arc(0.62, -0.95, 0.25, 0, 2*math.pi); ctx.fill()
    for lx, o in ((-0.35, 0), (-0.2, 1.6), (0.3, 0.8), (0.42, 2.4)):
        a = 0.45 * math.sin(ph + o)
        seg(ctx, [(lx, -0.55), (lx + 0.35 * math.sin(a), 0)], 0.1, P.c('#8b5a2b'))
    seg(ctx, [(-0.5, -0.8), (-0.8, -1.1 + 0.1 * math.sin(ph * 2))], 0.08, P.c('#8b5a2b'))
    ctx.restore()
    ctx.set_source_rgba(*P.c('#e63946'), 1); ctx.set_line_width(3)
    ctx.move_to(x + 0.62 * s, base - 0.95 * s)
    ctx.curve_to(x + 1.5 * s, base - 0.3 * s, owner_hand[0] - 60, owner_hand[1] + 40, *owner_hand)
    ctx.stroke()

def draw_temple(ctx, P, x, base, a):
    col = lerpc(P.c('#f3efe6', 0.55), (1, 1, 1), 0.2)
    ctx.save(); ctx.set_source_rgba(*col, a)
    ctx.rectangle(x - 150, base - 150, 300, 150); ctx.fill()
    for i, (dx, hh) in enumerate(((-120, 330), (0, 380), (120, 330))):
        ctx.rectangle(x + dx - 22, base - hh + 60, 44, hh - 60); ctx.fill()
        ctx.move_to(x + dx - 22, base - hh + 60); ctx.line_to(x + dx, base - hh - 40); ctx.line_to(x + dx + 22, base - hh + 60)
        ctx.close_path(); ctx.fill()
    ctx.set_source_rgba(*P.c('#e9c46a', 0.4), a)
    for dx, hh in ((-120, 330), (0, 380), (120, 330)):
        ctx.arc(x + dx, base - hh - 44, 7, 0, 2*math.pi); ctx.fill()
    ctx.restore()

# ---------------------------------------------------------------- scenery layers
HOR = 0.70 * H
ROAD_TOP, ROAD_BOT = 0.755 * H, 0.955 * H
LANE_FAR, LANE_NEAR = 0.815 * H, 0.925 * H

def draw_sky(ctx, P, t):
    g = cairo.LinearGradient(0, 0, 0, HOR)
    g.add_color_stop_rgb(0, *P.top); g.add_color_stop_rgb(1, *P.hor)
    ctx.set_source(g); ctx.rectangle(0, 0, W, H); ctx.fill()
    p = t / DUR
    # stars
    sa = smooth((p - 0.7) / 0.2)
    if sa > 0:
        for i in range(140):
            x, y = rnd(i, 1) * W, rnd(i, 2) * HOR * 0.85
            tw = 0.6 + 0.4 * math.sin(t * (1 + rnd(i, 3) * 3) + i)
            ctx.set_source_rgba(1, 1, 0.95, sa * tw * (0.3 + 0.7 * rnd(i, 4)))
            ctx.arc(x, y, 1 + 1.6 * rnd(i, 5), 0, 2*math.pi); ctx.fill()
    # sun
    sy = lerp(0.30 * H, HOR + 90, smooth(p / 0.64))
    if p < 0.7:
        sx = 0.80 * W
        glow(ctx, sx, sy, 420, (1, 0.8, 0.5), 0.45 * (1 - smooth((p - 0.5) / 0.2)))
        ctx.set_source_rgba(1, lerp(0.93, 0.55, smooth(p / 0.6)), lerp(0.7, 0.35, smooth(p / 0.6)), 1)
        ctx.arc(sx, sy, 78, 0, 2*math.pi); ctx.fill()
    # moon
    if p > 0.5:
        k = smooth((p - 0.5) / 0.5)
        mx, my = 0.16 * W + 0.05 * W * k, lerp(HOR + 100, 0.16 * H, k)
        glow(ctx, mx, my, 230, (0.85, 0.9, 1), 0.35 * k)
        ctx.set_source_rgb(0.96, 0.95, 0.88); ctx.arc(mx, my, 62, 0, 2*math.pi); ctx.fill()
        ctx.set_source_rgba(0.75, 0.74, 0.7, 0.5)
        for dx, dy, rr in ((-18, -12, 12), (15, 10, 9), (5, -25, 6), (-10, 20, 7)):
            ctx.arc(mx + dx, my + dy, rr, 0, 2*math.pi); ctx.fill()
    # clouds
    for i in range(6):
        cx = (rnd(i, 7) * (W + 800) - 0.02 * cam(t) - t * 6) % (W + 800) - 400
        cy = 0.08 * H + rnd(i, 8) * 0.30 * H
        col = lerpc(P.hor, (1, 1, 1), 0.35 * (1 - P.night))
        ctx.set_source_rgba(*col, 0.5 * (1 - 0.6 * P.night))
        for j in range(5):
            ctx.save(); ctx.translate(cx + j * 55 - 110, cy + math.sin(j * 1.7) * 12); ctx.scale(1.7, 0.7)
            ctx.arc(0, 0, 40 + 18 * rnd(i * 5 + j, 9), 0, 2*math.pi); ctx.restore(); ctx.fill()

def draw_hills(ctx, P, t):
    for layer, (f, base, amp, col, depth) in enumerate(((0.04, 0.62, 45, '#4a6b8a', 0.8), (0.10, 0.66, 40, '#3f6b4f', 0.6))):
        off = f * cam(t)
        ctx.move_to(-300, H)
        for sx in range(-300, W + 320, 20):
            wx = sx + off
            y = base * H - amp * (math.sin(wx / 310 + layer) * 0.6 + math.sin(wx / 127 + 2 * layer) * 0.3 + math.sin(wx / 53) * 0.1)
            ctx.line_to(sx, y)
        ctx.line_to(W + 300, H); ctx.close_path()
        ctx.set_source_rgb(*P.c(col, depth)); ctx.fill()

def draw_treeline(ctx, P, t):
    f, sp = 0.25, 55
    off = f * cam(t)
    k0 = int((off - 400) // sp)
    ctx.set_source_rgb(*P.c('#2f5a3a', 0.4))
    ctx.rectangle(-300, 0.70 * H, W + 600, 0.1 * H); ctx.fill()
    for k in range(k0, k0 + int((W + 800) / sp)):
        x = k * sp - off + rnd(k, 11) * 30
        r = 38 + 34 * rnd(k, 12)
        ctx.arc(x, 0.70 * H - r * 0.5 + 10 * rnd(k, 13), r, 0, 2*math.pi); ctx.fill()

def draw_trees(ctx, P, t):
    f, sp = 0.55, 150
    off = f * cam(t)
    k0 = int((off - 400) // sp)
    base = 0.745 * H
    for k in range(k0, k0 + int((W + 800) / sp)):
        if rnd(k, 21) < 0.22: continue
        x = k * sp - off + rnd(k, 22) * 60
        h = 170 + 150 * rnd(k, 23)
        col = ['#2d6a4f', '#40916c', '#1b4332', '#52796f', '#6a994e'][int(rnd(k, 24) * 5)]
        seg(ctx, [(x, base), (x, base - h * 0.55)], 14, P.c('#4a3728', 0.25), cairo.LINE_CAP_BUTT)
        ctx.set_source_rgb(*P.c(col, 0.25))
        if rnd(k, 25) < 0.3:
            for j in range(3):
                w = (0.34 - j * 0.08) * h; y0 = base - h * (0.25 + j * 0.22)
                ctx.move_to(x - w, y0); ctx.line_to(x, y0 - h * 0.38); ctx.line_to(x + w, y0); ctx.close_path(); ctx.fill()
        else:
            for j in range(4):
                a = j * 1.6 + rnd(k, 26) * 3
                ctx.arc(x + math.cos(a) * h * 0.13, base - h * 0.72 + math.sin(a) * h * 0.1, h * (0.2 + 0.05 * rnd(k, 27 + j)), 0, 2*math.pi)
                ctx.fill()

def draw_road(ctx, P, t):
    ctx.set_source_rgb(*P.c('#5c8a4a', 0.15)); ctx.rectangle(-400, 0.735 * H, W + 800, ROAD_TOP - 0.735 * H + 1); ctx.fill()
    ctx.set_source_rgb(*P.c('#3b3e45')); ctx.rectangle(-400, ROAD_TOP, W + 800, ROAD_BOT - ROAD_TOP); ctx.fill()
    ctx.set_source_rgb(*P.c('#2e3d27')); ctx.rectangle(-400, ROAD_BOT, W + 800, H * 1.2); ctx.fill()
    ctx.set_source_rgb(*P.c('#9a9a92')); ctx.rectangle(-400, ROAD_BOT, W + 800, 10); ctx.fill()
    off = cam(t)
    ctx.set_source_rgb(*P.c('#e8e8e0')); ctx.rectangle(-400, ROAD_TOP + 8, W + 800, 4); ctx.fill()
    ctx.set_source_rgb(*P.c('#f2c230'))
    sp = 160; k0 = int((off - 400) // sp)
    ymid = (LANE_FAR + LANE_NEAR) / 2 - 18
    for k in range(k0, k0 + int((W + 800) / sp)):
        ctx.rectangle(k * sp - off, ymid, 80, 7); ctx.fill()
    # asphalt texture
    ctx.set_source_rgba(*P.c('#2a2c31'), 0.6)
    sp2 = 97; k0 = int((off - 400) // sp2)
    for k in range(k0, k0 + int((W + 800) / sp2)):
        x = k * sp2 - off + rnd(k, 31) * 80
        y = ROAD_TOP + 20 + rnd(k, 32) * (ROAD_BOT - ROAD_TOP - 40)
        ctx.rectangle(x, y, 18 + 30 * rnd(k, 33), 3); ctx.fill()
    # potholes: Ridge -> Ross
    if 94 < t < 104:
        sp3 = 230; k0 = int((off - 400) // sp3)
        for k in range(k0, k0 + int((W + 800) / sp3)):
            if rnd(k, 41) < 0.35: continue
            x = k * sp3 - off + rnd(k, 42) * 100
            y = ROAD_TOP + 30 + rnd(k, 43) * (ROAD_BOT - ROAD_TOP - 60)
            ctx.save(); ctx.translate(x, y); ctx.scale(1, 0.32)
            ctx.set_source_rgb(*P.c('#1c1d20')); ctx.arc(0, 0, 26 + 22 * rnd(k, 44), 0, 2*math.pi); ctx.fill()
            ctx.restore()

def draw_foreground(ctx, P, t):
    f, sp = 1.4, 260
    off = f * cam(t)
    k0 = int((off - 400) // sp)
    for k in range(k0, k0 + int((W + 800) / sp)):
        x = k * sp - off + rnd(k, 51) * 120
        ctx.set_source_rgb(*P.c('#1f3a1c'))
        for j in range(5):
            a = -math.pi / 2 + (j - 2) * 0.3
            ctx.move_to(x + j * 8 - 16, H + 5); ctx.line_to(x + j * 8 - 16 + 70 * math.cos(a), H - 60 * (0.6 + 0.4 * rnd(k * 5 + j, 52)))
            ctx.line_to(x + j * 8 - 8, H + 5); ctx.close_path(); ctx.fill()

def draw_objects_far_edge(ctx, P, t):
    base = ROAD_TOP + 4
    off = SIGN_F * cam(t)
    for ts, text, xf in SIGNS:
        x = at_screen(xf, ts, SIGN_F) - off
        if -300 < x < W + 300: draw_sign(ctx, P, x, base, text)
    for ts, xf, keys in LIGHTS:
        x = at_screen(xf, ts, SIGN_F) - off
        if -300 < x < W + 400: draw_light(ctx, P, x, base, light_state(keys, t), t)
    for tg in GATES:
        x = at_screen(0.60, tg, SIGN_F) - off
        if -500 < x < W + 300: draw_gate(ctx, P, x, base, t, tg)
    x = at_screen(GRADE[1], GRADE[0], SIGN_F) - off
    if -200 < x < W + 200:
        seg(ctx, [(x, base), (x, base - 200)], 7, P.c('#8a8f96'), cairo.LINE_CAP_BUTT)
        ctx.save(); ctx.translate(x, base - 250); ctx.rotate(math.pi / 4)
        ctx.rectangle(-55, -55, 110, 110); ctx.set_source_rgb(*P.c('#ffcc00')); ctx.fill_preserve()
        ctx.set_source_rgb(0.1, 0.1, 0.1); ctx.set_line_width(5); ctx.stroke(); ctx.restore()
        ctx.select_font_face('Anton'); ctx.set_font_size(52); ctx.set_source_rgb(0.1, 0.1, 0.1)
        ctx.move_to(x - 36, base - 228); ctx.show_text('7%')
    # pedestrians on the far path (they walk left slowly -> slightly slower scroll)
    wf = SIGN_F
    if 48 < t < 60:
        x = at_screen(0.78, 52.3, wf) - off - (t - 52.3) * 40
        draw_walker(ctx, P, x, base - 6, t, '#c77dff', 0.95, stroller=True)
        if 51.7 < t < 56: speech(ctx, x + 40, base - 215, 'SORRY!', t - 51.7, '#ffffff', '#1b1b1b', 26)
    if 85 < t < 96:
        x = at_screen(0.72, 89.8, wf) - off - (t - 89.8) * 40
        draw_walker(ctx, P, x, base - 4, t, '#ff8fab', 0.95)
        draw_walker(ctx, P, x + 70, base - 2, t + 0.3, '#4cc9f0', 1.02)
        draw_dog(ctx, P, x - 330, base, t, (x + 12, base - 100))

# ---------------------------------------------------------------- overlays
def outlined(ctx, text, x, y, fill, stroke=(0.05, 0.05, 0.1), lw=10, a=1.0):
    ctx.move_to(x, y); ctx.text_path(text)
    ctx.set_source_rgba(*stroke, 0.85 * a); ctx.set_line_width(lw); ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.stroke_preserve(); ctx.set_source_rgba(*fill, a); ctx.fill()

def speech(ctx, x, y, text, age, bg, fg, size=34):
    k = ease_pop(age / 0.25)
    if k <= 0.01: return
    ctx.save(); ctx.translate(x, y); ctx.scale(k, k)
    ctx.select_font_face('Permanent Marker'); ctx.set_font_size(size)
    ext = ctx.text_extents(text)
    w, h = ext.x_advance + 30, size + 22
    rrect(ctx, -w / 2, -h / 2, w, h, 16); ctx.set_source_rgb(*hexc(bg)); ctx.fill()
    ctx.move_to(-12, h / 2 - 1); ctx.line_to(-24, h / 2 + 22); ctx.line_to(6, h / 2 - 1); ctx.close_path(); ctx.fill()
    ctx.set_source_rgb(*hexc(fg)); ctx.move_to(-ext.x_advance / 2, size * 0.36); ctx.show_text(text)
    ctx.restore()

STICKERS = [  # t0, t1, text, x, y, rot, bg, fg, size
    (9.6, 14.6, '6:15 PM', .20, .20, -.08, '#ffd23f', '#1b1b1b', 60),
    (11.9, 14.6, 'x 38', .62, .20, .07, '#e63946', '#ffffff', 64),
    (62.3, 66.0, "NOT A RACE*", .60, .17, .06, '#ffffff', '#e63946', 46),
    (64.2, 66.0, "*legs disagree", .62, .25, -.04, '#1b1b1b', '#ffd23f', 34),
    (72.0, 81.5, '25 MI  ·  600 FT', .22, .19, -.05, '#2ec4b6', '#0b1d1f', 46),
    (75.0, 81.5, 'EVERY WEDNESDAY', .24, .27, .04, '#ffd23f', '#1b1b1b', 38),
    (99.0, 101.6, 'POTHOLE CHART', .60, .21, .06, '#1b1b1b', '#ffd23f', 44),
    (111.2, 114.3, 'OH YOU SUCK', .60, .21, .08, '#e63946', '#ffffff', 50),
    (117.7, 124.0, 'LAPTOP · BADGE · SHOES', .58, .22, -.05, '#264653', '#e9c46a', 38),
    (127.1, 130.7, '. . .', .70, .54, 0, '#ffffff', '#1b1b1b', 40),
    (127.6, 130.7, '. . .', .45, .52, 0, '#ffffff', '#1b1b1b', 40),
    (128.1, 130.7, '. . .', .86, .53, 0, '#ffffff', '#1b1b1b', 40),
    (130.0, 130.7, '', 0, 0, 0, '#ffffff', '#ffffff', 10),
    (137.2, 141.0, "NOT A RACE*", .60, .17, .06, '#ffffff', '#e63946', 46),
    (139.2, 141.0, "*legs disagree", .62, .25, -.04, '#1b1b1b', '#ffd23f', 34),
    (166.4, 171.0, '26 MPH', .60, .20, .06, '#ff6b35', '#ffffff', 60),
    (186.5, 190.0, 'JUMP!', .82, .50, -.06, '#e63946', '#ffffff', 52),
    (195.3, 198.5, "CAUGHT 'EM", .60, .21, .05, '#2ec4b6', '#0b1d1f', 48),
    (212.3, 218.0, "NOT A RACE*", .60, .17, .06, '#ffffff', '#e63946', 46),
    (214.3, 218.0, "*legs disagree", .62, .25, -.04, '#1b1b1b', '#ffd23f', 34),
    (234.0, 239.0, '77:00', .60, .21, -.05, '#ffd23f', '#1b1b1b', 72),
]

def sticker(ctx, t, s):
    t0, t1, text, xf, yf, rot, bg, fg, size = s
    if not text or not (t0 <= t < t1 + 0.25): return
    k = ease_pop((t - t0) / 0.3) * (1 - smooth((t - t1) / 0.25))
    if k <= 0.01: return
    ctx.save(); ctx.translate(xf * W, yf * H); ctx.rotate(rot + 0.02 * math.sin(t * 3)); ctx.scale(k, k)
    font = 'Anton' if size >= 50 else 'Permanent Marker'
    ctx.select_font_face(font); ctx.set_font_size(size)
    ext = ctx.text_extents(text)
    w, h = ext.x_advance + 44, size * 1.45
    rrect(ctx, -w / 2 + 6, -h / 2 + 8, w, h, 12); ctx.set_source_rgba(0, 0, 0, 0.35); ctx.fill()
    rrect(ctx, -w / 2, -h / 2, w, h, 12); ctx.set_source_rgb(*hexc(bg)); ctx.fill()
    ctx.set_source_rgb(*hexc(fg)); ctx.move_to(-ext.x_advance / 2, size * 0.38); ctx.show_text(text)
    ctx.restore()

def draw_tag(ctx, t, tag, pos):
    t0, t1, rid, name, sub = tag
    if not (t0 <= t < t1 + 0.2) or rid not in pos: return
    k = ease_pop((t - t0) / 0.3) * (1 - smooth((t - t1) / 0.2))
    if k <= 0.01: return
    x, y = pos[rid]
    x = min(W - 140, max(140, x))
    y -= 30 + 6 * math.sin(t * 4)
    ctx.save(); ctx.translate(x, y); ctx.scale(k, k)
    ctx.select_font_face('Anton'); ctx.set_font_size(40); e1 = ctx.text_extents(name)
    ctx.select_font_face('Bebas Neue'); ctx.set_font_size(26); e2 = ctx.text_extents(sub)
    w = max(e1.x_advance, e2.x_advance) + 36
    rrect(ctx, -w / 2, -104, w, 86, 10); ctx.set_source_rgb(0.07, 0.07, 0.1); ctx.fill()
    ctx.move_to(-12, -19); ctx.line_to(0, 0); ctx.line_to(12, -19); ctx.close_path(); ctx.fill()
    ctx.set_source_rgb(*hexc('#ffd23f')); ctx.select_font_face('Anton'); ctx.set_font_size(40)
    ctx.move_to(-e1.x_advance / 2, -58); ctx.show_text(name)
    ctx.set_source_rgb(1, 1, 1); ctx.select_font_face('Bebas Neue'); ctx.set_font_size(26)
    ctx.move_to(-e2.x_advance / 2, -28); ctx.show_text(sub)
    ctx.restore()

def draw_calendar(ctx, t):
    if not (176.0 <= t < 184.4): return
    k = ease_pop((t - 176.0) / 0.3) * (1 - smooth((t - 184.2) / 0.2))
    if k <= 0.01: return
    day = 'TUE?' if t < 177.7 else 'THU?' if t < 179.4 else '???' if t < 180.9 else 'WED'
    ch = max(176.1, 177.7 if t >= 177.7 else 0, 179.4 if t >= 179.4 else 0, 180.9 if t >= 180.9 else 0)
    flip = smooth((t - ch) / 0.18)
    ctx.save(); ctx.translate(0.12 * W, 0.34 * H); ctx.rotate(-0.05); ctx.scale(k, k)
    rrect(ctx, -120, -110, 240, 230, 16); ctx.set_source_rgb(1, 1, 1); ctx.fill()
    ctx.rectangle(-120, -110, 240, 60); ctx.set_source_rgb(*hexc('#e63946')); ctx.fill()
    ctx.set_source_rgb(1, 1, 1); ctx.select_font_face('Bebas Neue'); ctx.set_font_size(40)
    e = ctx.text_extents('RIDE DAY'); ctx.move_to(-e.x_advance / 2, -66); ctx.show_text('RIDE DAY')
    ctx.save(); ctx.scale(1, max(0.05, flip))
    ctx.set_source_rgb(0.1, 0.1, 0.12); ctx.select_font_face('Anton'); ctx.set_font_size(96)
    e = ctx.text_extents(day); ctx.move_to(-e.x_advance / 2, 75 / max(0.05, flip) * flip); ctx.show_text(day)
    ctx.restore()
    if day == 'WED':
        seg(ctx, [(-95, 30), (95, 45)], 10, hexc('#e63946'))
    ctx.restore()

def draw_hud(ctx, t):
    k = smooth((t - 9.3) / 0.6) * (1 - smooth((t - 243.0) / 1.0))
    if k <= 0: return
    fi = min(len(MILES) - 1, int(t * FPS))
    x0, y0, w, h = W - 486, 36 - (1 - k) * 320, 450, 262
    rrect(ctx, x0, y0, w, h, 22); ctx.set_source_rgba(0.04, 0.05, 0.08, 0.82); ctx.fill()
    ctx.set_source_rgba(1, 1, 1, 0.15); ctx.set_line_width(2); rrect(ctx, x0, y0, w, h, 22); ctx.stroke()
    def lab(s, x, y):
        ctx.select_font_face('Bebas Neue'); ctx.set_font_size(22); ctx.set_source_rgba(1, 1, 1, 0.55)
        ctx.move_to(x, y); ctx.show_text(s)
    def val(s, x, y, size, col=(1, 1, 1)):
        ctx.select_font_face('Anton'); ctx.set_font_size(size); ctx.set_source_rgb(*col)
        ctx.move_to(x, y); ctx.show_text(s)
    mph = keyed(SPEED, t) + 0.35 * math.sin(t * 1.3) + 0.2 * math.sin(t * 3.1)
    lab('SPEED  MPH', x0 + 26, y0 + 38); val(f'{mph:4.1f}', x0 + 24, y0 + 130, 90, hexc('#ffd23f'))
    lab('TIME', x0 + 280, y0 + 38); val(ride_clock(t) + ' PM', x0 + 280, y0 + 82, 38)
    lab('DIST  MI', x0 + 280, y0 + 112); val(f'{MILES[fi]:.1f}', x0 + 280, y0 + 154, 38)
    lab(f'CLIMB  {int(climb[fi])} FT', x0 + 26, y0 + 166)
    px0, py0, pw, ph = x0 + 26, y0 + 180, w - 52, 58
    e_lo, e_hi = elev.min() - 10, elev.max() + 10
    def prof():
        ctx.move_to(px0, py0 + ph)
        for j in range(0, len(MILES), 30):
            ctx.line_to(px0 + pw * MILES[j] / 25.2, py0 + ph - ph * (elev[j] - e_lo) / (e_hi - e_lo))
        ctx.line_to(px0 + pw, py0 + ph); ctx.close_path()
    prof(); ctx.set_source_rgba(1, 1, 1, 0.14); ctx.fill()
    ctx.save(); ctx.rectangle(px0, py0 - 5, pw * MILES[fi] / 25.2, ph + 10); ctx.clip()
    prof(); ctx.set_source_rgba(*hexc('#2ec4b6'), 0.8); ctx.fill(); ctx.restore()
    mx = px0 + pw * MILES[fi] / 25.2; my = py0 + ph - ph * (elev[fi] - e_lo) / (e_hi - e_lo)
    ctx.set_source_rgb(*hexc('#ffd23f')); ctx.arc(mx, my, 7, 0, 2*math.pi); ctx.fill()

def draw_section(ctx, t):
    st, name = section_at(t)
    if t < 9.3 or t > 243.5: return
    age = t - st
    k = smooth(age / 0.4)
    ctx.save(); ctx.translate(40 - (1 - k) * 300, 58)
    ctx.select_font_face('Bebas Neue'); ctx.set_font_size(26); ctx.set_source_rgba(1, 1, 1, 0.7)
    ctx.move_to(0, 0); ctx.show_text('GEEZER GOON  ·  G2  ·  WEDNESDAY')
    ctx.select_font_face('Anton'); ctx.set_font_size(54)
    outlined(ctx, name, 0, 64, hexc('#ffd23f'), lw=6)
    e = ctx.text_extents(name)
    ctx.set_source_rgb(*hexc('#ffd23f')); ctx.rectangle(0, 78, e.x_advance * k, 5); ctx.fill()
    ctx.restore()

SHOUTS = ('GATE UP!',)
def layout_line(ctx, words, size, maxw):
    ctx.select_font_face('Anton'); ctx.set_font_size(size)
    sp = ctx.text_extents(' ').x_advance + size * 0.12
    rows, cur, cw = [], [], 0
    for i, wd in enumerate(words):
        ww = ctx.text_extents(wd).x_advance
        if cur and cw + sp + ww > maxw:
            rows.append((cur, cw)); cur, cw = [], 0
        cw += (sp if cur else 0) + ww; cur.append((i, wd, ww))
    rows.append((cur, cw))
    return rows, sp

def draw_lyrics(ctx, t, pul):
    idx = None
    for i, ln in enumerate(LINES):
        if ln[0] <= t: idx = i
    if idx is None or t > 244.3: return
    items = [(idx, 0.0)]
    if idx > 0 and t - LINES[idx][0] < 0.2:
        items.append((idx - 1, (t - LINES[idx][0]) / 0.2))
    for li, out in items:
        ln = LINES[li]
        text = ln[2].upper()
        words, starts, win = line_words([ln[0], ln[1], text])
        shout = text in SHOUTS
        size = 150 if shout else 78
        rows, sp = layout_line(ctx, words, size, W * 0.78)
        yc = 0.40 * H - (len(rows) - 1) * size * 0.6 - out * 60
        a = (1 - out) ** 2 if out else smooth((t - ln[0]) / 0.22)
        if shout:
            shake = 14 * math.exp(-(t - ln[0]) / 0.25)
            ctx.save(); ctx.translate(W / 2 + shake * math.sin(t * 90), yc); ctx.rotate(-0.04)
            k = ease_pop((t - ln[0]) / 0.18); ctx.scale(k, k)
            ctx.select_font_face('Anton'); ctx.set_font_size(size)
            e = ctx.text_extents(text)
            outlined(ctx, text, -e.x_advance / 2, size * 0.35, hexc('#e63946'), stroke=(1, 1, 1), lw=16, a=a)
            ctx.restore(); continue
        for r_i, (row, rw) in enumerate(rows):
            x = W / 2 - rw / 2
            y = yc + r_i * size * 1.18
            for (wi, wd, ww) in row:
                ts = starts[wi]
                nxt = starts[wi + 1] if wi + 1 < len(words) else ts + 0.45
                if t < ts:
                    col, al, sc = (1, 1, 1), 0.28, 1.0
                else:
                    kk = ease_pop((t - ts) / 0.16)
                    active = t < nxt + 0.05
                    col = hexc('#ffd23f') if active else (1, 1, 1)
                    al, sc = 1.0, kk * (1 + (0.06 * pul if active else 0))
                ctx.save(); ctx.translate(x + ww / 2, y)
                ctx.scale(sc, sc); ctx.select_font_face('Anton'); ctx.set_font_size(size)
                outlined(ctx, wd, -ww / 2, 0, col, lw=12, a=al * a)
                ctx.restore()
                x += ww + sp

def draw_title(ctx, t):
    if t > 10.0: return
    a = smooth(t / 1.2) * (1 - smooth((t - 8.6) / 1.2))
    ctx.save()
    ctx.set_source_rgba(0, 0, 0, 0.35 * a); ctx.rectangle(0, 0, W, H); ctx.fill()
    ctx.select_font_face('Anton')
    for i, ch in enumerate('GEEZER GOON'):
        pass
    ctx.set_font_size(230)
    txt = 'GEEZER GOON'
    e = ctx.text_extents(txt)
    k = 1 + 0.04 * t
    ctx.translate(W / 2, 0.40 * H); ctx.scale(k, k)
    outlined(ctx, txt, -e.x_advance / 2, 60, hexc('#ffd23f'), lw=18, a=a)
    ctx.select_font_face('Bebas Neue'); ctx.set_font_size(50)
    s = 'WEDNESDAY NIGHT  ·  KENSINGTON, MD  ·  25 MILES'
    e = ctx.text_extents(s); outlined(ctx, s, -e.x_advance / 2, 150, (1, 1, 1), lw=6, a=a)
    ctx.restore()

def draw_endcard(ctx, t):
    if t < 243.2: return
    a = smooth((t - 243.2) / 1.5)
    ctx.set_source_rgba(0.02, 0.02, 0.06, 0.55 * a); ctx.rectangle(0, 0, W, H); ctx.fill()
    ctx.save(); ctx.translate(W / 2, 0.36 * H)
    ctx.select_font_face('Anton'); ctx.set_font_size(170)
    e = ctx.text_extents('GEEZER GOON'); outlined(ctx, 'GEEZER GOON', -e.x_advance / 2, 40, hexc('#ffd23f'), lw=14, a=a)
    ctx.select_font_face('Bebas Neue'); ctx.set_font_size(58)
    s = 'SAME TIME  ·  SAME CORNER  ·  SAME OLD TOWN'
    e = ctx.text_extents(s); outlined(ctx, s, -e.x_advance / 2, 140, (1, 1, 1), lw=6, a=a)
    ctx.set_font_size(44)
    s2 = 'NEXT WEDNESDAY  ·  6:15 PM'
    e = ctx.text_extents(s2); outlined(ctx, s2, -e.x_advance / 2, 205, hexc('#2ec4b6'), lw=6, a=smooth((t - 244.5) / 1.0))
    ctx.restore()
    fade = smooth((t - 248.0) / 1.3)
    if fade > 0:
        ctx.set_source_rgba(0, 0, 0, fade); ctx.rectangle(0, 0, W, H); ctx.fill()

def draw_speedlines(ctx, t):
    k = max(smooth((t - 156) / 8) * (1 - smooth((t - 199) / 2)), 0)
    if k <= 0: return
    k *= 0.5 + 0.5 * smooth((t - 184) / 3)
    for i in range(26):
        y = 0.08 * H + rnd(i, 61) * 0.62 * H
        L = 200 + 400 * rnd(i, 62)
        x = W - ((t * (1800 + 1200 * rnd(i, 63)) + rnd(i, 64) * W * 2) % (W * 1.6))
        g = cairo.LinearGradient(x, y, x + L, y)
        g.add_color_stop_rgba(0, 1, 1, 1, 0); g.add_color_stop_rgba(1, 1, 1, 1, 0.35 * k)
        ctx.set_source(g); ctx.rectangle(x, y, L, 2 + 2 * rnd(i, 65)); ctx.fill()

def draw_vignette(ctx):
    g = cairo.RadialGradient(W / 2, H / 2, H * 0.45, W / 2, H / 2, H * 1.05)
    g.add_color_stop_rgba(0, 0, 0, 0, 0); g.add_color_stop_rgba(1, 0, 0, 0, 0.55)
    ctx.set_source(g); ctx.rectangle(0, 0, W, H); ctx.fill()

# ---------------------------------------------------------------- frame
def render(t, surf=None):
    surf = surf or cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    P = Pal(t)
    pul = pulse(t) * min(1.0, 0.4 + energy(t))
    draw_sky(ctx, P, t)
    # world (tilted on the hill, shaken on the gate shouts)
    ctx.save()
    shake = 0.0
    for tg in GATES:
        if 0 <= t - tg < 0.6: shake = 10 * math.exp(-(t - tg) / 0.15)
    ctx.translate(shake * math.sin(t * 83), -2.5 * pul + shake * math.cos(t * 71))
    ang = math.radians(hill_angle(t))
    ctx.translate(W / 2, LANE_NEAR); ctx.rotate(ang); ctx.translate(-W / 2, -LANE_NEAR)
    draw_hills(ctx, P, t)
    draw_temple(ctx, P, at_screen(TEMPLE[1], TEMPLE[0], 0.10) - 0.10 * cam(t), 0.66 * H, 0.9)
    draw_treeline(ctx, P, t)
    draw_trees(ctx, P, t)
    draw_road(ctx, P, t)
    draw_objects_far_edge(ctx, P, t)
    lights = smooth((t / DUR - 0.68) / 0.12)
    pos, order = {}, []
    for r in RIDERS:
        x = rider_x(r, t)
        vx = (rider_x(r, t + 0.08) - rider_x(r, t - 0.08)) / 0.16
        passing = min(1.0, abs(vx) / 260)
        lane_y = LANE_NEAR if r['lane'] == 0 else LANE_FAR
        sway_amp = 0 if (r['id'] == 'gary' and 42.7 < t < 46.3) else 1
        sway = sway_amp * (14 * math.sin(t * 0.7 + sum(map(ord, r['id'])) % 7) + 6 * math.sin(t * 1.9 + len(r['id'])))
        bump = 0.0
        if 94.5 < t < 101.8:
            bump = 5 * max(0, math.sin(t * 17 + x * 0.01)) ** 8
        gf = max([smooth(1 - abs(t - (w[0] + 3.0)) / 3.0) for w in GATE_WIN] + [0])
        lane_y += (18 if r['lane'] == 1 else -18) * gf
        y = lane_y + passing * 34 - bump
        order.append((y, x + sway, r, passing))
    order.sort(key=lambda o: o[0])
    for y, x, r, passing in order:
        s = (175 if r['lane'] == 0 else 128) * r['s'] * (1 + 0.04 * passing)
        wang = cam(t) / (0.34 * s)
        crank = wang / 1.7 + sum(map(ord, r['id'])) % 10
        if r['id'] == 'angelo': crank = wang / 1.35
        if r['id'] == 'gary' and 42.7 < t < 46.3: crank = cam(42.7) / (0.34 * s) / 1.7 + sum(map(ord, r['id'])) % 10  # coasting
        depth = 0.0 if r['lane'] == 0 else 0.22
        draw_rider(ctx, P, r, x, y, s, crank, wang, depth, t, lights)
        pos[r['id']] = (x + 0.2 * s, y - 1.55 * s)
    for tag in TAGS: draw_tag(ctx, t, tag, pos)
    for tg in GATES:
        if tg <= t < tg + 2.2: speech(ctx, pos['karim'][0] - 20, pos['karim'][1] - 30, 'GATE UP!', t - tg, '#e63946', '#ffffff', 40)
    if 83.0 <= t < 85.6: speech(ctx, pos['phil'][0] + 60, pos['phil'][1] - 40, 'MAKE IT!!', t - 83.0, '#ffd23f', '#1b1b1b', 40)
    if 151.5 <= t < 156.0: speech(ctx, pos['me'][0] + 70, pos['me'][1] + 10, '?!', t - 151.5, '#ffffff', '#1b1b1b', 44)
    if 186.4 <= t < 189.5: speech(ctx, pos['karim'][0] - 30, pos['karim'][1] - 20, 'GO!', t - 186.4, '#e63946', '#ffffff', 40)
    draw_foreground(ctx, P, t)
    ctx.restore()
    draw_speedlines(ctx, t)
    # chorus flash
    for st, name in SECTIONS:
        if 'CHORUS' in name and 'PRE' not in name and 0 <= t - st < 0.4:
            ctx.set_source_rgba(1, 1, 1, 0.3 * (1 - (t - st) / 0.4)); ctx.rectangle(0, 0, W, H); ctx.fill()
    draw_vignette(ctx)
    for s in STICKERS: sticker(ctx, t, s)
    draw_calendar(ctx, t)
    draw_gate_map(ctx, t)
    draw_lyrics(ctx, t, pul)
    draw_section(ctx, t)
    draw_hud(ctx, t)
    draw_title(ctx, t)
    draw_endcard(ctx, t)
    return surf

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'still':
        for a in sys.argv[2:]:
            render(float(a)).write_to_png(f'still_{a}.png')
    elif mode == 'chunk':
        f0, f1, out = int(sys.argv[2]), min(int(sys.argv[3]), NF), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgra',
                              '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
                              '-crf', '19', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
        for f in range(f0, f1):
            render(f / FPS, surf)
            surf.flush(); p.stdin.write(bytes(surf.get_data()))
            if f % 300 == 0: print(out, f, flush=True)
        p.stdin.close(); p.wait()
