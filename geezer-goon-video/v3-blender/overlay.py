"""Lyrics, HUD, name tags, stickers and cards drawn over the Blender frames with Pillow.

    python3 overlay.py FRAMES_DIR OUT_DIR [f0 f1] [--procs N]

ffmpeg here has no libass/drawtext, so everything is drawn per frame. Screen positions for riders come from the
markers_*.json files Blender writes next to the frames.
"""
import os, sys, json, math, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C
import shots as SH

W, H = 1280, 720
K = W / 1920.0                      # v2 was laid out at 1920x1080
FONTS = os.path.join(HERE, 'fonts')
_fc = {}


def font(name, px):
    key = (name, int(round(px)))
    if key not in _fc:
        _fc[key] = ImageFont.truetype(os.path.join(FONTS, {'anton': 'anton.ttf', 'bebas': 'bebasneue.ttf', 'marker': 'permanentmarker.ttf'}[name]), key[1])
    return _fc[key]


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def pop_out(x):
    x = clamp(x)
    c1 = 1.9; c3 = c1 + 1
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2


def text_w(f, s):
    return f.getlength(s)


def outlined(d, xy, s, f, fill, stroke=(10, 8, 30, 235), sw=6, anchor='ls'):
    d.text(xy, s, font=f, fill=fill, stroke_width=sw, stroke_fill=stroke, anchor=anchor)


# ----------------------------------------------------------------------------------------------- lyrics
LYR = [dict(t0=a, t1=b, text=s.upper()) for a, b, s in C.D['lines']]


def line_words(l):
    words = l['text'].split(' ')
    win = min((l['t1'] - l['t0']) * 0.93, 0.072 * len(l['text']) + 0.3)
    w = [len(x) + 2 for x in words]
    tot = sum(w)
    acc = 0.0
    st, en = [], []
    for i, x in enumerate(words):
        st.append(l['t0'] + acc / tot * win)
        acc += w[i]
        en.append(l['t0'] + acc / tot * win)
    return words, st, en


def draw_lyrics(img, t, cx=0.40, y=0.30, maxw=0.56, size=None):
    idx = -1
    for i, l in enumerate(LYR):
        if l['t0'] <= t:
            idx = i
    if idx < 0 or t > 244.6:
        return
    l = LYR[idx]
    age = t - l['t0']
    shout = l['text'] == 'GATE UP!'
    fade_in = smooth(age / 0.18)
    nxt = LYR[idx + 1]['t0'] if idx + 1 < len(LYR) else None
    out_a = 1 - smooth((t - (nxt - 0.12)) / 0.12) if nxt else 1 - smooth((t - l['t1']) / 0.5)
    a = fade_in * clamp(out_a)
    if a <= 0.01:
        return
    words, st, en = line_words(l)
    size = size or (190 * K if shout else (74 if len(l['text']) > 46 else 86) * K)
    f = font('anton', size)
    CXL = cx * W
    yMid = y * H
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if shout:
        k = pop_out(age / 0.2)
        tmp = Image.new('RGBA', (int(W * 0.7), int(size * 2)), (0, 0, 0, 0))
        td = ImageDraw.Draw(tmp)
        outlined(td, (tmp.width / 2, size * 1.3), 'GATE UP!', f, (255, 59, 79, 255), stroke=(255, 255, 255, 240), sw=int(size * 0.11), anchor='ms')
        tmp = tmp.resize((max(1, int(tmp.width * k)), max(1, int(tmp.height * k))), Image.BICUBIC).rotate(2.6, resample=Image.BICUBIC, expand=True)
        shake = 14 * K * math.exp(-age / 0.25) * math.sin(t * 90)
        layer.alpha_composite(tmp, (int(CXL + shake - tmp.width / 2), int(yMid - tmp.height / 2)))
        _paste(img, layer, a)
        return
    sp = text_w(f, ' ') + size * 0.1
    maxpx = maxw * W
    rows, cur, cw = [], [], 0.0
    for i, wd in enumerate(words):
        ww = text_w(f, wd)
        if cur and cw + sp + ww > maxpx:
            rows.append((cur, cw)); cur, cw = [], 0.0
        cw += (sp if cur else 0) + ww
        cur.append((i, wd, ww))
    rows.append((cur, cw))
    top = yMid - (len(rows) - 1) * size * 0.59 + (1 - fade_in) * 22 * K
    sw = max(2, int(size * 0.075))
    for ri, (row, rw) in enumerate(rows):
        x = CXL - rw / 2
        y0 = top + ri * size * 1.17
        for i, wd, ww in row:
            prog = clamp((t - st[i]) / max(0.08, en[i] - st[i]))
            on = t >= st[i]; done = t >= en[i]
            pop = 1 + 0.07 * math.exp(-(t - st[i]) / 0.13) if on else 1.0
            base = (255, 255, 255, 255) if on else (255, 255, 255, 110)
            pad = int(size * 0.5)
            tw, th = int(ww + pad * 2), int(size * 1.8)
            tmp = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
            td = ImageDraw.Draw(tmp)
            td.text((pad, size * 1.2 + 4), wd, font=f, fill=(0, 0, 0, 0), stroke_width=sw + 2, stroke_fill=(10, 8, 30, 235), anchor='ls')
            td.text((pad, size * 1.2), wd, font=f, fill=base, stroke_width=sw, stroke_fill=(10, 8, 30, 235), anchor='ls')
            if on and not done:
                hl = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
                hd = ImageDraw.Draw(hl)
                hd.text((pad, size * 1.2), wd, font=f, fill=(255, 210, 63, 255), anchor='ls')
                mask = Image.new('L', (tw, th), 0)
                ImageDraw.Draw(mask).rectangle((0, 0, pad + (ww + 8) * prog, th), fill=255)
                hl.putalpha(ImageChops.multiply(hl.getchannel('A'), mask))
                tmp.alpha_composite(hl)
            elif done:
                g = 0.35 * math.exp(-(t - en[i]) / 0.5)
                if g > 0.02:
                    hl = Image.new('RGBA', (tw, th), (0, 0, 0, 0))
                    ImageDraw.Draw(hl).text((pad, size * 1.2), wd, font=f, fill=(255, 210, 63, int(255 * g)), anchor='ls')
                    tmp.alpha_composite(hl)
            if abs(pop - 1) > 0.003:
                tmp = tmp.resize((int(tw * pop), int(th * pop)), Image.BICUBIC)
            layer.alpha_composite(tmp, (int(x + ww / 2 - tmp.width / 2), int(y0 - size * 1.2 * pop)))
            x += ww + sp
    _paste(img, layer, a)


def _paste(img, layer, a=1.0):
    if a < 0.999:
        al = layer.getchannel('A').point(lambda v: int(v * a))
        layer.putalpha(al)
    img.alpha_composite(layer)


# ----------------------------------------------------------------------------------------------- HUD
ALT = np.array(C.R['alt'])
SEC = C.SECTIONS
RT = C.R


def section_at(t):
    s = SEC[0]
    for x in SEC:
        if x[0] <= t + 1e-6:
            s = x
    return s


def clock_str(t):
    if t < 9.5:
        sec = -45 + t * 4.7
    else:
        s = C.odo(t)
        # cumulative ride time along the route scaled to the 77 minutes in the lyric
        sec = (s / C.LEN) * 77 * 60
    tot = 6 * 3600 + 15 * 60 + int(sec)
    h = (tot // 3600) % 12 or 12
    m = (tot // 60) % 60
    return f'{h}:{m:02d}'


def climb_ft(t):
    s = C.odo(t)
    gain = np.cumsum(np.clip(np.diff(ALT), 0, None))
    gain = gain * (600 / 3.28084) / gain[-1]
    i = int(clamp(s / RT['step'], 0, len(gain) - 1))
    return gain[i] * 3.28084


def mph_at(t, shot):
    """Real riding speed at the shot's location (the pack's speed), eased in at the start."""
    v = shot.get('v')
    if isinstance(v, (list, tuple)):
        k = (t - shot['t0']) / max(1e-6, shot['t1'] - shot['t0'])
        v = v[0] + (v[1] - v[0]) * clamp(k)
    if v is None:
        v = min(12.0, max(7.0, C.route_speed(C.odo(0.5 * (shot['t0'] + shot['t1'])))))
    return v * 2.23694


def draw_hud(img, t, shot, hide=False):
    k = smooth((t - 9.6) / 0.6) * (1 - smooth((t - 242.5) / 1.0)) * (0 if hide else 1)
    if k <= 0.01:
        return
    w, h = int(450 * K), int(262 * K)
    x0 = W - w - int(22 * K)
    y0 = int(34 * K - (1 - k) * 320 * K)
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=int(22 * K), fill=(8, 10, 16, 210), outline=(255, 255, 255, 40), width=2)
    lab = lambda s, x, y: d.text((x, y), s, font=font('bebas', 22 * K), fill=(255, 255, 255, 140), anchor='ls')
    val = lambda s, x, y, px, col=(255, 255, 255, 255): d.text((x, y), s, font=font('anton', px * K), fill=col, anchor='ls')
    mph = mph_at(t, shot) * (0 if t < 19 else min(1, (t - 19) / 5))
    lab('SPEED  MPH', x0 + 26 * K, y0 + 38 * K)
    val(f'{mph:4.1f}', x0 + 24 * K, y0 + 130 * K, 90, (255, 210, 63, 255))
    lab('TIME', x0 + 280 * K, y0 + 38 * K)
    val(clock_str(t) + ' PM', x0 + 280 * K, y0 + 82 * K, 38)
    lab('DIST  MI', x0 + 280 * K, y0 + 112 * K)
    val(f'{C.odo(t) / 1609.344:.1f}', x0 + 280 * K, y0 + 154 * K, 38)
    lab(f'CLIMB  {int(round(climb_ft(t)))} FT', x0 + 26 * K, y0 + 166 * K)
    if C.odo_rate(t) > 150:
        d.text((x0 + 232 * K, y0 + 166 * K), '>> TIME-LAPSE', font=font('bebas', 20 * K), fill=(255, 59, 79, 255), anchor='ls')
    px0, py0, pw, ph = x0 + 26 * K, y0 + 180 * K, w - 52 * K, 58 * K
    aMin, aMax = 28, 118
    n = len(ALT)
    pts = [(px0 + pw * i / (n - 1), py0 + ph - ph * (ALT[i] - aMin) / (aMax - aMin)) for i in range(0, n, 8)]
    d.polygon([(px0, py0 + ph)] + pts + [(px0 + pw, py0 + ph)], fill=(255, 255, 255, 36))
    frac = C.odo(t) / C.LEN
    done = [p for p in pts if p[0] <= px0 + pw * frac]
    if len(done) > 1:
        d.polygon([(px0, py0 + ph)] + done + [(done[-1][0], py0 + ph)], fill=(46, 196, 182, 215))
    mi = int(clamp(C.odo(t) / RT['step'], 0, n - 1))
    cx, cy = px0 + pw * frac, py0 + ph - ph * (ALT[mi] - aMin) / (aMax - aMin)
    d.ellipse((cx - 7 * K, cy - 7 * K, cx + 7 * K, cy + 7 * K), fill=(255, 210, 63, 255))
    # section label top-left
    st, name = section_at(t)
    kk = smooth((t - st) / 0.4)
    ox = 40 * K - (1 - kk) * 300 * K
    d.text((ox, 58 * K), 'GEEZER GOON  ·  G2  ·  WEDNESDAY', font=font('bebas', 26 * K), fill=(255, 255, 255, 190), anchor='ls')
    outlined(d, (ox, 122 * K), name, font('anton', 54 * K), (255, 210, 63, 255), stroke=(8, 8, 24, 230), sw=int(7 * K))
    e = text_w(font('anton', 54 * K), name)
    d.rectangle((ox, 134 * K, ox + e * kk, 139 * K), fill=(255, 210, 63, 255))
    _paste(img, layer, 1.0)


# ----------------------------------------------------------------------------------------------- callouts
TAGS = [('karim', 'KARIM', 'ON THE FRONT', 14.7, 22.5), ('hulk', 'THE HULK', 'BACK AND STRONG', 36.4, 39.5), ('phil', 'PHIL', 'TAKING A PULL', 39.6, 42.7),
        ('gary', 'GARY', 'NOT MOVING', 42.7, 46.2), ('angelo', 'ANGELO', 'ONE GEAR', 114.4, 120.9), ('me', 'ME', 'HANGING ON', 169.6, 171.0), ('me', 'ME', 'FAT TIRES', 192.1, 195.3)]
STICKERS = [  # t0, t1, text, xfrac, yfrac, rot, fill, ink, size(v2 px)
    (36.8, 39.5, 'SEGMENTS TO OWN', 0.56, 0.62, 0.05, (255, 210, 63), (27, 27, 27), 46),
    (62.3, 66.1, 'NOT A RACE*', 0.63, 0.66, 0.06, (255, 255, 255), (230, 57, 70), 50), (63.9, 66.1, '*legs disagree', 0.63, 0.74, -0.04, (27, 27, 27), (255, 210, 63), 34),
    (137.2, 143.7, 'NOT A RACE*', 0.63, 0.66, 0.06, (255, 255, 255), (230, 57, 70), 50), (139.4, 143.7, '*legs disagree', 0.63, 0.74, -0.04, (27, 27, 27), (255, 210, 63), 34),
    (205.2, 211.3, 'NOT A RACE*', 0.6, 0.66, 0.06, (255, 255, 255), (230, 57, 70), 50), (207.2, 211.3, '*legs disagree', 0.6, 0.74, -0.04, (27, 27, 27), (255, 210, 63), 34),
    (111.2, 114.3, 'OH YOU SUCK', 0.6, 0.66, 0.07, (230, 57, 70), (255, 255, 255), 52), (120.9, 124.1, 'A WEEK OF MONDAYS', 0.5, 0.7, -0.04, (38, 70, 83), (233, 196, 106), 40),
    (186.6, 190.2, 'JUMP!', 0.62, 0.66, -0.06, (230, 57, 70), (255, 255, 255), 56), (195.4, 198.5, "CAUGHT 'EM", 0.58, 0.68, 0.05, (46, 196, 182), (11, 29, 31), 50),
    (72.0, 75.0, '25 MI · 600 FT', 0.3, 0.8, -0.04, (46, 196, 182), (11, 29, 31), 46),
    (97.7, 101.6, 'POTHOLE CHART', 0.3, 0.78, -0.03, (255, 122, 26), (20, 20, 22), 50),
    (42.9, 46.2, '0 CM', 0.27, 0.62, 0.0, (255, 59, 79), (255, 255, 255), 54),
    (124.4, 127.0, '1 BIKE LENGTH', 0.35, 0.7, 0.0, (255, 59, 79), (255, 255, 255), 44),
    (160.0, 169.6, 'THE GAP', 0.2, 0.78, -0.03, (255, 59, 79), (255, 255, 255), 46),
    (78.5, 81.9, 'EVERY WEDNESDAY', 0.32, 0.84, -0.03, (255, 210, 63), (27, 27, 27), 52),
    (176.2, 184.2, 'TUESDAY? THURSDAY?', 0.4, 0.86, 0.02, (46, 196, 182), (11, 29, 31), 52),
]
BUBBLES = [  # t0, t1, text, who, dx, dy(px at 720p), fill, ink, size
    (33.1, 35.0, 'OW.', 'hulk', 60, -40, (255, 255, 255), (27, 27, 27), 42), (36.6, 39.5, 'BACK.', 'hulk', 60, -40, (255, 210, 63), (27, 27, 27), 42),
    (83.0, 85.4, 'MAKE IT!!', 'phil', 70, -50, (255, 210, 63), (27, 27, 27), 44), (170.0, 176.5, '?!', 'me', 50, -50, (255, 255, 255), (27, 27, 27), 52),
    (186.5, 189.6, 'GO!', 'karim', -40, -50, (230, 57, 70), (255, 255, 255), 46),
]


def draw_sticker(img, t, st):
    t0, t1, text, xf, yf, rot, fill, ink, sz = st
    if not (t0 <= t < t1 + 0.3):
        return
    age = t - t0
    k = pop_out(age / 0.25) * (1 - smooth((t - t1) / 0.3))
    if k <= 0.02:
        return
    f = font('anton', sz * K * 0.9)
    tw = text_w(f, text)
    pad = 20 * K
    box = Image.new('RGBA', (int(tw + pad * 2), int(sz * K * 1.5)), (0, 0, 0, 0))
    bd = ImageDraw.Draw(box)
    bd.rounded_rectangle((0, 0, box.width - 1, box.height - 1), radius=int(10 * K), fill=fill + (255,))
    bd.text((box.width / 2, box.height * 0.74), text, font=f, fill=ink + (255,), anchor='ms')
    box = box.resize((max(1, int(box.width * k)), max(1, int(box.height * k))), Image.BICUBIC).rotate(math.degrees(-rot), resample=Image.BICUBIC, expand=True)
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    layer.alpha_composite(box, (int(xf * W - box.width / 2), int(yf * H - box.height / 2)))
    sh = layer.filter(ImageFilter.GaussianBlur(6 * K))
    sh = Image.merge('RGBA', (Image.new('L', (W, H), 0), Image.new('L', (W, H), 0), Image.new('L', (W, H), 0), sh.getchannel('A').point(lambda v: int(v * 0.45))))
    img.alpha_composite(sh, (int(4 * K), int(6 * K)))
    img.alpha_composite(layer)


def draw_tag(img, t, tag, mk):
    rid, name, sub, a, b = tag
    if not (a <= t < b + 0.2) or rid not in mk:
        return
    x, y, depth = mk[rid]
    if depth <= 0 or not (0.02 < x < 0.98 and 0.02 < y < 0.98):
        return
    k = pop_out((t - a) / 0.3) * (1 - smooth((t - b) / 0.2))
    if k <= 0.02:
        return
    px, py = x * W, y * H - 10 * K
    fn, fs = font('anton', 30 * K), font('bebas', 22 * K)
    w = max(text_w(fn, name), text_w(fs, sub)) + 30 * K
    h = 62 * K
    layer = Image.new('RGBA', (int(w + 10), int(h + 24 * K)), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((0, 0, w, h), radius=int(8 * K), fill=(12, 14, 22, 235), outline=(255, 210, 63, 255), width=3)
    d.text((15 * K, 34 * K), name, font=fn, fill=(255, 210, 63, 255), anchor='ls')
    d.text((15 * K, 56 * K), sub, font=fs, fill=(255, 255, 255, 220), anchor='ls')
    d.polygon([(w * 0.12, h), (w * 0.12 + 18 * K, h), (w * 0.12 + 4 * K, h + 20 * K)], fill=(255, 210, 63, 255))
    layer = layer.resize((max(1, int(layer.width * k)), max(1, int(layer.height * k))), Image.BICUBIC)
    img.alpha_composite(layer, (int(px - 10 * K * k), int(py - layer.height)))


def draw_bubble(img, t, bb, mk):
    t0, t1, text, who, dx, dy, fill, ink, sz = bb
    if not (t0 <= t < t1) or who not in mk:
        return
    x, y, depth = mk[who]
    if depth <= 0 or not (0.02 < x < 0.98 and 0.02 < y < 0.98):
        return
    age = t - t0
    k = pop_out(age / 0.2) * (1 - smooth((t - (t1 - 0.2)) / 0.2))
    f = font('anton', sz * K * 0.85)
    tw = text_w(f, text)
    box = Image.new('RGBA', (int(tw + 36 * K), int(sz * K * 1.4)), (0, 0, 0, 0))
    bd = ImageDraw.Draw(box)
    bd.rounded_rectangle((0, 0, box.width - 1, box.height - 1), radius=int(box.height / 2), fill=fill + (255,), outline=(20, 20, 22, 255), width=3)
    bd.text((box.width / 2, box.height * 0.72), text, font=f, fill=ink + (255,), anchor='ms')
    box = box.resize((max(1, int(box.width * k)), max(1, int(box.height * k))), Image.BICUBIC)
    img.alpha_composite(box, (int(x * W + dx * K - box.width / 2), int(y * H + dy * K - box.height / 2)))


def draw_bigwords(img, t):
    """'GEEZER GOON' punched in on each sung word of the chorus."""
    for l in LYR:
        if l['text'].startswith('GEEZER GOON') and l['t0'] - 0.1 <= t <= l['t1'] + 0.2:
            words, st, en = line_words(l)
            for i, wd in enumerate(words):
                wd = wd.replace(',', '')
                age = t - st[i]
                if 0 <= age < (en[i] - st[i] + 0.6):
                    k = pop_out(age / 0.18) * (1 - smooth((age - 0.35) / 0.3))
                    if k <= 0.02:
                        continue
                    size = 150 * K
                    f = font('anton', size)
                    tmp = Image.new('RGBA', (int(text_w(f, wd) + size), int(size * 1.7)), (0, 0, 0, 0))
                    td = ImageDraw.Draw(tmp)
                    outlined(td, (tmp.width / 2, size * 1.25), wd, f, (255, 210, 63, 255) if wd == 'GEEZER' else (255, 122, 89, 255), stroke=(10, 8, 30, 235), sw=int(size * 0.07), anchor='ms')
                    tmp = tmp.resize((max(1, int(tmp.width * k)), max(1, int(tmp.height * k))), Image.BICUBIC).rotate(-3 if i % 2 == 0 else 3, resample=Image.BICUBIC, expand=True)
                    al = tmp.getchannel('A').point(lambda v: int(v * 0.85))
                    tmp.putalpha(al)
                    xf = 0.27 if i % 2 == 0 else 0.6
                    img.alpha_composite(tmp, (int(xf * W - tmp.width / 2), int(H * 0.52 - tmp.height / 2)))


def draw_cards(img, t):
    # end card
    if t >= 243.2:
        a = smooth((t - 243.2) / 1.4)
        ov = Image.new('RGBA', (W, H), (3, 3, 12, int(255 * 0.62 * a)))
        img.alpha_composite(ov)
        layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        outlined(d, (W / 2, H * 0.40), 'GEEZER GOON', font('anton', 210 * K), (255, 210, 63, 255), stroke=(10, 8, 30, 240), sw=int(18 * K), anchor='ms')
        outlined(d, (W / 2, H * 0.40 + 110 * K), 'SAME TIME  ·  SAME CORNER  ·  SAME OLD TOWN', font('bebas', 64 * K), (255, 255, 255, 255), stroke=(10, 8, 30, 240), sw=int(9 * K), anchor='ms')
        if t > 244.6:
            a2 = smooth((t - 244.6) / 1.0)
            l2 = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            outlined(ImageDraw.Draw(l2), (W / 2, H * 0.40 + 190 * K), 'NEXT WEDNESDAY  ·  6:15 PM', font('bebas', 52 * K), (46, 196, 182, 255), stroke=(10, 8, 30, 240), sw=int(8 * K), anchor='ms')
            _paste(layer, l2, a2)
        _paste(img, layer, a)
        f = smooth((t - 248.0) / 1.3)
        if f > 0:
            img.alpha_composite(Image.new('RGBA', (W, H), (0, 0, 0, int(255 * f))))
    # next wednesday title over the final map
    if 216.54 <= t < 224.3:
        k = smooth((t - 216.54 - 0.6) / 0.6)
        if k > 0:
            layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            outlined(d, (W * 0.04, H * 0.33), 'NEXT WEDNESDAY', font('anton', 150 * K), (255, 210, 63, 255), stroke=(5, 8, 18, 240), sw=int(16 * K), anchor='ls')
            outlined(d, (W * 0.04, H * 0.33 + 84 * K), '6:15 PM  ·  SAME ROUTE', font('bebas', 70 * K), (255, 255, 255, 255), stroke=(5, 8, 18, 240), sw=int(9 * K), anchor='ls')
            _paste(img, layer, k)
    # riders counter during the gather
    if 9.6 <= t < 19:
        n = int(round(38 * smooth((t - 9.6) / 5.2)))
        k = pop_out((t - 9.6) / 0.3) * (1 - smooth((t - 18.2) / 0.6))
        if k > 0.02:
            f = font('anton', 90 * K * 0.8)
            s = f'{n} / 38 RIDERS'
            tw = text_w(f, s) + 60 * K
            box = Image.new('RGBA', (int(tw), int(120 * K * 0.8)), (0, 0, 0, 0))
            bd = ImageDraw.Draw(box)
            bd.rounded_rectangle((0, 0, box.width - 1, box.height - 1), radius=int(14 * K), fill=(230, 57, 70, 255))
            bd.text((box.width / 2, box.height * 0.76), s, font=f, fill=(255, 255, 255, 255), anchor='ms')
            box = box.resize((max(1, int(box.width * k)), max(1, int(box.height * k))), Image.BICUBIC).rotate(2.3, resample=Image.BICUBIC, expand=True)
            img.alpha_composite(box, (int(W * 0.2 - box.width / 2), int(H * 0.17 - box.height / 2)))


# ----------------------------------------------------------------------------------------------- frame
def flash(img, t, shot):
    age = t - shot['t0']
    if shot.get('flash') or age < 0:
        pass
    # a short white punch on section starts and on every GATE UP
    starts = [s[0] for s in SEC] + [46.28, 101.65, 224.28]
    for s0 in starts:
        a = t - s0
        if 0 <= a < 0.18:
            img.alpha_composite(Image.new('RGBA', (W, H), (255, 255, 255, int(255 * 0.45 * (1 - a / 0.18)))))


def lyric_pos(shot):
    kind = shot['kind']
    if kind in ('calendar',):
        return dict(cx=0.40, y=0.16)
    if kind == 'map':
        return dict(cx=0.40, y=0.16)
    if shot.get('computer'):
        return dict(cx=0.30, y=0.46, maxw=0.44, size=64 * K)
    if 26.4 < shot['t0'] + 0.01 < 29.3 and shot.get('title'):
        return dict(cx=0.40, y=0.46)
    return dict(cx=0.40, y=0.30)


def compose(frame_path, t, markers):
    shot = SH.shot_at(t)
    img = Image.open(frame_path).convert('RGBA')
    if img.size != (W, H):
        img = img.resize((W, H), Image.LANCZOS)
    mk = markers.get(int(round(t * C.FPS)), {})
    draw_bigwords(img, t)
    for tg in TAGS:
        draw_tag(img, t, tg, mk)
    for bb in BUBBLES:
        draw_bubble(img, t, bb, mk)
    for st in STICKERS:
        draw_sticker(img, t, st)
    draw_hud(img, t, shot, hide=(shot.get('computer') is not None and shot['t0'] > 100) or (26.2 < t < 29.0))
    draw_lyrics(img, t, **lyric_pos(shot))
    draw_cards(img, t)
    flash(img, t, shot)
    return img.convert('RGB')


def load_markers(frames_dir):
    mk = {}
    if ':' in frames_dir:                      # lowest priority first so the earlier folders override
        for d in reversed(frames_dir.split(':')):
            mk.update(load_markers(d))
        return mk
    jl = os.path.join(frames_dir, 'markers.jsonl')
    if os.path.exists(jl):
        for line in open(jl):
            try:
                d = json.loads(line)
                mk[int(d['f'])] = {r: tuple(xy) for r, xy in d['m'].items()}
            except Exception:
                pass
    for p in glob.glob(os.path.join(frames_dir, 'markers_*.json')):
        try:
            for k, v in json.load(open(p)).items():
                mk[int(k)] = {r: tuple(xy) for r, xy in v.items()}
        except Exception:
            pass
    return mk


def find_src(frames_dir, f):
    """frames_dir may be several folders joined by ':' ; the first folder that has the frame wins (re-renders go first)."""
    for d in frames_dir.split(':'):
        for ext in ('jpg', 'png'):
            p = os.path.join(d, f'f_{f:05d}.{ext}')
            if os.path.exists(p):
                return p
    return None


def work(args):
    frames_dir, out_dir, frames, markers = args
    for f in frames:
        src = find_src(frames_dir, f)
        dst = os.path.join(out_dir, f'f_{f:05d}.png')
        if os.path.exists(dst) or src is None:
            continue
        compose(src, f / C.FPS, markers).save(dst, compress_level=3)
    return len(frames)


_G = {}


def _init(frames_dir, markers):
    _G['dir'] = frames_dir
    _G['mk'] = markers


def _one(f):
    src = find_src(_G['dir'], f)
    if src is None:
        return f, None
    return f, compose(src, f / C.FPS, _G['mk']).tobytes()


_NOISE = os.environ.get('GG_NOISE', '3')
POST = ("hue=s='if(between(t,127.1,130.8),0.18,1)',eq=saturation=1.1:contrast=1.04:gamma=0.98,"
        "vignette=angle=PI/5:eval=init" + (f",noise=alls={_NOISE}:allf=t" if _NOISE != '0' else ""))


def run_pipe(frames_dir, out_mp4, f0, f1, procs):
    """Compose frames in parallel and stream them, in order, into one ffmpeg encode (no intermediate PNGs)."""
    import multiprocessing as mp, subprocess
    markers = load_markers(frames_dir)
    frames = list(range(f0, f1 + 1))
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-framerate', str(C.FPS), '-i', '-']
    audio = os.path.join(HERE, 'song.m4a')
    if os.path.exists(audio):                  # audio from the same offset as the first frame (stream copy cuts on frame boundaries)
        cmd += ['-ss', f'{f0 / C.FPS:.3f}', '-i', audio, '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '128k', '-shortest']
    cmd += ['-vf', POST, '-c:v', 'libx264', '-preset', 'medium', '-crf', os.environ.get('GG_CRF', '19'), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out_mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    last = None
    n = 0
    with mp.Pool(procs, initializer=_init, initargs=(frames_dir, markers)) as pool:
        for f, buf in pool.imap(_one, frames, chunksize=4):
            if buf is None:                 # a gap: repeat the previous frame rather than stall
                buf = last if last is not None else bytes(W * H * 3)
            last = buf
            ff.stdin.write(buf)
            n += 1
            if n % 300 == 0:
                print('piped', n, '/', len(frames), flush=True)
    ff.stdin.close()
    ff.wait()
    print('encoded', out_mp4, 'rc', ff.returncode, flush=True)


if __name__ == '__main__':
    import multiprocessing as mp
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    procs = 2
    if '--procs' in sys.argv:
        procs = int(sys.argv[sys.argv.index('--procs') + 1])
        args = [a for a in args if a != str(procs)]
    if '--pipe' in sys.argv:
        out_mp4 = sys.argv[sys.argv.index('--pipe') + 1]
        args = [a for a in args if a != out_mp4]
        fdir = args[0]
        f0 = int(args[1]) if len(args) > 1 else 0
        f1 = int(args[2]) if len(args) > 2 else C.NF - 1
        run_pipe(fdir, out_mp4, f0, f1, procs)
        sys.exit(0)
    fdir, odir = args[0], args[1]
    f0 = int(args[2]) if len(args) > 2 else 0
    f1 = int(args[3]) if len(args) > 3 else C.NF - 1
    os.makedirs(odir, exist_ok=True)
    markers = load_markers(fdir)
    frames = [f for f in range(f0, f1 + 1)]
    chunks = [frames[i::procs] for i in range(procs)]
    with mp.Pool(procs) as pool:
        pool.map(work, [(fdir, odir, ch, markers) for ch in chunks])
    print('overlay done', len(frames))
