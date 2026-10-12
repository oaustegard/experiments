"""Frames for The Regulator: Ken Burns over the chosen stills, with subtitles, speaker tags,
clock cards, title, cast, evidence board and end cards drawn in pycairo.

  python3 render.py preview T1 T2 ...   -> preview/t<T>.png
  python3 render.py video [WORKERS]     -> picture.mp4 (no audio), rendered in parallel parts
"""
import json, math, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import cairo
from PIL import Image
from cairokit import Shaper

HERE = Path(__file__).resolve().parent
W, H = 1280, 720
TL = json.load(open(HERE / "timeline.json"))
FPS, XF, TOTAL = TL["fps"], TL["xf"], TL["duration"]
SC = TL["scenes"]
CHOSEN = json.load(open(HERE / "chosen.json"))

F = HERE / "fonts"
SERIF = Shaper(str(F / "cormorant-garamond-latin-500-normal.ttf"))
SERIF_B = Shaper(str(F / "cormorant-garamond-latin-600-normal.ttf"))
ITAL = Shaper(str(F / "cormorant-garamond-latin-500-italic.ttf"))
MONO = Shaper(str(F / "jetbrains-mono-latin-400-normal.ttf"))

ENAMEL = (0.94, 0.93, 0.89)
INK = (0.09, 0.12, 0.16)
BLUE = (0.16, 0.31, 0.55)
BRASS = (0.83, 0.66, 0.36)
PALE = (0.93, 0.91, 0.86)

CLOCK = dict(kitchen="ROSA'S CLOCK", morbier="HALL CLOCK", regulator="REGULATOR", wF="FÉLIX'S WATCH",
             wB="GIRARD'S WATCH", wD="DOCTOR'S WATCH", wL="LINA'S WATCH", repeater="THE REPEATER")
WHO = dict(O="ODILE MONNIER", F="FÉLIX MONNIER", L="LINA RUEDI", D="DR CASIMIR VAUTHIER", B="BASTIEN GIRARD", R="ROSA HÄFELI")
NAME = dict(O="Odile Monnier", F="Félix Monnier", L="Lina Ruedi", D="Dr Casimir Vauthier", B="Bastien Girard", R="Rosa Häfeli")
ROLE = dict(O="the wife", F="the nephew", L="the régleuse", D="the physician", B="the watch merchant", R="the maid")

BOARD = [
 ("At dinner, the temple strikes nine", "Félix's watch 8:40  ·  Lina's 9:00  ·  the doctor's 9:02  ·  Girard's not shown"),
 ("Lina, in the atelier", "hall clock strikes ten as the regulator shows 9:48  ·  regulator agrees with the temple at ten  ·  sent down at 10:20 by the regulator, went straight down"),
 ("Rosa, in the kitchen", "believes her clock is a quarter fast  ·  Lina arrives 10:47  ·  piano 10:58 to 11:11  ·  hall clock strikes eleven at 11:13"),
 ("Lina, in the kitchen", "piano 10:58 to 11:11 by Rosa's clock  ·  saw the master with Rosa's clock that afternoon"),
 ("The doctor", "left the salon at 10:45 by his watch; the piano began as he went out and stopped when he came back  ·  hall clock struck eleven a moment later"),
 ("Odile", "the piano began as he went out and stopped when he came back  ·  hall clock struck eleven a moment later"),
 ("Félix, in the library", "steps up 10:15  ·  thump and repeater 10:18, ten low notes  ·  steps down 10:21  ·  Girard arrives 10:32"),
 ("Girard, in the street", "out at 10:21  ·  piano starts 10:29  ·  faint bells from the atelier 10:34  ·  in at 10:46, the hall clock just struck eleven and shows 11:02"),
 ("The rules", "one of the six lied  ·  beliefs about clocks may be wrong  ·  the temple bell is true  ·  no room is more than five minutes from any other  ·  the blow fell as the repeater sounded"),
]

_imgs = {}
def still(shot):
    if shot not in _imgs:
        im = Image.open(HERE / CHOSEN[shot]).convert("RGB")
        s = max(W / im.width, H / im.height)
        _imgs[shot] = im.resize((round(im.width * s * 1.25), round(im.height * s * 1.25)), Image.LANCZOS)
    return _imgs[shot]

yy, xx = np.mgrid[0:H, 0:W]
VIG = (1 - 0.42 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) ** 1.4).clip(0.45, 1)[..., None].astype(np.float32)

def ease(p): return p * p * (3 - 2 * p)

def kenburns(shot, kb, p):
    im = still(shot); S = im.width / (W * 1.25) * (W / (im.width / 1.25)) if False else 1.25
    p = ease(min(max(p, 0), 1))
    z, cx, cy = 1.0, 0.5, 0.5
    if kb == "in": z = 1.0 + 0.13 * p
    elif kb == "out": z = 1.13 - 0.13 * p
    elif kb == "left": z, cx = 1.1, 0.56 - 0.12 * p
    elif kb == "right": z, cx = 1.1, 0.44 + 0.12 * p
    elif kb == "up": z, cy = 1.1, 0.58 - 0.16 * p
    else: z = 1.04
    cover = min(im.width / W, im.height / H)          # source pixels per output pixel at z = 1
    vw, vh = W * cover / z, H * cover / z
    x0 = min(max(cx * im.width - vw / 2, 0), im.width - vw)
    y0 = min(max(cy * im.height - vh / 2, 0), im.height - vh)
    a = vw / W
    out = im.transform((W, H), Image.AFFINE, (a, 0, x0, 0, a, y0), Image.BICUBIC)
    return np.asarray(out, dtype=np.float32) / 255.0

def text(c, shaper, s, x, y, size, rgb, a=1.0, anchor="l"):
    w = shaper.width(s, size)
    if anchor == "c": x -= w / 2
    elif anchor == "r": x -= w
    c.new_path(); shaper.text_path(c, s, x, y, size); c.set_source_rgba(*rgb, a); c.fill()
    return w

def wrap(shaper, s, size, maxw):
    out, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if shaper.width(t, size) > maxw and cur: out.append(cur); cur = w
        else: cur = t
    return out + [cur]

def dial(c, x, y, r, reading, a=1.0):
    h, m = map(int, reading.split(":")); hh = h % 12 + m / 60
    c.set_source_rgba(*ENAMEL, a); c.arc(x, y, r, 0, 2 * math.pi); c.fill()
    c.set_line_width(max(1.2, r * 0.06)); c.set_source_rgba(*BRASS, a); c.arc(x, y, r, 0, 2 * math.pi); c.stroke()
    for i in range(60):
        if r < 40 and i % 5: continue
        ang = i / 60 * 2 * math.pi; r1 = r * 0.9; r2 = r * (0.78 if i % 5 == 0 else 0.85)
        c.set_line_width(r * (0.035 if i % 5 == 0 else 0.015)); c.set_source_rgba(*INK, a)
        c.move_to(x + r1 * math.sin(ang), y - r1 * math.cos(ang)); c.line_to(x + r2 * math.sin(ang), y - r2 * math.cos(ang)); c.stroke()
    c.set_line_cap(cairo.LINE_CAP_ROUND)
    for ang, ln, wd in ((hh / 12 * 2 * math.pi, 0.5, 0.075), (m / 60 * 2 * math.pi, 0.76, 0.045)):
        c.set_line_width(r * wd); c.set_source_rgba(*BLUE, a)
        c.move_to(x, y); c.line_to(x + r * ln * math.sin(ang), y - r * ln * math.cos(ang)); c.stroke()
    c.set_source_rgba(*BLUE, a); c.arc(x, y, r * 0.07, 0, 2 * math.pi); c.fill()

def lerp_reading(a, b, p):
    ha, ma = map(int, a.split(":")); hb, mb = map(int, b.split(":"))
    ta, tb = ha * 60 + ma, hb * 60 + mb
    t = round(ta + (tb - ta) * ease(min(max(p, 0), 1)))
    return f"{t // 60}:{t % 60:02d}"

def card(c, x, y, key, reading, a, p):
    """A chip: dial, timepiece name, reading. 'A=B' turns the dial from A back to true time B."""
    if "=" in reading:
        r0, r1 = reading.split("="); shown = lerp_reading(r0, r1, (p - 0.15) / 0.5)
        big, small = f"{r0} → {r1}", "· TRUE TIME"
    else:
        shown, big, small = reading, reading, ""
    w = 280 if "=" in reading else 222
    c.set_source_rgba(0.05, 0.07, 0.10, 0.74 * a); c.rectangle(x, y, w, 66); c.fill()
    c.set_source_rgba(*BRASS, 0.8 * a); c.set_line_width(1); c.rectangle(x + 0.5, y + 0.5, w - 1, 65); c.stroke()
    dial(c, x + 34, y + 33, 24, shown, a)
    text(c, MONO, CLOCK[key] + (" " + small if small else ""), x + 68, y + 25, 11, BRASS, a)
    if "=" in reading:  # the mono face has no arrow glyph: draw one
        w0 = text(c, MONO, r0, x + 68, y + 50, 21, PALE, a)
        ax = x + 68 + w0 + 8; c.set_source_rgba(*BRASS, a); c.set_line_width(1.6)
        c.move_to(ax, y + 43); c.line_to(ax + 20, y + 43); c.stroke()
        c.move_to(ax + 14, y + 38); c.line_to(ax + 20, y + 43); c.line_to(ax + 14, y + 48); c.stroke()
        text(c, MONO, r1, ax + 28, y + 50, 21, PALE, a)
    else:
        text(c, MONO, big, x + 68, y + 50, 21, PALE, a)

def scene_at(t):
    for sc in SC:
        if sc["t0"] <= t < sc["t1"]: return sc
    return SC[-1]

def overlay(t):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H); c = cairo.Context(s)
    sc = scene_at(t)
    fade = max(0, min(1, (t - sc["t0"]) / 0.8, (sc["t1"] - t) / 0.6))
    if sc["kind"] == "title":
        a = max(0, min(1, (t - sc["t0"] - 0.6) / 1.2, (sc["t1"] - t) / 0.8))
        c.set_source_rgba(0, 0, 0, 0.35 * a); c.rectangle(0, 0, W, H); c.fill()
        text(c, SERIF_B, "T H E   R E G U L A T O R", W / 2, H / 2 + 6, 64, PALE, a, "c")
        text(c, ITAL, "La Chaux-de-Fonds, 31 December 1899", W / 2, H / 2 + 52, 26, BRASS, a, "c")
    if sc["kind"] == "cast":
        text(c, SERIF_B, NAME[sc["who"]], 70, 110, 46, PALE, fade)
        text(c, ITAL, ROLE[sc["who"]], 72, 146, 25, BRASS, fade)
    if sc["kind"] == "board":
        a = min(1, (t - sc["t0"]) / 1.0)
        c.set_source_rgba(0.03, 0.05, 0.08, 0.88 * a); c.rectangle(0, 0, W, H); c.fill()
        text(c, MONO, "THE EVIDENCE  ·  PAUSE HERE TO SOLVE", 80, 64, 14, BRASS, a)
        y = 104
        for head, body in BOARD:
            text(c, SERIF_B, head, 80, y, 21, PALE, a)
            for ln in wrap(SERIF, body, 19, 800):
                text(c, SERIF, ln, 390, y, 19, (0.82, 0.82, 0.78), a); y += 23
            y += 11
        text(c, ITAL, "Who struck Aurèle Monnier, and at what true minute?", W / 2, H - 30, 25, BRASS, a, "c")
    if sc["kind"] == "end":
        a = min(1, (t - sc["t0"]) / 1.2)
        c.set_source_rgba(0, 0, 0, 0.55 * a); c.rectangle(0, 0, W, H); c.fill()
        rows = [("The Regulator", SERIF_B, 50, PALE, -70),
                ("Written by Muninn. Pictures by Gemini 3 Pro Image. Voices by Gemini 3.8 Flash TTS.", ITAL, 22, PALE, -12),
                ("Every clue checked by Claude Haiku 5.5; the solution proved unique with z3.", ITAL, 22, PALE, 20),
                ("Work the case yourself: muninn.austegard.com/scratch/regulator", MONO, 15, BRASS, 70)]
        for s1, f, sz, col, dy in rows:
            text(c, f, s1, W / 2, H / 2 + dy, sz, col, a, "c")
    if sc.get("dial"):
        dial(c, W - 230, H / 2 - 40, 150, sc["dial"][1], 0.94 * fade)
    for ln in sc["lines"]:
        if ln["t0"] - 0.15 <= t <= ln["t1"] + 0.35:
            la = max(0, min(1, (t - ln["t0"] + 0.15) / 0.3, (ln["t1"] + 0.35 - t) / 0.3))
            rows = wrap(SERIF, ln["text"], 29, 1060)
            y0 = H - 34 - 34 * (len(rows) - 1)
            top = y0 - 36 - (24 if ln["spk"] != "M" else 0)
            g = cairo.LinearGradient(0, top - 30, 0, top + 20); g.add_color_stop_rgba(0, 0, 0, 0, 0); g.add_color_stop_rgba(1, 0, 0, 0, 0.5 * la)
            c.set_source(g); c.rectangle(0, top - 30, W, H); c.fill()
            if ln["spk"] != "M":
                text(c, MONO, WHO[ln["spk"]], W / 2, y0 - 34, 13, BRASS, la, "c")
            for k, r in enumerate(rows):
                text(c, SERIF, r, W / 2, y0 + 34 * k, 29, PALE, la, "c")
            lp = (t - ln["t0"]) / max(0.5, ln["t1"] - ln["t0"])
            for k, (key, reading) in enumerate(ln["cards"]):
                card(c, W - (280 if "=" in reading else 222) - 36, 36 + 78 * k, key, reading, la, lp)
    return s

def frame(t):
    sc = scene_at(t); i = sc["i"]
    p = (t - sc["t0"] + XF) / (sc["t1"] - sc["t0"] + XF)
    img = kenburns(sc["shot"], sc["kb"], p)
    if i > 0 and t - sc["t0"] < XF:
        pv = SC[i - 1]; q = (t - pv["t0"] + XF) / (pv["t1"] - pv["t0"] + XF)
        a = (t - sc["t0"]) / XF
        img = img * a + kenburns(pv["shot"], pv["kb"], q) * (1 - a)
    img = img * VIG
    o = overlay(t); arr = np.frombuffer(o.get_data(), np.uint8).reshape(H, o.get_stride() // 4, 4)[:, :W]
    al = arr[..., 3:4].astype(np.float32) / 255.0
    rgb = arr[..., [2, 1, 0]].astype(np.float32) / 255.0  # cairo is premultiplied BGRA
    img = img * (1 - al) + rgb
    fi = min(1, t / 1.5, (TOTAL - t) / 2.0)
    return (np.clip(img * fi, 0, 1) * 255).astype(np.uint8)

def part(args):
    k, a, b = args
    out = HERE / f"parts/p{k:02d}.mp4"
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    for f in range(a, b):
        ff.stdin.write(frame(f / FPS).tobytes())
    ff.stdin.close(); ff.wait(); return str(out)

if __name__ == "__main__":
    if sys.argv[1] == "preview":
        (HERE / "preview").mkdir(exist_ok=True)
        for t in map(float, sys.argv[2:]):
            Image.fromarray(frame(t)).save(HERE / f"preview/t{t:07.2f}.png"); print("preview", t)
    else:
        n = int(TOTAL * FPS); workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        (HERE / "parts").mkdir(exist_ok=True)
        cuts = [round(n * k / (workers * 3)) for k in range(workers * 3 + 1)]
        jobs = [(k, cuts[k], cuts[k + 1]) for k in range(len(cuts) - 1)]
        with Pool(workers) as pool:
            parts = pool.map(part, jobs)
        (HERE / "parts/list.txt").write_text("".join(f"file '{Path(p).name}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(HERE / "parts/list.txt"), "-c", "copy", str(HERE / "picture.mp4")], check=True)
        print("picture.mp4", n, "frames")
