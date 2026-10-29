"""Fake camera and microphone for the headless check (e2e.mjs).

fixtures/hand.jpg is a generated photo (Gemini, via Muninn's image tool): a drawn circle and square on paper, a
child's index finger inside the circle. fixtures/nohand.png is the same picture with the hand painted out and the
circle's hidden arc redrawn, so the page can see the whole drawing before the hand arrives, as it would in use.
fake.y4m: 1.5 s without the hand, then 8.5 s with it, 640x480 at 30 fps (Chromium loops it).
fake.wav: 10 s, room noise at -60 dBFS and seven taps, one a second from 3 s, at -20 dBFS.
"""
import numpy as np, wave
from PIL import Image, ImageDraw
from pathlib import Path
F = Path(__file__).resolve().parent / "fixtures"
img = Image.open(F / "hand.jpg").convert("RGB")
a = np.asarray(img).astype(np.float32)
r, g, b = a[..., 0], a[..., 1], a[..., 2]
H, W = r.shape
yy, xx = np.mgrid[0:H, 0:W]
# skin: redder than it is blue, not paper-white, below the circle's top, in the hand's columns
skin = (r - b > 25) & (r > 120) & (r < 250) & (yy > 380) & (xx > 340) & (xx < 700)
paper = (r > 200) & (g > 200) & (b > 200) & (yy > 600) & (yy < 820) & (xx > 150) & (xx < 330)   # paper near the hand
pc = a[paper].mean(0)
m = Image.fromarray((skin * 255).astype(np.uint8)).filter(__import__("PIL.ImageFilter", fromlist=["x"]).MaxFilter(15))
mask = np.asarray(m) > 0
out = a.copy(); out[mask] = pc
# the paper's lower edge runs behind the hand; continue it as the table colour below y=830
out[mask & (yy > 828)] = np.array([150, 105, 60], np.float32)
o = Image.fromarray(out.clip(0, 255).astype(np.uint8))
d = ImageDraw.Draw(o)
d.ellipse([214, 213, 650, 646], outline=(25, 25, 25), width=11)   # redraw the circle (only its hidden arc changes)
o.save(F / "nohand.png")
print("hand pixels painted out:", int(mask.sum()))

def yuv420(im):
    im = im.resize((640, 480), Image.BILINEAR)
    x = np.asarray(im).astype(np.float32)
    Y = 0.257 * x[..., 0] + 0.504 * x[..., 1] + 0.098 * x[..., 2] + 16
    U = -0.148 * x[..., 0] - 0.291 * x[..., 1] + 0.439 * x[..., 2] + 128
    V = 0.439 * x[..., 0] - 0.368 * x[..., 1] - 0.071 * x[..., 2] + 128
    sub = lambda c: c.reshape(240, 2, 320, 2).mean((1, 3))
    return b"".join(np.clip(c, 0, 255).astype(np.uint8).tobytes() for c in (Y, sub(U), sub(V)))
fa, fb = yuv420(o), yuv420(img)
with open(F / "fake.y4m", "wb") as f:
    f.write(b"YUV4MPEG2 W640 H480 F30:1 Ip A1:1 C420jpeg\n")
    for k in range(300):
        f.write(b"FRAME\n" + (fa if k < 45 else fb))
sr = 48000; rng = np.random.default_rng(3); x = 10 ** (-60 / 20) * rng.standard_normal(10 * sr)
for t in range(3, 10):
    n = t * sr; k = np.arange(int(0.012 * sr))
    x[n:n + len(k)] += 10 ** (-20 / 20) * np.exp(-k / (0.0025 * sr)) * (0.6 * rng.standard_normal(len(k)) + 0.8 * np.sin(2 * np.pi * 150 * k / sr))
with wave.open(str(F / "fake.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
