"""Score and mix at 48 kHz, all synthesized in numpy: a clock's tick that stops when the regulator
stops and runs under the evidence board, the temple bell, the repeater striking ten low notes, two
doubled and eight high (faint after Félix's line, plainly in the reveal), and a low drone that rises
through the reveal. Voice lines are placed from timeline.json. Writes mix.wav.
"""
import json
from pathlib import Path
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

HERE = Path(__file__).resolve().parent
SR = 48000
TL = json.load(open(HERE / "timeline.json"))
N = int(TL["duration"] * SR)
t = np.arange(N) / SR
rng = np.random.default_rng(1899)
S = TL["scenes"]
def at(part): return [s for s in S if s["part"] == part]
def scene(shot, part=None): return next(s for s in S if s["shot"] == shot and (part is None or s["part"] == part))

def place(buf, x, t0, g=1.0):
    i = int(t0 * SR); j = min(N, i + len(x))
    if j > i: buf[i:j] += g * x[: j - i]

def bell(f0=146.8, dur=7.0):
    tt = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(tt)
    for r, a, d in [(0.5, 0.6, 3.5), (1.0, 1.0, 2.6), (1.183, 0.5, 2.0), (1.506, 0.45, 1.6), (2.0, 0.35, 1.2), (2.514, 0.25, 0.9), (2.662, 0.2, 0.8), (3.011, 0.12, 0.6)]:
        out += a * np.sin(2 * np.pi * f0 * r * tt + rng.uniform(0, 6)) * np.exp(-tt / d)
    out *= np.minimum(1, tt / 0.004)
    return out / np.abs(out).max()

def ping(f0, dur=1.2, decay=0.45):
    tt = np.arange(int(dur * SR)) / SR
    x = sum(a * np.sin(2 * np.pi * f0 * r * tt) * np.exp(-tt / (decay / r ** 0.5)) for r, a in [(1, 1), (2.76, 0.35), (5.4, 0.12)])
    return x * np.minimum(1, tt / 0.002) / 1.5

def tick(hi):
    n = int(0.03 * SR); x = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.004 * SR))
    return sosfilt(butter(2, [2200, 5200] if hi else [1500, 3800], btype="band", fs=SR, output="sos"), x) * 4

def place_local(buf, x, t0):
    i = int(t0 * SR); j = min(len(buf), i + len(x)); buf[i:j] += x[: j - i]

def repeater(level):
    out = np.zeros(int(7.2 * SR)); x = 0.0
    for _ in range(10): place_local(out, ping(523.25, 1.4, 0.6), x); x += 0.42           # hours, low
    x += 0.25
    for _ in range(2): place_local(out, ping(523.25, 1.0, 0.4), x); place_local(out, ping(783.99, 1.0, 0.4), x + 0.16); x += 0.55  # quarters
    x += 0.15
    for _ in range(8): place_local(out, ping(1046.5, 0.8, 0.25), x); x += 0.19           # minutes, high
    return out * level

music = np.zeros(N); voice = np.zeros(N); fx = np.zeros(N)

for s in S:
    for ln in s["lines"]:
        y, sr = sf.read(HERE / ln["wav"], dtype="float32")
        place(voice, resample_poly(y, 2, 1), ln["t0"])
        if ln.get("chime"):
            place(fx, repeater(ln["chime"]["level"]), ln["chime"]["t"])

def ticks(a, b, g):
    for k, x in enumerate(np.arange(a, b, 1.0)):
        place(fx, tick(k % 2 == 0), x, g)
reg = scene("regulator")
ticks(0.4, reg["lines"][0]["t0"] + 2.6, 0.10)            # the clock stops with the regulator
a1 = at("act1")[0]["t0"]; a2end = at("act2")[-1]["t1"]
ticks(a1 + 2, a2end, 0.045)
board = next(s for s in S if s["kind"] == "board")
ticks(at("pause")[0]["t0"], board["t1"], 0.11)

title = next(s for s in S if s["kind"] == "title")
place(fx, bell(), title["t0"] + 0.5, 0.35)
place(fx, bell(), board["t0"] + 0.2, 0.22)
end = next(s for s in S if s["kind"] == "end")
for k in range(3): place(fx, bell(), end["t0"] - 6 + k * 2.8, 0.25)

def drone(a, b, g0, g1, notes=(73.42, 110.0, 87.31)):
    i, j = int(a * SR), int(b * SR); tt = t[i:j]
    x = sum(np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.003 * tt) for f in notes)
    x = sosfilt(butter(2, 900, fs=SR, output="sos"), x)
    music[i:j] += x * np.interp(tt, [a, a + 3, b - 3, b], [0, g0, g1, 0])
op = at("open"); drone(op[0]["t0"], op[-1]["t1"], 0.035, 0.03)
rv = at("reveal"); drone(rv[0]["t0"], rv[-1]["t1"], 0.025, 0.06, notes=(73.42, 110.0, 87.31, 146.83))
co = at("coda"); drone(co[0]["t0"], co[-1]["t1"], 0.03, 0.02, notes=(73.42, 110.0, 92.5))

env = np.convolve(np.abs(voice), np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
duck = 1 - 0.55 * np.clip(env / (env.max() * 0.08), 0, 1)
mix = voice * 0.9 + (music + fx * 0.8) * duck
mix /= np.abs(mix).max() / 0.9
sf.write(HERE / "mix.wav", mix.astype(np.float32), SR)
print("mix.wav", round(N / SR, 1), "s")
