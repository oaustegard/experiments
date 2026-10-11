"""Score and mix, all synthesized in numpy at 48 kHz.

Wind that rises on each flight, a low D drone, one plucked string at each arrival (a knot counted,
as the prologue asks), the snap of the cord at the cut, voice ducking. Writes score.wav and mix.wav.
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
rng = np.random.default_rng(11)

def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype="band", fs=SR, output="sos"), x)

def env_at(points):  # piecewise-linear envelope from (time, value) points
    ts, vs = zip(*points); return np.interp(t, ts, vs)

# wind: pink-ish noise through a band whose level follows the flights
noise = np.cumsum(rng.standard_normal(N)) ; noise -= sosfilt(butter(1, 20, btype="low", fs=SR, output="sos"), noise)
noise /= np.abs(noise).max()
wind_lo, wind_hi = bp(noise, 120, 700), bp(rng.standard_normal(N), 400, 2400) * 0.25
pts = [(0, 0.25)]
for s in TL["stops"]:
    pts += [(s["t0"] - 0.5, 0.22), ((s["t0"] + s["t1"]) / 2, 0.85), (s["t1"] + 0.8, 0.22)]
pts += [(TL["outro"], 0.25), (TL["outro"] + 6, 0.6), (TL["cut"], 0.7), (TL["cut"] + 0.3, 0.05), (TL["duration"], 0.0)]
gust = 0.75 + 0.25 * np.sin(2 * np.pi * t / 7.3) * np.sin(2 * np.pi * t / 3.1)
wind = (wind_lo + wind_hi) * env_at(pts) * gust
wind /= np.abs(wind).max() + 1e-9

# drone: D2, A2, D3 with slow beating, swells in and out
drone = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in [(73.42, 1.0, 0), (73.6, 0.6, 1), (110.0, 0.45, 2), (146.83, 0.3, 3)])
drone = bp(drone, 50, 400) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 23.0)) * env_at([(0, 0), (4, 1), (TL["cut"], 1), (TL["cut"] + 0.4, 0.2), (TL["cut"] + 4, 0.6), (TL["duration"], 0)])
drone /= np.abs(drone).max() + 1e-9

# a plucked string (Karplus-Strong) at each arrival; D dorian, walking down as the knots are counted
def pluck(freq, dur=4.0, damp=0.996):
    n = int(SR / freq); buf = rng.uniform(-1, 1, n); out = np.zeros(int(dur * SR))
    for i in range(len(out)):
        out[i] = buf[i % n]; buf[i % n] = damp * 0.5 * (buf[i % n] + buf[(i + 1) % n])
    return out * np.exp(-np.arange(len(out)) / (SR * 1.6))
scale = [293.66, 261.63, 246.94, 220.0, 196.0, 174.61, 164.81, 146.83]
strings = np.zeros(N)
for j, s in enumerate(TL["stops"]):
    p = pluck(scale[j % len(scale)] / (1 if j < len(scale) else 2)); i = int(s["t1"] * SR)
    strings[i:i + len(p)] += p[: N - i]
strings /= np.abs(strings).max() + 1e-9

# the cut: a short bright snap and a falling whip
i = int(TL["cut"] * SR); m = int(0.35 * SR)
snap = rng.standard_normal(m) * np.exp(-np.arange(m) / (SR * 0.03))
snap = bp(snap, 1500, 9000) + 0.6 * np.sin(2 * np.pi * np.cumsum(np.linspace(900, 120, m)) / SR) * np.exp(-np.arange(m) / (SR * 0.08))
cut = np.zeros(N); cut[i:i + m] = snap / np.abs(snap).max()

score = 0.20 * wind + 0.16 * drone + 0.30 * strings + 0.55 * cut

# voice, upsampled, and ducking from its envelope
v, vsr = sf.read(HERE / "narration.wav", dtype="float32"); v = resample_poly(v, SR, vsr)[:N]; v = np.pad(v, (0, N - len(v)))
k = int(0.25 * SR); venv = np.convolve(np.abs(v), np.ones(k) / k, "same"); venv /= venv.max()
duck = 1 - 0.55 * np.clip(venv * 6, 0, 1)
score = score * duck
sf.write(HERE / "score.wav", score.astype(np.float32), SR)
mix = 0.9 * v / (np.abs(v).max() + 1e-9) + score
mix /= np.abs(mix).max() / 0.89
sf.write(HERE / "mix.wav", mix.astype(np.float32), SR)
print("mix.wav", round(N / SR, 1), "s")
