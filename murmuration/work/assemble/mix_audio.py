"""Narration track from timeline.json (24 kHz lines -> 48 kHz), then music ducked under it."""
import json, os, subprocess, numpy as np, soundfile as sf
from scipy.signal import resample_poly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
tl = json.load(open(f"{ROOT}/timeline.json"))
SR = 48000
total = tl["total"]
n = int(round(total * SR))
voice = np.zeros(n, dtype=np.float64)
for s in tl["scenes"]:
    for l in s["lines"]:
        y, sr = sf.read(f"{ROOT}/{l['file']}")
        if y.ndim > 1: y = y.mean(1)
        y = resample_poly(y, SR, sr) if sr != SR else y
        e = int(0.015 * SR); y[:e] *= np.linspace(0, 1, e); y[-e:] *= np.linspace(1, 0, e)
        i = int(round((s["t0"] + l["start"]) * SR))
        voice[i:i + len(y)] += y[: max(0, n - i)]
peak = np.abs(voice).max()
voice = voice / peak * 0.85
sf.write(f"{ROOT}/audio/voice_track.wav", np.stack([voice, voice], 1).astype(np.float32), SR, subtype="FLOAT")
print("voice track", len(voice) / SR, "s")
