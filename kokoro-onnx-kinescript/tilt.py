"""Level and spectral tilt of one arm against another, averaged over the ten lines: total energy difference and
energy difference per band (dB), waveform correlation. Kokoro's vocoder noise makes two runs of one arm differ,
so every pair is read against a repeat of the same arm.    python3 tilt.py -> results/tilt.json"""
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BANDS = [(0, 1000), (1000, 3000), (3000, 6000), (6000, 12000)]
PAIRS = [("torch312b", "torch312"), ("onnxb", "onnx"), ("torch313", "torch312"),
         ("torch312-customstft", "torch312"), ("torch312-customstft-fixed", "torch312"),
         ("onnx-export", "torch312"), ("onnx-export", "torch312-customstft"),
         ("onnx", "torch312"), ("onnx", "torch312-customstft-fixed"),
         ("onnx-fp16", "torch312"), ("onnx-int8", "torch312")]


def wav(arm, sid):
    with wave.open(str(HERE / "out" / arm / f"{sid}.wav")) as f:
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float64) / 32767


def power(x, n=1024, hop=256):
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    return (np.abs(np.fft.rfft(frames, axis=1)) ** 2).sum(0)


f = np.fft.rfftfreq(1024, 1 / 24000)
res = {}
for a, b in PAIRS:
    if not (HERE / "out" / a).exists() or not (HERE / "out" / b).exists():
        continue
    pa = sum(power(wav(a, f"vo{i}")) for i in range(1, 11))
    pb = sum(power(wav(b, f"vo{i}")) for i in range(1, 11))
    corr = []
    for i in range(1, 11):
        xa, xb = wav(a, f"vo{i}"), wav(b, f"vo{i}")
        m = min(len(xa), len(xb))
        corr.append(np.corrcoef(xa[:m], xb[:m])[0, 1])
    row = {"total_db": round(10 * np.log10(pa.sum() / pb.sum()), 2), "corr": round(float(np.mean(corr)), 4)}
    for lo, hi in BANDS:
        k = (f >= lo) & (f < hi)
        row[f"{lo // 1000}-{hi // 1000}k_db"] = round(10 * np.log10(pa[k].sum() / pb[k].sum()), 2)
    res[f"{a} vs {b}"] = row
    print(f"{a + ' vs ' + b:42s}", row)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "tilt.json").write_text(json.dumps(res, indent=1))
