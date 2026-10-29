"""Turn the GPU render into 24-bit/96 kHz WAVs: the dry render and a version through a synthetic stone-square reverb.

    python3 master.py audio.npy OUTDIR      (audio: float32 [n, 2] at 96 kHz)
"""
import sys
from pathlib import Path
import numpy as np
from scipy import signal
import wave

SR = 96000


def impulse_response(seconds=3.2, seed=3):
    """Stereo IR: sparse early reflections, then decorrelated noise decaying faster in high bands."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR); t = np.arange(n) / SR
    ir = np.zeros((n, 2))
    edges = [20, 250, 1000, 3000, 7000, 14000, 40000]
    t60 = [2.6, 2.3, 1.9, 1.4, 0.9, 0.5]
    for lo, hi, T in zip(edges[:-1], edges[1:], t60):
        sos = signal.butter(4, [lo, min(hi, SR / 2 - 100)], "bandpass", fs=SR, output="sos")
        for c in range(2):
            ir[:, c] += signal.sosfilt(sos, rng.normal(0, 1, n)) * 10 ** (-3 * t / T)
    onset = np.clip((t - 0.028) / 0.06, 0, 1) ** 2         # diffuse tail builds after the early field
    ir *= onset[:, None]
    for _ in range(18):                                     # early reflections off stone walls
        d = rng.uniform(0.012, 0.09); g = rng.uniform(0.2, 0.6) * (0.012 / d) ** 0.5
        ir[int(d * SR), rng.integers(0, 2)] += g * rng.choice([-1, 1]) * 30
    return ir / np.sqrt((ir ** 2).sum(0)).max()


def save(x, path):
    x = x / np.abs(x).max() * 10 ** (-1 / 20)              # -1 dBFS peak
    q = np.clip(np.round(x * 8388607), -8388608, 8388607).astype("<i4").ravel()
    b = q.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()   # little-endian low 3 bytes = 24-bit PCM
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(3); w.setframerate(SR); w.writeframes(b)


if __name__ == "__main__":
    a = np.load(sys.argv[1]).astype(np.float64).reshape(-1, 2)
    out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    hp = signal.butter(2, 18, "highpass", fs=SR, output="sos")
    a = signal.sosfilt(hp, a, axis=0)
    save(a, out / "bells-dry-96k24.wav")
    ir = impulse_response()
    wet = np.stack([signal.fftconvolve(a[:, c], ir[:, c]) for c in range(2)], 1)
    dry = np.pad(a, ((0, len(wet) - len(a)), (0, 0)))
    mix = dry / np.abs(dry).max() + 0.55 * wet / np.abs(wet).max()
    save(mix, out / "bells-hall-96k24.wav")
    f, p = signal.welch(a[:, 0], SR, nperseg=8192)
    band = lambda lo, hi: 10 * np.log10(p[(f >= lo) & (f < hi)].mean() / p.max())
    print("dry peak", np.abs(a).max(), "| band energy dB re max: 20k-30k", round(band(20000, 30000), 1),
          "| 10k-20k", round(band(10000, 20000), 1), "| 1k-4k", round(band(1000, 4000), 1))
