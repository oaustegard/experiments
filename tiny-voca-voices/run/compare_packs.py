"""Packed tunable graphs against fp32, same noise: durations, pitch (Praat, cents), band levels, WER.

    python compare_packs.py onnx/packed phones.json tunable.json
"""
import json
import math
import sys
from pathlib import Path

import jiwer
import numpy as np
import onnxruntime as ort
import parselmouth
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

packed, phones, meta = Path(sys.argv[1]), json.load(open(sys.argv[2])), json.load(open(sys.argv[3]))
space = meta["voice_space"]
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
norm = lambda s: __import__("re").sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " ")).split()


def voice(mix, table):
    rows = np.array(space[table]["rows"], dtype=np.float32)
    return (np.array(mix, dtype=np.float32) @ rows)[None]


SETTINGS = {
    "F2": dict(mix=(1, 0)),
    "M2": dict(mix=(0, 1)),
    "M2-baritone": dict(mix=(0, 1), formant=0.92, shift=-5 * math.log(2) / 12, range=1.1, speed=0.95),
}


def run(session, p, s, noise):
    one = lambda v: np.array([v], dtype=np.float32)
    feeds = {"tokens": np.array([p["ids"]], dtype=np.int64), "stress": np.array([p["stress"]], dtype=np.int64),
             "voice": voice(s["mix"], "voice"), "frame_voice": voice(s["mix"], "frame_voice"),
             "speed": one(s.get("speed", 1.0)), "pitch_shift": one(s.get("shift", 0.0)), "pitch_range": one(s.get("range", 1.0)),
             "formant": one(s.get("formant", 1.0)), "tilt": one(0.0), "breath": one(0.0), "noise": noise}
    audio, durations = session.run(None, feeds)
    return audio[0], durations[0]


def f0(x):
    snd = parselmouth.Sound(x.astype(np.float64), 24000)
    return snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500).selected_array["frequency"]


def bands(x):
    spec = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, 1024)[::256] * np.hanning(1024), axis=1)) ** 2
    edges = np.geomspace(100, 11000, 9)
    hz = np.fft.rfftfreq(1024, 1 / 24000)
    return np.array([10 * np.log10(spec[:, (hz >= a) & (hz < b)].mean() + 1e-12) for a, b in zip(edges, edges[1:])])


rng = np.random.default_rng(0)
noise = rng.standard_normal((1, 256 * 3000)).astype(np.float32)
ref = ort.InferenceSession(str(packed / "tunable-noise-fp32.onnx"))
result = {}
for path in sorted(packed.glob("tunable-noise-*.onnx")):
    variant = path.stem.replace("tunable-noise-", "")
    sess = ort.InferenceSession(str(path))
    cents, band_err, dur_same, dur_total, wers, corr = [], [], 0, 0, [], []
    for name, s in SETTINGS.items():
        for p in phones:
            a, d = run(sess, p, s, noise)
            ra, rd = run(ref, p, s, noise)
            dur_same += int((d == rd).sum()); dur_total += len(d)
            if variant != "fp32":
                n = min(len(a), len(ra))
                fa, fr = f0(a[:n]), f0(ra[:n])
                both = (fa > 0) & (fr > 0)
                cents.extend(np.abs(1200 * np.log2(fa[both] / fr[both])))
                band_err.append(np.abs(bands(a[:n]) - bands(ra[:n])).max())
                corr.append(np.corrcoef(a[:n], ra[:n])[0, 1])
            hyp = " ".join(x.text for x in asr.transcribe(resample_poly(a / max(1, np.abs(a).max() / 0.99), 2, 3).astype(np.float32), beam_size=5, language="en")[0])
            wers.append(jiwer.wer(" ".join(norm(p["text"])), " ".join(norm(hyp))))
    r = {"bytes": path.stat().st_size, "durations_identical_pct": round(100 * dur_same / dur_total, 2),
         "wer_pct": round(100 * float(np.mean(wers)), 2)}
    if cents:
        r.update(pitch_cents_median=round(float(np.median(cents)), 1), pitch_cents_p90=round(float(np.percentile(cents, 90)), 1),
                 band_db_max=round(float(np.max(band_err)), 2), waveform_corr_min=round(float(np.min(corr)), 4))
    result[variant] = r
    print(variant, r, flush=True)
json.dump(result, open(packed / "compare.json", "w"), indent=1)
