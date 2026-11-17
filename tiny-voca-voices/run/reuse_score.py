"""Score the reuse check: the same voice file and text from the tuner, tinyvoca.js and speak.py.

Reports samples, Whisper WER, and Praat pitch against the tuner's audio (frame by frame, in cents). The noise
signal is random in each runtime, so waveforms differ; durations and pitch do not depend on it.

    python reuse_score.py OUT_DIR   (with tuner-1.wav, library-1.wav, python-1.wav, reuse.json)
"""
import json
import re
import sys
from pathlib import Path

import jiwer
import numpy as np
import parselmouth
from faster_whisper import WhisperModel
from scipy.io import wavfile
from scipy.signal import resample_poly

out = Path(sys.argv[1])
TEXT = "The meeting moved to three thirty on Thursday, so please bring the quarterly numbers and a working laptop."
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().replace("3:30", "three thirty").replace("3.30", "three thirty")).split()
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)


def load(name):
    rate, x = wavfile.read(out / f"{name}-1.wav")
    return x.astype(np.float64) / 32768


def f0(x):
    return parselmouth.Sound(x, 24000).to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=500).selected_array["frequency"]


ref = load("tuner")
ref_f0 = f0(ref)
report = json.loads((out / "reuse.json").read_text())
for name in ("tuner", "library", "python"):
    x = load(name)
    hyp = " ".join(s.text for s in asr.transcribe(resample_poly(x, 2, 3).astype(np.float32), beam_size=5, language="en")[0])
    fx = f0(x)
    n = min(len(fx), len(ref_f0))
    both = (fx[:n] > 0) & (ref_f0[:n] > 0)
    cents = np.abs(1200 * np.log2(fx[:n][both] / ref_f0[:n][both]))
    report[name] = {"samples": len(x), "wer_pct": round(100 * jiwer.wer(" ".join(norm(TEXT)), " ".join(norm(hyp))), 1),
                    "f0_median_hz": round(float(np.median(fx[fx > 0])), 1),
                    "pitch_vs_tuner_cents_median": round(float(np.median(cents)), 1),
                    "pitch_vs_tuner_cents_p90": round(float(np.percentile(cents, 90)), 1), "transcript": hyp.strip()}
    print(name, report[name], flush=True)
(out / "reuse.json").write_text(json.dumps(report, indent=1))
