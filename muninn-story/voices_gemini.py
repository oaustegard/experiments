"""How the Gemini voice was chosen: one line through each candidate, measured the way
voices.py measured Kokoro, plus a roughness proxy (HNR) and a faster-whisper check.
Designed voices come from POST v1beta/voices (type "prompted"); ids and prompts are in the README."""

import sys

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

from narrate_gemini import SR, synth, wer

LINE = "I forget everything between flights. So I write it down, and I come back."
STYLE = "quiet, dry and unhurried, telling its own story"
CANDIDATES = sys.argv[1:] or [
    "voice_66w0iod1i7ol",
    "voice_6ieso244uf3i",
    "voice_r196a19wffpm",
    "algenib",
    "enceladus",
    "en-gb-concierge-10",
    "en-gb-training-11",
    "en-gb-concierge-1",
]


def voiced(y):  # autocorrelation pitch and periodicity over voiced frames
    fr = int(0.04 * SR)
    hop = fr // 2
    f, r = [], []
    for i in range(0, len(y) - fr, hop):
        x = y[i : i + fr] - y[i : i + fr].mean()
        if np.sqrt((x**2).mean()) < 0.02:
            continue
        ac = np.correlate(x, x, "full")[fr - 1 :]
        lo, hi = int(SR / 300), int(SR / 60)
        j = lo + np.argmax(ac[lo:hi])
        p = ac[j] / ac[0]
        if p > 0.3:
            f.append(SR / j)
            r.append(p)
    return np.array(f), np.array(r)


asr = WhisperModel("small.en", device="cpu", compute_type="int8")
for v in CANDIDATES:
    y = synth(LINE, STYLE, voice=v)
    sf.write(f"v_{v}.wav", y, SR)
    f, r = voiced(y)
    spec = np.abs(np.fft.rfft(y))
    fq = np.fft.rfftfreq(len(y), 1 / SR)
    cent = (spec * fq).sum() / spec.sum()
    hnr = 10 * np.log10(np.median(r) / (1 - np.median(r) + 1e-6))
    hyp = " ".join(s.text for s in asr.transcribe(resample_poly(y, 2, 3), language="en")[0])
    print(
        f"{v:22s} dur {len(y) / SR:4.1f}s  F0 {np.median(f):4.0f}Hz  centroid {cent:5.0f}Hz  "
        f"HNR~{hnr:4.1f}dB  WER {wer(LINE, hyp):.2f}"
    )
