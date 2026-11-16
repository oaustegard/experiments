"""Score rendered voices: Whisper small.en WER, median F0 and formants (Praat)."""
import json
import re
import sys
from pathlib import Path

import jiwer
import numpy as np
import parselmouth
from scipy.io import wavfile
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

out = Path(sys.argv[1])
texts = Path(sys.argv[2]).read_text().strip().split("\n\n")
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().replace("3:30", "three thirty").replace("3.30", "three thirty").replace("-", " ")).split()

def load16(path):
    rate, x = wavfile.read(path)
    return resample_poly(x.astype(np.float32) / 32768, 2, 3).astype(np.float32)


GRID = np.geomspace(300, 5000, 400)


def envelope(path):
    """Mean cepstrally smoothed log spectrum of loud frames on a log-frequency grid.
    Liftering at 1.5 ms removes harmonics, so voices at different pitch compare."""
    rate, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768
    n, hop = 1024, 256
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    energy = (frames ** 2).sum(1)
    frames = frames[energy > np.percentile(energy, 50)]
    logmag = np.log(np.abs(np.fft.rfft(frames, axis=1)) + 1e-9)
    cep = np.fft.irfft(logmag, axis=1)
    keep = int(0.0015 * rate)
    cep[:, keep:-keep] = 0
    smooth = np.fft.rfft(cep, axis=1).real.mean(0)
    hz = np.fft.rfftfreq(n, 1 / rate)
    return np.interp(GRID, hz, smooth)


def formant_scale(env, base):
    """Scale k such that env(f) best matches base(f / k)."""
    best = None
    for k in np.arange(0.80, 1.2001, 0.005):
        shifted = np.interp(GRID / k, GRID, base, left=np.nan, right=np.nan)
        ok = ~np.isnan(shifted)
        a, b = env[ok] - env[ok].mean(), shifted[ok] - shifted[ok].mean()
        r = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
        best = max(best or (r, k), (r, k))
    return round(float(best[1]), 3)


rows = {}
envs = {}
for wav in sorted(out.glob("*-[0-9].wav")):
    name, index = wav.stem.rsplit("-", 1)
    ref = texts[int(index) - 1]
    hyp = " ".join(s.text for s in asr.transcribe(load16(wav), beam_size=5, language="en")[0])
    sound = parselmouth.Sound(str(wav))
    pitch = sound.to_pitch(pitch_floor=70, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    formant = sound.to_formant_burg(maximum_formant=5500)
    times = [pitch.get_time_from_frame_number(i + 1) for i in range(pitch.n_frames)
             if pitch.selected_array["frequency"][i] > 0]
    fs = np.array([[formant.get_value_at_time(k, t) for k in (1, 2, 3)] for t in times])
    hnr = parselmouth.praat.call(sound.to_harmonicity_cc(), "Get mean", 0, 0)
    envs.setdefault(name, []).append(envelope(wav))
    r = rows.setdefault(name, {"wer": [], "f0": [], "F": [], "hnr": [], "hyp": []})
    r["wer"].append(jiwer.wer(" ".join(norm(ref)), " ".join(norm(hyp))))
    r["f0"].extend(f0)
    r["F"].append(fs)
    r["hnr"].append(hnr)
    r["hyp"].append(hyp.strip())

summary = {}
print(f"{'voice':16} {'WER%':>5} {'F0':>5} {'F1':>5} {'F2':>5} {'F3':>5} {'HNR':>5} {'fmt×':>6}")
for name, r in rows.items():
    F = np.nanmedian(np.concatenate(r["F"]), axis=0)
    s = {"wer_pct": round(100 * float(np.mean(r["wer"])), 1), "f0": round(float(np.median(r["f0"]))),
         "F1": round(F[0]), "F2": round(F[1]), "F3": round(F[2]), "hnr_db": round(float(np.mean(r["hnr"])), 1),
         "transcripts": r["hyp"]}
    base = "M2" if name.startswith("M2") else "F2"
    s["formant_scale_vs_" + base] = formant_scale(np.mean(envs[name], 0), np.mean(envs[base], 0))
    summary[name] = s
    print(f"{name:16} {s['wer_pct']:5} {s['f0']:5} {s['F1']:5} {s['F2']:5} {s['F3']:5} {s['hnr_db']:5} {s['formant_scale_vs_' + base]:6}")
(out / "scores.json").write_text(json.dumps(summary, indent=1))
