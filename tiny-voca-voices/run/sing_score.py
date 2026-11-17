"""Score sung audio: Praat pitch against the score's notes, and Whisper on the lyrics.

Pitch: each voiced frame against the nearest note the melody uses (transposed), in cents. A note-level check
lines up the score's timing with the audio: the median pitch inside each note's middle half, against that note.

    python sing_score.py OUT_DIR   (with sing.json and VOICE.wav from sing_check.mjs)
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
run = json.loads((out / "sing.json").read_text())
score = run["score"]
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
words = re.sub(r"[^a-z' ]", " ", score["lyrics"].lower().replace("-", "")).split()


def hz(name):
    m = re.match(r"([A-G])([#b]?)(-?\d)", name)
    step = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[m[1]] + (m[2] == "#") - (m[2] == "b")
    return 440 * 2 ** ((12 * (int(m[3]) + 1) + step - 69) / 12)


# Timeline of notes in seconds (the sung audio starts at the first syllable's onset, so allow a lag search).
beat = 60 / score["bpm"]
timeline, t = [], 0.0
for line in score["notes"].split("\n"):
    for entry in line.split():
        for part in entry.split("~"):
            name, beats = part.split("/")
            d = float(beats) * beat
            if name.lower() != "r":
                timeline.append((t, t + d, hz(name)))
            t += d
report = {}
for voice, case in run["cases"].items():
    if "error" in case:
        report[voice] = case
        continue
    rate, x = wavfile.read(out / f"{voice}.wav")
    x = x.astype(np.float64) / 32768
    shift = 2 ** (case["transpose"] / 12)
    pitch = parselmouth.Sound(x, rate).to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=700)
    f0 = pitch.selected_array["frequency"]
    times = np.array([pitch.get_time_from_frame_number(i + 1) for i in range(len(f0))])
    notes = np.array(sorted({h for _, _, h in timeline})) * shift
    voiced = f0 > 0
    nearest = np.abs(1200 * np.log2(f0[voiced][:, None] / notes[None])).min(1)
    # Note level: best global lag in [-0.4, 0.4] s, then the median pitch over each note's middle half.
    def note_errors(lag):
        errs = []
        for a, b, h in timeline:
            lo, hi = a + lag + (b - a) * 0.25, a + lag + (b - a) * 0.75
            sel = voiced & (times >= lo) & (times <= hi)
            if sel.sum() >= 3:
                errs.append(1200 * np.log2(np.median(f0[sel]) / (h * shift)))
        return np.array(errs)
    lag = min(np.arange(-0.4, 0.41, 0.02), key=lambda L: np.median(np.abs(note_errors(L))) if len(note_errors(L)) else 1e9)
    errs = note_errors(lag)
    hyp = " ".join(s.text for s in asr.transcribe(resample_poly(x, 2, 3).astype(np.float32), beam_size=5, language="en")[0])
    report[voice] = {"transpose": case["transpose"], "seconds": round(case["seconds"], 2),
                     "frames_within_50_cents_of_a_score_note_pct": round(100 * float((nearest < 50).mean()), 1),
                     "notes_measured": int(len(errs)), "notes_total": len(timeline),
                     "note_error_cents_median": round(float(np.median(np.abs(errs))), 1),
                     "notes_within_50_cents": int((np.abs(errs) < 50).sum()), "lag_s": round(float(lag), 2),
                     "wer_pct": round(100 * jiwer.wer(" ".join(words), " ".join(re.sub(r"[^a-z' ]", " ", hyp.lower()).split())), 1),
                     "transcript": hyp.strip()}
    print(voice, report[voice], flush=True)
(out / "sing_scores.json").write_text(json.dumps(report, indent=1))
