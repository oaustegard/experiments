"""Pitch against the PyTorch reference: Praat F0 every 10 ms on the ten lines, compared frame by frame where both
are voiced. Word times are identical across arms, so frames line up. Run with a Python that has parselmouth.

    python pitch.py ARM ...   -> results/pitch.json (median and 90th-percentile |difference| in cents, voicing agreement)
"""
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth

HERE = Path(__file__).resolve().parent
REF = "torch312"


def f0(arm, sid):
    snd = parselmouth.Sound(str(HERE / "out" / arm / f"{sid}.wav"))
    return snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400).selected_array["frequency"]


out = {}
for arm in sys.argv[1:]:
    cents, agree, total = [], 0, 0
    for i in range(1, 11):
        a, b = f0(arm, f"vo{i}"), f0(REF, f"vo{i}")
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        both = (a > 0) & (b > 0)
        cents += list(np.abs(1200 * np.log2(a[both] / b[both])))
        agree += int(((a > 0) == (b > 0)).sum())
        total += n
    c = np.array(cents)
    out[arm] = {"median_cents": round(float(np.median(c)), 1), "p90_cents": round(float(np.percentile(c, 90)), 1),
                "voicing_agree": round(agree / total, 4), "frames": int(len(c))}
    print(arm, out[arm], flush=True)
(HERE / "results" / "pitch.json").write_text(json.dumps(out, indent=1))
