"""Compare where voicing starts in each chorus part with the lead (part 0), after removing each part's
deliberate delay. Voicing onsets mark vowel starts, so a shared timing plan puts them on the same frames.

    python chorus_onsets.py DIR    (part<k>.f32 and delays.json from run/chorus_sync.mjs)
"""
import json, sys
from pathlib import Path
import numpy as np, parselmouth

d = Path(sys.argv[1])
delays = json.loads((d / "delays.json").read_text())


def onsets(k):
    x = np.fromfile(d / f"part{k}.f32", dtype=np.float32).astype(np.float64)
    shift = int(round(delays[k] * 24))                      # samples of deliberate delay
    x = x[shift:]
    f0 = parselmouth.Sound(x, 24000).to_pitch(time_step=0.005, pitch_floor=60, pitch_ceiling=700).selected_array["frequency"]
    v = f0 > 0
    return np.flatnonzero(v[1:] & ~v[:-1]) * 5.0 + 5.0         # ms


lead = onsets(0)
for k in range(1, len(delays)):
    o = onsets(k)
    diffs = np.array([np.min(np.abs(o - t)) for t in lead]) if len(o) else np.array([np.inf])
    print(f"part {k}: {len(lead)} lead onsets, matched |diff| median {np.median(diffs):.0f} ms, "
          f"p90 {np.percentile(diffs, 90):.0f} ms, within 20 ms: {100 * np.mean(diffs <= 20):.0f}%")
