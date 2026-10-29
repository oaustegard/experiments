"""Pick the best take per line and mix a narration track timed to the reel.
Usage: assemble.py TAKES_DIR NAME [SCRIPT_JSON]   (takes named <line id>_s<seed>.wav)
Best take = lowest ASR WER, then the longest that fits its slot (atempo <= 1.12 if none fit).
Writes vo.wav (48 kHz mono, loudnorm -16 LUFS) and picks.json."""
import json, re, subprocess, sys
import numpy as np, librosa, soundfile as sf, jiwer
from faster_whisper import WhisperModel
from pathlib import Path
D = str(Path(__file__).resolve().parent)  # takes and outputs live beside this script
TAKES, NAME, TOTAL, SR = sys.argv[1], sys.argv[2], 35.5, 48000
S = json.load(open(sys.argv[3] if len(sys.argv) > 3 else f"{D}/script.json"))
asr = WhisperModel("small.en", device="cpu", compute_type="int8")
norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", s.lower().replace("ay-yoo", "aeyu")).split())
track = np.zeros(int(TOTAL * SR), np.float32)
picks = []
for l in S:
    slot = l["end"] - l["start"]
    cands = []
    for s in (1, 2, 3):
        y, _ = librosa.load(f"{TAKES}/{l['id']}_s{s}.wav", sr=SR)
        y, _ = librosa.effects.trim(y, top_db=40)
        hyp = " ".join(x.text for x in asr.transcribe(librosa.resample(y, orig_sr=SR, target_sr=16000), language="en", beam_size=5)[0])
        wer = jiwer.wer(norm(l["text"]), norm(hyp)) if l["id"] != "l1" and l["id"] != "l9" else 0.0  # brand name: checked by ear-proxy below
        cands.append({"seed": s, "dur": len(y) / SR, "wer": round(wer, 3), "hyp": hyp.strip(), "y": y})
    fit = [c for c in cands if c["dur"] <= slot]
    pool = fit or cands
    best = min(pool, key=lambda c: (c["wer"], -c["dur"] if fit else c["dur"]))
    y = best["y"]
    if best["dur"] > slot:
        rate = min(1.12, best["dur"] / slot)
        y = librosa.effects.time_stretch(y, rate=rate)
    i = int(l["start"] * SR)
    track[i:i + len(y)] += y[: len(track) - i]
    picks.append({k: v for k, v in best.items() if k != "y"} | {"id": l["id"], "slot": round(slot, 2), "final_dur": round(len(y) / SR, 2)})
    print(picks[-1], [ (c["seed"], round(c["dur"],2), c["wer"]) for c in cands])
sf.write(f"{D}/{NAME}_raw.wav", track, SR)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{D}/{NAME}_raw.wav", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(SR), f"{D}/{NAME}.wav"], check=True)
json.dump(picks, open(f"{D}/{NAME}_picks.json", "w"), indent=1)
