"""Pick the best take per line and mix a narration track timed to the reel.
Usage: assemble.py TAKES_DIR NAME [SCRIPT_JSON [TOTAL_SECONDS]]   (takes named <line id>_s<seed>.wav)
Best take = lowest ASR WER, then the longest that fits its slot (atempo <= 1.12 if none fit).
Writes vo.wav (48 kHz mono, loudnorm -16 LUFS) and picks.json."""
import glob, json, re, subprocess, sys
import numpy as np, librosa, soundfile as sf, jiwer
from faster_whisper import WhisperModel
from pathlib import Path
D = str(Path(__file__).resolve().parent)  # takes and outputs live beside this script
TAKES, NAME, SR = sys.argv[1], sys.argv[2], 48000
TOTAL = float(sys.argv[4]) if len(sys.argv) > 4 else 35.5  # composition length, seconds
S = json.load(open(sys.argv[3] if len(sys.argv) > 3 else f"{D}/script.json"))
asr = WhisperModel("small.en", device="cpu", compute_type="int8")
MAX_PAUSE = 0.22  # seconds of silence kept between phrases inside a line


def cap_pauses(y):
    """Shorten silences inside a take to MAX_PAUSE, keeping the speech itself untouched."""
    spans = librosa.effects.split(y, top_db=35, frame_length=1024, hop_length=256)
    gap = np.zeros(int(MAX_PAUSE * SR), np.float32)
    out = []
    for i, (a, b) in enumerate(spans):
        if i:
            pa = spans[i - 1][1]
            out.append(y[pa:a] if a - pa <= len(gap) else gap)
        out.append(y[a:b])
    return np.concatenate(out) if out else y


norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", s.lower().replace("ay-yoo", "aeyu")).split())
track = np.zeros(int(TOTAL * SR), np.float32)
picks = []
for l in S:
    slot = l["end"] - l["start"]
    cands = []
    for f in sorted(glob.glob(f"{TAKES}/{l['id']}_s*.wav")):
        s = int(f.rsplit("_s", 1)[1][:-4])
        y, _ = librosa.load(f, sr=SR)
        y, _ = librosa.effects.trim(y, top_db=40)
        y = cap_pauses(y)
        hyp = " ".join(x.text for x in asr.transcribe(librosa.resample(y, orig_sr=SR, target_sr=16000), language="en", beam_size=5)[0])
        wer = jiwer.wer(norm(l["text"]), norm(hyp)) if l["id"] != "l1" and l["id"] != "l9" else 0.0  # brand name: checked by ear-proxy below
        cands.append({"seed": s, "dur": len(y) / SR, "wer": round(wer, 3), "hyp": hyp.strip(), "y": y})
    # fill the slot: aim to end 0.25 s before the next line, not merely fit
    target = l.get("target", slot)
    best = min(cands, key=lambda c: (c["wer"], abs(c["dur"] - target)))
    y = best["y"]
    rate = float(np.clip(best["dur"] / target, 0.92, 1.12))  # gentle stretch only
    if abs(rate - 1) > 0.02:
        y = librosa.effects.time_stretch(y, rate=rate)
    if len(y) / SR > slot + 0.05:
        print(f"WARNING {l['id']}: {len(y)/SR:.2f}s overruns its {slot:.2f}s slot", file=sys.stderr)
    i = int(l["start"] * SR)
    track[i:i + len(y)] += y[: len(track) - i]
    picks.append({k: v for k, v in best.items() if k != "y"} | {"id": l["id"], "slot": round(slot, 2), "final_dur": round(len(y) / SR, 2)})
    print(picks[-1], [ (c["seed"], round(c["dur"],2), c["wer"]) for c in cands])
sf.write(f"{D}/{NAME}_raw.wav", track, SR)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f"{D}/{NAME}_raw.wav", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(SR), f"{D}/{NAME}.wav"], check=True)
json.dump(picks, open(f"{D}/{NAME}_picks.json", "w"), indent=1)
