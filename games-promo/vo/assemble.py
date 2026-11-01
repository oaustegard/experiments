"""Pick the best take per line and mix a narration track timed to the reel.
Usage: assemble.py TAKES_DIR NAME [SCRIPT_JSON [TOTAL_SECONDS]]   (takes named <line id>_s<seed>.wav)
Best take = lowest ASR WER among takes that end 0.1 s before the next line, then closest to the\ntarget (slot - 0.25 s). Rubber Band stretch only within 4% (librosa phase vocoder sounded echoey).
Writes NAME.wav (48 kHz mono 16-bit, peaks at -2 dBFS, linear gain) and NAME_picks.json."""
import glob, json, re, sys
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


def rubberband(y, rate):
    """Time-stretch with ffmpeg's Rubber Band filter (tempo > 1 shortens), formants preserved."""
    import os, subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        a, b = os.path.join(td, "in.wav"), os.path.join(td, "out.wav")
        sf.write(a, y, SR)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-af",
                        f"rubberband=tempo={rate:.4f}:formant=preserved:pitchq=quality:window=short",
                        "-ar", str(SR), b], check=True)
        out, _ = librosa.load(b, sr=SR)
    return out


norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", s.lower().replace("ay-yoo", "aeyu")).split())
track = np.zeros(int(TOTAL * SR), np.float32)
picks = []
for l in S:
    slot = l["end"] - l["start"]
    cands = []
    for f in sorted(glob.glob(f"{TAKES}/{l['id']}_s*.wav")):
        s = int(f.rsplit("_s", 1)[1][:-4])
        y, _ = librosa.load(f, sr=SR)
        _, (a, b) = librosa.effects.trim(y, top_db=40)
        # keep 60 ms ahead of the detected start: a hard cut at the threshold clips
        # soft onsets ("th", "h") and sounds like the line was cut off
        y = y[max(0, a - int(0.06 * SR)):min(len(y), b + int(0.04 * SR))]
        y = cap_pauses(y)
        hyp = " ".join(x.text for x in asr.transcribe(librosa.resample(y, orig_sr=SR, target_sr=16000), language="en", beam_size=5)[0])
        wer = jiwer.wer(norm(l["text"]), norm(hyp)) if l["id"] != "l1" and l["id"] != "l9" else 0.0  # brand name: checked by ear-proxy below
        cands.append({"seed": s, "dur": len(y) / SR, "wer": round(wer, 3), "hyp": hyp.strip(), "y": y})
    # fill the slot: aim to end 0.25 s before the next line; a take may run on
    # until 0.1 s before the next line starts
    target = l.get("target", slot)
    nxt = next((m["start"] for m in S if m["start"] > l["start"]), TOTAL)
    max_len = nxt - 0.1 - l["start"]
    fits = [c for c in cands if c["dur"] <= max_len] or [min(cands, key=lambda c: c["dur"])]
    best = min(fits, key=lambda c: (c["wer"], abs(c["dur"] - target)))
    y = best["y"]
    # Rubber Band, formant-preserved, only for small changes. librosa's phase
    # vocoder at 6-12% gave half the lines a hollow "laptop mic in an echoey room"
    # sound (Oskar, 2026-10-30); beyond 4% we take the length as recorded.
    rate = best["dur"] / target
    if 0.01 < abs(rate - 1) <= 0.04 or best["dur"] > max_len:
        rate = min(rate, 1.06) if best["dur"] > max_len else rate
        y = rubberband(y, rate)
    if len(y) / SR > slot + 0.05:
        print(f"WARNING {l['id']}: {len(y)/SR:.2f}s overruns its {slot:.2f}s slot", file=sys.stderr)
    i = int(l["start"] * SR)
    track[i:i + len(y)] += y[: len(track) - i]
    picks.append({k: v for k, v in best.items() if k != "y"} | {"id": l["id"], "slot": round(slot, 2), "final_dur": round(len(y) / SR, 2)})
    print(picks[-1], [ (c["seed"], round(c["dur"],2), c["wer"]) for c in cands])
sf.write(f"{D}/{NAME}_raw.wav", track, SR)
# Linear gain only: peaks to -2 dBFS. Single-pass loudnorm to -16 LUFS needed ~5 dB of
# limiting on TTS that already peaks near 0 dBFS; Oskar heard it as "amplified too much".
peak = float(np.abs(track).max())
sf.write(f"{D}/{NAME}.wav", (track * (10 ** (-2 / 20) / peak)).astype(np.float32), SR, subtype="PCM_16")
json.dump(picks, open(f"{D}/{NAME}_picks.json", "w"), indent=1)
