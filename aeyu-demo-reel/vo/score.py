"""Score TTS clips: duration, wpm, pitch liveliness (F0 std in semitones), pause share, ASR WER.
Usage: score.py lines.json clip_dir [clip_dir...]  (clips named <id>.wav; text looked up by suffix)"""
import json, re, sys, glob, os
import numpy as np, librosa, jiwer
from faster_whisper import WhisperModel
texts = {L["id"]: L["text"] for L in json.load(open(sys.argv[1]))}
asr = WhisperModel("small.en", device="cpu", compute_type="int8")
norm = lambda s: re.sub(r"[^a-z' ]", " ", s.lower()).split()
for d in sys.argv[2:]:
    for f in sorted(glob.glob(f"{d}/*.wav")):
        name = os.path.basename(f)[:-4]
        key = next((k for k in texts if name == k or name.endswith("_" + k.split("_")[-1]) and k.split("_")[-1] == name.split("_")[-1]), None)
        txt = texts.get(name) or next(v for k, v in texts.items() if k.split("_")[-1] == name.split("_")[-1])
        y, sr = librosa.load(f, sr=16000)
        dur = len(y) / sr
        f0, vf, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024)
        f0 = f0[vf & ~np.isnan(f0)]
        st = 12 * np.log2(f0 / np.median(f0))
        rms = librosa.feature.rms(y=y, frame_length=400, hop_length=160)[0]
        pause = float(np.mean(rms < 0.02 * rms.max()))
        hyp = " ".join(s.text for s in asr.transcribe(y.astype(np.float32), language="en", beam_size=5)[0])
        wer = jiwer.wer(" ".join(norm(txt)), " ".join(norm(hyp)))
        print(f"{d.split('/')[-1]:>8} {name:10} {dur:5.2f}s {len(txt.split())/dur*60:4.0f}wpm  F0med {np.median(f0):4.0f}Hz  F0sd {np.std(st):4.2f}st  silence {pause:.2f}  WER {wer:.2f}  | {hyp.strip()}")
