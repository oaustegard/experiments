import json, numpy as np, soundfile as sf
from kokoro_onnx import Kokoro
from script import SCENES
k = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
# My voice: mostly bm_fable (breathy, mid pitch, the rasp) with bm_lewis (the low register) mixed in.
VOICE = 0.65 * k.get_voice_style("bm_fable") + 0.35 * k.get_voice_style("bm_lewis")
# Spelling for the synthesizer only; subtitles keep the real names.
SAY = lambda s: s.replace("Huginn", "Hoogin").replace("Muninn", "Moonin")
SR = 24000; GAP = 0.55
timeline = []; audio = []
for key, sents in SCENES:
    sc = {"key": key, "lines": []}; t = 0.0; buf = [np.zeros(int(0.5 * SR), np.float32)]; t = 0.5
    for s in sents:
        y, sr = k.create(SAY(s), voice=VOICE, speed=0.92, lang="en-gb")
        assert sr == SR
        sc["lines"].append({"text": s, "start": t, "end": t + len(y) / SR})
        buf += [y.astype(np.float32), np.zeros(int(GAP * SR), np.float32)]; t += len(y) / SR + GAP
    buf.append(np.zeros(int(0.9 * SR), np.float32)); t += 0.9
    sc["dur"] = t; timeline.append(sc); audio.append(np.concatenate(buf))
    print(f"{key:8s} {t:5.1f}s")
full = np.concatenate(audio); sf.write("narration.wav", full, SR)
json.dump(timeline, open("timeline.json", "w"), indent=1)
print("total", round(len(full) / SR, 1), "s")
