import json, re, difflib
from faster_whisper import WhisperModel
n = json.load(open('narration.json'))
items = {"intro": n["intro"]}
for ri, r in enumerate(n["runs"]):
    items[f"r{ri}-title"] = r["title"]
    for bi, b in enumerate(r["beats"]): items[f"r{ri}-b{bi}"] = b["text"]
    items[f"r{ri}-end"] = r["end"]
m = WhisperModel("small.en", device="cpu", compute_type="int8")
norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("id", "id")).split()
bad = []
for k, t in items.items():
    import soundfile as sf, numpy as np
    from scipy.signal import resample_poly
    x, sr = sf.read(f"vo/{k}.wav", dtype="float32")
    if x.ndim > 1: x = x.mean(axis=1)
    x = resample_poly(x, 16000, sr).astype("float32")
    segs, _ = m.transcribe(x, beam_size=5)
    heard = " ".join(s.text for s in segs)
    a, b = norm(t), norm(heard)
    ops = [o for o in difflib.SequenceMatcher(None, a, b).get_opcodes() if o[0] != "equal"]
    diffs = [(" ".join(a[i1:i2]), " ".join(b[j1:j2])) for _, i1, i2, j1, j2 in ops]
    print(k, "OK" if not diffs else diffs)
    if diffs: bad.append(k)
print("needs review:", bad)
