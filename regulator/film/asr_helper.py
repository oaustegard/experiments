"""Transcribe takes from the end backwards so narrate.py finds them cached (it skips existing .txt)."""
import sys
from pathlib import Path
import soundfile as sf
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
asr = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=2)
for p in sorted(Path("takes").glob("*_[0-9].wav"), reverse=True):
    c = p.with_suffix(".txt")
    if c.exists(): continue
    y, _ = sf.read(p, dtype="float32")
    txt = " ".join(s.text for s in asr.transcribe(resample_poly(y, 2, 3), language="en", beam_size=1)[0])
    if not c.exists(): c.write_text(txt)
    print(p.name, flush=True)
