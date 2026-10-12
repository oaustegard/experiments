"""Speak every line in its speaker's designed voice: best of TAKES by faster-whisper WER.
  python3 narrate.py   -> takes/*.wav, narration.json [{s, l, spk, text, wav, dur, wer, heard}]
Reuses muninn-story/narrate_gemini.py (Interactions API through the CF gateway)."""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, soundfile as sf
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "muninn-story"))
import narrate_gemini as ng
import importlib.util as _iu
_sp = _iu.spec_from_file_location("film_script", HERE / "script.py"); _m = _iu.module_from_spec(_sp); _sp.loader.exec_module(_m)
SCENES = _m.SCENES  # by path: muninn-story has its own script.py on sys.path

VOICE = json.load(open(HERE / "voices.json"))
TAKES = int(os.environ.get("TAKES", "2"))
BASE = {
 "M": "a retired detective telling an old case across a table; measured, warm, natural pace, no dramatic pauses",
 "O": "cool, clipped and controlled, a widow refusing to show grief",
 "F": "nervous and quick, joking a little to hide fear",
 "L": "flat, exact and steady, like reading out measurements",
 "D": "genial, warm and assured, a cultivated man helping the police",
 "B": "blunt, gruff and impatient",
 "R": "frightened but proud, plain and earnest",
}
MOOD = {"open": "hushed, grave", "act1": "", "act2": "", "pause": "slower, direct to the listener", "reveal": "quietly certain, building", "coda": "quiet, sombre"}
LETTER = "quiet, steady, resigned; reading his own letter"

def say(t):  # spelling for the synthesizer only
    for a, b in [("Häfeli", "Haefeli"), ("régleuse", "ray-gluhz")]:
        t = t.replace(a, b)
    return t

def lines():
    for i, s in enumerate(SCENES):
        for j, (spk, text, _) in enumerate(s["lines"]):
            style = BASE[spk]
            if spk == "M" and MOOD[s["part"]]: style += "; " + MOOD[s["part"]]
            if spk == "D" and s["part"] == "coda": style = LETTER
            yield i, j, spk, text, style

def main():
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly
    T = HERE / "takes"; T.mkdir(exist_ok=True)
    L = list(lines())
    def take(a):
        i, j, spk, text, style, n = a
        p = T / f"{i:02d}_{j}_{n}.wav"
        if not p.exists():
            y = ng.synth(say(text), style, voice=VOICE[spk]); sf.write(p, y, ng.SR)
        return p
    jobs = [(*x, n) for x in L for n in range(TAKES)]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(take, jobs))
    asr = WhisperModel(os.environ.get("ASR", "small.en"), device="cpu", compute_type="int8")
    out = []
    for i, j, spk, text, style in L:
        best = None
        for n in range(TAKES):
            p = T / f"{i:02d}_{j}_{n}.wav"; c = p.with_suffix(".txt")
            if not c.exists():
                y, _ = sf.read(p, dtype="float32"); y16 = resample_poly(y, 2, 3)
                c.write_text(" ".join(s.text for s in asr.transcribe(y16, language="en", beam_size=1)[0]))
            heard = ng.norm_hyp(c.read_text()); w = ng.wer(text, heard)
            if best is None or w < best[0]: best = (w, p, heard)
        y, _ = sf.read(best[1], dtype="float32"); y = ng.cap_pauses(y, cap=0.6)
        fin = T / f"{i:02d}_{j}_best.wav"; sf.write(fin, y, ng.SR)
        out.append(dict(s=i, l=j, spk=spk, text=text, wav=str(fin.relative_to(HERE)), dur=round(len(y) / ng.SR, 3), wer=round(best[0], 3), heard=best[2]))
        print(f"{i:02d}.{j} {spk} wer={best[0]:.3f} {len(y)/ng.SR:5.1f}s", flush=True)
    json.dump(out, open(HERE / "narration.json", "w"), indent=1, ensure_ascii=False)
    print("total", round(sum(o["dur"] for o in out), 1), "s; mean wer", round(np.mean([o["wer"] for o in out]), 3))

if __name__ == "__main__":
    main()
