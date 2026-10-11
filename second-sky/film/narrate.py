"""Narrate the epic: one Gemini TTS call per sentence in Aun's voice, best of N takes by ASR word error.

  python3 narrate.py      -> takes/*.wav, narration.json [{seg, i, text, wav, dur, wer, heard}]

Reuses the gateway client from muninn-story/narrate_gemini.py. Takes are cached, so a rerun only
fills gaps. Voice: voice_r196a19wffpm, the stored "weathered old British storyteller" design.
"""
import json, os, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "muninn-story"))
import narrate_gemini as ng  # noqa: E402

VOICE = "voice_r196a19wffpm"
STYLE = "an old man telling a story by a winter fire; plain, unhurried but steady, not theatrical"
MOOD = {0: "quiet, beginning", 3: "rueful", 6: "low, ashamed", 9: "grieving softly", 13: "very quiet, near the end of his strength"}
TAKES = int(os.environ.get("TAKES", "2"))
SR = 24000


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [p for p in parts if p]


def segments():
    epic = json.load(open(HERE.parent / "work/epic.json"))
    segs = [dict(seg=0, cons=None, text=epic["prologue"])]
    segs += [dict(seg=i + 1, cons=s["id"], text=s["text"]) for i, s in enumerate(epic["stops"])]
    return epic, segs


def main():
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    _, segs = segments()
    jobs = []
    for s in segs:
        style = STYLE + ("; " + MOOD[s["seg"]] if s["seg"] in MOOD else "")
        for i, t in enumerate(sentences(s["text"])):
            jobs.append((s["seg"], i, t, style))
    (HERE / "takes").mkdir(exist_ok=True)

    def take(seg, i, t, style, n):
        p = HERE / f"takes/s{seg:02d}_{i:02d}_{n}.wav"
        if not p.exists():
            y = ng.cap_pauses(ng.synth(t, style, voice=VOICE), cap=0.6)
            sf.write(p, y, SR)
        return p

    # lines that scored badly last time get extra takes (RETAKE_WER, up to RETAKES)
    prev = {(l["seg"], l["i"]): l["wer"] for l in json.load(open(HERE / "narration.json"))} if (HERE / "narration.json").exists() else {}
    retake = float(os.environ.get("RETAKE_WER", "9"))
    ntakes = {(j[0], j[1]): int(os.environ.get("RETAKES", "4")) if prev.get((j[0], j[1]), 0) > retake else TAKES for j in jobs}
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda a: take(*a), [(*j, n) for j in jobs for n in range(ntakes[(j[0], j[1])])]))

    asr = WhisperModel("small.en", device="cpu", compute_type="int8")
    cache_p = HERE / "takes/asr.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    out = []
    for seg, i, t, style in jobs:
        best = None
        for n in range(ntakes[(seg, i)]):
            p = HERE / f"takes/s{seg:02d}_{i:02d}_{n}.wav"
            if p.name not in cache:
                y, _ = sf.read(p, dtype="float32")
                cache[p.name] = " ".join(x.text for x in asr.transcribe(resample_poly(y, 2, 3), language="en", beam_size=5)[0])
                cache_p.write_text(json.dumps(cache, indent=0))
            e = ng.wer(t, ng.norm_hyp(cache[p.name]))
            if best is None or e < best[0]:
                best = (e, p, cache[p.name])
        dur = sf.info(best[1]).duration
        out.append(dict(seg=seg, i=i, text=t, wav=best[1].name, dur=round(dur, 3), wer=round(best[0], 3), heard=best[2].strip()))
        print(f"s{seg:02d}.{i:02d} {dur:5.1f}s wer={best[0]:.2f} {t[:60]}", flush=True)
    (HERE / "narration.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("total speech", round(sum(o["dur"] for o in out) / 60, 2), "min;", sum(o["wer"] > 0.15 for o in out), "lines over 0.15 WER")


if __name__ == "__main__":
    main()
