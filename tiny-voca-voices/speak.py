"""Speak text in a tiny-voca voice file, with onnxruntime and dizione (no PyTorch).

    pip install onnxruntime numpy soundfile git+https://github.com/n4ze3m/dizione
    python speak.py my-voice.voice.json "Hello there." hello.wav

The voice file comes from the Voice Tuner (https://austegard.com/experiments/tiny-voca-voices/web/). The model
(10 MB) and dizione's dictionary (5 MB, from huggingface.co/n4ze3m/tiny-voca) are downloaded once into
~/.cache/tinyvoca; pass --cache to put them elsewhere. In your own code:

    from speak import TinyVoca
    tts = TinyVoca()
    audio = tts.speak("Hello there.", json.load(open("my-voice.voice.json")))   # float32, 24 kHz
"""
import argparse
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np

MODEL_URL = "https://austegard.com/experiments/tiny-voca-voices/web/tiny-voca-tunable.bin"
MODEL_SHA256 = "a3dcc3e6317e58621e4c1623992cffd471f67e5156519868dd5a31f69a50619d"
HF = "https://huggingface.co/n4ze3m/tiny-voca/resolve/d94d2a0338fd1e0eab7dd896ec0ad23d8cdf7c14/phonemizer/"
DATA = ["lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json", "lexicon.json", "sources.json"]
FORMAT = "tiny-voca-voice/1"
SAMPLE_RATE = 24000
MAX_TOKENS = 512
PEAK = 0.99

# Sentence splitting as in tiny-voca's infer.sentences / the browser demo.
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["')\]]))\s+(?=["'(\[]?[A-Z0-9])""")
_NOT_AN_END = re.compile(r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$",
                         re.IGNORECASE | re.ASCII)


def sentences(text):
    parts, current = [], ""
    for piece in _SENTENCE_END.split(re.sub(r"\s+", " ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [p for p in parts if any(c.isalnum() for c in p)]


def finished(text):
    body = text.rstrip()
    if not body:
        return text
    last = body.split()[-1]
    if any(c.isalnum() for c in last):
        marked = last.rstrip("\"'”’)]»")[-1:] in tuple(".!?…,;:")
    else:
        marked = any(c in ".!?…" for c in last) or last in (",", ";", ":")
    return text if marked else body + "."


def fetch(url, path, sha256=None):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            data = response.read()
        if sha256 and hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError(f"{url}: checksum mismatch")
        path.write_bytes(data)
    return path


def check_voice(voice):
    if voice.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} voice file")
    for key in ("voice", "frame_voice"):
        if len(voice.get(key, [])) != 192:
            raise ValueError(f"voice file: {key} must hold 192 numbers")
    if voice.get("model", {}).get("sha256", MODEL_SHA256) != MODEL_SHA256:
        raise ValueError("voice file was made for a different model file")
    return voice


class TinyVoca:
    def __init__(self, cache=Path.home() / ".cache" / "tinyvoca", model=None, threads=None):
        import onnxruntime as ort
        cache = Path(cache)
        data = cache / "phonemizer"
        for name in DATA:
            fetch(HF + name, data / name)
        os.environ.setdefault("DIZIONE_DATA", str(data))
        import dizione
        self.dizione = dizione
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        path = Path(model) if model else fetch(MODEL_URL, cache / "tiny-voca-tunable.bin", MODEL_SHA256)
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def _utterances(self, sentence):
        try:
            phones = self.dizione.phonemize(sentence)
        except ValueError:
            return []
        if len(phones.ids) <= MAX_TOKENS - 12:
            return [phones]
        pieces = re.split(r"(?<=[,;:—])\s+", sentence)
        if len(pieces) > 1:
            return [u for p in pieces for u in self._utterances(p)]
        words = sentence.split(" ")
        if len(words) < 2:
            return []
        half = (len(words) + 1) // 2
        return self._utterances(" ".join(words[:half])) + self._utterances(" ".join(words[half:]))

    def synthesize(self, ids, stress, voice):
        one = lambda v: np.array([v], dtype=np.float32)
        audio, _ = self.session.run(None, {
            "tokens": np.array([ids], dtype=np.int64), "stress": np.array([stress], dtype=np.int64),
            "voice": np.array([voice["voice"]], dtype=np.float32),
            "frame_voice": np.array([voice["frame_voice"]], dtype=np.float32),
            "speed": one(voice["speed"]), "pitch_shift": one(voice["pitch_shift"]),
            "pitch_range": one(voice["pitch_range"]), "formant": one(voice["formant"]),
            "tilt": one(voice["tilt"]), "breath": one(voice["breath"])})
        return audio[0]

    def speak(self, text, voice):
        check_voice(voice)
        chunks = [self.synthesize(list(u.ids), list(u.stress), voice)
                  for s in sentences(finished(text)) for u in self._utterances(s)]
        if not chunks:
            raise ValueError("tiny-voca only reads English")
        audio = np.concatenate(chunks)
        loudest = float(np.abs(audio).max())
        return audio * (PEAK / loudest) if loudest > PEAK else audio


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("voice", type=Path, help="a .voice.json file from the Voice Tuner")
    parser.add_argument("text")
    parser.add_argument("output", type=Path, help=".wav file to write")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "tinyvoca")
    parser.add_argument("--model", type=Path, help="a local copy of tiny-voca-tunable.bin")
    args = parser.parse_args()
    import soundfile as sf
    tts = TinyVoca(args.cache, args.model)
    audio = tts.speak(args.text, json.loads(args.voice.read_text()))
    sf.write(args.output, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"wrote {args.output} ({len(audio) / SAMPLE_RATE:.1f} s)")


if __name__ == "__main__":
    main()
