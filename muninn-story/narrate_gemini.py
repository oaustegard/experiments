"""Narration with Gemini 3.8 Flash TTS, a drop-in replacement for narrate.py (Kokoro).

Writes narration.wav + timeline.json in the same shape, so music.py and render.py run unchanged.
Calls go through Muninn's Cloudflare AI Gateway (CF_ACCOUNT_ID, CF_GATEWAY_ID, CF_API_TOKEN).

API notes, measured 2026-09-24:
- Use the Interactions API (POST v1beta/interactions). generateContent returns audio too,
  but a "Style: text" prefix is spoken aloud and systemInstruction is refused
  ("Developer instruction is not enabled for this model"); style belongs in a
  speech_metadata annotation on the text.
- The voice is a designed voice (POST v1beta/voices, type "prompted"), stored, expires 2027-09-24.
- Output is 24 kHz mono 16-bit WAV with leading/trailing silence, trimmed here so GAP is the gap.
"""

import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from script import SCENES

MODEL = "gemini-3.8-flash-tts"
VOICE = "voice_66w0iod1i7ol"  # "muninn-raven-rp", designed from VOICE_PROMPT below
VOICE_PROMPT = (
    "An ageless narrator who is secretly a raven that learned to speak: a low, dry, quietly "
    "amused male voice with a faint natural rasp at the edges of words, like a bird's throat. "
    "Soft southern British accent. Unhurried and intimate, close to the microphone, never "
    "theatrical or booming. Thoughtful pauses, a little wry."
)
BASE = "plain and low, natural speaking pace, no dramatic pauses"
MOOD = {
    "poem": "hushed, reciting something old",
    "empty": "matter-of-fact",
    "oskar": "warm, a little fond",
    "flights": "brisker, alert",
    "clock": "rueful, dry",
    "theseus": "thoughtful, unsure",
    "return": "quiet and warm, settling",
}


def SAY(s):  # spelling for the synthesizer only; subtitles keep the real names
    return s.replace("Huginn", "Hoogin").replace("Muninn", "Moonin")


SR = 24000
GAP = 0.55


def _url(path):
    return (
        f"https://gateway.ai.cloudflare.com/v1/{os.environ['CF_ACCOUNT_ID']}/"
        f"{os.environ['CF_GATEWAY_ID']}/google-ai-studio/v1beta/{path}"
    )


def synth(text, style, voice=VOICE):
    body = {
        "model": MODEL,
        "input": [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [{"type": "speech_metadata", "style": style}],
                    }
                ],
            }
        ],
        "response_format": {"type": "audio", "mime_type": "audio/wav", "sample_rate": SR},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    hdr = {"Content-Type": "application/json", "cf-aig-authorization": f"Bearer {os.environ['CF_API_TOKEN']}"}
    for a in range(5):
        r = requests.post(_url("interactions"), json=body, headers=hdr, timeout=180)
        if r.status_code not in (429, 500, 502, 503, 504):
            break
        time.sleep(2 * 2**a)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
    audio = [c for st in r.json()["steps"] for c in st.get("content", []) if c.get("type") == "audio"]
    y, sr = sf.read(io.BytesIO(base64.b64decode(audio[-1]["data"])), dtype="float32")
    assert sr == SR, sr
    return trim(y)


def trim(y, pad=0.06):
    k = int(0.02 * SR)
    env = np.convolve(np.abs(y), np.ones(k) / k, "same")
    on = np.nonzero(env > 0.04 * env.max())[0]
    a, b = max(0, on[0] - int(pad * SR)), min(len(y), on[-1] + int(pad * SR))
    return y[a:b]


def cap_pauses(y, cap=0.7, xf=0.01):
    """Shorten silent runs inside a line to `cap` seconds. The designed voice was prompted with
    'thoughtful pauses' and takes 1-1.8 s mid-line breaths that no style annotation removes."""
    k = int(0.02 * SR)
    env = np.convolve(np.abs(y), np.ones(k) / k, "same")
    quiet = env < 0.04 * env.max()
    out = []
    i = 0
    n = len(y)
    c = int(cap * SR)
    f = int(xf * SR)
    while i < n:
        j = i
        while j < n and quiet[j] == quiet[i]:
            j += 1
        seg = y[i:j]
        if quiet[i] and len(seg) > c:  # keep the edges of the silence, drop its middle
            a, b = seg[: c // 2].copy(), seg[-(c // 2) :].copy()
            r = np.linspace(1, 0, f)
            a[-f:] *= r
            b[:f] *= r[::-1]
            seg = np.concatenate([a, b])
        out.append(seg)
        i = j
    return np.concatenate(out)


def wer(ref, hyp):
    import difflib

    def n(s):
        return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).split()

    r, h = n(ref), n(hyp)
    m = sum(b.size for b in difflib.SequenceMatcher(None, r, h).get_matching_blocks())
    return 1 - m / len(r) + max(0, len(h) - m) / len(r)  # misses + insertions (a spoken style leak)


def norm_hyp(h):
    """Undo ASR spelling choices that are not speech errors: digits, the TTS respellings, Oscar."""
    from num2words import num2words

    h = re.sub(
        r"\d[\d,]*", lambda m: " " + num2words(int(m.group().replace(",", ""))).replace(",", "") + " ", h
    )
    for a, b in [
        ("Hoogin", "Huginn"),
        ("Hugin", "Huginn"),
        ("Moonin", "Muninn"),
        ("Munin", "Muninn"),
        ("Oscar", "Oskar"),
        ("back-up", "backup"),
        ("back up", "backup"),
    ]:
        h = re.sub(rf"\b{a}\b", b, h)
    return h


if __name__ == "__main__":
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    asr = WhisperModel(os.environ.get("ASR", "medium.en"), device="cpu", compute_type="int8")
    jobs = [
        (key, i, s, f"{BASE}; {MOOD[key]}" if key in MOOD else BASE)
        for key, sents in SCENES
        for i, s in enumerate(sents)
    ]
    os.makedirs("takes", exist_ok=True)  # every take is kept; reruns reuse them

    def take(key, i, s, style, n):
        p = f"takes/{key}_{i}_{n}.wav"
        if not os.path.exists(p):
            sf.write(p, synth(SAY(s), style), SR)
        return sf.read(p, dtype="float32")[0]

    ASR_CACHE = Path("takes/asr.json")
    cache = json.loads(ASR_CACHE.read_text()) if ASR_CACHE.exists() else {}

    def check(y, s, key):  # whisper wants 16 kHz; transcripts cached per take
        if key not in cache:
            cache[key] = " ".join(
                x.text for x in asr.transcribe(resample_poly(y, 2, 3), language="en", beam_size=5)[0]
            )
            ASR_CACHE.write_text(json.dumps(cache, indent=0))
        return wer(s, norm_hyp(cache[key])), cache[key]

    with ThreadPoolExecutor(3) as ex:
        list(ex.map(lambda j: take(*j, 0), jobs))
    out = {}
    for key, i, s, style in jobs:
        best = None
        for n in range(int(os.environ.get("TAKES", "4"))):
            y = take(key, i, s, style, n)
            e, hyp = check(y, s, f"{key}_{i}_{n}")
            if best is None or e < best[1]:
                best = (y, e, hyp, n)
            if e == 0:
                break
        out[(key, i)] = best[:3]
        if best[1] > 0:
            print(f"  {key}/{i} best take {best[3]} wer {best[1]:.2f}: {best[2]!r}", flush=True)

    timeline, audio, errs = [], [], []
    for key, sents in SCENES:
        sc = {"key": key, "lines": []}
        buf = [np.zeros(int(0.5 * SR), np.float32)]
        t = 0.5
        for i, s in enumerate(sents):
            y, e, hyp = out[(key, i)]
            errs.append(e)
            y = cap_pauses(y)
            sc["lines"].append({"text": s, "start": t, "end": t + len(y) / SR})
            buf += [y, np.zeros(int(GAP * SR), np.float32)]
            t += len(y) / SR + GAP
        buf.append(np.zeros(int(0.9 * SR), np.float32))
        t += 0.9
        sc["dur"] = t
        timeline.append(sc)
        audio.append(np.concatenate(buf))
        print(f"{key:8s} {t:5.1f}s", flush=True)
    full = np.concatenate(audio)
    sf.write("narration.wav", full, SR)
    Path("timeline.json").write_text(json.dumps(timeline, indent=1))
    print(
        "total",
        round(len(full) / SR, 1),
        "s; mean line WER",
        round(float(np.mean(errs)), 3),
        "max",
        round(float(np.max(errs)), 3),
    )
