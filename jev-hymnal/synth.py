"""Turn a run's choices into note events and a stereo WAV (numpy synthesis, no samples).

events(log) -> list of dicts {t, voice, midi, dur, vel, bar, step}; shared with render.py so the
picture and the sound come from the same events.
"""
import numpy as np
from hymnal import HARMONY, DRUMS, BASS, KEYS, LEAD

SR, BPM = 44100, 120
STEP = 60 / BPM / 4          # sixteenth = 0.125 s
BAR = STEP * 16              # 2.0 s
A_MINOR = [0, 2, 3, 5, 7, 8, 10]   # semitones from A


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def scale_note(chord, degree):
    """Scale degree counted from the chord root inside A minor (G# over E major), octave 5."""
    root = HARMONY[chord][1] % 12            # pitch class of the chord root
    scale = [(9 + s) % 12 for s in A_MINOR]  # A minor pitch classes
    if chord == "E":
        scale[scale.index(7)] = 8            # G -> G#
    i0 = scale.index(root)
    i = i0 + degree
    pc = scale[i % 7]
    octv = 5 + i // 7
    base = 12 * (octv + 1) + pc
    while base < 64:
        base += 12
    return base


def events(log):
    ev = []
    for r in log:
        b, c, t0 = r["bar"], r["choice"], (r["bar"] - 1) * BAR
        chord = c["harmony"]
        for voice, steps in DRUMS[c["drums"]][2].items():
            for s in steps:
                ev.append(dict(t=t0 + s * STEP, voice=voice, midi=None, dur=STEP, vel=1.0, bar=b, step=s))
        root = HARMONY[chord][1]
        for s, n, semi in BASS[c["bass"]][2]:
            ev.append(dict(t=t0 + s * STEP, voice="bass", midi=root + semi, dur=n * STEP, vel=1.0, bar=b, step=s))
        voicing = HARMONY[chord][2]
        arp = list(voicing) + [m + 12 for m in voicing]
        for s, n, what in KEYS[c["keys"]][2]:
            notes = voicing if what == "chord" else [arp[what]]
            kind = "pad" if c["keys"] == "pad" else "keys"
            for m in notes:
                ev.append(dict(t=t0 + s * STEP, voice=kind, midi=m, dur=n * STEP, vel=1.0, bar=b, step=s))
        for s, n, deg in LEAD[c["lead"]][2]:
            ev.append(dict(t=t0 + s * STEP, voice="lead", midi=scale_note(chord, deg), dur=n * STEP, vel=1.0, bar=b, step=s))
    return ev


def _env(n, a, d_rate):
    t = np.arange(n) / SR
    e = np.exp(-t * d_rate)
    na = max(1, int(a * SR))
    e[:na] *= np.linspace(0, 1, na)
    return e


def _lp(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    try:  # scipy when present: live mode renders each bar inside a deadline
        from scipy.signal import lfilter
        return lfilter([1 - a], [1, -a], x)
    except ImportError:
        pass
    y = np.empty_like(x); acc = 0.0
    for i in range(len(x)):   # one-pole lowpass; buffers are short
        acc = (1 - a) * x[i] + a * acc; y[i] = acc
    return y


RNG = np.random.default_rng(7)


def voice_wave(e):
    v, dur = e["voice"], e["dur"]
    if v == "kick":
        n = int(0.4 * SR); t = np.arange(n) / SR
        f = 45 + 110 * np.exp(-t * 30)
        return 0.7 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7), 0.0
    if v == "snare":
        n = int(0.25 * SR); t = np.arange(n) / SR
        noise = np.diff(RNG.standard_normal(n + 1))
        return (0.25 * noise * np.exp(-t * 22) + 0.3 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30)), 0.1
    if v in ("hat", "ohat", "shaker", "crash"):
        dec = {"hat": 70, "ohat": 12, "shaker": 35, "crash": 3}[v]
        n = int({"hat": 0.08, "ohat": 0.3, "shaker": 0.12, "crash": 1.8}[v] * SR)
        noise = np.diff(np.diff(RNG.standard_normal(n + 2)))
        gain = {"hat": 0.08, "ohat": 0.07, "shaker": 0.05, "crash": 0.09}[v]
        att = 0.02 if v == "shaker" else 0.001
        return gain * noise * _env(n, att, dec), (0.3 if v != "crash" else 0.0)
    f = mtof(e["midi"])
    n = int((dur + (0.6 if v == "pad" else 0.12)) * SR); t = np.arange(n) / SR
    if v == "bass":
        saw = 2 * ((t * f) % 1) - 1
        w = _lp(saw, 500) * 0.5
        env = np.minimum(1, np.maximum(0, (dur + 0.05 - t) / 0.05)) * _env(n, 0.005, 1.5)
        return w * env, 0.0
    if v == "pad":
        w = sum(2 * ((t * f * d) % 1) - 1 for d in (0.995, 1.0, 1.006)) / 3
        w = _lp(w, 1400) * 0.13
        rel = np.clip((dur + 0.6 - t) / 0.6, 0, 1)
        return w * np.minimum(1, t / 0.5) * rel, 0.0
    if v == "keys":
        w = (2 * np.abs(2 * ((t * f) % 1) - 1) - 1) * 0.12
        return w * _env(n, 0.003, 9 if dur <= STEP else 4), 0.0
    if v == "lead":
        vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1, t / 0.3)
        ph = np.cumsum(f * vib) / SR
        w = (np.sign(np.sin(2 * np.pi * ph)) * 0.5 + np.sin(2 * np.pi * ph) * 0.5)
        w = _lp(w, 2600) * 0.11
        rel = np.clip((dur + 0.1 - t) / 0.1, 0, 1)
        return w * np.minimum(1, t / 0.01) * rel, 0.0
    raise ValueError(v)


def render_audio(ev, seconds, normalize=True):
    """Mix events into stereo. normalize=False returns the raw sum (live mode applies the
    master stage, 0.9 * tanh(1.4 x), at playback so overlapping bar tails sum before it)."""
    out = np.zeros((int(seconds * SR) + SR, 2))
    lead_bus = np.zeros(len(out))
    for e in ev:
        w, pan = voice_wave(e)
        i = int(e["t"] * SR); j = min(len(out), i + len(w)); w = w[: j - i]
        l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
        out[i:j, 0] += w * l; out[i:j, 1] += w * r
        if e["voice"] in ("lead", "keys"):
            lead_bus[i:j] += w
    d = int(3 * STEP * SR)   # dotted-eighth ping-pong delay on lead and keys
    for k, g in enumerate((0.35, 0.22, 0.13, 0.08)):
        sh = d * (k + 1); ch = k % 2
        out[sh:, 1 - ch] += g * lead_bus[: len(out) - sh]
    if not normalize:
        return out.astype(np.float32)
    out = np.tanh(out * 1.4)
    return (out / np.max(np.abs(out)) * 0.9).astype(np.float32)


def write_wav(path, x):
    import wave
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
