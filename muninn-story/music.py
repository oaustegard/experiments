"""Score for the film: D dorian drone, pad, a Karplus-Strong lyre that answers in the
gaps between spoken lines, a frame drum for the flying scenes, wind throughout.
Everything is synthesized here; the timeline from narrate.py places it."""
import json, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

SR = 48000
TL = json.load(open("timeline.json"))
TAIL = 4.0                                    # music rings past the last word
starts = np.cumsum([0] + [s["dur"] for s in TL])
T = starts[-1] + TAIL
N = int(T * SR); t = np.arange(N) / SR
rng = np.random.default_rng(7)
L = np.zeros(N); R = np.zeros(N)

def hz(m): return 440 * 2 ** ((m - 69) / 12)
def add(sig, at, pan=0.0, gain=1.0):
    i = int(at * SR); j = min(N, i + len(sig)); sig = sig[: j - i] * gain
    L[i:j] += sig * np.sqrt((1 - pan) / 2); R[i:j] += sig * np.sqrt((1 + pan) / 2)
def lp(x, f, o=2): return sosfilt(butter(o, f, "low", fs=SR, output="sos"), x)
def bp(x, lo, hi, o=2): return sosfilt(butter(o, [lo, hi], "band", fs=SR, output="sos"), x)

# ---- per-scene intensity curve (0..1), smoothed
inten = {"ravens": .55, "poem": .35, "empty": .25, "ledger": .5, "boot": .45, "oskar": .6,
         "flights": .85, "clock": .35, "theseus": .55, "return": .8}
env = np.zeros(N)
for s, st in zip(TL, starts[:-1]):
    env[int(st * SR): int((st + s["dur"]) * SR)] = inten[s["key"]]
env[int(starts[-1] * SR):] = 0.5
k = int(3 * SR); env = np.convolve(env, np.ones(k) / k, "same")
fade = np.clip(t / 4, 0, 1) * np.clip((T - t) / 4, 0, 1)

# ---- drone: D2 + A2, detuned saws, low-passed, slow breathing
def saw(f, ph=0): return 2 * ((f * t + ph) % 1) - 1
dr = sum(saw(hz(38) * d, rng.random()) for d in (0.997, 1.0, 1.004)) \
   + 0.7 * sum(saw(hz(45) * d, rng.random()) for d in (0.998, 1.003))
dr = lp(dr, 520, 4) * (0.75 + 0.25 * np.sin(2 * np.pi * t / 11))
L += 0.035 * dr * fade * (0.6 + 0.6 * env); R += 0.035 * np.roll(dr, 900) * fade * (0.6 + 0.6 * env)

# ---- pad: one chord per scene, sines with slow vibrato, 2 s crossfades
CH = {"ravens": [50, 57, 62, 65], "poem": [48, 55, 60, 64], "empty": [45, 52, 57, 60],
      "ledger": [50, 57, 62, 69], "boot": [46, 53, 58, 62], "oskar": [43, 50, 55, 59],
      "flights": [50, 57, 60, 64], "clock": [45, 52, 55, 60], "theseus": [46, 53, 57, 62],
      "return": [50, 57, 62, 66]}                     # ends on D major: the raven comes home
for s, st in zip(TL, starts[:-1]):
    a, b = st - 1.0, st + s["dur"] + (TAIL if s["key"] == "return" else 1.0)
    i, j = int(max(0, a) * SR), min(N, int(b * SR)); tt = t[i:j] - t[i]; n = j - i
    w = np.clip(tt / 2, 0, 1) * np.clip((tt[-1] - tt) / 2, 0, 1)
    sig = sum(np.sin(2 * np.pi * hz(m) * tt * (1 + 0.002 * np.sin(2 * np.pi * (0.2 + 0.05 * q) * tt)))
              for q, m in enumerate(CH[s["key"]]))
    sig = lp(sig, 1800) * w * 0.02
    L[i:j] += sig; R[i:j] += np.roll(sig, 480)

# ---- lyre: Karplus-Strong plucks
def pluck(midi, dur=3.0, bright=0.5):
    f = hz(midi); p = int(SR / f); n = int(dur * SR)
    buf = rng.uniform(-1, 1, p); buf = lp(np.concatenate([buf] * 3), 2000 + 5000 * bright)[-p:]
    out = np.empty(n); d = 0.996
    for k_ in range(n):                      # plain loop is fine: a few dozen notes
        v = buf[k_ % p]; nx = buf[(k_ + 1) % p]; buf[k_ % p] = d * 0.5 * (v + nx); out[k_] = v
    return out * np.exp(-np.arange(n) / SR * 0.9)
THEME = [62, 69, 67, 65, 64, 62]                        # D A G F E D, dorian descent
ANS = {"ravens": [62, 69, 72, 69], "poem": [65, 64, 62], "empty": [57, 60, 62],
       "ledger": [62, 64, 65, 67, 69], "boot": [69, 67, 65], "oskar": [62, 66, 69],
       "flights": [69, 72, 74, 72, 69], "clock": [64, 62, 60], "theseus": [62, 69, 67, 65],
       "return": THEME}
cache = {}
for s, st in zip(TL, starts[:-1]):
    lines = s["lines"]
    gaps = [(lines[q]["end"], lines[q + 1]["start"]) for q in range(len(lines) - 1)]
    gaps.append((lines[-1]["end"], s["dur"] + 0.4))
    for gi, (g0, g1) in enumerate(gaps):
        notes = ANS[s["key"]] if gi == len(gaps) - 1 else ANS[s["key"]][:2]
        step = min(0.32, max(0.14, (g1 - g0 - 0.1) / max(1, len(notes))))
        for q, m in enumerate(notes):
            if m not in cache: cache[m] = pluck(m)
            add(cache[m], st + g0 + 0.05 + q * step, pan=0.35 * np.sin(q), gain=0.09)
# the whole theme once more, slowly, after the last word
end = starts[-1]
for q, m in enumerate(THEME + [74]):
    add(pluck(m, 4.0, 0.3), end + 0.2 + q * 0.42, pan=-0.3 + 0.1 * q, gain=0.1)

# ---- frame drum on the flying scenes (~72 bpm), wing-beat whooshes in the opening
def thump():
    n = int(0.6 * SR); tt = np.arange(n) / SR
    f = 55 + 60 * np.exp(-tt * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 7) + 0.15 * lp(rng.normal(0, 1, n), 800) * np.exp(-tt * 25)
th = thump()
for s, st in zip(TL, starts[:-1]):
    if s["key"] in ("ravens", "flights", "return"):
        b = 60 / 72; x = st + 1.0
        while x < st + s["dur"] - 0.5:
            add(th, x, gain=0.13); add(th, x + b * 0.5, gain=0.05); x += b * 2
    if s["key"] == "ravens":
        for q in range(9):
            n = int(0.45 * SR); w = np.sin(np.linspace(0, np.pi, n)) ** 2
            add(bp(rng.normal(0, 1, n), 300, 1500) * w, st + 2.0 + q * 0.62, pan=-0.6 + 0.15 * q, gain=0.05)

# ---- wind: band-passed noise with slow gusts
wind = bp(rng.normal(0, 1, N), 250, 1400)
gust = 0.5 + 0.5 * np.sin(2 * np.pi * t / 7.3) * np.sin(2 * np.pi * t / 3.1 + 1)
L += 0.018 * wind * gust * fade; R += 0.018 * np.roll(wind, 7000) * gust * fade

# ---- duck the score under the voice, then mix
voc, vsr = sf.read("narration.wav")
voc = resample_poly(voc, SR, vsr); voc = np.pad(voc, (0, max(0, N - len(voc))))[:N]
ve = np.convolve(np.abs(voc), np.ones(int(0.12 * SR)) / int(0.12 * SR), "same")
duck = 1 - 0.45 * np.clip(ve / 0.03, 0, 1)
duck = np.convolve(duck, np.ones(int(0.25 * SR)) / int(0.25 * SR), "same")
mus = np.stack([L, R], 1) * duck[:, None]
mus /= np.abs(mus).max() / 0.5
sf.write("music.wav", mus.astype(np.float32), SR)
mix = mus * 0.5 + voc[:, None] * 0.95 / np.abs(voc).max()
mix /= max(1.0, np.abs(mix).max() / 0.97)
sf.write("mix.wav", mix.astype(np.float32), SR)
print("T", round(T, 2), "peak", round(float(np.abs(mix).max()), 3),
      "voice rms", round(float(np.sqrt((voc ** 2).mean())), 4), "music rms", round(float(np.sqrt((mus ** 2).mean())), 4))
