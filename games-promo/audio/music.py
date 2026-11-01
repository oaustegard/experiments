"""Original chiptune score for the games promo, locked to the edit: one 4/4 bar = 66 frames
at 30 fps = 2.2 s (109.09 bpm). Pulse lead, triangle bass, noise drums, all synthesized,
like the games' own Web Audio scores. Writes music.wav (48 kHz stereo)."""
import numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt
SR, BAR = 48000, 2.2
BEAT = BAR / 4
TOTAL = 55.0
N = int(SR * (TOTAL + 0.5))
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(7)

def hz(m): return 440 * 2 ** ((m - 69) / 12)
def env(n, a=0.005, d=0.1, s=0.6, r=0.05, hold=None):
    t = np.arange(n) / SR; hold = hold if hold is not None else n / SR - r
    e = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) / d))
    rel = np.clip(1 - (t - hold) / r, 0, 1); return e * np.where(t > hold, rel, 1)
def pulse(f, n, duty=0.25):
    ph = (np.arange(n) * f / SR) % 1; return np.where(ph < duty, 1.0, -1.0) - (2 * duty - 1)
def tri(f, n):
    ph = (np.arange(n) * f / SR) % 1; return 4 * np.abs(ph - 0.5) - 1
def saw(f, n):
    ph = (np.arange(n) * f / SR) % 1; return 2 * ph - 1
def lp(x, fc, order=2):
    return sosfilt(butter(order, fc / (SR / 2), 'low', output='sos'), x)
def hp(x, fc, order=2):
    return sosfilt(butter(order, fc / (SR / 2), 'high', output='sos'), x)
def put(t, sig, g=1.0, pan=0.0):
    i = int(t * SR); j = min(N, i + len(sig))
    if i >= N or j <= i: return
    s = sig[: j - i] * g
    L[i:j] += s * np.sqrt(0.5 * (1 - pan)) ; R[i:j] += s * np.sqrt(0.5 * (1 + pan))

def note(kind, t, m, dur, g, pan=0.0, duty=0.25, a=0.004, d=0.12, s=0.55, vib=0):
    n = int(dur * SR); f = hz(m)
    if vib:
        tt = np.arange(n) / SR; f = f * (1 + 0.006 * np.sin(2 * np.pi * 5.5 * tt) * np.clip(tt / 0.25, 0, 1))
        ph = np.cumsum(f / SR) % 1
        w = (np.where(ph < duty, 1.0, -1.0) - (2 * duty - 1)) if kind == 'pulse' else 4 * np.abs(ph - 0.5) - 1
    else:
        w = {'pulse': lambda: pulse(f, n, duty), 'tri': lambda: tri(f, n), 'saw': lambda: lp(saw(f, n), 2400)}[kind]()
    put(t, w * env(n, a, d, s, min(0.04, dur * 0.3)), g, pan)

def kick(t, g=0.9):
    n = int(0.32 * SR); tt = np.arange(n) / SR
    f = 50 + 110 * np.exp(-tt / 0.035); ph = 2 * np.pi * np.cumsum(f) / SR
    put(t, np.sin(ph) * np.exp(-tt / 0.12) + 0.3 * np.exp(-tt / 0.004) * rng.standard_normal(n) * 0.2, g)
def snare(t, g=0.45):
    n = int(0.22 * SR); tt = np.arange(n) / SR
    nz = hp(rng.standard_normal(n), 1200) * np.exp(-tt / 0.06)
    body = np.sin(2 * np.pi * 190 * tt) * np.exp(-tt / 0.04)
    put(t, nz * 0.8 + body * 0.5, g)
def hat(t, g=0.12, open_=False, pan=0.25):
    n = int((0.18 if open_ else 0.05) * SR); tt = np.arange(n) / SR
    put(t, hp(rng.standard_normal(n), 7000) * np.exp(-tt / (0.06 if open_ else 0.012)), g, pan)
def crash(t, g=0.22):
    n = int(1.8 * SR); tt = np.arange(n) / SR
    s = hp(rng.standard_normal(n), 4000) * np.exp(-tt / 0.5); put(t, s, g, -0.3); put(t, s[::-1][::-1], g * 0.7, 0.3)
def riser(t0, t1, g=0.25):
    n = int((t1 - t0) * SR); tt = np.arange(n) / SR; x = tt / tt[-1]
    nz = rng.standard_normal(n); out = np.zeros(n); 
    for k in range(8):  # stepped band sweep
        a, b = k * n // 8, (k + 1) * n // 8
        out[a:b] = hp(nz[a:b], 300 + 900 * k)
    put(t0, out * x ** 2, g)
def whoosh(t, g=0.18):
    n = int(0.5 * SR); tt = np.arange(n) / SR
    w = lp(rng.standard_normal(n), 3000) * np.sin(np.pi * tt / tt[-1]) ** 2; put(t - 0.25, w, g, 0)

# chords as (root midi, minor?) per bar; A minor
Am, F, C, G, E, Dm = (57, 'm'), (53, ''), (48, ''), (55, ''), (52, ''), (50, 'm')
def triad(ch): r, q = ch; return [r, r + (3 if q == 'm' else 4), r + 7]

def bar_t(b): return b * BAR

# ---- section plan (bars) ----
prog_main = [Am, F, C, G]
prog_dark = [Am, Am, F, E]
lead_main = [  # (beat offset in bar, midi, beats) one phrase per 2 bars, original
    [(0, 76, 1), (1, 74, .5), (1.5, 72, .5), (2, 74, 1), (3, 69, 1)],
    [(0, 72, 1.5), (1.5, 74, .5), (2, 76, 1), (3, 79, 1)],
    [(0, 81, 1), (1, 79, .5), (1.5, 76, .5), (2, 79, 1.5), (3.5, 76, .5)],
    [(0, 74, 1), (1, 72, 1), (2, 71, 1), (3, 67, 1)],
]

def arp(b, ch, g=0.09, div=4, oct_=12, pan=0.35, duty=0.125):
    tones = triad(ch); seq = tones + [tones[1]]
    for k in range(4 * div):
        note('pulse', bar_t(b) + k * BEAT / div, seq[k % len(seq)] + oct_, BEAT / div * 0.9, g, pan, duty, d=0.05, s=0.3)
def bass(b, ch, g=0.32, pattern='8'):
    r = ch[0] - 12
    if pattern == '8':
        for k in range(8): note('tri', bar_t(b) + k * BEAT / 2, r + (12 if k % 2 else 0), BEAT / 2 * 0.85, g, d=0.2, s=0.8)
    elif pattern == '16':
        for k in range(16): note('tri', bar_t(b) + k * BEAT / 4, r + (12 if k % 4 == 2 else 0), BEAT / 4 * 0.8, g, d=0.1, s=0.7)
    elif pattern == 'long':
        note('tri', bar_t(b), r, BAR * 0.98, g * 1.1, d=1.5, s=0.8); note('pulse', bar_t(b), r - 12, BAR * 0.98, g * 0.18, duty=0.5, d=1.0, s=0.7)
def drums(b, style='full'):
    t = bar_t(b)
    if style == 'full':
        for k in (0, 2): kick(t + k * BEAT)
        kick(t + 2.5 * BEAT, 0.6)
        for k in (1, 3): snare(t + k * BEAT)
        for k in range(8): hat(t + k * BEAT / 2, 0.10 if k % 2 else 0.07, open_=(k == 7))
    elif style == 'half':
        kick(t); snare(t + 2 * BEAT, 0.55); kick(t + 3.5 * BEAT, 0.6)
        for k in range(16): hat(t + k * BEAT / 4, 0.05 + 0.04 * (k % 2 == 0))
    elif style == 'drive':
        for k in range(4): kick(t + k * BEAT, 0.8)
        for k in (1, 3): snare(t + k * BEAT)
        for k in range(16): hat(t + k * BEAT / 4, 0.06 + 0.05 * (k % 4 == 2))
    elif style == 'light':
        kick(t); kick(t + 2.5 * BEAT, 0.5)
        for k in (1, 3): hat(t + k * BEAT, 0.1, open_=True)
        for k in range(8): hat(t + k * BEAT / 2, 0.05)
def pad(b, ch, g=0.05, bars=1):
    for m in triad(ch):
        for det in (-0.06, 0.06):
            n = int(BAR * bars * SR); f = hz(m + det)
            w = lp(saw(f, n), 1400) * env(n, 0.3, 1.0, 0.8, 0.4)
            put(bar_t(b), w, g, det * 6)
def lead(b, phrase, g=0.13, oct_=0, duty=0.25):
    for off, m, ln in phrase:
        note('pulse', bar_t(b) + off * BEAT, m + oct_, ln * BEAT * 0.92, g, -0.15, duty, d=0.25, s=0.6, vib=1)
def sonar(t, m=88, g=0.14):
    n = int(1.2 * SR); tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * hz(m) * tt) * np.exp(-tt / 0.25)
    put(t, s, g, -0.4); put(t + 0.28, s * 0.5, g, 0.4); put(t + 0.56, s * 0.25, g, -0.4)

# Hook: bars 0-1 (0-4.4 s)
for b in range(2):
    pad(b, prog_main[b]); arp(b, prog_main[b], g=0.05, div=2)
    bass(b, prog_main[b], 0.18, 'long')
for k in range(8): hat(bar_t(1) + k * BEAT / 2, 0.05 + 0.01 * k)
riser(bar_t(1), bar_t(2), 0.2)
# Montage: bars 2-8 (7 shots, one per bar)
for i, b in enumerate(range(2, 9)):
    ch = prog_main[i % 4]
    drums(b, 'full'); bass(b, ch, 0.30); arp(b, ch, 0.06)
    if i >= 1: lead(b, lead_main[(i - 1) % 4], 0.10)
    whoosh(bar_t(b), 0.10)
crash(bar_t(2)); kick(bar_t(2), 1.0)
# Rally: bars 9-10, driving
for i, b in enumerate(range(9, 11)):
    ch = [Am, G][i]; drums(b, 'drive'); bass(b, ch, 0.3, '16'); arp(b, ch, 0.06, div=4)
crash(bar_t(9), 0.18)
# GPU: bars 11-14, dark half-time
for i, b in enumerate(range(11, 15)):
    ch = prog_dark[i]; drums(b, 'half'); bass(b, ch, 0.36, 'long'); pad(b, ch, 0.045)
    arp(b, ch, 0.05, div=4, oct_=0, duty=0.5)
crash(bar_t(11), 0.2); kick(bar_t(11), 1.0)
riser(bar_t(14) + BEAT * 2, bar_t(15), 0.15)
# Minesweeper: bars 15-16, sparse with sonar pings
for i, b in enumerate(range(15, 17)):
    ch = [Am, F][i]; pad(b, ch, 0.05); bass(b, ch, 0.16, 'long')
    kick(bar_t(b), 0.6); kick(bar_t(b) + 2 * BEAT, 0.5)
    for k in range(8): hat(bar_t(b) + k * BEAT / 2, 0.04)
    sonar(bar_t(b), 88 if i == 0 else 84)
    sonar(bar_t(b) + 2 * BEAT, 91, 0.10)
# Table: bars 17-20.5, light groove with chord stabs
tab = [C, G, Am, F]
for i, b in enumerate(range(17, 21)):
    ch = tab[i]; drums(b, 'light'); bass(b, ch, 0.24)
    for k in (0.5, 1.5, 2.5, 3.5):
        for m in triad(ch): note('pulse', bar_t(b) + k * BEAT, m + 12, BEAT * 0.3, 0.035, 0.3, 0.5, d=0.06, s=0.2)
    if i < 3: lead(b, lead_main[i], 0.07, oct_=-12, duty=0.5)
# Classics: bars 20.5-22, pure 8-bit callback
t0 = bar_t(20.5)
for k in range(12):
    m = [69, 72, 76, 81, 79, 76, 72, 76, 74, 71, 67, 71][k]
    note('pulse', t0 + k * BEAT / 2, m, BEAT / 2 * 0.9, 0.09, 0, 0.5, d=0.08, s=0.4)
    hat(t0 + k * BEAT / 2, 0.05)
for k in range(6): note('tri', t0 + k * BEAT / 2 * 2, [45, 45, 41, 41, 43, 43][k], BEAT * 0.9, 0.3, d=0.3, s=0.7)
kick(t0, 0.7); kick(t0 + 2 * BEAT, 0.7); snare(t0 + BEAT); snare(t0 + 3 * BEAT); snare(t0 + 5 * BEAT); snare(t0 + 5.5 * BEAT, 0.3)
# CTA: bars 22-24 full, final chord at bar 24 rings out
for i, b in enumerate(range(22, 24)):
    ch = [F, G][i]; drums(b, 'full'); bass(b, ch, 0.3); arp(b, ch, 0.06); lead(b, lead_main[2 + i], 0.10); pad(b, ch, 0.04)
crash(bar_t(22)); kick(bar_t(22), 1.0)
fin = bar_t(24)
crash(fin, 0.28); kick(fin, 1.0)
for m in [45, 57, 60, 64, 69, 72, 76]:
    n = int(3.5 * SR); w = pulse(hz(m), n, 0.25 if m > 60 else 0.5) * env(n, 0.005, 1.2, 0.0, 0.5)
    put(fin, lp(w, 3000), 0.06)
note('tri', fin, 33, 3.0, 0.4, d=1.5, s=0.3)

# master: gentle glue, fade the tail, peak to -1 dBFS
mix = np.stack([L, R], 1)[: int(SR * TOTAL)]
tail = np.linspace(1, 0, int(SR * 1.0)) ** 2; mix[-len(tail):] *= tail[:, None]
mix = np.tanh(mix * 1.2) / np.tanh(1.2)
mix *= 10 ** (-1 / 20) / np.abs(mix).max()
sf.write('music.wav', mix.astype(np.float32), SR, subtype='PCM_16')
print('music.wav', mix.shape[0] / SR, 's')
