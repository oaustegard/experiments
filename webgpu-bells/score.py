"""Build the mode table and strike list for a six-bell ring plus a bourdon.

Every sound is a sum of damped sinusoids (modal synthesis): each bell is ~50
doublet partials with church-bell ratios plus ~400 short clapper-transient
modes. The GPU job (job.js) renders the sum at 96 kHz; render_ref() is a float64
numpy reference for checking it.

    python3 score.py OUTDIR      writes modes.bin (f64 N x 6), strike_mode.bin (u32),
                                 strike_on.bin (u32), strike_amp.bin (f64), meta.json
"""
import json, sys
from pathlib import Path
import numpy as np

SR = 96000
rng = np.random.default_rng(7)

# Partials relative to the strike pitch (prime): ratio, nodal meridians n, amplitude, T60 factor.
PARTIALS = [
    (0.500, 2, 0.55, 1.00),  # hum
    (1.000, 2, 0.45, 0.62),  # prime
    (1.190, 3, 0.60, 0.48),  # tierce (minor third: the bell's sad colour)
    (1.500, 3, 0.22, 0.34),  # quint
    (2.000, 4, 1.00, 0.32),  # nominal
    (2.510, 4, 0.30, 0.22),  # deciem
    (2.670, 5, 0.28, 0.20),  # undeciem
    (3.010, 5, 0.45, 0.17),  # superquint
    (3.360, 6, 0.18, 0.14),
    (4.020, 6, 0.32, 0.12),  # octave nominal
]
for k, r in enumerate([4.55, 5.02, 5.37, 6.04, 6.62, 7.13, 8.06, 8.85, 9.47, 10.4, 11.3, 12.2, 13.4, 14.5,
                       15.9, 17.1, 18.6, 20.2, 21.9, 23.7]):
    PARTIALS.append((r, 7 + k // 2, 0.26 * (4.0 / r) ** 0.6, 0.11 * (4.5 / r) ** 0.9))


def contact(f, tau):
    """Magnitude spectrum of a half-sine force pulse of duration tau (clapper contact)."""
    x = 2 * f * tau
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.abs(np.cos(np.pi * f * tau) / (1 - x * x))
    return np.where(np.abs(1 - x * x) < 1e-6, np.pi / 4, h)


class Bell:
    def __init__(self, f0, t_hum, pan, loud, mic_sep=0.06, n_trans=400):
        self.modes = []  # (freq Hz, T60 s, n, theta0, gainL complex, gainR complex, amp)
        itd = pan * 0.0006                                # inter-mic delay from the bell's position
        gl, gr = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
        th_l, th_r = 0.9 - mic_sep / 2, 0.9 + mic_sep / 2  # listener angles around the bell axis
        for ratio, n, amp, tfac in PARTIALS:
            f = f0 * ratio * (1 + rng.normal(0, 0.003))   # imperfect tuning
            if f > 21000: continue
            split = f * rng.uniform(2e-4, 1.4e-3)          # doublet splitting -> warble
            th0 = rng.uniform(0, np.pi)
            for d, ph in ((-0.5, 0.0), (0.5, np.pi / 2)):  # cos / sin pair of the n-fold pattern
                fd = f + d * split
                pat = lambda th: np.cos(n * (th - th0) - ph)
                w = 2 * np.pi * fd
                self.modes.append((fd, t_hum * tfac * rng.uniform(0.9, 1.1), n, th0, ph,
                                   gl * pat(th_l) * np.exp(1j * w * itd / 2),
                                   gr * pat(th_r) * np.exp(-1j * w * itd / 2), loud * amp))
        # Clapper transient: dense high modes that die in tens of milliseconds.
        u = rng.uniform(0, 1, n_trans)
        ft = 2200 * (19000 / 2200) ** np.sqrt(u)           # denser toward the top
        for f in ft:
            t60 = rng.uniform(0.04, 0.35) * (3000 / f) ** 0.5
            n = int(rng.integers(6, 40)); th0 = rng.uniform(0, np.pi); ph = rng.choice([0, np.pi / 2])
            w = 2 * np.pi * f
            self.modes.append((f, t60, n, th0, ph,
                               gl * np.cos(n * (th_l - th0) - ph) * np.exp(1j * w * itd / 2),
                               gr * np.cos(n * (th_r - th0) - ph) * np.exp(-1j * w * itd / 2),
                               loud * 0.05 * rng.rayleigh(1.0) * (4000 / f) ** 0.3))

    def strike_amps(self, theta_s, tau, vel):
        out = []
        for f, t60, n, th0, ph, _, _, amp in self.modes:
            out.append(vel * amp * np.cos(n * (theta_s - th0) - ph) * contact(f, tau))
        return np.array(out)


def build():
    names = ["B4", "A4", "G4", "F#4", "E4", "D4"]
    midi = [71, 69, 67, 66, 64, 62]
    bells = []
    for i, m in enumerate(midi):
        f0 = 440 * 2 ** ((m - 69) / 12)
        size = (293.66 / f0)                                # tenor = 1
        bells.append(Bell(f0, t_hum=26 * size ** 0.8, pan=-0.75 + 1.5 * i / 5, loud=size ** 0.6))
    bourdon = Bell(110.0, t_hum=55, pan=0.1, loud=2.4, n_trans=500)
    allb = bells + [bourdon]

    # Rounds twice, then plain hunt down to back rounds.
    rows = ["123456", "123456", "214365", "241635", "426153", "462513", "645231", "654321"]
    gap, t = 0.205, 0.35
    strikes = []  # (bell index, time s, hand/back)
    for r, row in enumerate(rows):
        hand = r % 2 == 0
        if hand and r > 0: t += gap                         # handstroke gap
        for c in row:
            strikes.append((int(c) - 1, t + rng.normal(0, 0.006), hand)); t += gap
    t_end = t + 0.9
    strikes.append((6, t_end, True))
    dur = t_end + 7.0

    mode_rows, s_mode, s_on, s_amp = [], [], [], []
    base = []
    for b in allb:
        base.append(len(mode_rows))
        for f, t60, *_ , gL, gR, _a in b.modes:
            w = 2 * np.pi * f / SR
            sig = 6.907755 / (t60 * SR)
            mode_rows.append((w, sig, gL.real, gL.imag, gR.real, gR.imag))
    for bi, ts, hand in strikes:
        b = allb[bi]
        vel = rng.uniform(0.85, 1.0) * (1.0 if bi < 6 else 1.0)
        tau = 0.00032 * (1.6 if bi == 6 else 1.0) * rng.uniform(0.85, 1.15)
        th = rng.normal(0, 0.05) + (0.0 if hand else np.pi)
        amps = b.strike_amps(th, tau, vel)
        on = int(round(ts * SR))
        for k, a in enumerate(amps):
            if abs(a) < 1e-9: continue
            s_mode.append(base[bi] + k); s_on.append(on); s_amp.append(a)
    modes = np.array(mode_rows, dtype=np.float64)
    order = np.lexsort((np.array(s_on), np.array(s_mode)))
    return dict(modes=modes, s_mode=np.array(s_mode, np.uint32)[order], s_on=np.array(s_on, np.uint32)[order],
                s_amp=np.array(s_amp, np.float64)[order], n_samples=int(dur * SR), strikes=len(strikes))


def render_ref(sc, idx):
    """float64 reference at sample indices idx: returns (len(idx), 2)."""
    m = sc["modes"]; out = np.zeros((len(idx), 2))
    t = idx.astype(np.float64)
    for mi, on, a in zip(sc["s_mode"], sc["s_on"], sc["s_amp"]):
        w, s, lr, li, rr, ri = m[mi]
        dt = t - on; ok = dt >= 0; dt = np.where(ok, dt, 0)
        z = a * np.exp((-s + 1j * w) * dt) * ok   # response is Im(z * g)
        out[:, 0] += (z * (lr + 1j * li)).imag
        out[:, 1] += (z * (rr + 1j * ri)).imag
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    sc = build()
    sc["modes"].tofile(out / "modes.bin"); sc["s_mode"].tofile(out / "strike_mode.bin")
    sc["s_on"].tofile(out / "strike_on.bin"); sc["s_amp"].tofile(out / "strike_amp.bin")
    meta = dict(sr=SR, n_samples=sc["n_samples"], n_modes=len(sc["modes"]), n_strike_modes=len(sc["s_mode"]),
                strikes=sc["strikes"])
    (out / "meta.json").write_text(json.dumps(meta))
    print(meta)
