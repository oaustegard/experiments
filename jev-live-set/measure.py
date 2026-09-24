"""Objective proxies for 'thin': loudness (LUFS), low-end share, band balance, stereo width, crest."""
import sys, subprocess, re, wave, numpy as np, imageio_ffmpeg
def measure(path):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    o = subprocess.run([ff, "-hide_banner", "-i", path, "-af", "ebur128", "-f", "null", "-"], capture_output=True, text=True).stderr
    lufs = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", o)[-1])
    w = wave.open(path); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, 2).astype(np.float64) / 32768
    L, R = x[:, 0], x[:, 1]; M = (L + R) / 2; S = (L - R) / 2
    spec = np.abs(np.fft.rfft(M)) ** 2; f = np.fft.rfftfreq(len(M), 1 / sr); tot = spec.sum()
    bands = [(20, 120), (120, 500), (500, 2000), (2000, 8000), (8000, 20000)]
    share = [spec[(f >= a) & (f < b)].sum() / tot for a, b in bands]
    width = np.sqrt((S ** 2).mean()) / np.sqrt((M ** 2).mean())
    crest = 20 * np.log10(np.abs(M).max() / np.sqrt((M ** 2).mean()))
    oct_c = [125, 250, 500, 1000, 2000, 4000, 8000]
    lv = [10 * np.log10(spec[(f >= c / 2 ** .5) & (f < c * 2 ** .5)].sum() + 1e-12) for c in oct_c]
    slope = np.polyfit(np.log2(oct_c), lv, 1)[0]
    return dict(slope_db_oct=round(slope, 1), lufs=lufs, width=round(width, 3), crest_db=round(crest, 1), **{f"{a}-{b}Hz": round(s * 100, 1) for (a, b), s in zip(bands, share)})
for p in sys.argv[1:]: print(p, measure(p))
