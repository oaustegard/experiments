"""Duck the score under the narration and sum. Linear gains only (no loudnorm), peaks at -1 dBFS."""
import numpy as np, soundfile as sf, librosa
SR = 48000
vo, _ = librosa.load('../vo/vo_puck.wav', sr=SR, mono=True)
mu, _ = sf.read('music.wav'); mu = mu.astype(np.float32)
n = len(mu); vo = np.pad(vo, (0, max(0, n - len(vo))))[:n]
# voice activity envelope: 20 ms RMS, fast attack, slow release
hop = 480; rms = np.sqrt(np.convolve(vo ** 2, np.ones(hop) / hop, 'same'))
act = (rms > 10 ** (-38 / 20)).astype(float)
e = np.zeros(n); a, r = 1 - np.exp(-1 / (0.03 * SR)), 1 - np.exp(-1 / (0.35 * SR)); x = 0.0
# smooth per hop for speed
for i in range(0, n, hop):
    tgt = act[i]; k = a * hop if tgt > x else r * hop; x += (tgt - x) * min(1, k); e[i:i + hop] = x
duck = 10 ** ((-9 * e) / 20)
MUSIC_GAIN = 10 ** (-6 / 20)
out = mu * (MUSIC_GAIN * duck)[:, None] + vo[:, None] * 0.95
out *= 10 ** (-1 / 20) / np.abs(out).max()
sf.write('mix.wav', out.astype(np.float32), SR, subtype='PCM_16')
print('mix.wav', n / SR)
