"""WAV takes -> MP3, each before/after pair scaled by the same gain so loudness compares fairly."""
import glob, numpy as np, soundfile as sf, lameenc
for st in sorted({p.split('/')[-1].rsplit('-', 1)[0] for p in glob.glob('takes/*-before.wav')}):
    pair = {v: sf.read(f'takes/{st}-{v}.wav', dtype='float32') for v in ('before', 'after')}
    g = .95 / max(np.abs(x).max() for x, _ in pair.values())
    for v, (x, sr) in pair.items():
        enc = lameenc.Encoder(); enc.set_bit_rate(160); enc.set_in_sample_rate(sr); enc.set_channels(2); enc.set_quality(2)
        pcm = (np.clip(x * g, -1, 1) * 32767).astype('<i2')
        open(f'takes/{st}-{v}.mp3', 'wb').write(enc.encode(pcm.tobytes()) + enc.flush())
        print(st, v, f'gain {g:.2f}')
