"""Beat and energy envelopes for render.py: song16k.wav -> analysis.json (30 fps)."""
import json
import librosa
import numpy as np

y, sr = librosa.load('song16k.wav', sr=16000)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=256)
bt = librosa.frames_to_time(beats, sr=sr, hop_length=256)
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=256)
rms = librosa.feature.rms(y=y, hop_length=256)[0]
ft = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)
fps = 30
T = len(y) / sr
t = np.arange(int(T * fps)) / fps
r = np.interp(t, ft, rms)
o = np.interp(t, ft[:len(oenv)], oenv)
r, o = r / np.percentile(r, 99), o / np.percentile(o, 99)
json.dump({'tempo': float(np.atleast_1d(tempo)[0]), 'beats': bt.tolist(),
           'rms': np.clip(r, 0, 1.5).round(3).tolist(), 'onset': np.clip(o, 0, 1.5).round(3).tolist(),
           'dur': T}, open('analysis.json', 'w'))
print('tempo', tempo, 'beats', len(bt), 'dur', round(T, 2))
