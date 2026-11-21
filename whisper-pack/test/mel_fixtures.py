import numpy as np, pathlib
from transformers import WhisperFeatureExtractor
root = pathlib.Path(__file__).resolve().parents[1]
out = root / "test" / "fixtures"; out.mkdir(exist_ok=True)
a = np.load(root / "cache" / "ls-clean.npz", allow_pickle=True)["audio"]
clips = {f"utt{i}": a[i].astype(np.float32) for i in range(3)}
long = np.concatenate([a[i] for i in range(len(a))]).astype(np.float32)
assert len(long) >= 35 * 16000
clips["long35"] = long[: 35 * 16000]
t = np.arange(int(2.0 * 16000), int(4.0 * 16000)) / 16000
sig = np.zeros(4 * 16000, np.float32); sig[2 * 16000:] = 0.1 * np.sin(2 * np.pi * 440 * np.arange(2 * 16000) / 16000)
clips["silence_sine"] = sig
for name, x in clips.items():
    x.astype("<f4").tofile(out / f"{name}.f32")
    for n in (80, 128):
        fe = WhisperFeatureExtractor(feature_size=n)
        f = fe(x, sampling_rate=16000, return_tensors="np").input_features[0]
        assert f.shape == (n, 3000)
        f.astype("<f4").tofile(out / f"{name}.mel{n}.f32")
for n in (80, 128):
    WhisperFeatureExtractor(feature_size=n).mel_filters.astype("<f8").tofile(out / f"filters{n}.f64")
print("ok", {k: len(v) for k, v in clips.items()})
