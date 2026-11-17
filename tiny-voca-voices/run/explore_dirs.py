"""Do directions off the F2-M2 line give new, intelligible voices?

Random unit directions, orthogonal to F2-M2 in each embedding table, scaled to a
fraction of the F2-M2 distance and added to a stock voice.
"""
import json

import numpy as np
import torch
from scipy.io import wavfile

import voice_lab as vl
from tiny_voca.config import Config
from tiny_voca.infer import load_model
import tiny_voca.model as tm
from tiny_voca.text import set_frontend

set_frontend("own")
torch.set_num_threads(4)
model = load_model("w/tiny-voca.pt", check_frontend=False)
state = model.state_dict()
g = torch.Generator().manual_seed(7)
directions = {}
for key in ("voices.weight", "frame_voices.weight"):
    t = state[key]
    axis = (t[0] - t[1]) / (t[0] - t[1]).norm()
    ds = []
    for _ in range(4):
        d = torch.randn(t.shape[1], generator=g)
        d = d - (d @ axis) * axis
        for prev in ds:
            d = d - (d @ prev) * prev
        ds.append(d / d.norm())
    directions[key] = (torch.stack(ds), float((t[0] - t[1]).norm()))
voices = {}
for base, bi in (("F2", 0), ("M2", 1)):
    for i in range(4):
        for s in ((0.5, 1.0) if base == "F2" else (0.5,)):
            voices[f"{base}-d{i}-{s:g}"] = (bi, i, s)
names = list(model.config.speakers) + list(voices)
config = Config.from_dict({**model.config.to_dict(), "speakers": names})
grown = tm.TinyVoca(config)
for key in ("voices.weight", "frame_voices.weight"):
    ds, dist = directions[key]
    rows = [state[key][bi] + s * dist * ds[i] for bi, i, s in voices.values()]
    state[key] = torch.cat((state[key], torch.stack(rows)))
grown.load_state_dict(state)
grown.eval()
texts = open("lab/texts.txt").read().strip().split("\n\n")
import pathlib
out = pathlib.Path("out4")
out.mkdir(exist_ok=True)
report = {}
for name in names:
    f0s = []
    for t, text in enumerate(texts):
        wave, f0 = vl.render(grown, text, name, {}, seed=100 * t)
        wavfile.write(out / f"{name}-{t + 1}.wav", 24000, (wave.numpy() * 32767).astype(np.int16))
        f0s.append(f0)
    report[name] = round(float(torch.cat(f0s).median()), 1)
    print(name, report[name], flush=True)
json.dump(report, open(out / "pitch.json", "w"), indent=1)
