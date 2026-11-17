"""The tunable ONNX graph against PyTorch and against voice_lab.py, same noise.

    python check_tunable.py out/tunable-noise.onnx
"""
import json
import math
import sys

import numpy as np
import onnxruntime as ort
import torch

import voice_lab as vl
from export_tunable import Tunable
from tiny_voca.infer import load_model
from tiny_voca.text import phonemize, set_frontend

set_frontend("own")
model = load_model("w/tiny-voca.pt", check_frontend=False)
tunable = Tunable(model, external_noise=True).eval()
session = ort.InferenceSession(sys.argv[1])
c = model.config
text = "The meeting moved to three thirty on Thursday, so please bring the quarterly numbers."
phones = phonemize(text)
tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
CASES = {
    "stock-F2": dict(mix=(1.0, 0.0)),
    "stock-M2": dict(mix=(0.0, 1.0)),
    "F2-adult-mix": dict(mix=(0.8, 0.2), formant=0.95, semitones=-2.5, range=0.9, speed=0.95),
    "M2-baritone": dict(mix=(0.0, 1.0), formant=0.92, semitones=-5.0, range=1.1, speed=0.95),
    "F2-breathy": dict(mix=(1.0, 0.0), tilt=-1.5, breath=6.0),
    "extrapolated": dict(mix=(1.25, -0.25), formant=1.08, tilt=1.0),
}
report = {}
for name, k in CASES.items():
    shift = k.get("semitones", 0.0) * math.log(2) / 12
    # voice_lab path: TinyVoca.synthesize with a blended speaker row and the patched filter.
    preset = {"mix": {"F2": k["mix"][0], "M2": k["mix"][1]}, "warp": k.get("formant", 1.0), "tilt": k.get("tilt", 0.0),
              "breath": k.get("breath", 0.0)}
    grown, speaker = vl.with_voices(model, {"x": preset})
    sid = torch.tensor([grown.config.speakers.index(speaker["x"])])
    vl.SHAPE.update(warp=preset["warp"], tilt=preset["tilt"], breath=preset["breath"])
    lab, lengths, _ = grown.synthesize(tokens, stress, torch.tensor([len(phones)]), sid, speed=k.get("speed", 1.0),
                                       pitch_shift=shift, seed=0, pitch_range=k.get("range", 1.0))
    vl.SHAPE.update(warp=1.0, tilt=0.0, breath=0.0)
    lab = lab[0, :int(lengths[0])].numpy()
    noise = torch.zeros(1, c.hop * c.max_frames)
    noise[0, :len(lab)] = torch.randn(len(lab), generator=torch.Generator().manual_seed(0))
    one = lambda v: torch.tensor([v], dtype=torch.float32)
    mix = torch.tensor([k["mix"]], dtype=torch.float32)
    args = (tokens, stress, mix @ model.voices.weight.detach(), mix @ model.frame_voices.weight.detach(), one(k.get("speed", 1.0)), one(shift),
            one(k.get("range", 1.0)), one(k.get("formant", 1.0)), one(k.get("tilt", 0.0)), one(k.get("breath", 0.0)))
    with torch.no_grad():
        pt = tunable(*args, noise)[0][0].numpy()
    names = [i.name for i in session.get_inputs()]
    feeds = dict(zip(names, [a.numpy() for a in args] + [noise.numpy()]))
    ox = session.run(None, feeds)[0][0]
    peak = np.abs(pt).max()
    n = min(len(lab), len(pt), len(ox))
    report[name] = {"samples": [len(lab), len(pt), len(ox)],
                    "onnx_vs_torch_max_err_pct_peak": round(100 * float(np.abs(ox[:n] - pt[:n]).max() / peak), 3),
                    "torch_vs_lab_max_err_pct_peak": round(100 * float(np.abs(pt[:n] - lab[:n]).max() / peak), 3)}
    print(name, report[name], flush=True)
json.dump(report, open("out/check_tunable.json", "w"), indent=1)
