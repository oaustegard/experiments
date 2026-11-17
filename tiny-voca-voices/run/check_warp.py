"""Measure the formant scale the warp actually applies, at the filter itself."""
import numpy as np, torch
import voice_lab as vl
from tiny_voca.infer import load_model
from tiny_voca.text import phonemize, set_frontend
import tiny_voca.model as tm

set_frontend("own")
model = load_model("w/tiny-voca.pt", check_frontend=False)
captured = []
inner = vl.shaped_response
def spy(c, a, fft):
    r = inner(c, a, fft)
    if a is not None:
        captured.append(torch.log(r.abs() + 1e-9)[0].mean(0).numpy())
    return r
tm.cepstrum_response = spy
grid = np.geomspace(300, 5000, 400)
hz = np.linspace(0, 12000, 257)
phones = phonemize("She had never seen the sea before, and for a long while she stood at the edge of the water.")
for voice in (0, 1):
    envs = {}
    for k in (1.0, 0.9, 1.1):
        vl.SHAPE["warp"] = k
        captured.clear()
        model.synthesize(torch.tensor([phones.ids]), torch.tensor([phones.stress]), torch.tensor([len(phones)]), torch.tensor([voice]))
        envs[k] = np.interp(grid, hz, captured[0])
    def scale(env, base):
        best = (-2, 0)
        for s in np.arange(0.8, 1.2001, 0.005):
            sh = np.interp(grid / s, grid, base, left=np.nan, right=np.nan); ok = ~np.isnan(sh)
            a, b = env[ok] - env[ok].mean(), sh[ok] - sh[ok].mean()
            best = max(best, ((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()), s))
        return round(float(best[1]), 3), round(float(best[0]), 3)
    print("voice", voice, "warp0.9 ->", scale(envs[0.9], envs[1.0]), "warp1.1 ->", scale(envs[1.1], envs[1.0]))
