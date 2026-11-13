"""Does a remex-style rotation rescue the parts of Kokoro that plain int8 cannot pack (round 6)?

remex codes a vector by rotating it with a random orthogonal matrix, so every coordinate looks Gaussian and no
outlier sets the scale, then snapping each coordinate to Lloyd-Max levels for a Gaussian. For weights, that is
"incoherence processing" (QuIP#, QuaRot): W ~ Q(W R) R^T with R a random-sign Hadamard, undone once at load, so
inference runs at fp32 speed. Each arm packs only the named part and keeps the rest fp32; seeded as in
pack_probe.py.

    HF_HOME=<Kokoro-82M cache> /opt/kokoro-env/bin/python remex_probe.py   -> results/remex_probe.json
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
_src = (HERE / "pack_probe.py").read_text()
exec(_src[:_src.index("model = KModel")])  # q_rows, pack, packable, restore, speak, score  # noqa: S102


def hadamard(n):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


_R = {}


def rotation(n):
    """Random-sign Hadamard of size n (a power of two), fixed per size so it can be regenerated from a seed."""
    if n not in _R:
        signs = np.random.default_rng(n).choice([-1.0, 1.0], n)
        _R[n] = torch.tensor(hadamard(n) * signs[None, :], dtype=torch.float64)
    return _R[n]


def lloyd_max(bits, n=400_000):
    """Lloyd-Max levels for a standard Gaussian (1-D Lloyd iterations on samples)."""
    x = np.sort(np.random.default_rng(0).standard_normal(n))
    levels = np.quantile(x, (np.arange(2 ** bits) + 0.5) / 2 ** bits)
    for _ in range(100):
        edges = (levels[1:] + levels[:-1]) / 2
        idx = np.searchsorted(edges, x)
        levels = np.array([x[idx == k].mean() for k in range(2 ** bits)])
    return torch.tensor(levels)


LM = {b: lloyd_max(b) for b in (3, 4, 6)}


def snap(x, levels):
    edges = (levels[1:] + levels[:-1]) / 2
    return levels[torch.searchsorted(edges.contiguous(), x.contiguous())]


def rotated(code):
    """Rotate each row (padded to a power of two), code it, rotate back, crop."""
    def fn(w):
        flat = w.reshape(w.shape[0], -1).double()
        n = 1 << (flat.shape[1] - 1).bit_length()
        r = rotation(n)
        x = torch.nn.functional.pad(flat, (0, n - flat.shape[1])) @ r
        x = code(x)
        return (x @ r.T)[:, :flat.shape[1]].reshape(w.shape).to(w.dtype)
    return fn


def uniform(levels):
    return lambda x: q_rows(x, levels)


def gaussian(bits):
    def code(x):
        s = x.std(1, keepdim=True).clamp_min(1e-12)  # remex scales each row by its spread, then snaps
        return snap(x / s, LM[bits]) * s
    return code


def rel_err(fn, model, names):
    """Relative weight error ||W - Q(W)|| / ||W|| over the packed parameters."""
    num = den = 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if packable(n, p) and n.startswith(names):
                num += float(((fn(p) - p) ** 2).sum())
                den += float((p ** 2).sum())
    return round((num / den) ** 0.5, 5)


def voice(fn):
    return lambda v: fn(v.reshape(v.shape[0], -1)).reshape(v.shape)


model = KModel(repo_id="hexgrad/Kokoro-82M").eval()
ORIGINAL = {k: v.clone() for k, v in model.state_dict().items()}
base = speak(model)
DUR = ("predictor.", "bert", "text_encoder.")
SOUND = ("decoder.",)
schemes = {
    "int8 rows": uniform(127),
    "int8 rows, rotated": rotated(uniform(127)),
    "6-bit Lloyd-Max, rotated": rotated(gaussian(6)),
    "4-bit Lloyd-Max, rotated": rotated(gaussian(4)),
    "3-bit Lloyd-Max, rotated": rotated(gaussian(3)),
}
out = {}
for part, names in (("duration path", DUR), ("decoder", SOUND)):
    for label, fn in schemes.items():
        if part == "decoder" and label.startswith("int8"):
            continue  # round 5 covered int8 on the decoder
        name = f"{part}: {label}"
        out[name] = {"weight_rel_err": rel_err(fn, model, names), **score(base, speak(pack(model, fn,
                     skip=tuple(s for s in DUR + SOUND if s not in names))))}
        restore(model)
        print(f"{name:42s} {out[name]}", flush=True)
for label, fn in (("int8 rows", uniform(127)), ("int8 rows, rotated", rotated(uniform(127))),
                  ("6-bit Lloyd-Max, rotated", rotated(gaussian(6))), ("4-bit Lloyd-Max, rotated", rotated(gaussian(4)))):
    name = f"voice packs: {label}"
    out[name] = score(base, speak(model, voice(fn)))
    print(f"{name:42s} {out[name]}", flush=True)
(HERE / "results" / "remex_probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
