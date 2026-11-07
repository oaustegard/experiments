"""How much precision does the sampler need? Simulate storage formats in torch and compare 20-step
guided samples (same noise) with the f32 run.

Weights: f16 (what ships) or int8 symmetric per output channel (conv/linear weights only).
Activations: every conv output (and so every stored tensor the WGSL runtime keeps) rounded to f16
or bf16 — the storage format; arithmetic stays f32, as it would with pack2x16float storage.

    python3 run/quant_sim.py WEIGHTS.bin MANIFEST.json OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from export import NAMES
from model import sample
from ref_from_bin import load

CLASSES = ["marbled", "veined", "cracked", "honeycombed", "paisley", "woven", "bubbly", "scaly"]


def int8_weights(m):
    with torch.no_grad():
        for _name, p in m.named_parameters():
            if p.dim() >= 2:
                s = p.abs().amax(dim=tuple(range(1, p.dim())), keepdim=True).clamp_min(1e-12) / 127
                p.copy_((p / s).round().clamp(-127, 127) * s)


def act_hooks(m, dtype):
    hs = []
    for mod in m.modules():
        if isinstance(mod, torch.nn.Conv2d):
            hs.append(mod.register_forward_hook(lambda _m, _i, o: o.to(dtype).float()))
    return hs


def run(m, name, H, seed, act=None):
    hs = act_hooks(m, act) if act is not None else []
    g = torch.Generator().manual_seed(seed)
    n = torch.randn(1, 3, H, H, generator=g)
    if act is not None:
        n = n.to(act).float()
    x = sample(m, torch.tensor([NAMES.index(name)]), H, H, steps=20, cfg=3.0, pad="circular", noise=n)
    for h in hs:
        h.remove()
    return x


def psnr(a, b):
    return float(10 * torch.log10(4 / ((a - b) ** 2).mean()))


def main():
    wb, man, out = sys.argv[1:4]
    torch.set_num_threads(4)
    base = load(wb, man)
    q8 = load(wb, man)
    int8_weights(q8)
    arms = {"w16_a16": (base, torch.float16), "w16_abf16": (base, torch.bfloat16),
            "w8_a32": (q8, None), "w8_a16": (q8, torch.float16)}
    rows = []
    for i, c in enumerate(CLASSES):
        ref = run(base, c, 128, 200 + i)
        r = {"class": c}
        for k, (m, act) in arms.items():
            r[k] = psnr(ref, run(m, c, 128, 200 + i, act))
        rows.append(r)
        print(r, flush=True)
    summ = {k: float(np.median([r[k] for r in rows])) for k in arms}
    summ_min = {k: float(min(r[k] for r in rows)) for k in arms}
    Path(out).write_text(json.dumps({"median_psnr_db": summ, "min_psnr_db": summ_min, "rows": rows}, indent=1))
    print("median", summ, "\nmin", summ_min)


if __name__ == "__main__":
    main()
