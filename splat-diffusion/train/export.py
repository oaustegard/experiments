"""Export the splat denoiser for the WGSL runtime, and build test fixtures.

    python export.py weights ema.pt stats.json ../web/model     # weights.bin (f16) + manifest.json
    python export.py fixture ../web/test [K]                     # random weights + torch reference I/O
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import NCLASS, D, SplatDiT, sample

NAMES = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked", "crosshatched",
         "crystalline", "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved",
         "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled", "matted", "meshed", "paisley",
         "perforated", "pitted", "pleated", "polka-dotted", "porous", "potholed", "scaly", "smeared", "spiralled",
         "sprinkled", "stained", "stratified", "striped", "studded", "swirly", "veined", "waffled", "woven",
         "wrinkled", "zigzagged"]   # DTD label order (tanganke/dtd)


def write(sd, stats, K, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    man, chunks, off = {}, [], 0
    for k, v in sd.items():
        a = v.detach().float().cpu().numpy().astype(np.float16).ravel()
        man[k] = [off, list(v.shape)]
        chunks.append(a)
        off += a.size
    np.concatenate(chunks).tofile(out / "weights.bin")
    (out / "manifest.json").write_text(json.dumps({"n": off, "tensors": man, "K": K, "D": D, "repr": "opaque", "nclass": NCLASS,
                                                   "names": NAMES, "stats": stats}))
    print(f"{off} params, {(out / 'weights.bin').stat().st_size/1e6:.1f} MB")


def fixture(out: Path, K=64):
    torch.manual_seed(5)
    m = SplatDiT().eval()
    with torch.no_grad():
        for _n, p in m.named_parameters():
            p.copy_((torch.randn_like(p) / (p.shape[-1] ** 0.5 if p.dim() > 1 else 10)).half().float())
    stats = {"mean": [0, 0, 0, 0, 2.5, 0.0, 2.5, 0.0, 0.0, 0.0, 0.5, -1.0, 0.5],
             "std": [0.7] * 4 + [0.5, 3.0, 0.5, 0.3, 0.3, 0.3, 1.0, 1.0, 0.29]}
    write(m.state_dict(), stats, K, out)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, K, D, generator=g)
    with torch.no_grad():
        v = m(torch.cat([x, x]), torch.tensor([0.6, 0.6]), torch.tensor([9, NCLASS]))
    noise = torch.randn(1, K, D, generator=g)
    xs = sample(m, torch.tensor([4]), K, steps=3, cfg=2.5, noise=noise)
    import torch.nn.functional as F
    from splats import from_features, render, sort_by_depth
    mean, std = torch.tensor(stats["mean"]), torch.tensor(stats["std"])
    mu, A, col, o, s_, z = sort_by_depth(*(t[None] for t in from_features(xs[0] * std + mean)))
    mu, A, col, o, s_, z = mu[0], A[0], col[0], o[0], s_[0], z[0]
    # decoded splat layout, back-to-front: mu_x, mu_y, e^a, b, e^c, r, g, b, alpha, p, z, 0
    sp = torch.cat([mu, A[:, :1].exp(), A[:, 1:2], A[:, 2:3].exp(), col, (0.99 * torch.sigmoid(o))[:, None],
                    (1 + F.softplus(s_))[:, None], z[:, None], torch.zeros(K, 1)], -1)
    cases = [{"kind": "forward", "t": 0.6, "y": [9, NCLASS], "x": x.ravel().tolist(), "v": v.ravel().tolist()},
             {"kind": "sample", "steps": 3, "cfg": 2.5, "y": 4, "noise": noise.ravel().tolist(), "out": xs.ravel().tolist(),
              "splats": sp.ravel().tolist(),
              "paint64": (((render(mu[None], A[None], col[None], o[None], s_[None], 64, 64).clamp(-1, 1) + 1) * 127.5)[0]
                          .permute(1, 2, 0).ravel().tolist())}]
    (out / "cases.json").write_text(json.dumps(cases))
    print("fixture ok")


if __name__ == "__main__":
    if sys.argv[1] == "weights":
        sd = torch.load(sys.argv[2], map_location="cpu")
        st = json.loads(Path(sys.argv[3]).read_text())
        write(sd, {"mean": st["mean"], "std": st["std"]}, st["K"], Path(sys.argv[4]))
    else:
        fixture(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 64)
