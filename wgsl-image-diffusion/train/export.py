"""Export U-Net weights for the WGSL runtime, and build test fixtures.

    python export.py weights ema.pt ../web/model            # f16 weights.bin + manifest.json
    python export.py fixture ../web/test                     # random-weight model + torch reference I/O

weights.bin is little-endian float16, tensors back to back in state_dict order; manifest.json maps
each name to [offset_in_elements, shape] and carries the op list (model.layout()) the runtime follows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import NAMES, NCLASS, UNet, layout, sample


def write_weights(sd, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    man, chunks, off = {}, [], 0
    for k, v in sd.items():
        a = v.detach().float().cpu().numpy().astype(np.float16).ravel()
        man[k] = [off, list(v.shape)]
        chunks.append(a)
        off += a.size
    np.concatenate(chunks).tofile(out / "weights.bin")
    (out / "manifest.json").write_text(json.dumps({"n": off, "tensors": man, "nclass": NCLASS, "names": NAMES,
                                                   "layout": [list(o) for o in layout()]}))
    print(f"{off} params, {(out / 'weights.bin').stat().st_size/1e6:.1f} MB")


def load_bin(bin_path, man_path):
    """A UNet holding exactly the exported f16 weights, so torch references match the GPU's inputs."""
    w = np.fromfile(bin_path, np.float16).astype(np.float32)
    man = json.loads(Path(man_path).read_text())["tensors"]
    m = UNet().eval()
    m.load_state_dict({k: torch.from_numpy(w[o:o + int(np.prod(s))].reshape(s).copy()) for k, (o, s) in man.items()})
    return m


def fixture(out: Path):
    """Random (non-zero) weights so every layer, attention included, changes the output."""
    torch.manual_seed(7)
    m = UNet().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                fan = p[0].numel()
                p.copy_(torch.randn_like(p) / fan ** 0.5)
            else:
                gain = name.endswith(("n1.weight", "n2.weight", "norm.weight", "n_out.weight"))
                p.copy_(0.1 * torch.randn_like(p) + (1.0 if gain else 0.0))
    write_weights(m.state_dict(), out)
    m = load_bin(out / "weights.bin", out / "manifest.json")
    g = torch.Generator().manual_seed(3)
    cases = []
    # 48x80 gives L = 240 at 16x16-level attention (not a multiple of the 64-query workgroup or 16-key tile)
    for (H, W, y) in [(64, 64, [1, NCLASS]), (48, 80, [2, 0]), (32, 32, [0, NCLASS])]:
        x = torch.randn(len(y), 3, H, W, generator=g)
        t = torch.full((len(y),), 0.73)
        with torch.no_grad():
            v = m(x, t, torch.tensor(y))
        cases.append({"kind": "forward", "H": H, "W": W, "t": 0.73, "y": y,
                      "x": x.numpy().ravel().tolist(), "v": v.numpy().ravel().tolist()})
    noise = torch.randn(1, 3, 32, 32, generator=g)
    xs = sample(m, torch.tensor([2]), 32, 32, steps=3, cfg=2.5, noise=noise)
    cases.append({"kind": "sample", "H": 32, "W": 32, "steps": 3, "cfg": 2.5, "y": 2,
                  "noise": noise.numpy().ravel().tolist(), "out": xs.numpy().ravel().tolist()})
    (out / "cases.json").write_text(json.dumps(cases))
    print(f"{len(cases)} cases")


if __name__ == "__main__":
    if sys.argv[1] == "weights":
        sd = torch.load(sys.argv[2], map_location="cpu")
        write_weights(sd, Path(sys.argv[3]))
    elif sys.argv[1] == "fixture":
        fixture(Path(sys.argv[2]))
