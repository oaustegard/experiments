"""Export U-Net weights for the WGSL runtime, and build test fixtures.

    python export.py weights ema.pt ../web/model            # f16 weights.bin + manifest.json
    python export.py fixture ../web/test                     # random-weight model + torch reference I/O

weights.bin is little-endian float16, tensors back to back in state_dict order;
manifest.json maps each name to [offset_in_elements, shape].
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import NCLASS, PAD, UNet, sample

NAMES = ["banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked", "crosshatched",
         "crystalline", "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy", "grid", "grooved",
         "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled", "matted", "meshed", "paisley",
         "perforated", "pitted", "pleated", "polka-dotted", "porous", "potholed", "scaly", "smeared", "spiralled",
         "sprinkled", "stained", "stratified", "striped", "studded", "swirly", "veined", "waffled", "woven",
         "wrinkled", "zigzagged"]   # DTD label order (tanganke/dtd)


def write_weights(sd, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    man, chunks, off = {}, [], 0
    for k, v in sd.items():
        a = v.detach().float().cpu().numpy().astype(np.float16).ravel()
        man[k] = [off, list(v.shape)]
        chunks.append(a)
        off += a.size
    np.concatenate(chunks).tofile(out / "weights.bin")
    (out / "manifest.json").write_text(json.dumps({"n": off, "tensors": man, "nclass": NCLASS, "names": NAMES}))
    print(f"{off} params, {(out / 'weights.bin').stat().st_size/1e6:.1f} MB")


def write_weights_q8(sd, out: Path):
    """int8 weights, symmetric per output row, for every tensor with 2+ dims; f16 for the rest.

    weights_q8.bin holds, per tensor in state_dict order, either f16 values or f32 row scales
    followed by int8 values (each part padded to 4 bytes). manifest_q8.json carries the same
    "tensors" map as manifest.json (offsets into the dequantized f32 array) plus "parts" =
    [name, kind, byte_offset, rows] so the loader can rebuild that array.
    """
    out.mkdir(parents=True, exist_ok=True)
    man, parts, chunks, off, pos = {}, [], [], 0, 0
    for k, v in sd.items():
        a = v.detach().float().cpu().numpy()
        man[k] = [off, list(v.shape)]
        off += a.size
        if a.ndim >= 2:
            rows = a.reshape(a.shape[0], -1)
            sc = np.maximum(np.abs(rows).max(1), 1e-12) / 127
            q = np.clip(np.round(rows / sc[:, None]), -127, 127).astype(np.int8)
            b = sc.astype(np.float32).tobytes() + q.tobytes()
            parts.append([k, "q8", pos, int(a.shape[0])])
        else:
            b = a.astype(np.float16).tobytes()
            parts.append([k, "f16", pos, 0])
        b += b"\0" * (-len(b) % 4)
        chunks.append(b)
        pos += len(b)
    (out / "weights_q8.bin").write_bytes(b"".join(chunks))
    (out / "manifest_q8.json").write_text(json.dumps({"n": off, "tensors": man, "parts": parts,
                                                       "nclass": NCLASS, "names": NAMES}))
    print(f"q8: {pos/1e6:.1f} MB")


def f16_roundtrip(model):
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(p.half().float())
    return model


def fixture(out: Path):
    """Random (non-zero) weights so every layer matters, then reference outputs."""
    torch.manual_seed(7)
    m = UNet().eval()
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() > 1:
                fan = p[0].numel()
                p.copy_(torch.randn_like(p) / fan ** 0.5)
            else:
                p.copy_(0.1 * torch.randn_like(p) + (1.0 if name.endswith(("n1.weight", "n2.weight", "n_out.weight")) else 0.0))
    f16_roundtrip(m)
    write_weights(m.state_dict(), out)
    g = torch.Generator().manual_seed(3)
    cases = []
    for (H, W, pad) in [(64, 64, "zeros"), (48, 80, "circular"), (64, 64, "circular")]:
        x = torch.randn(2, 3, H, W, generator=g)
        t = torch.tensor([0.73, 0.73])
        y = torch.tensor([5, NCLASS])
        PAD["mode"] = pad
        with torch.no_grad():
            v = m(x, t, y)
        PAD["mode"] = "zeros"
        cases.append({"kind": "forward", "H": H, "W": W, "pad": pad, "t": 0.73, "y": [5, NCLASS],
                      "x": x.numpy().ravel().tolist(), "v": v.numpy().ravel().tolist()})
    # a short guided sampling run from fixed noise
    noise = torch.randn(1, 3, 32, 32, generator=g)
    xs = sample(m, torch.tensor([11]), 32, 32, steps=3, cfg=2.5, pad="circular", noise=noise)
    cases.append({"kind": "sample", "H": 32, "W": 32, "pad": "circular", "steps": 3, "cfg": 2.5, "y": 11,
                  "noise": noise.numpy().ravel().tolist(), "out": xs.numpy().ravel().tolist()})
    (out / "cases.json").write_text(json.dumps(cases))
    print(f"{len(cases)} cases")


if __name__ == "__main__":
    if sys.argv[1] == "weights":
        sd = torch.load(sys.argv[2], map_location="cpu")
        write_weights(sd, Path(sys.argv[3]))
        write_weights_q8(sd, Path(sys.argv[3]))
    elif sys.argv[1] == "fixture":
        fixture(Path(sys.argv[2]))
