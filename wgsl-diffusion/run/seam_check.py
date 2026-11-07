"""Does circular padding make the tiles seamless? Measure the jump across the wrap boundary.

For each sample, seam ratio = mean |x[:, 0] - x[:, -1]| (and the column equivalent), i.e. the step a
viewer sees where two copies of the tile meet, divided by the mean step between neighbouring
interior rows/columns. 1.0 means the seam is as smooth as the inside of the texture.

    python3 run/seam_check.py ema.pt results/seam.json [--n 12] [--size 128]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model import UNet, sample

PRETTY = ["marbled", "veined", "crystalline", "swirly", "paisley", "honeycombed", "scaly", "cracked",
          "lacelike", "bubbly", "woven", "braided"]


def seam_ratio(x):
    interior = 0.5 * ((x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean())
    wrap = 0.5 * ((x[..., 0, :] - x[..., -1, :]).abs().mean() + (x[..., :, 0] - x[..., :, -1]).abs().mean())
    return float(wrap / interior)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    from export import NAMES
    torch.set_num_threads(4)
    m = UNet().eval()
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    rows = []
    for i, name in enumerate(PRETTY[:a.n]):
        y = torch.tensor([NAMES.index(name)])
        g = torch.Generator().manual_seed(100 + i)
        noise = torch.randn(1, 3, a.size, a.size, generator=g)
        r = {"class": name}
        for pad in ("zeros", "circular"):
            x = sample(m, y, a.size, a.size, steps=a.steps, cfg=3.0, pad=pad, noise=noise)
            r[pad] = seam_ratio(x)
        rows.append(r)
        print(r, flush=True)
    med = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]
    summary = {"size": a.size, "steps": a.steps, "cfg": 3.0, "n": len(rows),
               "median_zeros": med("zeros"), "median_circular": med("circular"), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1))
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
