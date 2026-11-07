"""Gallery: 12 classes from the exported weights, 24 steps, guidance 3, each 128x128 tile drawn 2x2.

    python3 run/gallery.py web/model OUT.png
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from model import sample
from ref_from_bin import load
from splats import from_features, render

CLASSES = ["marbled", "veined", "crystalline", "swirly", "paisley", "honeycombed", "scaly", "cracked",
           "lacelike", "bubbly", "woven", "cobwebbed"]


def main():
    base, out = Path(sys.argv[1]), sys.argv[2]
    torch.set_num_threads(4)
    m, man = load(base / "weights.bin", base / "manifest.json")
    mean, std = torch.tensor(man["stats"]["mean"]), torch.tensor(man["stats"]["std"])
    cell, cols, pad, lab = 256, 4, 8, 22
    g = Image.new("RGB", (cols * (cell + pad) + pad, 3 * (cell + lab + pad) + pad), (14, 14, 16))
    d = ImageDraw.Draw(g)
    for k, c in enumerate(CLASSES):
        x = sample(m, torch.tensor([man["names"].index(c)]), man["K"], steps=24, cfg=3.0, seed=10 + k)
        mu, A, col = from_features(x[0] * std + mean)
        img = render(mu[None], A[None], col[None], 128, 128, chunk=4096).clamp(-1, 1)[0]
        a = ((img.permute(1, 2, 0).numpy() + 1) * 127.5).round().astype(np.uint8)
        r, cc = divmod(k, cols)
        x0, y0 = pad + cc * (cell + pad), pad + r * (cell + lab + pad)
        g.paste(Image.fromarray(np.tile(a, (2, 2, 1))), (x0, y0))
        d.text((x0 + 2, y0 + cell + 4), c, fill=(220, 215, 205))
    g.save(out)


if __name__ == "__main__":
    main()
