"""Gallery of 12 classes: 128x128 tiles, 20 steps, guidance 3, circular padding, each drawn 2x2.

    python3 run/gallery.py WEIGHTS.bin MANIFEST.json OUT.png
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from export import NAMES
from model import sample
from ref_from_bin import load

CLASSES = ["marbled", "veined", "crystalline", "swirly", "paisley", "honeycombed", "scaly", "cracked",
           "lacelike", "bubbly", "woven", "cobwebbed"]


def main():
    wb, man, out = sys.argv[1:4]
    torch.set_num_threads(4)
    m = load(wb, man)
    cell, cols, pad, lab = 256, 4, 8, 22
    rows = (len(CLASSES) + cols - 1) // cols
    g = Image.new("RGB", (cols * (cell + pad) + pad, rows * (cell + lab + pad) + pad), (14, 14, 16))
    d = ImageDraw.Draw(g)
    for k, c in enumerate(CLASSES):
        x = sample(m, torch.tensor([NAMES.index(c)]), 128, 128, steps=20, cfg=3.0, seed=10 + k, pad="circular")
        a = ((x[0].permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        r, cc = divmod(k, cols)
        x0, y0 = pad + cc * (cell + pad), pad + r * (cell + lab + pad)
        g.paste(Image.fromarray(np.tile(a, (2, 2, 1))), (x0, y0))
        d.text((x0 + 2, y0 + cell + 4), c, fill=(220, 215, 205))
    g.save(out)


if __name__ == "__main__":
    main()
