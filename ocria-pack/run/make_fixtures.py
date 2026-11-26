"""Raw gray fixtures for web/test/check_ocria.mjs: u32 width, u32 height (little-endian), then the pixels.

    python3 run/make_fixtures.py --gray GRAYDIR --list PAGES.txt --n 12 --out FIXDIR
"""
import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gray", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = a.list.read_text().split()[: a.n]
    for pid in ids:
        g = np.asarray(Image.open(a.gray / f"{pid}.png").convert("L"))
        h, w = g.shape
        (a.out / f"{pid}.gray").write_bytes(struct.pack("<II", w, h) + g.tobytes())
    print(f"{len(ids)} fixtures")


if __name__ == "__main__":
    main()
