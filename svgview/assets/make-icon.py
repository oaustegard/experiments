#!/usr/bin/env python3
"""Rebuild assets/svgview.ico from assets/icon.svg.

Rasterisation is done by svgview itself, so the icon and the viewer always
agree. Run after editing icon.svg:

    cargo build && python3 assets/make-icon.py

Windows Vista and later read PNG-compressed ICO entries at every size, which
keeps the file about 20x smaller than the equivalent BMP entries.
"""

import struct
import subprocess
import sys
from pathlib import Path

SIZES = [16, 24, 32, 48, 64, 128, 256]
ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parent


def find_svgview() -> Path:
    for profile in ("release", "debug"):
        for name in ("svgview", "svgview.exe"):
            candidate = ROOT / "target" / profile / name
            if candidate.exists():
                return candidate
    sys.exit("build svgview first: cargo build")


def main() -> None:
    exe = find_svgview()
    out_dir = ROOT / "target" / "icon"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for size in SIZES:
        png = out_dir / f"{size}.png"
        subprocess.run(
            [str(exe), str(ASSETS / "icon.svg"), "--png", str(png), "--width", str(size)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        images.append((size, png.read_bytes()))

    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, blobs = b"", b""
    for size, data in images:
        # 0 means 256 in the ICONDIRENTRY byte fields.
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)

    ico = ASSETS / "svgview.ico"
    ico.write_bytes(header + entries + blobs)
    print(f"{ico} ({len(header + entries + blobs)} bytes, {len(images)} sizes)")


if __name__ == "__main__":
    main()
