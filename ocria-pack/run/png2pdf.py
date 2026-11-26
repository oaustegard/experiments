"""Wrap each OmniParseBench image page in a one-page PDF so LiteParse's wasm build can read it.

The wasm build parses PDFs only. Each PNG becomes a PDF page whose single image is stored losslessly
(FlateDecode, 8-bit gray or RGB) at a page size of px * 72 / DPI points, so that LiteParse rendering
the page at --dpi DPI gets back the original pixels.

    python3 run/png2pdf.py --ds DS --out PDFDIR [--dpi 250]
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

from PIL import Image


def pdf_bytes(img: Image.Image, dpi: float) -> bytes:
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    elif img.mode in ("1", "I;16", "I", "F"):
        img = img.convert("L")
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    cs = "/DeviceGray" if img.mode == "L" else "/DeviceRGB"
    w, h = img.size
    data = zlib.compress(img.tobytes(), 6)
    pw, ph = w * 72.0 / dpi, h * 72.0 / dpi
    content = f"q {pw:.4f} 0 0 {ph:.4f} 0 0 cm /Im0 Do Q".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.4f} {ph:.4f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace {cs} "
         f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(data)} >>\nstream\n").encode() + data + b"\nendstream",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dpi", type=float, default=250.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in sorted((a.ds / "pages").glob("*.png")):
        dst = a.out / (png.stem + ".pdf")
        if dst.exists():
            continue
        with Image.open(png) as im:
            im.load()
            dst.write_bytes(pdf_bytes(im, a.dpi))
        n += 1
    print(f"wrapped {n} images")


if __name__ == "__main__":
    main()
