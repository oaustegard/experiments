#!/usr/bin/env python3
"""Render the three synthetic test pages into web/det/test/data/pages/ (git-ignored, not committed).

    python3 -I web/det/test/make_pages.py
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "web/det/test/data/pages"
FONTS = Path("/usr/share/fonts/truetype/dejavu")
WORDS = ("the model reads each line of the page and returns a probability map for text "
         "pixels with a threshold of two tenths over regions where glyphs are dense").split()


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def para(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def two_column(rng):
    im = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(im)
    f = font("DejaVuSerif.ttf", 22)
    d.text((120, 110), "Two-column article", font=font("DejaVuSerif-Bold.ttf", 34), fill="black")
    for col in (0, 1):
        x0 = 120 + col * 520
        y = 190
        for _ in range(9):
            words = para(rng, 60).split()
            line = ""
            for w in words:
                trial = (line + " " + w).strip()
                if d.textlength(trial, font=f) > 480:
                    d.text((x0, y), line, font=f, fill="black")
                    y += 32
                    line = w
                else:
                    line = trial
            d.text((x0, y), line, font=f, fill="black")
            y += 56
    return im


def heading_table(rng):
    im = Image.new("RGB", (1100, 1400), (252, 252, 250))
    d = ImageDraw.Draw(im)
    d.text((80, 90), "Quarterly results", font=font("DejaVuSans-Bold.ttf", 52), fill=(20, 20, 20))
    d.text((80, 160), "Summary of the regional figures reported in March.", font=font("DejaVuSans.ttf", 26), fill=(60, 60, 60))
    y = 230
    f = font("DejaVuSerif.ttf", 24)
    for _ in range(4):
        d.text((80, y), para(rng, 26), font=f, fill="black")
        y += 44
    y += 30
    rows = [("Region", "Units", "Revenue", "Change")] + [
        (r, str(rng.randint(100, 900)), f"{rng.randint(10, 999)},{rng.randint(100, 999)}", f"{rng.uniform(-9, 14):+.1f}%")
        for r in ("North", "South", "East", "West", "Central")]
    xs = [80, 330, 520, 760, 1000]
    fs = font("DejaVuSans.ttf", 20)
    for i, row in enumerate(rows):
        d.rectangle([xs[0], y, xs[-1], y + 40], fill=(230, 233, 240) if i == 0 else "white", outline=(120, 120, 120), width=2)
        for j, cell in enumerate(row):
            d.text((xs[j] + 12, y + 9), cell, font=fs, fill="black")
        y += 40
    for x in xs:
        d.line([(x, 230 + 4 * 44 + 30), (x, y)], fill=(120, 120, 120), width=2)
    return im


def dense_small(rng):
    im = Image.new("RGB", (1000, 1400), "white")
    d = ImageDraw.Draw(im)
    f = font("DejaVuSansMono.ttf", 15)
    fb = font("DejaVuSansMono-Bold.ttf", 15)
    d.rectangle([0, 0, 1000, 60], fill=(40, 60, 90))
    d.text((30, 20), "INVOICE 2026-0117   page 1 of 1", font=fb, fill="white")
    y = 90
    for i in range(44):
        d.text((30, y), f"{i + 1:03d}  SKU-{rng.randint(10000, 99999)}  {rng.choice(WORDS)[:14]:<14}"
                        f"  {rng.randint(1, 40):>3}  {rng.randint(1, 999):>6}.{rng.randint(0, 99):02d}",
               font=f, fill="black")
        if i % 9 == 8:
            d.rectangle([20, y + 22, 980, y + 26], fill=(180, 180, 180))
        y += 26 + (6 if i % 9 == 8 else 0)
    d.text((30, y + 30), "Notes: payment within 30 days; quote the invoice number on all correspondence.",
           font=font("DejaVuSans.ttf", 16), fill=(70, 70, 70))
    return im


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    for name, fn in (("text_two_column", two_column), ("heading_table", heading_table), ("dense_small", dense_small)):
        im = fn(rng)
        im.save(OUT / f"{name}.png")
        print(name, im.size)


if __name__ == "__main__":
    main()
