#!/usr/bin/env python3
"""Generate the two sample images the demo page bundles (web/samples/*.png). Original text, no scans."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "web" / "samples"
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"
S = "/usr/share/fonts/truetype/dejavu/DejaVuSerif{}.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def letter():
    im = Image.new("RGB", (900, 560), (250, 248, 242))
    d = ImageDraw.Draw(im)
    d.text((50, 40), "Harbour Light Cafe", font=font(S.format("-Bold"), 38), fill=(30, 30, 40))
    d.text((50, 100), "Opening hours for the autumn season", font=font(F.format(""), 24), fill=(60, 60, 70))
    body = ["Monday to Friday, 7:30 to 17:00.", "Saturday and Sunday, 9:00 to 15:00.",
            "Closed on public holidays.", "Order ahead by phone: 555-0142.",
            "Soup of the day is served from 11:30."]
    y = 170
    for t in body:
        d.text((50, y), t, font=font(S.format(""), 28), fill=(35, 35, 45))
        y += 52
    d.text((50, 470), "Thank you for visiting us this year.", font=font(F.format(""), 22), fill=(90, 90, 100))
    im.save(OUT / "letter.png", optimize=True)


def receipt():
    im = Image.new("RGB", (640, 760), (255, 255, 255))
    d = ImageDraw.Draw(im)
    mono = font("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26)
    d.text((140, 30), "CORNER MARKET", font=font(F.format("-Bold"), 34), fill=(0, 0, 0))
    d.text((60, 90), "12 Mill Road, Springfield", font=mono, fill=(40, 40, 40))
    rows = [("Bread", "3.20"), ("Whole milk", "1.85"), ("Cheddar", "6.40"), ("Apples", "4.10"),
            ("Coffee beans", "9.75"), ("Olive oil", "7.30")]
    y = 160
    for n, p in rows:
        d.text((60, y), n, font=mono, fill=(0, 0, 0))
        d.text((470, y), p, font=mono, fill=(0, 0, 0))
        y += 54
    d.line((60, y + 4, 580, y + 4), fill=(0, 0, 0), width=2)
    d.text((60, y + 24), "Total", font=font(F.format("-Bold"), 30), fill=(0, 0, 0))
    d.text((460, y + 24), "32.60", font=font(F.format("-Bold"), 30), fill=(0, 0, 0))
    d.text((90, y + 110), "Paid by card. Thank you!", font=mono, fill=(40, 40, 40))
    im.save(OUT / "receipt.png", optimize=True)


OUT.mkdir(parents=True, exist_ok=True)
letter()
receipt()
