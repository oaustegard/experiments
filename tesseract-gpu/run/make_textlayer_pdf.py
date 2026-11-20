"""Write web/samples/born-digital.pdf, the test PDF for the page's text-layer front end (web/pdftext.js).

  page 1  born-digital text in two columns, plus a picture of a scanned page: the layer covers the columns, the
          text inside the picture is in no layer and has to be OCR'd
  page 2  a scan with an invisible text layer whose text is wrong (as a bad OCR layer or a broken ToUnicode map
          gives): the layer must be rejected and the page OCR'd
  page 3  born-digital text only, with a superscript, no images: no OCR at all

The true text of each layer is written to results/textlayer_truth.json for run/check_textlayer.mjs.
    python3 run/make_textlayer_pdf.py
"""
import io
import json
import os
import random

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "web", "samples")
OUT = os.path.join(SAMPLES, "born-digital.pdf")
TRUTH = os.path.join(ROOT, "results", "textlayer_truth.json")

LEFT = [
    "The text layer of a born-digital PDF holds",
    "the characters the author typed, so a page",
    "that carries one needs no recognition at all.",
    "Reading it is exact and takes milliseconds.",
]
RIGHT = [
    "A scanned page has no such layer, and text",
    "inside a picture is not in the layer either,",
    "so those lines still go to the recognizer.",
    "Each line is checked against the image first.",
]
PAGE3 = [
    "Quarterly report, third quarter of 2026",
    "Revenue rose 12 percent over the prior year.",
    "Costs were flat apart from one-off items.",
    "The board met on 14 October and approved the plan.",
]


def jpeg(path, crop=None, quality=70):
    im = Image.open(path).convert("L")
    if crop:
        im = im.crop(crop)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return ImageReader(buf), im.size


def main():
    W, H = letter
    c = canvas.Canvas(OUT, pagesize=letter)
    c.setTitle("tesseract-gpu text-layer test")

    # page 1: two text columns, then a picture of the top of english.png
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, H - 72, "Born-digital page with a picture")
    c.setFont("Helvetica", 10.5)
    for i, (l, r) in enumerate(zip(LEFT, RIGHT)):
        y = H - 110 - 15 * i
        c.drawString(72, y, l)
        c.drawString(320, y, r)
    img, (iw, ih) = jpeg(os.path.join(SAMPLES, "english.png"), crop=(0, 0, 1275, 700))
    dw = W - 144
    c.drawImage(img, 72, 120, width=dw, height=dw * ih / iw)
    c.showPage()

    # page 2: french.png as a full-page scan under an invisible, wrong text layer
    img, _ = jpeg(os.path.join(SAMPLES, "french.png"))
    c.drawImage(img, 0, 0, width=W, height=H)
    rng = random.Random(7)
    t = c.beginText()
    t.setTextRenderMode(3)  # invisible, as OCR layers are
    t.setFont("Helvetica", 11)
    wrong = []
    for k in range(30):
        s = " ".join("".join(rng.choice("bcdfghjklmnpqrstvwxz") for _ in range(rng.randint(3, 8))) for _ in range(7))
        wrong.append(s)
        t.setTextOrigin(80, H - 90 - 20 * k)
        t.textLine(s)
    c.drawText(t)
    c.showPage()

    # page 3: text only, one superscript
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, H - 72, PAGE3[0])
    c.setFont("Helvetica", 11)
    for i, s in enumerate(PAGE3[1:]):
        y = H - 110 - 16 * i
        c.drawString(72, y, s)
        if i == 0:
            c.setFont("Helvetica", 7)
            c.drawString(72 + c.stringWidth(s, "Helvetica", 11) + 1, y + 4, "1")
            c.setFont("Helvetica", 11)
    c.showPage()
    c.save()

    truth = {
        "page1": {"title": "Born-digital page with a picture", "columns": LEFT + RIGHT},
        "page2": {"wrong_layer": wrong},
        "page3": {"lines": PAGE3, "superscript_after": PAGE3[1]},
    }
    os.makedirs(os.path.dirname(TRUTH), exist_ok=True)
    with open(TRUTH, "w") as f:
        json.dump(truth, f, indent=1)
    print(OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
