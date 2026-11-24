#!/usr/bin/env python3
"""Render synthetic text-line crops for the recognizer parity tests (data only, git-ignored).

    python3 -I web/rec/test/make_lines.py            # -> web/rec/test/fixtures/

Each line is drawn with PIL (DejaVu, Liberation and WenQuanYi fonts; English, French, German,
Spanish, Chinese, digits and mixed lines; dark and coloured text on light and tinted grounds), then
preprocessed as PaddleOCR's RecResizeImg does for one line: resize to height 48 keeping the aspect
ratio (width ceil(48 w / h), bilinear), BGR, x / 255, (x - 0.5) / 0.5. No right padding, so the
widths span ~100 to ~1200 and the engine sees odd widths too. A batch fixture puts four lines,
zero-padded on the right to the widest, in one [4, 3, 48, W] input.

Writes fixtures/index.json, fixtures/<id>/{x.f32, line.png} and fixtures/batch4/{x.f32, meta.json}.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures"
FONTS = Path("/usr/share/fonts/truetype")
LATIN = ["dejavu/DejaVuSans.ttf", "dejavu/DejaVuSerif.ttf", "liberation/LiberationSans-Regular.ttf",
         "liberation/LiberationSerif-Regular.ttf", "liberation/LiberationMono-Regular.ttf", "dejavu/DejaVuSans-Bold.ttf"]
CJK = ["wqy/wqy-zenhei.ttc"]
LINES = [
    ("en", "Invoice"),
    ("en", "Total: $1,284.50"),
    ("en", "The quick brown fox jumps over the lazy dog."),
    ("en", "Please return the signed form by 12 March 2025."),
    ("en", "Section 4.2: Results of the second trial (n = 318) held."),
    ("en", "Call +1 (617) 555-0142 or write to help@example.com"),
    ("fr", "Résumé"),
    ("fr", "Le café était très chaud ce matin-là, près de la fenêtre."),
    ("fr", "Où est la bibliothèque municipale ? À côté de l'église."),
    ("de", "Straße"),
    ("de", "Die Größe der Wohnung beträgt 84 Quadratmeter."),
    ("de", "Bitte überprüfen Sie die Angaben auf Seite 7."),
    ("es", "¿Dónde está la estación de tren?"),
    ("es", "El niño comió una manzana en el jardín."),
    ("zh", "中文"),
    ("zh", "今天天气很好"),
    ("zh", "我们明天下午三点在会议室讨论这个项目的进展情况。"),
    ("mix", "2024-11-24 08:15 ID#A7731"),
    ("mix", "PP-OCRv6 识别测试 123"),
    ("en", "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor."),
]
BATCH = [1, 7, 15, 17]


def render(text, font_path, size, fg, bg, rng):
    font = ImageFont.truetype(str(FONTS / font_path), size)
    l, t, r, b = font.getbbox(text)
    pad_x, pad_y = rng.randint(3, 12), rng.randint(3, 9)
    img = Image.new("RGB", (r - l + 2 * pad_x, b - t + 2 * pad_y), bg)
    ImageDraw.Draw(img).text((pad_x - l, pad_y - t), text, font=font, fill=fg)
    return img


def preprocess(img, H=48):
    w, h = img.size
    W = int(math.ceil(H * w / h))
    a = np.asarray(img.resize((W, H), Image.BILINEAR), np.float32)[:, :, ::-1]   # RGB -> BGR
    x = (a / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
    return np.ascontiguousarray(x.transpose(2, 0, 1), np.float32)


def main():
    rng = random.Random(20261124)
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    xs = []
    for i, (lang, text) in enumerate(LINES):
        fonts = CJK if lang in ("zh", "mix") and any(ord(c) > 0x2e80 for c in text) else LATIN
        font = fonts[i % len(fonts)]
        size = rng.randint(22, 40)
        if i % 4 == 3:   # coloured text on a tinted ground
            fg = (rng.randint(0, 90), rng.randint(0, 90), rng.randint(60, 160))
            bg = (rng.randint(215, 255), rng.randint(200, 245), rng.randint(180, 230))
        else:
            g = rng.randint(0, 60)
            fg, bg = (g, g, g), (rng.randint(225, 255),) * 3
        img = render(text, font, size, fg, bg, rng)
        x = preprocess(img)
        lid = f"l{i:02d}_{lang}"
        d = OUT / lid
        d.mkdir(exist_ok=True)
        x.tofile(d / "x.f32")
        img.save(d / "line.png")
        index.append({"id": lid, "lang": lang, "text": text, "font": font, "size": size, "W": int(x.shape[2])})
        xs.append(x)
    Wmax = max(xs[i].shape[2] for i in BATCH)
    xb = np.zeros((len(BATCH), 3, 48, Wmax), np.float32)
    for j, i in enumerate(BATCH):
        xb[j, :, :, : xs[i].shape[2]] = xs[i]
    d = OUT / "batch4"
    d.mkdir(exist_ok=True)
    xb.tofile(d / "x.f32")
    (d / "meta.json").write_text(json.dumps({"lines": [index[i]["id"] for i in BATCH], "B": len(BATCH), "W": Wmax}) + "\n")
    (OUT / "index.json").write_text(json.dumps({"lines": index, "batch": {"id": "batch4", "B": len(BATCH), "W": Wmax}},
                                               ensure_ascii=False, indent=1) + "\n")
    ws = sorted(e["W"] for e in index)
    print(f"{len(index)} lines, widths {ws}; batch4 W {Wmax} -> {OUT}")


if __name__ == "__main__":
    main()
