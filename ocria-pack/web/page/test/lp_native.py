# LiteParse 2.21.2 native (ONNX Runtime 1.31.1) OCR items for the 40 reference PDFs, layout off.
import json
import os

from liteparse import LiteParse

ids = open("ref40.txt").read().split()
p = LiteParse(ocr_backend="ocria-nano", dpi=250, output_format="json", quiet=True, keep_headers_footers=True,
              image_mode="off", num_workers=1)
for i in ids:
    r = p.parse(f"pdf/{i}.pdf")
    pg = r.pages[0]
    items = [{"text": t.text, "x": t.x, "y": t.y, "w": t.width, "h": t.height, "conf": t.confidence}
             for t in pg.text_items]
    with open(f"out/{i}.json", "w") as f:
        json.dump({"page_id": i, "width": pg.width, "height": pg.height, "text": pg.text, "text_items": items}, f)
    print(i, len(items), flush=True)
