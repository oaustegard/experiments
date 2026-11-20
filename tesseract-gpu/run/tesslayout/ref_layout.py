"""Tesseract's own layout-only line boxes via tesserocr, for the same pages as web/test/check_tesslayout.mjs.

python3 data/tesslayout/ref_layout.py <tessdata_dir> <page.png>...  ->  JSON on stdout:
{"tesseract": "<version>", "pages": {"<page path>": {"lines": [{"box": [x0,y0,x1,y1], "block_start": bool}]}}}
Pages are converted to 8-bit grey with PIL's 'L' (BT.601), the same luma the browser side uses.
"""
import json
import sys

from PIL import Image
from tesserocr import PSM, RIL, PyTessBaseAPI, tesseract_version


def layout(api, path):
    img = Image.open(path).convert("L")
    api.SetImage(img)
    it = api.AnalyseLayout()
    if it is None:  # Tesseract's answer for an empty page
        return []
    out = []
    while True:
        x0, y0, x1, y1 = it.BoundingBox(RIL.TEXTLINE)
        out.append({"box": [int(x0), int(y0), int(x1), int(y1)], "block_start": bool(it.IsAtBeginningOf(RIL.BLOCK))})
        if not it.Next(RIL.TEXTLINE):
            break
    return out


def main():
    tessdata, paths = sys.argv[1], sys.argv[2:]
    api = PyTessBaseAPI(path=tessdata, lang="eng", psm=PSM.AUTO)
    pages = {p: {"lines": layout(api, p)} for p in paths}
    api.End()
    print(json.dumps({"tesseract": tesseract_version().split()[1], "pages": pages}))


if __name__ == "__main__":
    main()
