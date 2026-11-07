"""Turn the bridge bank (bank/bank.npy from job tc-bank-1) into the files the harness and page read.

bank/bank.bin + bank/meta.json  raw tiles for run/build_table.mjs and run/eval.mjs
bank/bank.png                   all tiles, lossless: one row per class, one column per seed (the bank of record)
web/bank.png                    seed 0 of each class, 8 per row: the page's instant textures
To rebuild bank.bin from the committed PNG: python3 run/bank_files.py --from-png
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

d = Path(__file__).resolve().parents[1]
r = json.load(open(d / "bank/result.json"))["outputs"]
names, S, seeds = r["names"], r["S"], r["seeds"]
C = len(names)
if "--from-png" in sys.argv:
    sheet = np.asarray(Image.open(d / "bank/bank.png").convert("RGB"))
    a = sheet.reshape(C, S, seeds, S, 3).transpose(0, 2, 1, 3, 4)
else:
    a = np.load(d / "bank/bank.npy").reshape(C, seeds, S, S, 3)
    Image.fromarray(a.transpose(0, 2, 1, 3, 4).reshape(C * S, seeds * S, 3)).save(d / "bank/bank.png", optimize=True)
a.astype(np.uint8).tofile(d / "bank/bank.bin")
json.dump({"names": names, "S": S, "seeds": seeds, "steps": r["steps"], "cfg": 3, "weights": "q8", "seed_base": 1000,
           "job": "tc-bank-1", "device": "Mac", "median_ms": r["median_ms"]}, open(d / "bank/meta.json", "w"), indent=1)
cols = 8
rows = (C + cols - 1) // cols
atlas = np.zeros((rows * S, cols * S, 3), np.uint8)
for c in range(C):
    atlas[(c // cols) * S:(c // cols + 1) * S, (c % cols) * S:(c % cols + 1) * S] = a[c, 0]
Image.fromarray(atlas).save(d / "web/bank.png", optimize=True)
print("ok", a.shape)
