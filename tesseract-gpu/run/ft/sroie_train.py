"""SROIE receipt lines for fine-tuning, from receipts the evaluation set does not use.

    python3 run/ft/sroie_train.py          # -> data/ft/sroie/*.png + data/ft/sroie/labels.jsonl (src "sroie")

The evaluation set (data/real, run/make_real.py) takes its 400 SROIE lines from the first nine receipts of the
train shard (sroie-0000 to sroie-0008). Training lines here come from receipts 30 onward of the same shard, so no
receipt is in both; the test shard is not read. Line filters and crops follow make_real.py (3+ characters, 8+ px,
horizontal, slope under 3 degrees, 2 px padding).

SROIE's transcriptions are uppercase even where the receipt prints lowercase. A line is dropped when its label
has no lowercase letters but the stock eng model reads it with more lowercase than uppercase letters, so the
network is not taught to answer lowercase print in capitals. Licence: the mirror's card says MIT; the upstream
SROIE licence was not checked (notes/data_real.md).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "run"))
sys.path.insert(0, str(ROOT))
from make_real import MAX_SLOPE_DEG, MIN_CHARS, MIN_H, PAD, image_of, load_rows, sroie_lines  # noqa: E402

OUT = ROOT / "data/ft/sroie"
FIRST = 30


def main():
    import torch
    from server.torch_engine import TorchTess
    from tessgpu import decode, lineprep
    torch.set_num_threads(2)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = TorchTess(str(ROOT / "models/best/eng.traineddata"), device=dev, mode="cudnn")
    OUT.mkdir(parents=True, exist_ok=True)
    recs = load_rows("arvindrajan92/sroie_document_understanding",
                     ["data/train-00000-of-00001-66201d4afd6b73ca.parquet"])
    print(f"{len(recs)} receipts in the train shard; using {FIRST}..{len(recs) - 1}", flush=True)
    rows, dropped = [], {"case": 0, "filter": 0}
    batch = []

    def flush():
        crops = [lineprep.clean_line(c) for c, _ in batch]
        for (crop, meta), p in zip(batch, eng.recognize(crops)):
            pred = decode.greedy(p, eng.codec)
            lo = sum(ch.islower() for ch in pred)
            up = sum(ch.isupper() for ch in pred)
            if not any(ch.islower() for ch in meta["text"]) and lo > up:
                dropped["case"] += 1
                continue
            from PIL import Image
            Image.fromarray(crop_raw[meta["file"]]).save(OUT / meta["file"])
            meta["file"] = meta["file"]
            rows.append(meta)
        batch.clear()

    crop_raw = {}
    for d in range(FIRST, len(recs)):
        rec = recs[d]
        img = None
        for k, (text, box, slope) in enumerate(sroie_lines(rec)):
            text = text.strip()
            x0, y0, x1, y1 = box
            if len(text) < MIN_CHARS or y1 - y0 < MIN_H or (y1 - y0) > (x1 - x0) or abs(((slope + 90) % 180) - 90) > MAX_SLOPE_DEG:
                dropped["filter"] += 1
                continue
            if img is None:
                img = image_of(rec).convert("L")
            crop = (max(0, math.floor(x0) - PAD), max(0, math.floor(y0) - PAD),
                    min(img.width, math.ceil(x1) + PAD), min(img.height, math.ceil(y1) + PAD))
            if crop[2] <= crop[0] or crop[3] <= crop[1]:
                dropped["filter"] += 1
                continue
            a = np.asarray(img.crop(crop))
            name = f"sroie-{d:04d}-{k:03d}.png"
            crop_raw[name] = a
            batch.append((a, {"file": name, "text": text, "src": "sroie", "receipt": d}))
            if len(batch) >= 200:
                flush()
                crop_raw.clear()
        if d % 50 == 0:
            print(f"receipt {d}: {len(rows)} lines kept, dropped {dropped}", flush=True)
    if batch:
        flush()
    with open(OUT / "labels.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} lines; dropped {dropped}", flush=True)


if __name__ == "__main__":
    main()
