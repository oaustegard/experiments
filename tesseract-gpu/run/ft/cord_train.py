"""CORD v2 receipt lines for fine-tuning (naver-clova-ix/cord-v2, CC BY 4.0).

    python3 run/ft/cord_train.py            # -> data/ft/cord/*.png + data/ft/cord/labels.jsonl (src "cord")

Train and validation splits (900 receipt photos); the test split is not read. Each "valid_line" field's words are
merged into visual lines by run/make_real.py's cluster_lines and filtered and cropped as that script does (3+
characters, 8+ px high, horizontal, top edge under 3 degrees, 2 px padding). Quads are in image pixels (checked
against meta.image_size). Receipts numbered 0 mod 10 are held out for validation by run/ft/train.py.
"""
import io
import os
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "run"))
from make_real import MAX_SLOPE_DEG, PAD, cluster_lines, keep  # noqa: E402

OUT = ROOT / "data/ft/cord"
REPO = "naver-clova-ix/cord-v2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.endswith(".parquet") and ("/train-" in f or "/validation-" in f)
                   and os.environ.get("CORD_SPLIT", "") in f)
    rows, n, dropped = [], 0, {}
    for f in files:
        for rec in pq.read_table(hf_hub_download(REPO, f, repo_type="dataset")).to_pylist():
            img = Image.open(io.BytesIO(rec["image"]["bytes"])).convert("L")
            gt = json.loads(rec["ground_truth"])
            for g, field in enumerate(gt.get("valid_line", [])):
                items = []
                for w in field["words"]:
                    q = w["quad"]
                    items.append((w["text"], (min(q["x1"], q["x4"]), min(q["y1"], q["y2"]), max(q["x2"], q["x3"]), max(q["y3"], q["y4"]))))
                    slope = math.degrees(math.atan2(q["y2"] - q["y1"], max(1, q["x2"] - q["x1"])))
                    if abs(slope) > MAX_SLOPE_DEG:
                        items = None
                        break
                if not items:
                    dropped["rotated"] = dropped.get("rotated", 0) + 1
                    continue
                for k, (text, box, _) in enumerate(cluster_lines(items)):
                    why = keep(text, box, None)
                    if why:
                        dropped[why] = dropped.get(why, 0) + 1
                        continue
                    x0, y0, x1, y1 = box
                    crop = img.crop((max(0, x0 - PAD), max(0, y0 - PAD), min(img.width, x1 + PAD), min(img.height, y1 + PAD)))
                    name = f"cord-{n:04d}-{g:03d}-{k}.png"
                    crop.save(OUT / name)
                    rows.append({"file": name, "text": " ".join(text.split()), "src": "cord", "receipt": n})
            n += 1
        print(f"{f}: {n} receipts, {len(rows)} lines, dropped {dropped}", flush=True)
    with open(OUT / "labels.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} lines from {n} receipts; dropped {dropped}")


if __name__ == "__main__":
    main()
