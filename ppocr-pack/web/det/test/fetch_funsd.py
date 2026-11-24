#!/usr/bin/env python3
"""Save two FUNSD test-split form scans into web/det/test/data/pages/ (research use only; git-ignored).

Dataset: nielsr/funsd on Hugging Face (FUNSD, Jaume et al. 2019; research use only).

    python3 -I web/det/test/fetch_funsd.py
"""
import io
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "web/det/test/data/funsd_raw/test.parquet"
URL = "https://huggingface.co/datasets/nielsr/funsd/resolve/main/data/test-00000-of-00001.parquet"
OUT = ROOT / "web/det/test/data/pages"


def main():
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    table = pq.read_table(RAW, columns=["id", "image"])
    OUT.mkdir(parents=True, exist_ok=True)
    for i in range(2):
        blob = table.column("image")[i].as_py()["bytes"]
        im = Image.open(io.BytesIO(blob)).convert("RGB")
        im.save(OUT / f"funsd_{i}.png")
        print(f"funsd_{i}", table.column("id")[i].as_py(), im.size)


if __name__ == "__main__":
    main()
