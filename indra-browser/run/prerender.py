"""Every OmniParseBench page as the grayscale image Indra's OCR path reads.

A PDF page renders as indra-ocr's Converter renders it in mode="ocr" (pdfium,
grayscale, 200 DPI lowered past 100 MP: document.Converter.page_scale); an image
page is read as cv2.IMREAD_GRAYSCALE, as pipeline.read_images reads it. The
fp32 reference, every packed arm and the browser engine then read the same
pixels, so their outputs can be compared exactly.

    python3 run/prerender.py --ds /tmp/opb-data/ds --out /tmp/indra-eval/gray
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DPI = 200.0
MAX_PAGE_PIXELS = 100e6


def render(args):
    src, dst = args
    import cv2
    import numpy as np

    if dst.exists():
        return None
    if not src.exists():
        return f"{src.name}: missing"
    try:
        with open(src, "rb") as f:
            head = f.read(1024)
        if b"%PDF-" in head:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(str(src))
            try:
                page = doc[0]
                w, h = page.get_size()
                k = min(DPI / 72.0, (MAX_PAGE_PIXELS / max(w * h, 1.0)) ** 0.5)
                arr = page.render(scale=k, grayscale=True).to_numpy()
                gray = np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)
            finally:
                doc.close()
        else:
            gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return f"{src.name}: unreadable"
        tmp = dst.with_suffix(".tmp.png")
        cv2.imwrite(str(tmp), gray, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        tmp.rename(dst)
    except Exception as e:  # noqa: BLE001
        return f"{src.name}: {type(e).__name__}: {e}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import pandas as pd

    m = pd.read_parquet(a.ds / "manifest.parquet")
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(a.ds / r.doc_path, a.out / f"{r.page_id}.png") for r in m.itertuples()]
    with ProcessPoolExecutor(a.workers) as ex:
        errs = [e for e in ex.map(render, jobs, chunksize=8) if e]
    print(f"{len(jobs)} pages, {len(errs)} errors")
    for e in errs[:20]:
        print(" ", e)


if __name__ == "__main__":
    main()
