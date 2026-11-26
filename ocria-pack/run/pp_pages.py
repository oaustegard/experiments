"""PP-OCRv6 tiny (fp32 ONNX, onnxruntime) on the same page images as the OCRia arms, one JSON per page in the
shape run/grade.py reads.

The pipeline is ppocr-pack's reference (ppocr-pack/run/ref_det.py and ref_pipeline.py, which follow PaddleOCR
line by line): the detector at the page's own resolution (long side capped at 4000, PaddleX's default
limit), DBPostProcess with the model's inference.yml settings, crops in PaddleOCR's sorted order, the
recognizer at height 48 with greedy CTC. PP-OCR writes plain lines, so the page's markdown is its lines
joined by newlines and it has no layout blocks.

    python3 run/pp_pages.py --gray GRAYDIR --list PAGES.txt --models ../ppocr-pack/models --out OUT [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PP_RUN = Path(__file__).resolve().parents[2] / "ppocr-pack" / "run"
_S = {}


def _init(models: str, tier: str = "tiny"):
    sys.path.insert(0, str(PP_RUN))
    import onnxruntime as ort
    import yaml

    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    m = Path(models)
    det_cfg = yaml.safe_load((m / f"{tier}_det/inference.yml").read_text())
    rec_cfg = yaml.safe_load((m / f"{tier}_rec/inference.yml").read_text())
    norm = next(t["NormalizeImage"] for t in det_cfg["PreProcess"]["transform_ops"] if "NormalizeImage" in t)
    chars = rec_cfg["PostProcess"]["character_dict"]
    _S.update(det=ort.InferenceSession(str(m / f"{tier}_det/inference.onnx"), o, providers=["CPUExecutionProvider"]),
              rec=ort.InferenceSession(str(m / f"{tier}_rec/inference.onnx"), o, providers=["CPUExecutionProvider"]),
              post=det_cfg["PostProcess"] | {"min_size": 3}, mean=norm["mean"], std=norm["std"],
              chars=["<blank>"] + list(chars) + [" "])


def _page(job):
    pid, gray_path, out_path = job
    import cv2
    import ref_det
    import ref_pipeline as RP

    t0 = time.perf_counter()
    rec = {"page_id": pid}
    try:
        g = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        rgb = np.ascontiguousarray(np.repeat(g[:, :, None], 3, axis=2))
        h0, w0 = g.shape
        x, _, h1, w1 = ref_det.preprocess(rgb, min(max(h0, w0), 4000), _S["mean"], _S["std"])
        prob = _S["det"].run(None, {_S["det"].get_inputs()[0].name: x[None]})[0][0, 0]
        p = _S["post"]
        dbp = RP.DBPostProcess(p["thresh"], p["box_thresh"], p["max_candidates"], p["unclip_ratio"], p["min_size"])
        raw_map, raw_scores = dbp.boxes_from_bitmap(prob, prob > p["thresh"])
        boxes, _, _ = RP.filter_tag_det_res(RP.to_page(raw_map, w1, h1, w0, h0), raw_scores, (h0, w0))
        order = RP.sorted_boxes(boxes)
        lines = []
        for i in order:
            crop = RP.get_rotate_crop_image(rgb, boxes[i])
            xr, _ = RP.resize_norm_img(crop[:, :, ::-1])
            probs = _S["rec"].run(None, {_S["rec"].get_inputs()[0].name: xr[None]})[0][0]
            ids = probs.argmax(-1)
            keep = [k for j, k in enumerate(ids) if k != 0 and (j == 0 or k != ids[j - 1])]
            text = "".join(_S["chars"][k] for k in keep if k < len(_S["chars"])).strip()
            if text:
                lines.append(text)
        rec |= {"markdown": "\n".join(lines), "width": w0, "height": h0, "blocks": [], "items": len(lines)}
    except Exception as e:  # noqa: BLE001 - a failed page fails its tests
        import traceback
        rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
    rec["ms"] = round(1000 * (time.perf_counter() - t0))
    out_path.write_text(json.dumps(rec, ensure_ascii=False))
    return "error" not in rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gray", type=Path, required=True)
    ap.add_argument("--list", type=Path, required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tier", default="tiny", choices=["tiny", "small"])
    a = ap.parse_args()
    (a.out / "pages").mkdir(parents=True, exist_ok=True)
    jobs = [(pid, a.gray / f"{pid}.png", a.out / "pages" / f"{pid}.json") for pid in a.list.read_text().split()]
    jobs = [j for j in jobs if not j[2].exists()]
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.models, a.tier)) as ex:
        ok = sum(ex.map(_page, jobs, chunksize=2))
    print(f"{len(jobs)} pages, {ok} ok")


if __name__ == "__main__":
    main()
