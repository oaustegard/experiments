#!/usr/bin/env python3
"""Reference for web/pipeline.js: everything between the detector's probability map and the recognizer.

    python3 -I run/ref_pipeline.py                      # both models, every fixture page
    python3 -I run/ref_pipeline.py tiny funsd_0

Follows PaddleOCR line by line, with OpenCV and pyclipper doing the geometry:

  DBPostProcess (ppocr/postprocess/db_postprocess.py; PaddleX's text_detection is the same code)
      binarize at `thresh`; cv2.findContours(RETR_LIST, CHAIN_APPROX_SIMPLE); get_mini_boxes
      (cv2.minAreaRect); drop sside < 3; box_score_fast (cv2.fillPoly mask mean of the probability
      map); drop score < box_thresh; unclip (shapely area and length, pyclipper JT_ROUND offset of
      area * unclip_ratio / length); get_mini_boxes again, drop sside < 5; map to page pixels
  filter_tag_det_res (tools/infer/predict_system.py): order_points_clockwise, clip, drop <= 3 px
  sorted_boxes: sort by (y, x) of the top-left corner, then the 10 px left-to-right bubble pass
  get_rotate_crop_image: getPerspectiveTransform + warpPerspective(INTER_CUBIC, BORDER_REPLICATE),
      rot90 when h / w >= 1.5
  resize_norm_img (tools/infer/predict_rec.py), shape (3, 48, 320): resize to 48 high keeping the
      aspect ratio (bilinear), BGR, (x / 255 - 0.5) / 0.5, zero pad right to max(320, floor(48 * ratio))

One deliberate difference: this port pads the image to multiples of 32 instead of stretching it, so a
map pixel (x, y) is page pixel (x * src_w / w1, y * src_h / h1) with (h1, w1) the resized, unpadded
size; PaddleOCR's formula divides by the map size and multiplies by the page size.

Input per page and model: web/det/test/data/fixtures/det_<page>/<model>.ref.f32 (the probability map,
written by run/ref_det.py). Output next to it: <model>.pipe.json (boxes, scores, order, crop sizes) and
<model>.crops.bin (the crops, RGB bytes), read by web/det/test/check_pipeline.mjs. results/
pipeline_parity.json gets the JS-vs-Python numbers from that script plus, here, how many boxes the
packed detector finds against the fp32 one.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "web/det/models"
FIX_DIR = ROOT / "web/det/test/data/fixtures"
RESULTS = ROOT / "results/pipeline_parity.json"
REC_SHAPE = (3, 48, 320)


# ------------------------------------------------------------------ DBPostProcess


class DBPostProcess:
    """PaddleOCR's DBPostProcess, score_mode "fast", no dilation, quad boxes."""

    def __init__(self, thresh, box_thresh, max_candidates, unclip_ratio, min_size=3):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size

    def boxes_from_bitmap(self, pred, bitmap):
        """Boxes in map pixels (float, 4 x 2) with their scores, before the map-to-page step."""
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            box = self.unclip(points, self.unclip_ratio)
            if len(box) > 1:
                continue
            box = np.array(box).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            boxes.append(np.array(box))
            scores.append(float(score))
        return boxes, scores

    @staticmethod
    def unclip(box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return offset.Execute(distance)

    @staticmethod
    def get_mini_boxes(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        if points[1][1] > points[0][1]:
            index_1, index_4 = 0, 1
        else:
            index_1, index_4 = 1, 0
        if points[3][1] > points[2][1]:
            index_2, index_3 = 2, 3
        else:
            index_2, index_3 = 3, 2
        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    @staticmethod
    def box_score_fast(bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin: ymax + 1, xmin: xmax + 1], mask)[0]


def to_page(boxes, w1, h1, src_w, src_h):
    """Map-pixel boxes to int32 page pixels: round half to even, clip to the page (PaddleOCR's step)."""
    out = []
    for b in boxes:
        b = np.array(b, dtype=np.float64)
        b[:, 0] = np.clip(np.round(b[:, 0] * src_w / w1), 0, src_w)
        b[:, 1] = np.clip(np.round(b[:, 1] * src_h / h1), 0, src_h)
        out.append(b.astype("int32"))
    return out


# ------------------------------------------------------------------ boxes -> ordered quads


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


def filter_tag_det_res(dt_boxes, scores, image_shape):
    """order_points_clockwise, clip, drop tiny boxes. Returns (boxes, scores, index into dt_boxes)."""
    img_height, img_width = image_shape[0:2]
    boxes, kept_scores, src = [], [], []
    for i, box in enumerate(dt_boxes):
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        boxes.append(box)
        kept_scores.append(scores[i])
        src.append(i)
    return boxes, kept_scores, src


def sorted_boxes(dt_boxes):
    """Top to bottom, then left to right within 10 px. Returns the permutation as indices."""
    order = sorted(range(len(dt_boxes)), key=lambda i: (dt_boxes[i][0][1], dt_boxes[i][0][0]))
    _b = list(order)
    for i in range(len(_b) - 1):
        for j in range(i, -1, -1):
            if abs(dt_boxes[_b[j + 1]][0][1] - dt_boxes[_b[j]][0][1]) < 10 and \
                    dt_boxes[_b[j + 1]][0][0] < dt_boxes[_b[j]][0][0]:
                _b[j], _b[j + 1] = _b[j + 1], _b[j]
            else:
                break
    return _b


# ------------------------------------------------------------------ crop and recognizer input


def get_rotate_crop_image(img, points):
    points = np.array(points, dtype="float32")
    w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return np.ascontiguousarray(dst)


def resize_norm_img(img_bgr, max_wh_ratio=None):
    """PaddleOCR's resize_norm_img for one crop. Returns (float32 (3, 48, W), resized_w)."""
    imgC, imgH, imgW = REC_SHAPE
    if max_wh_ratio is None:
        max_wh_ratio = max(imgW / imgH, img_bgr.shape[1] / float(img_bgr.shape[0]))
    imgW = int(imgH * max_wh_ratio)
    h, w = img_bgr.shape[:2]
    ratio = w / float(h)
    resized_w = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))
    resized = cv2.resize(img_bgr, (resized_w, imgH)).astype("float32")
    resized = resized.transpose(2, 0, 1) / 255
    resized -= 0.5
    resized /= 0.5
    padding = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding[:, :, 0:resized_w] = resized
    return padding, resized_w


# ------------------------------------------------------------------ per page


def load_prob(page_dir, model, meta, long_side):
    s = long_side / max(meta["h"], meta["w"])
    h1, w1 = max(1, int(math.floor(meta["h"] * s + 0.5))), max(1, int(math.floor(meta["w"] * s + 0.5)))
    H, W = -(-h1 // 32) * 32, -(-w1 // 32) * 32
    raw = np.fromfile(page_dir / f"{model}.ref.f32", "<f4")
    assert raw.size == H * W, (raw.size, H, W)
    return raw.reshape(H, W), h1, w1


def process(prob, h1, w1, src_w, src_h, post, rgb):
    """Everything after the detector. Returns a dict (JSON-able) and the crops (list of RGB arrays)."""
    dbp = DBPostProcess(post["thresh"], post["box_thresh"], post["max_candidates"], post["unclip_ratio"], post["min_size"])
    raw_map, raw_scores = dbp.boxes_from_bitmap(prob, prob > post["thresh"])
    raw = to_page(raw_map, w1, h1, src_w, src_h)
    boxes, scores, src = filter_tag_det_res(raw, raw_scores, (src_h, src_w))
    order = sorted_boxes(boxes)
    final = [boxes[i] for i in order]
    final_scores = [scores[i] for i in order]
    crops = [get_rotate_crop_image(rgb, b) for b in final]
    rec, recin = [], []
    for k, c in enumerate(crops):
        x, rw = resize_norm_img(c[:, :, ::-1])
        rec.append({"w": int(c.shape[1]), "h": int(c.shape[0]), "resized_w": int(rw), "W": int(x.shape[2])})
        if k < 4:          # the first crops' recognizer input, to check recInput() numerically
            recin.append(x)
    return {
        "recin_floats": int(sum(x.size for x in recin)), "recin_data": np.concatenate([x.ravel() for x in recin]) if recin else np.zeros(0, np.float32),
        "raw_count": len(raw), "raw": [b.tolist() for b in raw], "raw_scores": raw_scores,
        "boxes": [b.astype(int).tolist() for b in final], "scores": final_scores,
        "order_from_filtered": order, "crops": rec,
    }, crops


def match_boxes(a, b):
    """Greedy best-IoU match of axis-aligned bounding boxes of quads a to quads b. -> list of (ia, ib, iou)."""
    def bb(q):
        q = np.array(q)
        return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()

    def iou(p, q):
        iw = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        ih = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        inter = iw * ih
        union = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - inter
        return inter / union if union > 0 else 0.0

    A, B = [bb(q) for q in a], [bb(q) for q in b]
    used, out = set(), []
    for i, p in enumerate(A):
        best, bj = 0.0, -1
        for j, q in enumerate(B):
            if j in used:
                continue
            v = iou(p, q)
            if v > best:
                best, bj = v, j
        if bj >= 0 and best > 0.5:
            used.add(bj)
            out.append((i, bj, best))
    return out


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="*", help="model names (tiny, small) and page names; default all")
    a = ap.parse_args(argv)
    names = [x for x in a.items if x in ("tiny", "small")] or ["tiny", "small"]
    pages = [x for x in a.items if x not in ("tiny", "small")] or sorted(
        d.name[4:] for d in FIX_DIR.glob("det_*") if d.is_dir())
    if not pages:
        raise SystemExit("no fixtures; run run/ref_det.py first")
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results.setdefault("pack_vs_fp32", {})
    results["task"] = "web/pipeline.js against run/ref_pipeline.py (PaddleOCR DBPostProcess, sorted_boxes, crops)"
    for name in names:
        man = json.loads((MODELS_DIR / name / "manifest.json").read_text())
        post = man["postprocess"]
        for page in pages:
            d = FIX_DIR / f"det_{page}"
            meta = json.loads((d / "meta.json").read_text())
            src_w, src_h = meta["w"], meta["h"]
            rgba = np.fromfile(d / "rgba.bin", np.uint8).reshape(src_h, src_w, 4)
            rgb = np.ascontiguousarray(rgba[:, :, :3])
            prob, h1, w1 = load_prob(d, name, meta, meta["long_side"])
            res, crops = process(prob, h1, w1, src_w, src_h, post, rgb)
            res.pop("recin_data").astype("<f4").tofile(d / f"{name}.recin.f32")
            res.update({"page": page, "model": name, "w": src_w, "h": src_h, "H": prob.shape[0], "W": prob.shape[1],
                        "h1": h1, "w1": w1, "post": post, "cv2": cv2.__version__, "pyclipper": pyclipper.__version__})
            (d / f"{name}.pipe.json").write_text(json.dumps(res) + "\n")
            blob = b"".join(c.tobytes() for c in crops)
            (d / f"{name}.crops.bin").write_bytes(blob)
            # the same page through the fp32 detector: how many boxes does packing change?
            ort = np.fromfile(d / f"{name}.ort.f32", "<f4").reshape(prob.shape)
            res32, _ = process(ort, h1, w1, src_w, src_h, post, rgb)
            res32.pop("recin_data")
            m = match_boxes(res["boxes"], res32["boxes"])
            dev = [float(np.abs(np.array(res["boxes"][i], float) - np.array(res32["boxes"][j], float)).max()) for i, j, _ in m]
            rec = {"boxes_pack": len(res["boxes"]), "boxes_fp32": len(res32["boxes"]), "matched": len(m),
                   "median_corner_shift_px": float(np.median(dev)) if dev else None,
                   "max_corner_shift_px": max(dev) if dev else None}
            results["pack_vs_fp32"].setdefault(name, {})[page] = rec
            print(f"{name:5s} {page:16s} raw {res['raw_count']:3d}  boxes {len(res['boxes']):3d}  "
                  f"fp32 {len(res32['boxes']):3d} matched {len(m):3d} max shift {rec['max_corner_shift_px']}", flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    print("->", RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
