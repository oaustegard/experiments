"""Table check: rendered pages -> lines with word boxes -> tables as HTML, scored against the truth.

Pages (data/table_pages/*.png, rendered here with PIL from the truth below):
  table_simple     4 rows x 5 columns, header row, all cells short
  table_multi      5 rows x 4 columns, multi-word cells, right-aligned numeric columns
  paragraph_table  paragraph, 3 x 3 table, paragraph
  two_columns      two text columns, no table: must NOT be detected as a table

Per page the pipeline runs as server/pipeline.read_page does, step by step:
  PP-OCRv6 tiny detector (server.pipeline._detect, onnxruntime) -> tessgpu.layout (lines, order)
  -> line crops padded by PAD -> TorchTess exact mode (models/best/eng.traineddata)
  -> tessgpu.beam.decode_with_x (ids + character start timesteps)
  -> tessgpu.words.line_words (word boxes) -> tessgpu.tables.build (tables, text lines).

Scores:
  cell accuracy  share of the truth cells whose text matches after whitespace collapse, compared
                 position by position (row r, column c) with the detected table matched to the
                 truth table. A truth table that is not found scores all its cells as wrong.
  detection      found (a detected table overlaps the truth table by IoU >= 0.5), missed (a truth
                 table with no match), false (a detected table with no truth table).

Writes results/tables_check.json and data/table_pages/*.png.
Exit status is 0 when every page runs; the numbers are reported, not gated.

Usage: python3 run/check_tables.py
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from server import pipeline as P  # noqa: E402
from server.torch_engine import TorchTess  # noqa: E402
from tessgpu import beam, layout as L, ref, tables, words  # noqa: E402

PAGES_DIR = os.path.join(ROOT, "data", "table_pages")
RESULTS = os.path.join(ROOT, "results", "tables_check.json")
SERIF = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"
SANS = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
IOU_BAR = 0.5

BODY = ("The detector finds text as blobs of probability and the layout step turns those blobs "
        "into lines that a reader would follow. Each line is a rotated rectangle, and the "
        "rectangle is then widened so that the glyphs are fully covered.")
BODY2 = ("A paragraph break is a horizontal gap larger than a single line, so the lines of one "
         "paragraph stay in one block and the blocks follow each other in reading order.")


# ---------------------------------------------------------------- rendering

def _wrap(draw, text, font, width):
    out, cur = [], ""
    for w in text.split():
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out


def _para(d, x, y, text, font, width, leading):
    for ln in _wrap(d, text, font, width):
        d.text((x, y), ln, font=font, fill=(0, 0, 0))
        y += leading
    return y


def _table(d, top, cols, rows, font, bold, row_h):
    """cols: [(left, right, 'l'|'r')]; rows: list of lists. Returns (truth box, rows)."""
    boxes = []
    for ri, row in enumerate(rows):
        f = bold if ri == 0 else font
        y = top + ri * row_h
        for (left, right, al), text in zip(cols, row):
            tw = d.textlength(text, font=f)
            x = left if al == "l" else right - tw
            d.text((x, y), text, font=f, fill=(0, 0, 0))
            boxes.append(d.textbbox((x, y), text, font=f))
    box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
           max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return [int(v) for v in box], rows


def page_table_simple(path):
    img = Image.new("RGB", (1240, 700), "white")
    d = ImageDraw.Draw(img)
    font, bold = ImageFont.truetype(SANS, 25), ImageFont.truetype(SANS_BOLD, 25)
    cols = [(110, 110, "l"), (330, 330, "l"), (500, 500, "l"), (670, 670, "l"), (840, 840, "l")]
    rows = [["Region", "Q1", "Q2", "Q3", "Q4"],
            ["North", "120", "135", "150", "162"],
            ["South", "98", "104", "111", "119"],
            ["East", "210", "198", "205", "220"]]
    box, rows = _table(d, 150, cols, rows, font, bold, 60)
    img.save(path)
    return [{"box": box, "rows": rows}]


def page_table_multi(path):
    img = Image.new("RGB", (1240, 800), "white")
    d = ImageDraw.Draw(img)
    font, bold = ImageFont.truetype(SANS, 25), ImageFont.truetype(SANS_BOLD, 25)
    cols = [(110, 330, "l"), (430, 700, "l"), (780, 930, "r"), (980, 1130, "r")]
    rows = [["Item", "Net Revenue Q3", "Units sold", "Share"],
            ["Widgets", "3 480 k", "1 240", "34.5 %"],
            ["Gadget Pro Max", "2 115 k", "912", "21.0 %"],
            ["Sprocket Set", "4 020 k", "1 587", "39.2 %"],
            ["Total", "9 615 k", "3 739", "100.0 %"]]
    box, rows = _table(d, 150, cols, rows, font, bold, 62)
    img.save(path)
    return [{"box": box, "rows": rows}]


def page_paragraph_table(path):
    img = Image.new("RGB", (1240, 900), "white")
    d = ImageDraw.Draw(img)
    font, bold = ImageFont.truetype(SERIF, 25), ImageFont.truetype(SANS_BOLD, 25)
    y = _para(d, 110, 120, BODY, font, 1020, 36) + 40
    cols = [(110, 110, "l"), (420, 420, "l"), (700, 700, "l")]
    rows = [["Sensor", "Reading", "Status"],
            ["Temp A", "21.4 C", "OK"],
            ["Temp B", "22.1 C", "OK"],
            ["Pressure", "1.02 bar", "Check"]]
    box, rows = _table(d, y, cols, rows, font, bold, 60)
    y = box[3] + 50
    _para(d, 110, y, BODY2, font, 1020, 36)
    img.save(path)
    return [{"box": box, "rows": rows}]


def page_two_columns(path):
    img = Image.new("RGB", (1240, 1000), "white")
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(SERIF, 25)
    for x in (110, 690):
        y = 130
        for para in ([BODY, BODY2] if x == 110 else [BODY2, BODY]):
            y = _para(d, x, y, para, font, 540, 34) + 26
    img.save(path)
    return []


PAGES = [
    ("table_simple", page_table_simple),
    ("table_multi", page_table_multi),
    ("paragraph_table", page_paragraph_table),
    ("two_columns", page_two_columns),
]


# ---------------------------------------------------------------- pipeline pieces

def crops_of(gray, boxes):
    """Padded, inverted line crops as server.pipeline._recognise makes them, with their boxes."""
    H, W = gray.shape
    crops, crop_boxes, idx = [], [], []
    for i, ln in enumerate(boxes):
        x0, y0, x1, y1 = ln["box"]
        x0, y0 = max(0, x0 - P.PAD), max(0, y0 - P.PAD)
        x1, y1 = min(W, x1 + P.PAD), min(H, y1 + P.PAD)
        if x1 <= x0 or y1 <= y0:
            continue
        c = np.ascontiguousarray(gray[y0:y1, x0:x1])
        if ref.needs_invert(c):
            c = np.ascontiguousarray(255 - c)
        crops.append(c)
        crop_boxes.append([x0, y0, x1, y1])
        idx.append(i)
    return crops, crop_boxes, idx


def read_lines(gray, eng, sess):
    """Page -> reading-order lines with words: [{"box", "words", "text"}]."""
    prob, sx, sy, w, h = P._detect(sess, gray)
    res = L.layout(prob, sx, sy, w, h)
    boxes = [res["lines"][i] for i in res["order"]]
    crops, crop_boxes, idx = crops_of(gray, boxes)
    probs = eng.recognize(crops) if crops else []
    out = []
    unichars = eng.model.unicharset
    by_idx = {}
    for i, cb, p in zip(idx, crop_boxes, probs):
        ids, xs = beam.decode_with_x(p, eng.model)
        ws = words.line_words(ids, xs, unichars, boxes[i]["box"], cb, eng.H)
        by_idx[i] = (ws, "".join(unichars[u] for u in ids))
    for i, ln in enumerate(boxes):
        ws, raw = by_idx.get(i, ([], ""))
        out.append({"box": list(ln["box"]), "block": ln["block"], "words": ws,
                    "text": " ".join(raw.split())})
    return out


# ---------------------------------------------------------------- scoring

def norm(s):
    return " ".join(str(s).split())


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def score(truth, blocks):
    det = [b for b in blocks if b["kind"] == "table"]
    truth_cells = sum(len(r) for t in truth for r in t["rows"])
    correct = 0
    used = set()
    found = 0
    cell_log = []
    for t in truth:
        best, best_iou = None, 0.0
        for k, d in enumerate(det):
            if k in used:
                continue
            v = iou(t["box"], d["box"])
            if v > best_iou:
                best, best_iou = k, v
        if best is None or best_iou < IOU_BAR:
            for r, row in enumerate(t["rows"]):
                for c, cell in enumerate(row):
                    cell_log.append({"r": r, "c": c, "truth": cell, "pred": None})
            continue
        used.add(best)
        found += 1
        pred = det[best]["rows"]
        for r, row in enumerate(t["rows"]):
            for c, cell in enumerate(row):
                p = pred[r][c] if r < len(pred) and c < len(pred[r]) else None
                ok = p is not None and norm(p) == norm(cell)
                correct += ok
                if not ok:
                    cell_log.append({"r": r, "c": c, "truth": cell, "pred": p})
    false = len(det) - len(used)
    return {
        "truth_tables": len(truth),
        "detected_tables": len(det),
        "found": found,
        "missed": len(truth) - found,
        "false": false,
        "truth_cells": truth_cells,
        "correct_cells": correct,
        "cell_accuracy": round(correct / truth_cells, 4) if truth_cells else None,
        "wrong_cells_first10": cell_log[:10],
        "detected_grids": [d["rows"] for d in det],
    }


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    os.makedirs(PAGES_DIR, exist_ok=True)
    eng = TorchTess(P.DEFAULT_MODEL, device="cpu", mode="exact")
    sess = P._det_session("tiny")
    pages = {}
    tot = {"truth_tables": 0, "found": 0, "missed": 0, "false": 0,
           "truth_cells": 0, "correct_cells": 0}
    for name, render in PAGES:
        path = os.path.join(PAGES_DIR, name + ".png")
        truth = render(path)
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        t1 = time.time()
        lines = read_lines(gray, eng, sess)
        blocks = tables.build(lines)
        s = score(truth, blocks)
        s["wall_seconds"] = round(time.time() - t1, 2)
        s["lines"] = [{"box": ln["box"], "block": ln["block"], "text": ln["text"], "words": ln["words"]}
                  for ln in lines]
        s["html"] = tables.to_html(blocks)
        pages[name] = s
        for k in tot:
            tot[k] += s[k]
        print(f"{name:16s} lines={len(lines):3d} truth_tables={s['truth_tables']} "
              f"found={s['found']} missed={s['missed']} false={s['false']} "
              f"cells={s['correct_cells']}/{s['truth_cells']} "
              f"accuracy={s['cell_accuracy']}", flush=True)
    tot["cell_accuracy"] = round(tot["correct_cells"] / tot["truth_cells"], 4)
    out = {
        "what": "tessgpu table pipeline on rendered pages: cell accuracy and table detection",
        "model": os.path.relpath(P.DEFAULT_MODEL, ROOT),
        "detector": "models/det/tiny",
        "params": {"GAP_FRAC": tables.GAP_FRAC, "ROW_GAP_FRAC": tables.ROW_GAP_FRAC,
                   "MIN_ROWS": tables.MIN_ROWS, "MIN_COLS": tables.MIN_COLS,
                   "IOU_BAR": IOU_BAR, "PAD": P.PAD},
        "totals": tot,
        "pages": pages,
        "wall_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"totals={json.dumps(tot)} wall={out['wall_seconds']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
