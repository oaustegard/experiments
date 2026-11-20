#!/usr/bin/env python3
"""Build real-scan printed-text line crops with ground truth -> data/real/.

Sources (Hugging Face Hub, see notes/data_real.md for licences):
  * FUNSD      nielsr/funsd        English forms, word boxes + entity tags
  * SROIE      arvindrajan92/sroie_document_understanding
               (mirror of ICDAR-2019 SROIE receipts; OCR text boxes per line)

FUNSD has only word boxes, so words are merged into lines: first by the set's
own entity grouping (contiguous B-/I- spans of one class, and runs of O words),
then by vertical overlap inside each group. SROIE already gives one text box
per line, so its boxes are used directly.

Filters, applied to every candidate line:
  * fewer than 3 characters after strip -> skipped
  * vertical (height > width) or rotated (top edge slope > MAX_SLOPE_DEG) -> skipped
  * box height under 8 px -> skipped
Crops get PAD px of padding, are converted to 8-bit greyscale PNG, and at most
CAP lines are kept per set.

Run:  python3 run/make_real.py            (from tesseract-gpu/, or any cwd)
Does not run tesseract.
"""
import io
import json
import math
import re
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "real"
CAP = 400
PAD = 2
MIN_CHARS = 3
MIN_H = 8
MAX_SLOPE_DEG = 3.0


def load_rows(repo, files, revision=None):
    """Download parquet shards from the Hub and return their rows as dicts."""
    rows = []
    for f in files:
        kw = {"revision": revision} if revision else {}
        path = hf_hub_download(repo, f, repo_type="dataset", **kw)
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def image_of(rec):
    img = rec["image"]
    data = img["bytes"] if isinstance(img, dict) else img
    return Image.open(io.BytesIO(data))


def group_words(words, boxes, tags):
    """Split a reading-order word list into the set's entity groups.

    A new group starts at every B- tag, on a change of class, and at the start
    of each run of O (other) words."""
    groups, prev = [], "<start>"
    for w, b, t in zip(words, boxes, tags):
        cls = "O" if t == "O" else t[2:]
        if not groups or t.startswith("B-") or cls != prev:
            groups.append([])
        groups[-1].append((w, b))
        prev = cls
    return groups


def cluster_lines(items):
    """Merge (text, box) items of one group into visual lines.

    Items are sorted by centre height; an item joins the current line when its
    vertical extent overlaps the line's by more than half the smaller height."""
    items = sorted(items, key=lambda it: (it[1][1] + it[1][3]) / 2)
    lines = []  # each: {"items": [...], "y0", "y1"}
    for text, (x0, y0, x1, y1) in items:
        h = max(y1 - y0, 1)
        if lines:
            cur = lines[-1]
            overlap = min(y1, cur["y1"]) - max(y0, cur["y0"])
            if overlap > 0.5 * min(h, cur["y1"] - cur["y0"]):
                cur["items"].append((text, (x0, y0, x1, y1)))
                cur["y0"], cur["y1"] = min(cur["y0"], y0), max(cur["y1"], y1)
                continue
        lines.append({"items": [(text, (x0, y0, x1, y1))], "y0": y0, "y1": y1})
    out = []
    for ln in lines:
        its = sorted(ln["items"], key=lambda it: it[1][0])
        text = " ".join(t for t, _ in its)
        box = (min(b[0] for _, b in its), min(b[1] for _, b in its),
               max(b[2] for _, b in its), max(b[3] for _, b in its))
        out.append((text, box, None))
    return out


def funsd_lines(rec):
    """Yield (text, box, slope_deg) for each merged line in a FUNSD form.

    nielsr/funsd stores LayoutLM-style boxes scaled to 0-1000 on both axes (x reaches 931 on a
    754-px-wide page), so they are mapped back to pixels before anything else uses them."""
    w, h = image_of(rec).size
    boxes = [(x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000) for x0, y0, x1, y1 in rec["bboxes"]]
    groups = group_words(rec["words"], boxes, rec["ner_tags_names"])
    for g in groups:
        for text, box, _ in cluster_lines(g):
            yield text, box, 0.0


def sroie_lines(rec):
    """Yield (text, box, slope_deg) from SROIE's per-line OCR boxes (4-point polygons)."""
    for entry in rec["ocr"]:
        pts = entry["box"]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = (min(xs), min(ys), max(xs), max(ys))
        # slope of the top edge, p0 -> p1
        dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        slope = math.degrees(math.atan2(dy, dx))
        yield entry["text"], box, slope


def keep(text, box, slope):
    """Return a reason string if the line must be skipped, else None."""
    if len(text.strip()) < MIN_CHARS:
        return "short"
    x0, y0, x1, y1 = box
    if y1 - y0 < MIN_H:
        return "short_box"
    if (y1 - y0) > (x1 - x0):
        return "vertical"
    if slope is not None and abs(((slope + 90) % 180) - 90) > MAX_SLOPE_DEG:
        return "rotated"
    return None


def emit(set_name, lang, recs, line_fn, doc_key, labels):
    """Crop, filter and write up to CAP lines for one set; append label rows."""
    set_dir = OUT / set_name
    if set_dir.exists():
        shutil.rmtree(set_dir)
    set_dir.mkdir(parents=True)
    n = 0
    skipped = {}
    for d, rec in enumerate(recs):
        if n >= CAP:
            break
        img = None
        doc = re.sub(r"[^A-Za-z0-9_-]+", "_", str(rec[doc_key]) if doc_key else f"{d:04d}")
        for k, (text, box, slope) in enumerate(line_fn(rec)):
            if n >= CAP:
                break
            reason = keep(text, box, slope)
            if reason:
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            if img is None:
                img = image_of(rec)
            x0, y0, x1, y1 = box
            crop = (max(0, math.floor(x0) - PAD), max(0, math.floor(y0) - PAD),
                    min(img.width, math.ceil(x1) + PAD), min(img.height, math.ceil(y1) + PAD))
            if crop[2] <= crop[0] or crop[3] <= crop[1]:
                skipped["empty_crop"] = skipped.get("empty_crop", 0) + 1
                continue
            line_id = f"{set_name}-{doc}-{k:03d}"
            path = set_dir / f"{line_id}.png"
            img.crop(crop).convert("L").save(path, format="PNG")
            labels.append({
                "id": line_id,
                "file": path.relative_to(ROOT).as_posix(),
                "lang": lang,
                "cond": "real",
                "set": set_name,
                "text": text.strip(),
            })
            n += 1
    print(f"{set_name}: wrote {n} lines; skipped {skipped}")


def main():
    labels = []

    # FUNSD: English forms. Entity tag names come from the parquet schema.
    funsd = []
    funsd_files = {}
    for split in ["train", "test"]:
        funsd_files[split] = hf_hub_download(
            "nielsr/funsd", f"default/{split}/0000.parquet",
            repo_type="dataset", revision="refs/convert/parquet")
    tag_names = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION",
                 "B-ANSWER", "I-ANSWER"]
    for path in funsd_files.values():
        for rec in pq.read_table(path).to_pylist():
            rec["ner_tags_names"] = [tag_names[i] for i in rec["ner_tags"]]
            funsd.append(rec)
    emit("funsd", "eng", funsd, funsd_lines, "id", labels)
    del funsd

    # SROIE: English receipts, one OCR text box per line.
    sroie = load_rows("arvindrajan92/sroie_document_understanding",
                      ["data/train-00000-of-00001-66201d4afd6b73ca.parquet"])
    emit("sroie", "eng", sroie, sroie_lines, None, labels)
    del sroie

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "labels.jsonl", "w", encoding="utf-8") as fh:
        for row in labels:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"labels.jsonl: {len(labels)} rows")


if __name__ == "__main__":
    main()
