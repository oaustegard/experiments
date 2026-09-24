#!/usr/bin/env python3
"""Build HotpotQA long-document fixtures rendered as compressed page sheets.

Mirrors LensVLM's (arXiv 2605.07019) eval construction: each sample's 10
paragraphs (2 gold + 8 HotpotQA distractors) are padded with paragraphs from
other samples to ~8k nominal tokens, placed before/after the original block
so evidence adjacency is preserved. The text is paginated at a fixed amount per
page, so the page count is the same at every compression rate. Compression
shrinks each page image, as in the paper.

Each document becomes ONE contact-sheet PNG (all pages in a labelled grid) so a
subagent reads it with one Read call. Sheet tokens use Claude's vision formula,
ceil(W/28) * ceil(H/28), and the page scale is searched so the whole sheet,
labels and gutters included, spends total_text_tokens / ratio image tokens.
"Nominal" text tokens are chars/4; Claude's current tokenizer produces more
tokens than that, so true compression is somewhat higher than the label.

Outputs (under data/):
  manifest.json          samples: id, question, answers, n_pages, gt_pages
  pages.bin              zlib+base64 page texts (so a casual cat reveals nothing)
  sheets/<ratio>/<id>.png
  sheet_stats.json       per ratio: sheet px, image tokens, effective ratio
"""
import base64
import json
import math
import random
import re
import sys
import zlib
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SEED = 20260924
N_SAMPLES = 20
TARGET_CHARS = 32000          # ~8k nominal tokens
BASE_FONT_PX = 14
BASE_W = 560                  # base page width, px
MARGIN = 8
LINE_H = 17
LINES_PER_PAGE = 28
RATIOS = [5, 10, 15]
PATCH = 28
LABEL_PX = 13
GAP = 3


def img_tokens(w, h):
    return math.ceil(w / PATCH) * math.ceil(h / PATCH)


def paragraphs(row):
    out = []
    for title, sents in zip(row["context"]["title"], row["context"]["sentences"]):
        text = "".join(sents).strip()
        if text:
            out.append((title, sents))
    return out


def para_text(title, sents):
    return f"[{title}] " + "".join(sents).strip()


def wrap(text, font, width):
    """Greedy word wrap of one paragraph into lines no wider than width px."""
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        trial = w if not cur else cur + " " + w
        if font.getlength(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def select_samples(rows, rng):
    pool = [r for r in rows
            if r["level"] == "hard"
            and r["answer"].strip().lower() not in ("yes", "no")
            and len(r["answer"]) <= 60]
    rng.shuffle(pool)
    return pool[:N_SAMPLES], pool[N_SAMPLES:N_SAMPLES + 600]


def build_doc(row, filler_rows, rng):
    own = [para_text(t, s) for t, s in paragraphs(row)]
    own_titles = set(row["context"]["title"])
    filler = []
    for fr in filler_rows:
        for t, s in paragraphs(fr):
            if t not in own_titles:
                filler.append(para_text(t, s))
    rng.shuffle(filler)
    need = TARGET_CHARS - sum(len(p) + 1 for p in own)
    pad = []
    for p in filler:
        if need <= 0:
            break
        pad.append(p)
        need -= len(p) + 1
    k = rng.randint(0, len(pad))
    return pad[:k] + own + pad[k:]


def paginate(paras, font):
    """Return list of pages, each a list of lines; and doc text with page map."""
    lines = []
    for p in paras:
        lines.extend(wrap(p, font, BASE_W - 2 * MARGIN))
    return [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]


def gt_pages(row, pages):
    """Pages containing any supporting-fact sentence (matched on a normalised
    32-char probe, since wrapping splits sentences across lines/pages)."""
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    page_txt = [norm(" ".join(pg)) for pg in pages]
    title_to_sents = dict(zip(row["context"]["title"], row["context"]["sentences"]))
    hits = set()
    for title, sid in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]):
        sents = title_to_sents.get(title, [])
        if sid >= len(sents):
            continue
        sent = norm(sents[sid])
        if not sent:
            continue
        probes = [sent[:32], sent[-32:]] if len(sent) > 40 else [sent]
        for i, t in enumerate(page_txt):
            if any(p in t for p in probes):
                hits.add(i + 1)
    return sorted(hits)


def render_page(lines, font):
    h = 2 * MARGIN + LINES_PER_PAGE * LINE_H
    im = Image.new("L", (BASE_W, h), 255)
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.text((MARGIN, MARGIN + i * LINE_H), ln, fill=0, font=font)
    return im


def grid_dims(n):
    cols = math.ceil(math.sqrt(n * 1.1))
    return cols, math.ceil(n / cols)


def compose_sheet(page_imgs, scale, label_font):
    pw = max(1, round(BASE_W * scale))
    ph = max(1, round(page_imgs[0].height * scale))
    cols, rows = grid_dims(len(page_imgs))
    cell_w, cell_h = pw + 2 + GAP, LABEL_PX + 2 + ph + 2 + GAP
    W, H = cols * cell_w + GAP, rows * cell_h + GAP
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(page_imgs):
        c, r = i % cols, i // cols
        x, y = GAP + c * cell_w, GAP + r * cell_h
        d.text((x, y), f"P{i + 1}", fill=0, font=label_font)
        y += LABEL_PX + 2
        d.rectangle([x, y, x + pw + 1, y + ph + 1], outline=150)
        sheet.paste(im.resize((pw, ph), Image.LANCZOS), (x + 1, y + 1))
    return sheet


def fit_scale(page_imgs, target_tokens, label_font):
    lo, hi = 0.05, 1.5
    for _ in range(40):
        mid = (lo + hi) / 2
        s = compose_sheet(page_imgs, mid, label_font)
        if img_tokens(*s.size) > target_tokens:
            hi = mid
        else:
            lo = mid
    return lo


def main():
    rng = random.Random(SEED)
    rows = pq.read_table(DATA / "hotpot_val.parquet").to_pylist()
    samples, filler_rows = select_samples(rows, rng)
    font = ImageFont.truetype(FONT, BASE_FONT_PX)
    label_font = ImageFont.truetype(FONT_BOLD, LABEL_PX)
    manifest, texts, stats = [], {}, {str(r): [] for r in RATIOS}
    for n, row in enumerate(samples):
        sid = f"h{n:02d}"
        paras = build_doc(row, rng.sample(filler_rows, 40), rng)
        pages = paginate(paras, font)
        gts = gt_pages(row, pages)
        if not gts:
            sys.exit(f"{sid}: no gt page found for {row['id']}")
        page_texts = ["\n".join(pg) for pg in pages]
        chars = sum(len(t) for t in page_texts)
        text_tokens = chars / 4
        imgs = [render_page(pg, font) for pg in pages]
        for r in RATIOS:
            scale = fit_scale(imgs, text_tokens / r, label_font)
            sheet = compose_sheet(imgs, scale, label_font)
            out = DATA / "sheets" / str(r) / f"{sid}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            sheet.save(out, optimize=True)
            tok = img_tokens(*sheet.size)
            stats[str(r)].append({"id": sid, "px": sheet.size, "img_tokens": tok,
                                  "text_tokens": round(text_tokens),
                                  "ratio": round(text_tokens / tok, 2),
                                  "page_px": [round(BASE_W * scale), round(imgs[0].height * scale)]})
        texts[sid] = page_texts
        manifest.append({"id": sid, "hotpot_id": row["id"], "question": row["question"],
                         "answer": row["answer"], "type": row["type"],
                         "n_pages": len(pages), "gt_pages": gts,
                         "text_tokens": round(text_tokens)})
        print(sid, len(pages), "pages", gts, row["answer"][:40])
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=1))
    blob = base64.b64encode(zlib.compress(json.dumps(texts).encode(), 9))
    (DATA / "pages.bin").write_bytes(blob)
    summ = {}
    for r, lst in stats.items():
        summ[r] = {"mean_ratio": round(sum(x["ratio"] for x in lst) / len(lst), 2),
                   "mean_img_tokens": round(sum(x["img_tokens"] for x in lst) / len(lst)),
                   "mean_page_px": [round(sum(x["page_px"][i] for x in lst) / len(lst)) for i in (0, 1)],
                   "per_doc": lst}
        print(r, "x:", {k: v for k, v in summ[r].items() if k != "per_doc"})
    (DATA / "sheet_stats.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
