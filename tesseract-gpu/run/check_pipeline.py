"""Pipeline check: five pages through both layouts and the tesseract CLI, scored by CER.

Pages (data/pipeline/pages/):
  two_column, heading_table, rotated_1deg   rendered here; the ground truth is the text rendered
  funsd_0, funsd_1                          copied from data/layout_pages/ when present; the ground
                                            truth is FUNSD's word list in reading order inferred
                                            from the word boxes (an approximation of reading order)

Methods per page:
  det   server.pipeline.read_page(layout="det")   PP-OCRv6 lines + GPU recogniser (CPU here)
  tess  server.pipeline.read_page(layout="tess")  Tesseract layout + GPU recogniser (CPU here)
  cli   tesseract page.png - --psm 3 --oem 1 -l eng

CER = Levenshtein distance / reference length, after collapsing whitespace on both sides.
Reported: CER against the ground truth, CER against the tesseract CLI text, and ms per page.

Writes:
  results/pipeline_check.json   per-page, per-method numbers and the environment
  data/pipeline/check/          each method's text, for inspection

Usage: python3 run/check_pipeline.py     (exits 0 when every page and method ran)
"""

import glob
import io
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

from server.pipeline import DEFAULT_MODEL, _cli_env as cli_env, read_page  # noqa: E402

PAGES_DIR = os.path.join(ROOT, 'data', 'pipeline', 'pages')
CHECK_DIR = os.path.join(ROOT, 'data', 'pipeline', 'check')
LAYOUT_PAGES = os.path.join(ROOT, 'data', 'layout_pages')
FUNSD_PARQUET = os.path.join(LAYOUT_PAGES, 'funsd_src', 'data', 'test-00000-of-00001.parquet')
RESULTS = os.path.join(ROOT, 'results', 'pipeline_check.json')

SERIF = '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'
SANS = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
SANS_BOLD = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'

BODY = (
    'The detector finds text as blobs of probability and the layout step turns those blobs into '
    'lines that a reader would follow. Each line is a rotated rectangle, and the rectangle is '
    'then widened by the unclip ratio so that the glyphs are fully covered. Columns are found '
    'from the widest vertical gap that runs through every box on the page, and rows inside a '
    'column are read from the top, one row at a time, from left to right.'
)
BODY2 = (
    'A paragraph break is a horizontal gap larger than a single line, so the lines of one '
    'paragraph stay in one block and the blocks follow each other in reading order. Small gaps '
    'between words on one line are bridged when fragments share a baseline band.'
)
TABLE = [
    ('Region', 'Units', 'Revenue'),
    ('North', '1 240', '3 480 k'),
    ('South', '912', '2 115 k'),
    ('East', '1 587', '4 020 k'),
    ('West', '640', '1 090 k'),
]


# ---------------------------------------------------------------- rendering

def _wrap(draw, text, font, width):
    lines, cur = [], ''
    for w in text.split():
        trial = (cur + ' ' + w).strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _para(draw, x, y, text, font, width, leading):
    for ln in _wrap(draw, text, font, width):
        draw.text((x, y), ln, font=font, fill=(0, 0, 0))
        y += leading
    return y


def render_two_column(path):
    """Two columns of body text under a heading. Reading order: all of the left, then the right."""
    img = Image.new('RGB', (1240, 1754), 'white')
    d = ImageDraw.Draw(img)
    d.text((110, 120), 'Notes on Column Layout', font=ImageFont.truetype(SANS_BOLD, 46), fill=(0, 0, 0))
    body = ImageFont.truetype(SERIF, 25)
    left = [BODY, BODY2, BODY]
    right = [BODY2, BODY, BODY2]
    for x, paras in ((110, left), (690, right)):
        y = 230
        for para in paras:
            y = _para(d, x, y, para, body, 540, 34) + 26
    img.save(path)
    return ' '.join(left + right)


def render_heading_table(path):
    """Heading, a paragraph, then a small table read row by row."""
    img = Image.new('RGB', (1240, 1100), 'white')
    d = ImageDraw.Draw(img)
    d.text((110, 110), 'Quarterly Results', font=ImageFont.truetype(SANS_BOLD, 44), fill=(0, 0, 0))
    body = ImageFont.truetype(SANS, 25)
    bold = ImageFont.truetype(SANS_BOLD, 25)
    y = _para(d, 110, 200, BODY + ' ' + BODY2, body, 1020, 36) + 40
    xs = [110, 560, 860]
    for ri, row in enumerate(TABLE):
        f = bold if ri == 0 else body
        for ci, cell in enumerate(row):
            if ci == 0:
                d.text((xs[0], y), cell, font=f, fill=(0, 0, 0))
            else:
                right = xs[ci] + (230 if ci == 1 else 300)
                d.text((right - d.textlength(cell, font=f), y), cell, font=f, fill=(0, 0, 0))
        if ri == 0:
            d.line([(110, y + 40), (1130, y + 40)], fill=(0, 0, 0), width=2)
        y += 56
    img.save(path)
    cells = ' '.join(' '.join(row) for row in TABLE)
    return 'Quarterly Results ' + BODY + ' ' + BODY2 + ' ' + cells


def render_rotated(path):
    """Three paragraphs, rotated by one degree (counter-clockwise, white fill)."""
    base = Image.new('RGB', (1240, 900), 'white')
    d = ImageDraw.Draw(base)
    body = ImageFont.truetype(SERIF, 27)
    y = 120
    for para in (BODY, BODY2, BODY):
        y = _para(d, 110, y, para, body, 1020, 38) + 30
    base.rotate(1.0, resample=Image.BICUBIC, expand=True, fillcolor='white').save(path)
    return ' '.join((BODY, BODY2, BODY))


# ---------------------------------------------------------------- FUNSD

def reading_order(words, bboxes):
    """Words in reading order: rows by centre height (within half a median box height), then x."""
    if not words:
        return ''
    hs = sorted(b[3] - b[1] for b in bboxes)
    h = hs[len(hs) // 2] or 1
    items = sorted(zip(words, bboxes), key=lambda wb: (wb[1][1] + wb[1][3]) / 2)
    rows, row_yc, row = [], None, []
    for w, b in items:
        yc = (b[1] + b[3]) / 2
        if row and abs(yc - row_yc) > 0.5 * h:
            rows.append(row)
            row = []
        if not row:
            row_yc = yc
        row.append((b[0], w))
    if row:
        rows.append(row)
    return ' '.join(w for r in rows for _, w in sorted(r))


def funsd_pages(out_dir):
    """Copy FUNSD forms 0 and 1 from data/layout_pages/ with their word-list ground truth."""
    if not os.path.exists(FUNSD_PARQUET):
        return {}
    import pyarrow.parquet as pq
    rows = pq.read_table(FUNSD_PARQUET, columns=['id', 'words', 'bboxes']).to_pylist()
    out = {}
    for row in rows[:2]:
        name = 'funsd_%s' % row['id']
        src = os.path.join(LAYOUT_PAGES, name + '.png')
        if not os.path.exists(src):
            continue
        dst = os.path.join(out_dir, name + '.png')
        shutil.copyfile(src, dst)
        out[name] = (dst, reading_order(row['words'], row['bboxes']))
    return out


# ---------------------------------------------------------------- scoring

def norm(s):
    return ' '.join(s.split())


def cer(hyp, ref):
    r = norm(ref)
    if not r:
        raise ValueError('empty reference text')
    return Levenshtein.distance(norm(hyp), r) / len(r)


def tesseract_cli(path):
    t0 = time.perf_counter()
    out = subprocess.run(['tesseract', path, '-', '--psm', '3', '--oem', '1', '-l', 'eng'],
                         capture_output=True, text=True, check=True, env=cli_env()).stdout
    return out, (time.perf_counter() - t0) * 1000


def load_grey(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.uint8)


# ---------------------------------------------------------------- main

def main():
    os.makedirs(PAGES_DIR, exist_ok=True)
    os.makedirs(CHECK_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    pages = {}   # name -> (png path, ground truth, ground-truth source)
    p = os.path.join(PAGES_DIR, 'two_column.png')
    pages['two_column'] = (p, render_two_column(p), 'rendered')
    p = os.path.join(PAGES_DIR, 'heading_table.png')
    pages['heading_table'] = (p, render_heading_table(p), 'rendered')
    p = os.path.join(PAGES_DIR, 'rotated_1deg.png')
    pages['rotated_1deg'] = (p, render_rotated(p), 'rendered')
    for name, (path, gt) in funsd_pages(PAGES_DIR).items():
        pages[name] = (path, gt, 'FUNSD words, reading order from boxes')

    # Warm up both layouts on the first page so the model load is not in the timings.
    first = load_grey(pages['two_column'][0])
    read_page(first, layout='det')
    read_page(first, layout='tess')

    report = {
        'what': 'server.pipeline.read_page on five pages vs ground truth and the tesseract CLI',
        'model': os.path.relpath(DEFAULT_MODEL, ROOT),
        'device': 'cpu', 'mode': 'exact', 'det': 'tiny', 'pad_px': 4,
        'pages': {},
    }
    methods = ('det', 'tess', 'cli')
    rows = []
    for name, (path, gt, gt_src) in pages.items():
        gray = load_grey(path)
        hyp = {}
        ms = {}
        engine = {}
        for m, layout in (('det', 'det'), ('tess', 'tess')):
            res = read_page(gray, layout=layout)
            hyp[m] = res['text']
            ms[m] = res['ms']['total']
            engine[m] = res['layout_engine']
        hyp['cli'], ms['cli'] = tesseract_cli(path)

        cer_gt = {m: cer(hyp[m], gt) for m in methods}
        cer_cli = {m: cer(hyp[m], hyp['cli']) for m in ('det', 'tess')}
        for m in methods:
            with open(os.path.join(CHECK_DIR, '%s.%s.txt' % (name, m)), 'w', encoding='utf-8') as f:
                f.write(hyp[m] + '\n')
        report['pages'][name] = {
            'gt_source': gt_src,
            'gt_chars': len(norm(gt)),
            'layout_engine': engine,
            'cer_vs_gt': {m: round(cer_gt[m], 4) for m in methods},
            'cer_vs_cli': {m: round(cer_cli[m], 4) for m in cer_cli},
            'ms': {m: round(ms[m], 1) for m in methods},
        }
        rows.append((name, cer_gt, cer_cli, ms))

    # Summary table
    head = '%-14s %8s %8s %8s | %9s %9s | %8s %8s %8s'
    print(head % ('page', 'GT det', 'GT tess', 'GT cli', 'cli-det', 'cli-tess',
                  'ms det', 'ms tess', 'ms cli'))
    for name, cg, cc, ms in rows:
        print(head % (name, '%.3f' % cg['det'], '%.3f' % cg['tess'], '%.3f' % cg['cli'],
                      '%.3f' % cc['det'], '%.3f' % cc['tess'],
                      '%.0f' % ms['det'], '%.0f' % ms['tess'], '%.0f' % ms['cli']))
    means = {
        'GT det': statistics.mean(r[1]['det'] for r in rows),
        'GT tess': statistics.mean(r[1]['tess'] for r in rows),
        'GT cli': statistics.mean(r[1]['cli'] for r in rows),
        'cli-det': statistics.mean(r[2]['det'] for r in rows),
        'cli-tess': statistics.mean(r[2]['tess'] for r in rows),
        'ms det': statistics.mean(r[3]['det'] for r in rows),
        'ms tess': statistics.mean(r[3]['tess'] for r in rows),
        'ms cli': statistics.mean(r[3]['cli'] for r in rows),
    }
    print(head % ('mean', '%.3f' % means['GT det'], '%.3f' % means['GT tess'],
                  '%.3f' % means['GT cli'], '%.3f' % means['cli-det'], '%.3f' % means['cli-tess'],
                  '%.0f' % means['ms det'], '%.0f' % means['ms tess'], '%.0f' % means['ms cli']))
    report['summary_mean'] = {k: round(v, 4) for k, v in means.items()}
    report['pages_run'] = len(rows)
    with open(RESULTS, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print('pages %d; wrote %s' % (len(rows), os.path.relpath(RESULTS, ROOT)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
