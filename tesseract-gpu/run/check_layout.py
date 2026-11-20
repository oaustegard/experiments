"""Layout check: test pages -> PP-OCRv6 tiny detector maps -> tessgpu.layout -> results.

Writes:
  data/layout_pages/   rendered test pages, FUNSD forms, numbered overlays (*_overlay.png)
  data/layout_cases/   the probability maps (<name>.f32, raw little-endian float32) and their metadata
  results/layout_check.json

run/check_layout.mjs recomputes the boxes in JavaScript from data/layout_cases and compares them
with the Python result stored in results/layout_check.json.

Usage: python3 run/check_layout.py   (re-executes itself under python3 -I, as the detector is loaded)
"""

import json
import math
import os
import sys
import time

if not sys.flags.isolated:
    os.execv(sys.executable, [sys.executable, '-I', os.path.abspath(__file__)] + sys.argv[1:])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from tessgpu import layout as L  # noqa: E402

PAGES_DIR = os.path.join(ROOT, 'data', 'layout_pages')
CASES_DIR = os.path.join(ROOT, 'data', 'layout_cases')
RESULTS = os.path.join(ROOT, 'results', 'layout_check.json')
MODEL = os.path.join(ROOT, 'models', 'det', 'tiny', 'inference.onnx')
FUNSD_PARQUET = os.path.join(PAGES_DIR, 'funsd_src', 'data', 'test-00000-of-00001.parquet')

SERIF = '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf'
SANS = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
SANS_BOLD = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
DEJAVU = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

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


# ---------------------------------------------------------------- pages

def _wrap(draw, text, font, width):
    words = text.split()
    lines, cur = [], ''
    for w in words:
        trial = (cur + ' ' + w).strip()
        if draw.textlength(trial, font=font) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _block(draw, x, y, text, font, width, leading):
    for ln in _wrap(draw, text, font, width):
        draw.text((x, y), ln, font=font, fill=(0, 0, 0))
        y += leading
    return y


def render_two_column(path):
    img = Image.new('RGB', (1240, 1754), 'white')
    d = ImageDraw.Draw(img)
    head = ImageFont.truetype(SANS_BOLD, 46)
    body = ImageFont.truetype(SERIF, 25)
    d.text((110, 120), 'Notes on Column Layout', font=head, fill=(0, 0, 0))
    y0 = 230
    cols = [(110, 540), (690, 540)]
    for ci, (x, w) in enumerate(cols):
        y = y0
        for pi, para in enumerate([BODY, BODY2, BODY, BODY2 if ci == 0 else BODY]):
            y = _block(d, x, y, para, body, w, 34) + 26
    img.save(path)


def render_heading_table(path):
    img = Image.new('RGB', (1240, 1100), 'white')
    d = ImageDraw.Draw(img)
    head = ImageFont.truetype(SANS_BOLD, 44)
    body = ImageFont.truetype(SANS, 25)
    bold = ImageFont.truetype(SANS_BOLD, 25)
    d.text((110, 110), 'Quarterly Results', font=head, fill=(0, 0, 0))
    y = _block(d, 110, 200, BODY + ' ' + BODY2, body, 1020, 36) + 40
    rows = [
        ('Region', 'Units', 'Revenue'),
        ('North', '1 240', '3 480 k'),
        ('South', '912', '2 115 k'),
        ('East', '1 587', '4 020 k'),
        ('West', '640', '1 090 k'),
    ]
    xs = [110, 560, 860]
    for ri, row in enumerate(rows):
        f = bold if ri == 0 else body
        for ci, cell in enumerate(row):
            if ci == 0:
                d.text((xs[0], y), cell, font=f, fill=(0, 0, 0))
            else:
                right = xs[ci] + (230 if ci == 1 else 300)
                tw = d.textlength(cell, font=f)
                d.text((right - tw, y), cell, font=f, fill=(0, 0, 0))
        if ri == 0:
            d.line([(110, y + 40), (1130, y + 40)], fill=(0, 0, 0), width=2)
        y += 56
    img.save(path)


def render_rotated(path):
    base = Image.new('RGB', (1240, 900), 'white')
    d = ImageDraw.Draw(base)
    body = ImageFont.truetype(SERIF, 27)
    y = 120
    for para in [BODY, BODY2, BODY]:
        y = _block(d, 110, y, para, body, 1020, 38) + 30
    rot = base.rotate(1.0, resample=Image.BICUBIC, expand=True, fillcolor='white')
    rot.save(path)


def extract_funsd(out_dir):
    import io
    import pyarrow.parquet as pq
    rows = pq.read_table(FUNSD_PARQUET, columns=['id', 'image']).to_pylist()
    names = []
    for row in rows[:2]:
        im = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
        name = 'funsd_%s' % row['id'].replace('/', '_')
        im.save(os.path.join(out_dir, name + '.png'))
        names.append(name)
    return names


def ensure_pages():
    os.makedirs(PAGES_DIR, exist_ok=True)
    pages = {
        'two_column': os.path.join(PAGES_DIR, 'two_column.png'),
        'heading_table': os.path.join(PAGES_DIR, 'heading_table.png'),
        'rotated_1deg': os.path.join(PAGES_DIR, 'rotated_1deg.png'),
    }
    if not os.path.exists(pages['two_column']):
        render_two_column(pages['two_column'])
    if not os.path.exists(pages['heading_table']):
        render_heading_table(pages['heading_table'])
    if not os.path.exists(pages['rotated_1deg']):
        render_rotated(pages['rotated_1deg'])
    if not os.path.exists(FUNSD_PARQUET):
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download('nielsr/funsd', 'data/test-00000-of-00001.parquet', repo_type='dataset',
                            local_dir=os.path.join(PAGES_DIR, 'funsd_src'))
        except Exception as e:  # FUNSD is optional; the synthetic pages still run
            print('FUNSD not fetched (%s); continuing without it' % type(e).__name__)
    if os.path.exists(FUNSD_PARQUET):
        for name in extract_funsd(PAGES_DIR):
            pages[name] = os.path.join(PAGES_DIR, name + '.png')
    return pages


# ---------------------------------------------------------------- detector

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # applied to B, G, R in that order
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def detect(sess, img):
    w0, h0 = img.size
    ratio = 1280 / max(w0, h0)
    cw, ch = round(w0 * ratio), round(h0 * ratio)
    im = img.resize((cw, ch), Image.BILINEAR)
    bgr = np.asarray(im, dtype=np.float32)[:, :, ::-1] / np.float32(255.0)
    x = (bgr - MEAN) / STD
    Hp = int(math.ceil(ch / 32) * 32)
    Wp = int(math.ceil(cw / 32) * 32)
    pad = np.zeros((Hp, Wp, 3), dtype=np.float32)
    pad[:ch, :cw] = x
    tensor = np.ascontiguousarray(pad.transpose(2, 0, 1)[None])
    out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    prob = out[0, 0].astype(np.float32)
    assert prob.shape == (Hp, Wp), prob.shape
    return prob, {'page_w': w0, 'page_h': h0, 'content_w': cw, 'content_h': ch,
                  'map_w': Wp, 'map_h': Hp, 'ratio': ratio}


# ---------------------------------------------------------------- overlay

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40)]


def overlay(page_path, out_path, res):
    img = Image.open(page_path).convert('RGB')
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(DEJAVU, max(12, img.size[0] // 70))
    by_id = {ln['id']: ln for ln in res['lines']}
    for rank, lid in enumerate(res['order']):
        ln = by_id[lid]
        col = PALETTE[ln['block'] % len(PALETTE)]
        pts = [tuple(p) for p in ln['quad']]
        d.line(pts + [pts[0]], fill=col, width=2)
        x, y = min(p[0] for p in pts), min(p[1] for p in pts)
        label = str(rank)
        tw = d.textlength(label, font=font)
        d.rectangle([x - 2, y - 2 - font.size, x + tw + 2, y], fill=col)
        d.text((x, y - font.size - 2), label, font=font, fill=(255, 255, 255))
    img.save(out_path)


# ---------------------------------------------------------------- main

def main():
    os.makedirs(CASES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    pages = ensure_pages()
    sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
    report = {
        'model': 'models/det/tiny/inference.onnx (PP-OCRv6_tiny_det)',
        'onnxruntime': ort.__version__,
        'params': dict(L.DEFAULTS),
        'pages': {},
    }
    total = 0
    for name in sorted(pages):
        img = Image.open(pages[name]).convert('RGB')
        t0 = time.time()
        prob, meta = detect(sess, img)
        t_det = time.time() - t0
        sx = meta['page_w'] / meta['content_w']
        sy = meta['page_h'] / meta['content_h']
        prob.astype('<f4').tofile(os.path.join(CASES_DIR, name + '.f32'))
        case = dict(meta, scale_x=sx, scale_y=sy, H=meta['map_h'], W=meta['map_w'])
        with open(os.path.join(CASES_DIR, name + '.json'), 'w') as f:
            json.dump(case, f, indent=1)

        t0 = time.time()
        res = L.layout(prob, sx, sy, meta['page_w'], meta['page_h'])
        t_lay = time.time() - t0
        ov = os.path.join(PAGES_DIR, name + '_overlay.png')
        overlay(pages[name], ov, res)
        total += len(res['lines'])
        report['pages'][name] = {
            'page': [meta['page_w'], meta['page_h']],
            'map': [meta['map_w'], meta['map_h']],
            'n_lines': len(res['lines']),
            'n_blocks': len(res['blocks']),
            'overlay': os.path.relpath(ov, ROOT),
            'seconds': {'detector': round(t_det, 3), 'layout': round(t_lay, 3)},
            'result': res,
        }
        print('%-14s page %4dx%-4d map %4dx%-4d lines %3d blocks %2d  (layout %.3fs)' % (
            name, meta['page_w'], meta['page_h'], meta['map_w'], meta['map_h'],
            len(res['lines']), len(res['blocks']), t_lay))

    # Two-column page: every left-column body line is read before every right-column body line.
    tc = report['pages'].get('two_column')
    if tc:
        lines = {ln['id']: ln for ln in tc['result']['lines']}
        pos = {lid: r for r, lid in enumerate(tc['result']['order'])}
        body = [ln for ln in lines.values() if ln['box'][1] > 200]
        left = [pos[ln['id']] for ln in body if ln['box'][0] < 600]
        right = [pos[ln['id']] for ln in body if ln['box'][0] >= 600]
        tc['column_order'] = {
            'left_lines': len(left), 'right_lines': len(right),
            'left_all_before_right': bool(left and right and max(left) < min(right)),
        }
        print('two_column column order:', tc['column_order'])

    report['total_lines'] = total
    with open(RESULTS, 'w') as f:
        json.dump(report, f, indent=1)
    print('wrote', os.path.relpath(RESULTS, ROOT), 'and', len(pages), 'cases')
    return 0


if __name__ == '__main__':
    sys.exit(main())
