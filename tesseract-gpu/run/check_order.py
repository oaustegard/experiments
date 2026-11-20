"""Reading-order check: column bands (order='bands') against the recursive XY-cut (default).

Pages with known reading order -> detector maps -> tessgpu.layout in both modes -> Kendall's tau.

Pages:
  two_column, heading_table, rotated_1deg   from run/check_layout.py (data/layout_pages, data/layout_cases)
  three_column, header_break, figure_caption  rendered here into data/order_pages/
  funsd_0, funsd_1                          layout only: FUNSD has word boxes and NER tags, no reading order

Ground truth is what this script drew. Every text draw is recorded with its position and a
unit number, and a per-line ID map is kept alongside the page image. The ID map goes through the
same rotation as the rotated page, so a detector line is matched to the rendered line it covers.
Units are the reading-order blocks: the GT order is (unit, y, x).

Scoring: the predicted order of matched GT lines (first detector line per GT line wins) against
the true order. tau = 1 - 2 * inversions / (m choose 2) over the m matched GT lines. coverage is
m over the number of GT lines. Pages without GT get tau None.

Writes results/order_check.json and data/order_pages/ (rendered pages, detector maps, overlays).
run/check_order.mjs recomputes both modes in JavaScript and checks them against this file.

Usage: python3 run/check_order.py   (re-executes itself under python3 -I, as check_layout does)
"""

import json
import os
import sys
import tempfile

if not sys.flags.isolated:
    os.execv(sys.executable, [sys.executable, '-I', os.path.abspath(__file__)] + sys.argv[1:])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'run')]

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from PIL import Image, ImageDraw as PILDraw, ImageFont  # noqa: E402

import check_layout as cl  # noqa: E402  (renderers, detector, overlay: the layout check's own code)
from tessgpu import layout as L  # noqa: E402

PAGES_DIR = os.path.join(ROOT, 'data', 'layout_pages')
CASES_DIR = os.path.join(ROOT, 'data', 'layout_cases')
ORDER_DIR = os.path.join(ROOT, 'data', 'order_pages')
RESULTS = os.path.join(ROOT, 'results', 'order_check.json')
LAYOUT_CHECK = os.path.join(ROOT, 'results', 'layout_check.json')
MODEL = cl.MODEL
MODES = ('xycut', 'bands')
MATCH_SHARE = 0.3  # a detector line is matched to a GT line when that GT line covers this share of its box


# ---------------------------------------------------------------- recording draw

class Rec:
    """Draws onto the page like ImageDraw, and records each text draw as one GT line."""

    def __init__(self, img):
        self.img = img
        self.d = PILDraw.Draw(img)
        self.id_img = Image.new('I', img.size, 0)
        self.idd = PILDraw.Draw(self.id_img)
        self.lines = []
        self.unit = 0

    def __getattr__(self, name):
        return getattr(self.d, name)

    def textlength(self, s, font=None):
        return self.d.textlength(s, font=font)

    def text(self, xy, s, font=None, fill=None, **kw):
        self.d.text(xy, s, font=font, fill=fill)
        if not s.strip():
            return
        l, t, r, b = font.getbbox(s)
        x, y = float(xy[0]), float(xy[1])
        gid = len(self.lines) + 1
        self.lines.append({'text': s, 'unit': self.unit, 'x': x, 'y': y, 'gid': gid,
                           'box': [round(x + l), round(y + t), round(x + r), round(y + b)]})
        self.idd.rectangle([x + l, y + t, x + r - 1, y + b - 1], fill=gid)


class _DrawFactory:
    """Stands in for check_layout's ImageDraw module while an existing renderer runs."""

    def __init__(self):
        self.recs = []

    def Draw(self, img):
        rec = Rec(img)
        self.recs.append(rec)
        return rec


# ---------------------------------------------------------------- new pages

W3, H3 = 1240, 1754
W2, H2 = 1240, 2100
W4, H4 = 1240, 2300


def _paras(rec, x, y, paras, font, width, leading, unit):
    rec.unit = unit
    for para in paras:
        y = cl._block(rec, x, y, para, font, width, leading) + 24
    return y


def render_three_column(rec):
    """Title, then three columns of two paragraphs each, 60 px gutters."""
    head = ImageFont.truetype(cl.SANS_BOLD, 40)
    body = ImageFont.truetype(cl.SERIF, 22)
    rec.unit = 0
    rec.text((80, 100), 'Three Columns of Notes', font=head, fill=(0, 0, 0))
    plan = [(cl.BODY, cl.BODY2), (cl.BODY2, cl.BODY), (cl.BODY, cl.BODY2)]
    for ci, x in enumerate((80, 460, 840)):
        y = _paras(rec, x, 200, plan[ci], body, 320, 30, unit=1 + ci)
        assert y < H3 - 60, y


def render_header_break(rec):
    """Title, two columns, a full-width section header, two columns."""
    head = ImageFont.truetype(cl.SANS_BOLD, 40)
    sect = ImageFont.truetype(cl.SANS_BOLD, 32)
    body = ImageFont.truetype(cl.SERIF, 24)
    rec.unit = 0
    rec.text((110, 100), 'Two Columns and a Section Break', font=head, fill=(0, 0, 0))
    lx, rx, cw = 110, 660, 470
    yl = _paras(rec, lx, 200, (cl.BODY2, cl.BODY), body, cw, 33, unit=1)
    yr = _paras(rec, rx, 200, (cl.BODY, cl.BODY2), body, cw, 33, unit=2)
    yh = max(yl, yr) + 20
    rec.unit = 3
    rec.text((110, yh), 'Section Two: Results Across Both Pilot Sites', font=sect, fill=(0, 0, 0))
    y0 = yh + 90
    yl = _paras(rec, lx, y0, (cl.BODY, cl.BODY2), body, cw, 33, unit=4)
    yr = _paras(rec, rx, y0, (cl.BODY2, cl.BODY), body, cw, 33, unit=5)
    assert max(yl, yr) < H2 - 60, max(yl, yr)


def render_figure_caption(rec):
    """Title, two columns, a full-width figure with a full-width caption, two columns."""
    head = ImageFont.truetype(cl.SANS_BOLD, 40)
    cap = ImageFont.truetype(cl.SANS, 24)
    body = ImageFont.truetype(cl.SERIF, 24)
    rec.unit = 0
    rec.text((110, 100), 'Measuring Throughput Across Two Sites', font=head, fill=(0, 0, 0))
    lx, rx, cw = 110, 660, 470
    yl = _paras(rec, lx, 200, (cl.BODY, cl.BODY2), body, cw, 33, unit=1)
    yr = _paras(rec, rx, 200, (cl.BODY2, cl.BODY), body, cw, 33, unit=2)
    yf = max(yl, yr) + 20
    # figure: a grey panel with no text in it, so it contributes no ground truth lines
    rec.rectangle([110, yf, 1130, yf + 300], fill=(236, 236, 236), outline=(150, 150, 150), width=2)
    for k in range(5):
        x = 170 + k * 180
        rec.rectangle([x, yf + 300 - 40 - 30 * (k % 3 + 1), x + 90, yf + 260], fill=(200, 200, 200))
    rec.unit = 3
    rec.text((110, yf + 320), 'Figure 3. Throughput per column across both pilot sites, in lines per second.',
             font=cap, fill=(0, 0, 0))
    y0 = yf + 320 + 70
    yl = _paras(rec, lx, y0, (cl.BODY, cl.BODY2), body, cw, 33, unit=4)
    yr = _paras(rec, rx, y0, (cl.BODY2, cl.BODY), body, cw, 33, unit=5)
    assert max(yl, yr) < H4 - 60, max(yl, yr)


NEW_PAGES = [
    ('three_column', W3, H3, render_three_column, 'three columns, no spanning element except the title'),
    ('header_break', W2, H2, render_header_break, 'two columns with a full-width section header in the middle'),
    ('figure_caption', W4, H4, render_figure_caption, 'two columns with a full-width figure caption under a figure'),
]

# Existing pages: the unit rule is written from what check_layout draws (x and y of each text draw).
EXISTING_PAGES = [
    ('two_column', cl.render_two_column, False,
     lambda ln: 0 if ln['y'] < 200 else (1 if ln['x'] < 600 else 2),
     'title, then left column top to bottom, then right column'),
    ('heading_table', cl.render_heading_table, False,
     lambda ln: 0 if ln['y'] < 150 else 1,
     'title, then full-width paragraph and table, row by row, left to right'),
    ('rotated_1deg', cl.render_rotated, True,
     lambda ln: 1,
     'single column, rotated 1 degree; lines top to bottom'),
]
NO_GT_PAGES = ['funsd_0', 'funsd_1']


# ---------------------------------------------------------------- ground truth

def gt_for(rec, unit_rule):
    lines = []
    for ln in rec.lines:
        lines.append(dict(ln, unit=unit_rule(ln) if unit_rule else ln['unit']))
    order = sorted(range(len(lines)), key=lambda k: (lines[k]['unit'], lines[k]['y'], lines[k]['x']))
    rank_of_gid = {lines[k]['gid']: r for r, k in enumerate(order)}
    index_of_gid = {lines[k]['gid']: k for k in range(len(lines))}
    true_text = [lines[k]['text'] for k in order]
    return lines, rank_of_gid, index_of_gid, true_text


def match_boxes(idmap, boxes):
    """For each detector box, the GT id of the line that covers it most, or None."""
    H, W = idmap.shape
    out = []
    for x0, y0, x1, y1 in boxes:
        xa, xb = max(0, x0), min(W, x1)
        ya, yb = max(0, y0), min(H, y1)
        if xb <= xa or yb <= ya:
            out.append(None)
            continue
        reg = idmap[ya:yb, xa:xb].ravel()
        reg = reg[reg > 0]
        if reg.size == 0:
            out.append(None)
            continue
        counts = np.bincount(reg)
        g = int(counts.argmax())
        share = counts[g] / float((xb - xa) * (yb - ya))
        out.append(g if share >= MATCH_SHARE else None)
    return out


def kendall_tau(ranks):
    """tau-a over a sequence of distinct ranks, or None when fewer than two items."""
    m = len(ranks)
    if m < 2:
        return None
    inv = 0
    for i in range(m):
        for j in range(i + 1, m):
            if ranks[i] > ranks[j]:
                inv += 1
    return 1.0 - 2.0 * inv / (m * (m - 1) / 2.0)


def score(res, gid_of_line, rank_of_gid, index_of_gid, lines, true_text):
    seq, seen = [], set()
    for lid in res['order']:
        g = gid_of_line[lid]
        if g is None or g in seen:
            continue
        seen.add(g)
        seq.append(g)
    ranks = [rank_of_gid[g] for g in seq]
    return {
        'tau': kendall_tau(ranks),
        'matched': len(seq),
        'gt_lines': len(lines),
        'coverage': len(seq) / float(len(lines)),
        'unmatched_detector_lines': sum(1 for g in gid_of_line if g is None),
        'text_in_order': [lines[index_of_gid[g]]['text'] for g in seq],
        'true_text': true_text,
    }


# ---------------------------------------------------------------- pages

def detect_page(sess, img):
    prob, meta = cl.detect(sess, img)
    sx = meta['page_w'] / meta['content_w']
    sy = meta['page_h'] / meta['content_h']
    return prob, dict(meta, scale_x=sx, scale_y=sy, H=meta['map_h'], W=meta['map_w'])


def existing_gt(sess, name, render, rotate, unit_rule):
    """Re-render an existing page with the recording draw; check it is the PNG the detector read."""
    fac = _DrawFactory()
    real = cl.ImageDraw
    cl.ImageDraw = fac
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, name + '.png')
            render(out)
            rendered = Image.open(out).convert('RGB')
            idmap = fac.recs[0].id_img
            if rotate:
                idmap = idmap.rotate(1.0, resample=Image.NEAREST, expand=True, fillcolor=0)
    finally:
        cl.ImageDraw = real
    stored = Image.open(os.path.join(PAGES_DIR, name + '.png')).convert('RGB')
    same = rendered.size == stored.size and np.array_equal(np.asarray(rendered), np.asarray(stored))
    assert idmap.size == stored.size, (idmap.size, stored.size)
    return fac.recs[0], np.asarray(idmap), same


def main():
    os.makedirs(ORDER_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    sess = ort.InferenceSession(MODEL, providers=['CPUExecutionProvider'])
    with open(LAYOUT_CHECK) as f:
        layout_check = json.load(f)

    report = {
        'question': 'Does the column-band order (order="bands") beat the recursive XY-cut order on pages '
                    'with columns, full-width headers and full-width captions?',
        'modes': {'xycut': 'recursive XY-cut (default)',
                  'bands': 'column bands: anything-to-text lineify / detectColumnCuts / orderByColumns'},
        'metric': "Kendall's tau over matched GT lines (predicted order vs true order); "
                  'coverage = matched GT lines / GT lines',
        'match_share': MATCH_SHARE,
        'params': dict(L.DEFAULTS),
        'onnxruntime': ort.__version__,
        'pages': {},
    }

    def run_layouts(name, prob, meta):
        res = {}
        for mode in MODES:
            res[mode] = L.layout(prob, meta['scale_x'], meta['scale_y'], meta['page_w'], meta['page_h'],
                                 order=mode)
            res[mode] = json.loads(json.dumps(res[mode]))
        return res

    def page_entry(name, source, prob, meta, res, gt=None, note=None):
        entry = {'source': source, 'page': [meta['page_w'], meta['page_h']],
                 'map': [meta['W'], meta['H']], 'case_dir': source, 'layouts': res, 'modes': {}}
        if note:
            entry['note'] = note
        for mode in MODES:
            if gt is None:
                entry['modes'][mode] = {'tau': None, 'reason': 'no reading-order ground truth'}
                continue
            lines, rank_of_gid, index_of_gid, true_text, gid_of_line = gt
            boxes = [ln['box'] for ln in res[mode]['lines']]
            entry['modes'][mode] = score(res[mode], gid_of_line, rank_of_gid, index_of_gid, lines, true_text)
            entry['modes'][mode]['band_column'] = [[ln['id'], ln.get('band'), ln.get('column')]
                                                   for ln in res[mode]['lines']]
        if gt is not None:
            entry['gt_lines'] = [{'text': ln['text'], 'unit': ln['unit'], 'box': ln['box']} for ln in gt[0]]
        return entry

    # existing pages: detector maps are the ones check_layout wrote; re-detect to confirm they match
    for name, render, rotate, unit_rule, note in EXISTING_PAGES:
        rec, idmap, same = existing_gt(sess, name, render, rotate, unit_rule)
        lines, rank_of_gid, index_of_gid, true_text = gt_for(rec, unit_rule)
        prob = np.fromfile(os.path.join(CASES_DIR, name + '.f32'), dtype='<f4')
        meta = json.load(open(os.path.join(CASES_DIR, name + '.json')))
        prob = prob.reshape(meta['H'], meta['W'])
        img = Image.open(os.path.join(PAGES_DIR, name + '.png')).convert('RGB')
        re_prob, _ = detect_page(sess, img)
        res = run_layouts(name, prob, meta)
        gid_of_line = match_boxes(idmap, [ln['box'] for ln in res['bands']['lines']])
        entry = page_entry(name, 'layout_cases', prob, meta, res,
                           gt=(lines, rank_of_gid, index_of_gid, true_text, gid_of_line), note=note)
        entry['gt_render_matches_png'] = bool(same)
        entry['case_matches_detector'] = bool(np.array_equal(re_prob, prob))
        lc = layout_check['pages'][name]['result']
        entry['xycut_matches_layout_check'] = json.loads(json.dumps(res['xycut'])) == json.loads(json.dumps(lc))
        report['pages'][name] = entry
        print('%-14s GT %3d lines  %s' % (name, len(lines), note))

    # new pages: render, detect, record ground truth, write map and overlay
    for name, w, h, render, note in NEW_PAGES:
        img = Image.new('RGB', (w, h), 'white')
        rec = Rec(img)
        render(rec)
        png = os.path.join(ORDER_DIR, name + '.png')
        img.save(png)
        img = Image.open(png).convert('RGB')
        prob, meta = detect_page(sess, img)
        prob.astype('<f4').tofile(os.path.join(ORDER_DIR, name + '.f32'))
        with open(os.path.join(ORDER_DIR, name + '.json'), 'w') as f:
            json.dump(meta, f, indent=1)
        lines, rank_of_gid, index_of_gid, true_text = gt_for(rec, None)
        idmap = np.asarray(rec.id_img)
        res = run_layouts(name, prob, meta)
        gid_of_line = match_boxes(idmap, [ln['box'] for ln in res['bands']['lines']])
        entry = page_entry(name, 'order_pages', prob, meta, res,
                           gt=(lines, rank_of_gid, index_of_gid, true_text, gid_of_line), note=note)
        report['pages'][name] = entry
        overlay = os.path.join(ORDER_DIR, name + '_bands_overlay.png')
        cl.overlay(png, overlay, res['bands'])
        with open(os.path.join(ORDER_DIR, name + '_gt.json'), 'w') as f:
            json.dump({'units_are_reading_blocks': True, 'lines': entry['gt_lines']}, f, indent=1)
        print('%-14s GT %3d lines  %s' % (name, len(lines), note))

    # FUNSD: layout only, both modes, for the JS parity check
    for name in NO_GT_PAGES:
        img = Image.open(os.path.join(PAGES_DIR, name + '.png')).convert('RGB')
        prob = np.fromfile(os.path.join(CASES_DIR, name + '.f32'), dtype='<f4')
        meta = json.load(open(os.path.join(CASES_DIR, name + '.json')))
        prob = prob.reshape(meta['H'], meta['W'])
        res = run_layouts(name, prob, meta)
        report['pages'][name] = page_entry(name, 'layout_cases', prob, meta, res, gt=None,
                                           note='FUNSD form: no reading-order ground truth, not scored')
        print('%-14s not scored (no reading-order ground truth)' % name)

    scored = [n for n, p in report['pages'].items() if p['modes'][MODES[0]].get('tau') is not None]
    summary = {}
    for mode in MODES:
        taus = [report['pages'][n]['modes'][mode]['tau'] for n in scored]
        summary[mode] = {'mean_tau': sum(taus) / len(taus), 'min_tau': min(taus),
                         'min_coverage': min(report['pages'][n]['modes'][mode]['coverage'] for n in scored)}
    report['scored_pages'] = scored
    report['summary'] = summary
    with open(RESULTS, 'w') as f:
        json.dump(report, f, indent=1)
        f.write('\n')

    print()
    print('%-16s %10s %10s %10s %10s' % ('page', 'tau xycut', 'tau bands', 'cov xycut', 'cov bands'))
    for n in report['pages']:
        p = report['pages'][n]
        if n in scored:
            print('%-16s %10.4f %10.4f %10.3f %10.3f' % (
                n, p['modes']['xycut']['tau'], p['modes']['bands']['tau'],
                p['modes']['xycut']['coverage'], p['modes']['bands']['coverage']))
        else:
            print('%-16s %10s %10s   (not scored)' % (n, '-', '-'))
    print('wrote', os.path.relpath(RESULTS, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
