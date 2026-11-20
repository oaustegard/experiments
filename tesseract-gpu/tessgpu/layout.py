"""Text-line boxes and reading order from a text-probability map, without OpenCV.

layout(prob, scale_x, scale_y, page_w, page_h, **overrides) -> dict

  prob            float array [H, W], values 0..1, at the detector's input scale
  scale_x/scale_y page pixels per map pixel (map pixel * scale = page pixel)
  page_w, page_h  page size in pixels

Three steps:
  1. Boxes: PaddleOCR's DBPostProcess semantics (binarise at thresh, 8-connected
     components, minimum-area rectangle, box score = mean probability inside,
     drop low score and small boxes, unclip by area*ratio/perimeter, scale to page).
  2. Lines: split components across a clear gutter, then merge fragments of one line.
  3. Reading order: recursive XY-cut (vertical cuts before horizontal), then rows
     top to bottom and left to right inside each block.

web/layout.js is a line-by-line port. Both use only + - * / and sqrt on doubles, and
the probability sums are exact integers (probabilities quantised to 1e-6), so the two
implementations give identical integer boxes and identical order.
"""

import math

import numpy as np

Q = 1000000  # probabilities are quantised to 1e-6 before summing, so box scores are exact

DEFAULTS = {
    'thresh': 0.2,         # DBPostProcess thresh: binarisation
    'box_thresh': 0.4,     # DBPostProcess box_thresh: minimum mean probability in the box
    'unclip_ratio': 1.4,   # DBPostProcess unclip_ratio
    'max_candidates': 3000,
    'min_size': 3,         # minimum short side in map pixels (PaddleOCR min_size)
    'split_low': 0.1,      # gutter column: at most this fraction of the component height
    'split_gap': 1.5,      # split a component across a gutter at least this many heights wide
    'merge_overlap': 0.6,  # fragments: vertical overlap as a fraction of the shorter box
    'merge_gap': 1.0,      # fragments: horizontal gap, in units of the shorter box height
    'merge_hratio': 1.5,   # fragments: tallest / shortest box height
    'col_gap': 1.5,        # XY-cut: vertical cut needs a gap this many median line heights wide
    'row_gap': 0.6,        # XY-cut: horizontal cut needs a gap this many median line heights
}


# ---------------------------------------------------------------- geometry

def _hull(pts):
    """Convex hull, counter-clockwise, monotone chain. Duplicates removed."""
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _extents(hull, ux, uy):
    vx, vy = -uy, ux
    s = [x * ux + y * uy for x, y in hull]
    t = [x * vx + y * vy for x, y in hull]
    return min(s), max(s), min(t), max(t)


def _rect_on(hull, ux, uy):
    """Rectangle of the hull with one side along (ux, uy). Long side is always u."""
    smin, smax, tmin, tmax = _extents(hull, ux, uy)
    if tmax - tmin > smax - smin:
        ux, uy = -uy, ux
    if ux < 0 or (ux == 0 and uy < 0):
        ux, uy = -ux, -uy
    smin, smax, tmin, tmax = _extents(hull, ux, uy)
    vx, vy = -uy, ux
    smid = (smin + smax) / 2
    tmid = (tmin + tmax) / 2
    cx = smid * ux + tmid * vx
    cy = smid * uy + tmid * vy
    W = smax - smin
    H = tmax - tmin
    return {'cx': cx, 'cy': cy, 'ux': ux, 'uy': uy, 'W': W, 'H': H, 'area': W * H}


def _min_rect(hull):
    """Minimum-area enclosing rectangle; candidate sides are the hull edges (rotating calipers)."""
    best = None
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        dx = bx - ax
        dy = by - ay
        L = math.sqrt(dx * dx + dy * dy)
        r = _rect_on(hull, dx / L, dy / L)
        if best is None or r['area'] < best['area']:
            best = r
    return best


def _corners(cx, cy, ux, uy, W, H):
    vx, vy = -uy, ux
    ax, ay = (W / 2) * ux, (W / 2) * uy
    bx, by = (H / 2) * vx, (H / 2) * vy
    return [
        (cx - ax - bx, cy - ay - by),
        (cx + ax - bx, cy + ay - by),
        (cx + ax + bx, cy + ay + by),
        (cx - ax + bx, cy - ay + by),
    ]


def _rnd(v):
    return math.floor(v + 0.5)


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _dist(p, q):
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    return math.sqrt(dx * dx + dy * dy)


def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------------------------------------------------------- step 1: boxes

def _components(mask):
    """8-connected components, in raster order of their first pixel; pixels sorted (y, x)."""
    H, W = mask.shape
    m = mask.ravel()
    seen = bytearray(H * W)
    comps = []
    for start in np.flatnonzero(m).tolist():
        if seen[start]:
            continue
        seen[start] = 1
        stack = [start]
        pix = []
        while stack:
            k = stack.pop()
            y, x = divmod(k, W)
            pix.append((x, y))
            for yy in (y - 1, y, y + 1):
                if yy < 0 or yy >= H:
                    continue
                for xx in (x - 1, x, x + 1):
                    if xx < 0 or xx >= W:
                        continue
                    kk = yy * W + xx
                    if m[kk] and not seen[kk]:
                        seen[kk] = 1
                        stack.append(kk)
        pix.sort(key=lambda p: (p[1], p[0]))
        comps.append(pix)
    return comps


def _split(pix, opt):
    """Split a component across gutters: runs of near-empty columns at least split_gap heights wide.

    Widest run first, then each side recursively. The bridge pixels inside the gutter are dropped.
    Pieces with fewer than 3 pixels are dropped.
    """
    if len(pix) < 3:
        return []
    xs = [p[0] for p in pix]
    ys = [p[1] for p in pix]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    hc = y1 - y0 + 1
    n = x1 - x0 + 1
    cnt = [0] * n
    for x, _ in pix:
        cnt[x - x0] += 1
    best = None
    i = 1
    while i < n - 1:
        if cnt[i] <= opt['split_low'] * hc:
            j = i
            while j + 1 < n - 1 and cnt[j + 1] <= opt['split_low'] * hc:
                j += 1
            g = j - i + 1
            if g >= opt['split_gap'] * hc and (best is None or g > best[0]):
                best = (g, i, j)
            i = j + 1
        else:
            i += 1
    if best is None:
        return [pix]
    _, a, b = best
    left = [p for p in pix if p[0] - x0 < a]
    right = [p for p in pix if p[0] - x0 > b]
    return _split(left, opt) + _split(right, opt)


def _score(q, H, W, r):
    """Mean quantised probability over the pixel centres inside the rectangle (exact integer sum)."""
    cs = _corners(r['cx'], r['cy'], r['ux'], r['uy'], r['W'], r['H'])
    xs = [c[0] for c in cs]
    ys = [c[1] for c in cs]
    x0 = max(0, math.floor(min(xs)))
    x1 = min(W - 1, math.ceil(max(xs)))
    y0 = max(0, math.floor(min(ys)))
    y1 = min(H - 1, math.ceil(max(ys)))
    if x1 < x0 or y1 < y0:
        return None
    X = np.arange(x0, x1 + 1, dtype=np.float64)[None, :]
    Y = np.arange(y0, y1 + 1, dtype=np.float64)[:, None]
    dx = X - r['cx']
    dy = Y - r['cy']
    ux, uy = r['ux'], r['uy']
    vx, vy = -uy, ux
    s = dx * ux + dy * uy
    t = dx * vx + dy * vy
    inside = (np.abs(s) <= r['W'] / 2) & (np.abs(t) <= r['H'] / 2)
    n = int(np.count_nonzero(inside))
    if n == 0:
        return None
    S = int(q[y0:y1 + 1, x0:x1 + 1][inside].sum())
    return S / (n * Q)


def _db_box(pix, q, H, W, opt):
    """DBPostProcess for one component: returns (score, unclipped corners in map pixels) or None."""
    hull = _hull(pix)
    if len(hull) < 3:
        return None
    r = _min_rect(hull)
    if min(r['W'], r['H']) < opt['min_size']:
        return None
    score = _score(q, H, W, r)
    if score is None or score < opt['box_thresh']:
        return None
    d = r['W'] * r['H'] * opt['unclip_ratio'] / (2 * (r['W'] + r['H']))
    We = r['W'] + 2 * d
    He = r['H'] + 2 * d
    if min(We, He) < opt['min_size'] + 2:
        return None
    return score, _corners(r['cx'], r['cy'], r['ux'], r['uy'], We, He)


def _line_from_quad(quad, score, pw, ph):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    box = [
        _clamp(_rnd(min(xs)), 0, pw),
        _clamp(_rnd(min(ys)), 0, ph),
        _clamp(_rnd(max(xs)), 0, pw),
        _clamp(_rnd(max(ys)), 0, ph),
    ]
    return {'box': box, 'quad': quad, 'score': score}


# ---------------------------------------------------------------- step 2: lines

def _mergeable(a, b, opt):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ha = ay1 - ay0
    hb = by1 - by0
    hmin = min(ha, hb)
    hmax = max(ha, hb)
    if hmin <= 0:
        return False
    overlap = min(ay1, by1) - max(ay0, by0)
    if overlap < opt['merge_overlap'] * hmin:
        return False
    if hmax > opt['merge_hratio'] * hmin:
        return False
    gap = max(ax0, bx0) - min(ax1, bx1)
    return gap <= opt['merge_gap'] * hmin


def _merge_lines(lines, opt):
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _mergeable(lines[i]['box'], lines[j]['box'], opt):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for root in sorted(groups):
        mem = groups[root]
        if len(mem) == 1:
            merged.append(lines[mem[0]])
            continue
        pts = []
        for i in mem:
            pts.extend(lines[i]['quad'])
        r = _min_rect(_hull(pts))
        quad = _corners(r['cx'], r['cy'], r['ux'], r['uy'], r['W'], r['H'])
        score = 0.0
        for i in mem:
            score += lines[i]['score']
        score = score / len(mem)
        box = [
            min(lines[i]['box'][0] for i in mem),
            min(lines[i]['box'][1] for i in mem),
            max(lines[i]['box'][2] for i in mem),
            max(lines[i]['box'][3] for i in mem),
        ]
        merged.append({'box': box, 'quad': quad, 'score': score})
    return merged


# ---------------------------------------------------------------- step 3: reading order

def _widest_gap(ids, boxes, axis):
    """Widest gap in the union of intervals along axis (0 = x, 1 = y). Returns (width, start, end)."""
    order = sorted(ids, key=lambda i: (boxes[i][axis], i))
    best = None
    reach = boxes[order[0]][axis + 2]
    for i in order[1:]:
        a0 = boxes[i][axis]
        if a0 > reach:
            g = a0 - reach
            if best is None or g > best[0]:
                best = (g, reach, a0)
        reach = max(reach, boxes[i][axis + 2])
    return best


def _xycut(ids, boxes, mh, opt, blocks):
    if len(ids) > 1:
        g = _widest_gap(ids, boxes, 0)
        if g is not None and g[0] >= opt['col_gap'] * mh:
            left = [i for i in ids if boxes[i][2] <= g[1]]
            right = [i for i in ids if boxes[i][0] >= g[2]]
            _xycut(left, boxes, mh, opt, blocks)
            _xycut(right, boxes, mh, opt, blocks)
            return
        g = _widest_gap(ids, boxes, 1)
        if g is not None and g[0] >= opt['row_gap'] * mh:
            top = [i for i in ids if boxes[i][3] <= g[1]]
            bottom = [i for i in ids if boxes[i][1] >= g[2]]
            _xycut(top, boxes, mh, opt, blocks)
            _xycut(bottom, boxes, mh, opt, blocks)
            return
    blocks.append(sorted(ids))


def _rows(ids, yd, boxes):
    """Inside a block: group boxes on one row (deskewed centre), rows top to bottom, then left to right."""
    items = sorted(ids, key=lambda i: (yd[i], boxes[i][0], i))
    rows = []  # [sum of yd, count, min height, members]
    for i in items:
        h = boxes[i][3] - boxes[i][1]
        if rows:
            r = rows[-1]
            mean = r[0] / r[1]
            if abs(yd[i] - mean) < 0.5 * min(h, r[2]):
                r[0] += yd[i]
                r[1] += 1
                r[2] = min(r[2], h)
                r[3].append(i)
                continue
        rows.append([yd[i], 1, h, [i]])
    out = []
    for r in rows:
        out.extend(sorted(r[3], key=lambda i: (boxes[i][0], i)))
    return out


# ---------------------------------------------------------------- driver

def layout(prob, scale_x, scale_y, page_w, page_h, **overrides):
    opt = dict(DEFAULTS)
    opt.update(overrides)
    p = np.asarray(prob, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError('prob must be a 2-D [H, W] map')
    H, W = p.shape
    q = np.floor(p * 1e6 + 0.5).astype(np.int64)
    mask = p > opt['thresh']

    comps = _components(mask)
    pieces = []
    for c in comps:
        pieces.extend(_split(c, opt))
    pieces = pieces[:opt['max_candidates']]

    lines = []
    for pix in pieces:
        d = _db_box(pix, q, H, W, opt)
        if d is None:
            continue
        score, corners = d
        quad = [(x * scale_x, y * scale_y) for x, y in corners]
        lines.append(_line_from_quad(quad, score, page_w, page_h))
    n_db = len(lines)

    lines.sort(key=lambda L: (L['box'][1], L['box'][0], L['box'][3], L['box'][2]))
    lines = _merge_lines(lines, opt)
    lines.sort(key=lambda L: (L['box'][1], L['box'][0], L['box'][3], L['box'][2]))
    n = len(lines)

    boxes = [L['box'] for L in lines]
    slopes = []
    for L in lines:
        p0, p1 = L['quad'][0], L['quad'][1]
        dx = p1[0] - p0[0]
        if abs(dx) > 1e-9:
            slopes.append((p1[1] - p0[1]) / dx)
    skew = _median(slopes) if slopes else 0.0
    # line height from the rotated rectangle's short side, so a 1-degree slant does not inflate it
    mh = _median([_dist(L['quad'][1], L['quad'][2]) for L in lines]) if n else 0.0
    yd = [((b[1] + b[3]) / 2) - skew * ((b[0] + b[2]) / 2) for b in boxes]

    blocks = []
    if n:
        _xycut(list(range(n)), boxes, mh, opt, blocks)
    order = []
    block_of = [0] * n
    for bi, b in enumerate(blocks):
        for i in _rows(b, yd, boxes):
            block_of[i] = bi
            order.append(i)

    out_lines = []
    for i, L in enumerate(lines):
        out_lines.append({
            'id': i,
            'box': L['box'],
            'quad': [[round2(x), round2(y)] for x, y in L['quad']],
            'score': round4(L['score']),
            'block': block_of[i],
        })
    return {
        'page': {'w': page_w, 'h': page_h},
        'lines': out_lines,
        'order': order,
        'blocks': blocks,
        'skew': round6(skew),
        'median_h': mh,
        'n_components': len(comps),
        'n_pieces': len(pieces),
        'n_db_boxes': n_db,
    }


def round2(v):
    return math.floor(v * 100 + 0.5) / 100


def round4(v):
    return math.floor(v * 10000 + 0.5) / 10000


def round6(v):
    return math.floor(v * 1000000 + 0.5) / 1000000
