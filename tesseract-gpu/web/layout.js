/*
 * Text-line boxes and reading order from a text-probability map, without OpenCV.
 * Line-by-line port of tessgpu/layout.py. Same integer boxes and same order on the same map.
 *
 *   layout(prob, H, W, scaleX, scaleY, pageW, pageH, opts) -> {lines, order, blocks, ...}
 *     prob            Float32Array or Array of length H*W, values 0..1, at the detector's input scale
 *     scaleX, scaleY  page pixels per map pixel
 *     pageW, pageH    page size in pixels
 *
 * Only + - * / and Math.sqrt on doubles are used (no trig, no libraries), and probability sums are
 * exact integers (probabilities quantised to 1e-6), so Python and JS agree bit for bit.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  else root.TessLayout = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const Q = 1000000;

  const DEFAULTS = {
    thresh: 0.2,
    box_thresh: 0.4,
    unclip_ratio: 1.4,
    max_candidates: 3000,
    min_size: 3,
    split_low: 0.1,
    split_gap: 1.5,
    merge_overlap: 0.6,
    merge_gap: 1.0,
    merge_hratio: 1.5,
    col_gap: 1.5,
    row_gap: 0.6,
  };

  function cmpPt(a, b) {
    return (a[0] - b[0]) || (a[1] - b[1]);
  }

  // Sorted, de-duplicated points: the same set Python's sorted(set(pts)) gives.
  function uniqSorted(pts) {
    const p = pts.slice().sort(cmpPt);
    const out = [];
    for (let i = 0; i < p.length; i++) {
      if (i === 0 || cmpPt(p[i - 1], p[i]) !== 0) out.push(p[i]);
    }
    return out;
  }

  function hull(pts) {
    const p = uniqSorted(pts);
    if (p.length < 3) return p;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const q of p) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
      lower.push(q);
    }
    const upper = [];
    for (let i = p.length - 1; i >= 0; i--) {
      const q = p[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
      upper.push(q);
    }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  function extents(h, ux, uy) {
    const vx = -uy, vy = ux;
    let smin = Infinity, smax = -Infinity, tmin = Infinity, tmax = -Infinity;
    for (const pt of h) {
      const s = pt[0] * ux + pt[1] * uy;
      const t = pt[0] * vx + pt[1] * vy;
      if (s < smin) smin = s;
      if (s > smax) smax = s;
      if (t < tmin) tmin = t;
      if (t > tmax) tmax = t;
    }
    return [smin, smax, tmin, tmax];
  }

  function rectOn(h, ux, uy) {
    let e = extents(h, ux, uy);
    if (e[3] - e[2] > e[1] - e[0]) {
      const nx = -uy, ny = ux;
      ux = nx; uy = ny;
    }
    if (ux < 0 || (ux === 0 && uy < 0)) {
      ux = -ux; uy = -uy;
    }
    e = extents(h, ux, uy);
    const vx = -uy, vy = ux;
    const smid = (e[0] + e[1]) / 2;
    const tmid = (e[2] + e[3]) / 2;
    const cx = smid * ux + tmid * vx;
    const cy = smid * uy + tmid * vy;
    const W = e[1] - e[0];
    const H = e[3] - e[2];
    return { cx, cy, ux, uy, W, H, area: W * H };
  }

  function minRect(h) {
    let best = null;
    const n = h.length;
    for (let i = 0; i < n; i++) {
      const a = h[i], b = h[(i + 1) % n];
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const L = Math.sqrt(dx * dx + dy * dy);
      const r = rectOn(h, dx / L, dy / L);
      if (best === null || r.area < best.area) best = r;
    }
    return best;
  }

  function corners(cx, cy, ux, uy, W, H) {
    const vx = -uy, vy = ux;
    const ax = (W / 2) * ux, ay = (W / 2) * uy;
    const bx = (H / 2) * vx, by = (H / 2) * vy;
    return [
      [cx - ax - bx, cy - ay - by],
      [cx + ax - bx, cy + ay - by],
      [cx + ax + bx, cy + ay + by],
      [cx - ax + bx, cy - ay + by],
    ];
  }

  function dist(p, q) {
    const dx = q[0] - p[0], dy = q[1] - p[1];
    return Math.sqrt(dx * dx + dy * dy);
  }

  function rnd(v) { return Math.floor(v + 0.5); }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

  function median(vals) {
    const s = vals.slice().sort((a, b) => a - b);
    const n = s.length;
    if (n % 2) return s[(n - 1) / 2];
    return (s[n / 2 - 1] + s[n / 2]) / 2;
  }

  // 8-connected components in raster order of first pixel; pixels sorted (y, x).
  function components(mask, H, W) {
    const seen = new Uint8Array(H * W);
    const comps = [];
    for (let start = 0; start < H * W; start++) {
      if (!mask[start] || seen[start]) continue;
      seen[start] = 1;
      const stack = [start];
      const pix = [];
      while (stack.length) {
        const k = stack.pop();
        const y = Math.floor(k / W), x = k - y * W;
        pix.push([x, y]);
        for (let yy = y - 1; yy <= y + 1; yy++) {
          if (yy < 0 || yy >= H) continue;
          for (let xx = x - 1; xx <= x + 1; xx++) {
            if (xx < 0 || xx >= W) continue;
            const kk = yy * W + xx;
            if (mask[kk] && !seen[kk]) {
              seen[kk] = 1;
              stack.push(kk);
            }
          }
        }
      }
      pix.sort((a, b) => (a[1] - b[1]) || (a[0] - b[0]));
      comps.push(pix);
    }
    return comps;
  }

  function split(pix, opt) {
    if (pix.length < 3) return [];
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of pix) {
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    const hc = y1 - y0 + 1;
    const n = x1 - x0 + 1;
    const cnt = new Array(n).fill(0);
    for (const [x] of pix) cnt[x - x0] += 1;
    let best = null;
    let i = 1;
    while (i < n - 1) {
      if (cnt[i] <= opt.split_low * hc) {
        let j = i;
        while (j + 1 < n - 1 && cnt[j + 1] <= opt.split_low * hc) j += 1;
        const g = j - i + 1;
        if (g >= opt.split_gap * hc && (best === null || g > best[0])) best = [g, i, j];
        i = j + 1;
      } else {
        i += 1;
      }
    }
    if (best === null) return [pix];
    const a = best[1], b = best[2];
    const left = pix.filter((p) => p[0] - x0 < a);
    const right = pix.filter((p) => p[0] - x0 > b);
    return split(left, opt).concat(split(right, opt));
  }

  function score(q, H, W, r) {
    const cs = corners(r.cx, r.cy, r.ux, r.uy, r.W, r.H);
    let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
    for (const c of cs) {
      if (c[0] < mnx) mnx = c[0];
      if (c[0] > mxx) mxx = c[0];
      if (c[1] < mny) mny = c[1];
      if (c[1] > mxy) mxy = c[1];
    }
    const x0 = Math.max(0, Math.floor(mnx));
    const x1 = Math.min(W - 1, Math.ceil(mxx));
    const y0 = Math.max(0, Math.floor(mny));
    const y1 = Math.min(H - 1, Math.ceil(mxy));
    if (x1 < x0 || y1 < y0) return null;
    const ux = r.ux, uy = r.uy, vx = -uy, vy = ux;
    const hw = r.W / 2, hh = r.H / 2;
    let n = 0, S = 0;
    for (let y = y0; y <= y1; y++) {
      const dy = y - r.cy;
      for (let x = x0; x <= x1; x++) {
        const dx = x - r.cx;
        const s = dx * ux + dy * uy;
        const t = dx * vx + dy * vy;
        if (Math.abs(s) <= hw && Math.abs(t) <= hh) {
          n += 1;
          S += q[y * W + x];
        }
      }
    }
    if (n === 0) return null;
    return S / (n * Q);
  }

  function dbBox(pix, q, H, W, opt) {
    const h = hull(pix);
    if (h.length < 3) return null;
    const r = minRect(h);
    if (Math.min(r.W, r.H) < opt.min_size) return null;
    const sc = score(q, H, W, r);
    if (sc === null || sc < opt.box_thresh) return null;
    const d = r.W * r.H * opt.unclip_ratio / (2 * (r.W + r.H));
    const We = r.W + 2 * d;
    const He = r.H + 2 * d;
    if (Math.min(We, He) < opt.min_size + 2) return null;
    return { score: sc, corners: corners(r.cx, r.cy, r.ux, r.uy, We, He) };
  }

  function lineFromQuad(quad, sc, pw, ph) {
    let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
    for (const p of quad) {
      if (p[0] < mnx) mnx = p[0];
      if (p[0] > mxx) mxx = p[0];
      if (p[1] < mny) mny = p[1];
      if (p[1] > mxy) mxy = p[1];
    }
    const box = [
      clamp(rnd(mnx), 0, pw),
      clamp(rnd(mny), 0, ph),
      clamp(rnd(mxx), 0, pw),
      clamp(rnd(mxy), 0, ph),
    ];
    return { box, quad, score: sc };
  }

  function mergeable(a, b, opt) {
    const ha = a[3] - a[1], hb = b[3] - b[1];
    const hmin = Math.min(ha, hb), hmax = Math.max(ha, hb);
    if (hmin <= 0) return false;
    const overlap = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
    if (overlap < opt.merge_overlap * hmin) return false;
    if (hmax > opt.merge_hratio * hmin) return false;
    const gap = Math.max(a[0], b[0]) - Math.min(a[2], b[2]);
    return gap <= opt.merge_gap * hmin;
  }

  function mergeLines(lines, opt) {
    const n = lines.length;
    const parent = [];
    for (let i = 0; i < n; i++) parent.push(i);
    const find = (i) => {
      while (parent[i] !== i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
      }
      return i;
    };
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (mergeable(lines[i].box, lines[j].box, opt)) {
          const ri = find(i), rj = find(j);
          if (ri !== rj) parent[Math.max(ri, rj)] = Math.min(ri, rj);
        }
      }
    }
    const groups = new Map();
    for (let i = 0; i < n; i++) {
      const r = find(i);
      if (!groups.has(r)) groups.set(r, []);
      groups.get(r).push(i);
    }
    const roots = Array.from(groups.keys()).sort((a, b) => a - b);
    const merged = [];
    for (const root of roots) {
      const mem = groups.get(root);
      if (mem.length === 1) {
        merged.push(lines[mem[0]]);
        continue;
      }
      let pts = [];
      for (const i of mem) pts = pts.concat(lines[i].quad);
      const r = minRect(hull(pts));
      const quad = corners(r.cx, r.cy, r.ux, r.uy, r.W, r.H);
      let s = 0;
      for (const i of mem) s += lines[i].score;
      s = s / mem.length;
      const box = [
        Math.min(...mem.map((i) => lines[i].box[0])),
        Math.min(...mem.map((i) => lines[i].box[1])),
        Math.max(...mem.map((i) => lines[i].box[2])),
        Math.max(...mem.map((i) => lines[i].box[3])),
      ];
      merged.push({ box, quad, score: s });
    }
    return merged;
  }

  function widestGap(ids, boxes, axis) {
    const order = ids.slice().sort((a, b) => (boxes[a][axis] - boxes[b][axis]) || (a - b));
    let best = null;
    let reach = boxes[order[0]][axis + 2];
    for (let k = 1; k < order.length; k++) {
      const i = order[k];
      const a0 = boxes[i][axis];
      if (a0 > reach) {
        const g = a0 - reach;
        if (best === null || g > best[0]) best = [g, reach, a0];
      }
      if (boxes[i][axis + 2] > reach) reach = boxes[i][axis + 2];
    }
    return best;
  }

  function xycut(ids, boxes, mh, opt, blocks) {
    if (ids.length > 1) {
      let g = widestGap(ids, boxes, 0);
      if (g !== null && g[0] >= opt.col_gap * mh) {
        const left = ids.filter((i) => boxes[i][2] <= g[1]);
        const right = ids.filter((i) => boxes[i][0] >= g[2]);
        xycut(left, boxes, mh, opt, blocks);
        xycut(right, boxes, mh, opt, blocks);
        return;
      }
      g = widestGap(ids, boxes, 1);
      if (g !== null && g[0] >= opt.row_gap * mh) {
        const top = ids.filter((i) => boxes[i][3] <= g[1]);
        const bottom = ids.filter((i) => boxes[i][1] >= g[2]);
        xycut(top, boxes, mh, opt, blocks);
        xycut(bottom, boxes, mh, opt, blocks);
        return;
      }
    }
    blocks.push(ids.slice().sort((a, b) => a - b));
  }

  function rows(ids, yd, boxes) {
    const items = ids.slice().sort((a, b) => (yd[a] - yd[b]) || (boxes[a][0] - boxes[b][0]) || (a - b));
    const rs = []; // [sum of yd, count, min height, members]
    for (const i of items) {
      const h = boxes[i][3] - boxes[i][1];
      if (rs.length) {
        const r = rs[rs.length - 1];
        const mean = r[0] / r[1];
        if (Math.abs(yd[i] - mean) < 0.5 * Math.min(h, r[2])) {
          r[0] += yd[i];
          r[1] += 1;
          r[2] = Math.min(r[2], h);
          r[3].push(i);
          continue;
        }
      }
      rs.push([yd[i], 1, h, [i]]);
    }
    const out = [];
    for (const r of rs) {
      out.push(...r[3].slice().sort((a, b) => (boxes[a][0] - boxes[b][0]) || (a - b)));
    }
    return out;
  }

  function round2(v) { return Math.floor(v * 100 + 0.5) / 100; }
  function round4(v) { return Math.floor(v * 10000 + 0.5) / 10000; }
  function round6(v) { return Math.floor(v * 1000000 + 0.5) / 1000000; }

  function layout(prob, H, W, scaleX, scaleY, pageW, pageH, overrides) {
    const opt = Object.assign({}, DEFAULTS, overrides || {});
    if (prob.length !== H * W) throw new Error('prob must have length H*W');
    const q = new Float64Array(H * W);
    const mask = new Uint8Array(H * W);
    for (let k = 0; k < H * W; k++) {
      const p = prob[k];
      q[k] = Math.floor(p * 1e6 + 0.5);
      mask[k] = p > opt.thresh ? 1 : 0;
    }

    const comps = components(mask, H, W);
    let pieces = [];
    for (const c of comps) pieces = pieces.concat(split(c, opt));
    pieces = pieces.slice(0, opt.max_candidates);

    let lines = [];
    for (const pix of pieces) {
      const d = dbBox(pix, q, H, W, opt);
      if (d === null) continue;
      const quad = d.corners.map(([x, y]) => [x * scaleX, y * scaleY]);
      lines.push(lineFromQuad(quad, d.score, pageW, pageH));
    }
    const nDb = lines.length;

    const byBox = (A, B) => (A.box[1] - B.box[1]) || (A.box[0] - B.box[0]) || (A.box[3] - B.box[3]) || (A.box[2] - B.box[2]);
    lines.sort(byBox);
    lines = mergeLines(lines, opt);
    lines.sort(byBox);
    const n = lines.length;

    const boxes = lines.map((L) => L.box);
    const slopes = [];
    for (const L of lines) {
      const p0 = L.quad[0], p1 = L.quad[1];
      const dx = p1[0] - p0[0];
      if (Math.abs(dx) > 1e-9) slopes.push((p1[1] - p0[1]) / dx);
    }
    const skew = slopes.length ? median(slopes) : 0.0;
    // line height from the rotated rectangle's short side, so a 1-degree slant does not inflate it
    const mh = n ? median(lines.map((L) => dist(L.quad[1], L.quad[2]))) : 0.0;
    const yd = boxes.map((b) => ((b[1] + b[3]) / 2) - skew * ((b[0] + b[2]) / 2));

    const blocks = [];
    if (n) xycut(Array.from({ length: n }, (_, i) => i), boxes, mh, opt, blocks);
    const order = [];
    const blockOf = new Array(n).fill(0);
    blocks.forEach((b, bi) => {
      for (const i of rows(b, yd, boxes)) {
        blockOf[i] = bi;
        order.push(i);
      }
    });

    const outLines = lines.map((L, i) => ({
      id: i,
      box: L.box,
      quad: L.quad.map(([x, y]) => [round2(x), round2(y)]),
      score: round4(L.score),
      block: blockOf[i],
    }));
    return {
      page: { w: pageW, h: pageH },
      lines: outLines,
      order,
      blocks,
      skew: round6(skew),
      median_h: mh,
      n_components: comps.length,
      n_pieces: pieces.length,
      n_db_boxes: nDb,
    };
  }

  return { layout, DEFAULTS };
});
