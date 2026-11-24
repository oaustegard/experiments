// Page pipeline around the PP-OCRv6 detector and recognizer: image in, text lines out.
//
//   import { TextPipeline } from "./pipeline.js";
//   const pipe = new TextPipeline({ det, rec });          // det: DetNet (web/det/engine.js), rec: see below
//   const lines = await pipe.run(canvasOrImageData);      // [{ box, text, score, detScore }], reading order
//   // box: four [x, y] page pixels, clockwise from the top-left corner.
//
// Stages, each one PaddleOCR's own (run/ref_pipeline.py is the Python reference and the check):
//   det.run             probability map of the page (long side 1280, zero padded to multiples of 32)
//   dbPostProcess       DBPostProcess: binarize at thresh, contours, min-area rectangle, fast box score,
//                       unclip by unclip_ratio (Clipper's round offset), min size, box_thresh
//   filterBoxes         order_points_clockwise, clip to the page, drop boxes of 3 px or less
//   sortedBoxes         top to bottom, then left to right within 10 px
//   cropQuad            get_rotate_crop_image: perspective warp (bicubic), rotate 90 degrees when h / w >= 1.5
//   recInput            resize_norm_img: 48 px high, BGR, (x / 255 - 0.5) / 0.5, zero pad right to max(320, w)
//   rec.run / rec.decode
//
// The recognizer interface (web/rec/engine.js RecNet has it):
//   await rec.run(x, B, W)    x: Float32Array [B, 3, 48, W]  ->  { B, T, V, probs }   probs: Float32Array [B, T, V]
//   rec.decode(probs, T, V, B)   -> B entries, each a string or { text, score }
// score is the line's recognizer confidence. When decode gives plain strings the pipeline computes it from
// probs as PaddleOCR does (mean of the best probability over the steps that emit a character).
//
// Nothing here touches the DOM except toRGBA() for canvas input, so the geometry runs in Node.
// The detector's probability map is the only GPU result; thresholds come from det.post (the model's
// inference.yml, copied into the manifest by run/export_det.py) and can be overridden in options.post.

const REC_H = 48, REC_MIN_W = 320;

// ------------------------------------------------------------------ image input

// RGBA bytes of an ImageData-like { data, width, height }, a canvas, an ImageBitmap or an <img>.
export function toRGBA(image) {
  if (image && image.data && image.width && image.height) {
    if (image.data.length !== 4 * image.width * image.height) throw new Error("image data is not RGBA");
    return { data: image.data, width: image.width, height: image.height };
  }
  const w = image.width || image.naturalWidth, h = image.height || image.naturalHeight;
  if (!w || !h) throw new Error("unsupported image input");
  const cv = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(w, h) : Object.assign(document.createElement("canvas"), { width: w, height: h });
  const ctx = cv.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  const d = ctx.getImageData(0, 0, w, h);
  return { data: d.data, width: w, height: h };
}

// ------------------------------------------------------------------ contours

// cv2.findContours(RETR_LIST, CHAIN_APPROX_SIMPLE) for a binary mask: Suzuki and Abe's border following,
// outer borders and hole borders both. Calls onContour(points) once per border in raster order of its
// start pixel, with the points OpenCV's contour array holds: the border pixels where the direction of the
// walk changes, starting at the border's first pixel. (cv2 lists the contours last found first; the
// caller reverses.)
export function findContours(mask, W, H, onContour) {
  const S = W + 2;
  const f = new Int32Array(S * (H + 2));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (mask[y * W + x]) f[(y + 1) * S + x + 1] = 1;
  // neighbours clockwise (y down) starting north: N NE E SE S SW W NW
  const DX = [0, 1, 1, 1, 0, -1, -1, -1], DY = [-1, -1, 0, 1, 1, 1, 0, -1];
  const OFF = DY.map((dy, k) => dy * S + DX[k]);
  const dirOf = (d) => { for (let k = 0; k < 8; k++) if (OFF[k] === d) return k; return -1; };
  let nbd = 1;
  for (let i = 1; i <= H; i++) {
    for (let j = 1; j <= W; j++) {
      const p = i * S + j, v = f[p];
      if (v === 0) continue;
      let from;
      if (v === 1 && f[p - 1] === 0) from = 6;          // outer border start; the zero pixel is west
      else if (v >= 1 && f[p + 1] === 0) from = 2;      // hole border start; the zero pixel is east
      else continue;
      nbd++;
      // 3.1: clockwise from the zero neighbour, first non-zero neighbour (i1, j1)
      let first = -1;
      for (let t = 1; t <= 8; t++) {
        const k = (from + t) & 7;
        if (f[p + OFF[k]] !== 0) { first = p + OFF[k]; break; }
      }
      if (first < 0) { f[p] = -nbd; onContour([[j - 1, i - 1]]); continue; }
      const cells = [];
      let p2 = first, p3 = p;
      for (;;) {
        // 3.3: counterclockwise from the neighbour after p2, first non-zero neighbour p4
        const d2 = dirOf(p2 - p3);
        let p4 = -1, eastZero = false;
        for (let t = 1; t <= 8; t++) {
          const k = (d2 - t + 8) & 7;
          const q = p3 + OFF[k];
          if (f[q] !== 0) { p4 = q; break; }
          if (k === 2) eastZero = true;
        }
        // 3.4
        if (eastZero) f[p3] = -nbd;
        else if (f[p3] === 1) f[p3] = nbd;
        cells.push(p3);
        // 3.5
        if (p4 === p && p3 === first) break;
        p2 = p3; p3 = p4;
      }
      onContour(chainSimple(cells, S));
    }
  }
}

// CHAIN_APPROX_SIMPLE: keep the cells where the step direction (to the previous cell, to the next) changes.
function chainSimple(cells, S) {
  const n = cells.length;
  const pt = (c) => [(c % S) - 1, Math.floor(c / S) - 1];
  if (n < 3) return cells.map(pt);
  const out = [];
  for (let i = 0; i < n; i++) {
    const a = cells[(i + n - 1) % n], b = cells[i], c = cells[(i + 1) % n];
    if (b - a !== c - b) out.push(pt(b));
  }
  return out;
}

// ------------------------------------------------------------------ convex hull (cv2.convexHull)

const sgn = (v) => (v > 0) - (v < 0);

// Sklansky's scan over the sorted points P[start..end] (OpenCV's Sklansky_); the hull indices go to
// stack[off ...]; returns their count.
function sklansky(P, start, end, stack, off, nsign, sign2) {
  const incr = end > start ? 1 : -1;
  let pprev = start, pcur = pprev + incr, pnext = pcur + incr, size = 3;
  if (start === end || (P[start][0] === P[end][0] && P[start][1] === P[end][1])) { stack[off] = start; return 1; }
  stack[off] = pprev; stack[off + 1] = pcur; stack[off + 2] = pnext;
  end += incr;
  while (pnext !== end) {
    const cury = P[pcur][1], by = P[pnext][1] - cury;
    if (sgn(by) !== nsign) {
      const ax = P[pcur][0] - P[pprev][0], ay = cury - P[pprev][1], bx = P[pnext][0] - P[pcur][0];
      const convexity = ay * bx - ax * by;
      if (sgn(convexity) === sign2 && (ax !== 0 || ay !== 0)) {
        pprev = pcur; pcur = pnext; pnext += incr;
        stack[off + size++] = pnext;
      } else if (pprev === start) {
        pcur = pnext; stack[off + 1] = pcur; pnext += incr; stack[off + 2] = pnext;
      } else {
        stack[off + size - 2] = pnext;
        pcur = pprev; pprev = stack[off + size - 4]; size--;
      }
    } else {
      pnext += incr;
      stack[off + size - 1] = pnext;
    }
  }
  return size - 1;
}

// cv2.convexHull(points, clockwise=False) for integer points, port of OpenCV's convexHull_: the same
// hull vertices in the same order, including the cyclic shift that makes the indices ascend or descend.
// minAreaRect's float32 arithmetic depends on where the hull starts, so the order matters here.
export function convexHull(pts) {
  const total = pts.length;
  if (total === 0) return [];
  const order = Array.from({ length: total }, (_, i) => i).sort((a, b) => pts[a][0] - pts[b][0] || pts[a][1] - pts[b][1] || a - b);
  const P = order.map((i) => pts[i]);
  let miny = 0, maxy = 0;
  for (let i = 1; i < total; i++) {
    if (P[miny][1] > P[i][1]) miny = i;
    if (P[maxy][1] < P[i][1]) maxy = i;
  }
  let hull = [];
  const stack = new Int32Array(total + 2);
  if (P[0][0] === P[total - 1][0] && P[0][1] === P[total - 1][1]) return [pts[order[0]]];
  let tlOff = 0, tlCount = sklansky(P, 0, maxy, stack, tlOff, -1, 1);
  let trOff = tlCount, trCount = sklansky(P, total - 1, maxy, stack, trOff, -1, -1);
  [tlOff, trOff] = [trOff, tlOff]; [tlCount, trCount] = [trCount, tlCount];     // counterclockwise
  for (let i = 0; i < tlCount - 1; i++) hull.push(stack[tlOff + i]);
  for (let i = trCount - 1; i > 0; i--) hull.push(stack[trOff + i]);
  const stopIdx = trCount > 2 ? stack[trOff + 1] : tlCount > 2 ? stack[tlOff + tlCount - 2] : -1;
  const blOff = 0, blCount0 = sklansky(P, 0, miny, stack, blOff, 1, -1);
  const brOff = blCount0, brCount0 = sklansky(P, total - 1, miny, stack, brOff, 1, 1);
  let blCount = blCount0, brCount = brCount0;
  if (stopIdx >= 0) {
    const check = blCount > 2 ? stack[blOff + 1] : blCount + brCount > 2 ? stack[brOff + 2 - blCount] : -1;
    if (check === stopIdx || (check >= 0 && P[check][0] === P[stopIdx][0] && P[check][1] === P[stopIdx][1])) {
      blCount = Math.min(blCount, 2); brCount = Math.min(brCount, 2);
    }
  }
  for (let i = 0; i < blCount - 1; i++) hull.push(stack[blOff + i]);
  for (let i = brCount - 1; i > 0; i--) hull.push(stack[brOff + i]);
  const nout = hull.length;
  hull = hull.map((h) => order[h]);              // indices into the caller's array
  if (nout >= 3) {
    let minIdx = 0, maxIdx = 0, lt = 0;
    for (let i = 1; i < nout; i++) {
      const idx = hull[i];
      lt += hull[i - 1] < idx ? 1 : 0;
      if (lt > 1 && lt <= i - 2) break;
      if (idx < hull[minIdx]) minIdx = i;
      if (idx > hull[maxIdx]) maxIdx = i;
    }
    const mmdist = Math.abs(maxIdx - minIdx);
    if ((mmdist === 1 || mmdist === nout - 1) && (lt <= 1 || lt >= nout - 2)) {
      const ascending = (maxIdx + 1) % nout === minIdx;
      const i0 = ascending ? minIdx : maxIdx;
      if (i0 > 0) {
        const tmp = [];
        let j = i0, ok = true;
        for (let i = 0; i < nout; i++) {
          const curr = hull[j];
          tmp.push(curr);
          const nextJ = j + 1 < nout ? j + 1 : 0;
          if (i < nout - 1 && ascending !== (curr < hull[nextJ])) { ok = false; break; }
          j = nextJ;
        }
        if (ok) hull = tmp;
      }
    }
  }
  return hull.map((i) => pts[i]);
}

// ------------------------------------------------------------------ min-area rectangle (cv2.minAreaRect)

const fr = Math.fround;

// OpenCV's rotatingCalipers for CALIPERS_MINAREARECT on a counterclockwise hull, in float32 like the C++
// (every operation rounded with Math.fround). Returns [corner, vector1, vector2] as float pairs.
function rotatingCalipers(P) {
  const n = P.length;
  let left = 0, bottom = 0, right = 0, top = 0;
  let px = P[0][0], py = P[0][1];
  let leftX = px, rightX = px, topY = py, bottomY = py;
  const vx = new Float64Array(n), vy = new Float64Array(n), inv = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    if (px < leftX) { leftX = px; left = i; }
    if (px > rightX) { rightX = px; right = i; }
    if (py > topY) { topY = py; top = i; }
    if (py < bottomY) { bottomY = py; bottom = i; }
    const qx = P[(i + 1) % n][0], qy = P[(i + 1) % n][1];
    const dx = qx - px, dy = qy - py;
    vx[i] = fr(dx); vy[i] = fr(dy); inv[i] = fr(1 / Math.sqrt(dx * dx + dy * dy));
    px = qx; py = qy;
  }
  let baseA = 1, baseB = 0;
  const seq = [bottom, right, top, left];
  let minArea = 3.4028234663852886e38, best = null;
  // rotated vectors of the four calipers' edges: as is, 90 degrees clockwise, 180, 90 counterclockwise
  const rot = (m, s) => (m === 0 ? [vx[s], vy[s]] : m === 1 ? [vy[s], -vx[s]] : m === 2 ? [-vx[s], -vy[s]] : [-vy[s], vx[s]]);
  const isRight = (a, b) => fr(fr(a[1] * b[0]) + fr(-a[0] * b[1])) < 0;   // a rotated 90 degrees clockwise, dotted with b
  for (let k = 0; k < n; k++) {
    let main = 0;
    const rv = [rot(0, seq[0]), rot(1, seq[1]), rot(2, seq[2]), rot(3, seq[3])];
    for (let i = 1; i < 4; i++) if (isRight(rv[i], rv[main])) main = i;
    const pi = seq[main];
    const lx = fr(vx[pi] * inv[pi]), ly = fr(vy[pi] * inv[pi]);
    if (main === 0) { baseA = lx; baseB = ly; }
    else if (main === 1) { baseA = ly; baseB = -lx; }
    else if (main === 2) { baseA = -lx; baseB = -ly; }
    else { baseA = -ly; baseB = lx; }
    seq[main] = seq[main] + 1 === n ? 0 : seq[main] + 1;
    let dx = P[seq[1]][0] - P[seq[3]][0], dy = P[seq[1]][1] - P[seq[3]][1];
    const width = fr(fr(dx * baseA) + fr(dy * baseB));
    dx = P[seq[2]][0] - P[seq[0]][0]; dy = P[seq[2]][1] - P[seq[0]][1];
    const height = fr(fr(-dx * baseB) + fr(dy * baseA));
    const area = fr(width * height);
    if (area <= minArea) { minArea = area; best = [seq[3], baseA, width, baseB, height, seq[0]]; }
  }
  const [li, A1, width, B1, height, bi] = best;
  const A2 = -B1, B2 = A1;
  const C1 = fr(fr(A1 * P[li][0]) + fr(P[li][1] * B1));
  const C2 = fr(fr(A2 * P[bi][0]) + fr(P[bi][1] * B2));
  const idet = fr(1 / fr(fr(A1 * B2) - fr(A2 * B1)));
  const px0 = fr(fr(fr(C1 * B2) - fr(C2 * B1)) * idet), py0 = fr(fr(fr(A1 * C2) - fr(A2 * C1)) * idet);
  return [[px0, py0], [fr(A1 * width), fr(B1 * width)], [fr(A2 * height), fr(B2 * height)]];
}

// cv2.minAreaRect then cv2.boxPoints for a contour of integer points: { corners, sside }, where corners are
// the rectangle's four corners (float32 values, OpenCV's order) and sside the shorter side; no corners
// and sside 0 when the hull is a point or a segment. Port of OpenCV 4.x / 5.x (float32 arithmetic included,
// so the corners carry the same rounding noise: 592.99994 where the exact answer is 593).
export function minAreaRect(contour) {
  const hull = convexHull(contour);
  if (hull.length < 3) return { corners: null, sside: 0 };
  const [c, o1, o2] = rotatingCalipers(hull);
  const cx = fr(c[0] + fr(fr(o1[0] + o2[0]) * 0.5)), cy = fr(c[1] + fr(fr(o1[1] + o2[1]) * 0.5));
  let w = fr(Math.sqrt(o2[0] * o2[0] + o2[1] * o2[1])), h = fr(Math.sqrt(o1[0] * o1[0] + o1[1] * o1[1]));
  let angle = -Math.PI / 2;
  if (o1[0] === 0 && o1[1] > 0) [w, h] = [h, w];
  else angle = -Math.atan2(o1[0], o1[1]);
  const deg = fr(angle * 180 / Math.PI);
  const rad = deg * Math.PI / 180;                       // RotatedRect::points
  const b = fr(fr(Math.cos(rad)) * 0.5), a = fr(fr(Math.sin(rad)) * 0.5);
  const ah = fr(a * h), aw = fr(a * w), bh = fr(b * h), bw = fr(b * w);
  return {
    corners: [[fr(fr(cx - ah) - bw), fr(fr(cy + bh) - aw)], [fr(fr(cx + ah) - bw), fr(fr(cy - bh) - aw)],
      [fr(fr(cx + ah) + bw), fr(fr(cy - bh) + aw)], [fr(fr(cx - ah) + bw), fr(fr(cy + bh) + aw)]],
    sside: Math.min(w, h),
  };
}

// PaddleOCR's get_mini_boxes: the rectangle's corners as [top-left, top-right, bottom-right, bottom-left]
// (by x, then by y within each side) and its shorter side.
export function getMiniBoxes(contour) {
  const r = minAreaRect(contour);
  if (!r.corners) return { box: null, sside: 0 };
  const p = r.corners.slice().sort((a, b) => a[0] - b[0]);
  const [i1, i4] = p[1][1] > p[0][1] ? [0, 1] : [1, 0];
  const [i2, i3] = p[3][1] > p[2][1] ? [2, 3] : [3, 2];
  return { box: [p[i1], p[i2], p[i3], p[i4]], sside: r.sside };
}

// ------------------------------------------------------------------ box score (cv2.fillPoly port)

const XY_SHIFT = 16, XY_ONE = 1 << 16;

// OpenCV's 8-connected line iterator (leftToRight): sets mask pixels along (x0, y0)-(x1, y1).
function drawLine(mask, mw, mh, x0, y0, x1, y1) {
  let dx = x1 - x0, dy = y1 - y0;
  if (dx < 0) { [x0, y0, x1, y1] = [x1, y1, x0, y0]; dx = -dx; dy = -dy; }
  const sy = dy >= 0 ? 1 : -1;
  dy = Math.abs(dy);
  const majorX = dx >= dy;
  if (!majorX) { const t = dx; dx = dy; dy = t; }
  let err = dx - 2 * dy, x = x0, y = y0;
  for (let c = dx + 1; c > 0; c--) {
    if (x >= 0 && x < mw && y >= 0 && y < mh) mask[y * mw + x] = 1;
    const m = err < 0;
    err += -2 * dy + (m ? 2 * dx : 0);
    if (majorX) { x++; if (m) y += sy; } else { y += sy; if (m) x++; }
  }
}

// cv2.fillPoly(mask, [poly], 1) for integer vertices: the polygon's edges drawn with the line iterator,
// plus the span from ceil(x_left) to floor(x_right) on every scan line between pairs of edges (x in
// 16.16 fixed point, slope truncated), as OpenCV's FillEdgeCollection does. Exact for polygons inside the
// mask; OpenCV clips edges that leave it before drawing, this draws only the pixels that fall inside.
export function fillPoly(poly, mw, mh) {
  const mask = new Uint8Array(mw * mh);
  const n = poly.length;
  const edges = [];
  let [px, py] = poly[n - 1];
  for (let i = 0; i < n; i++) {
    const [qx, qy] = poly[i];
    drawLine(mask, mw, mh, px, py, qx, qy);
    if (py !== qy) {
      const x0 = px * XY_ONE, x1 = qx * XY_ONE;
      const dx = Math.trunc((x1 - x0) / (qy - py));
      edges.push(py < qy ? { y0: py, y1: qy, x: x0, dx } : { y0: qy, y1: py, x: x1, dx });
    }
    px = qx; py = qy;
  }
  if (edges.length >= 2) {
    let ymin = Infinity, ymax = -Infinity;
    for (const e of edges) { if (e.y0 < ymin) ymin = e.y0; if (e.y1 > ymax) ymax = e.y1; }
    ymax = Math.min(ymax, mh);
    for (let y = Math.max(ymin, 0); y < ymax; y++) {
      const xs = [];
      for (const e of edges) if (e.y0 <= y && y < e.y1) xs.push(e.x + (y - e.y0) * e.dx);
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const a = Math.max(Math.floor((xs[k] + XY_ONE - 1) / XY_ONE), 0), b = Math.min(Math.floor(xs[k + 1] / XY_ONE), mw - 1);
        for (let x = a; x <= b; x++) mask[y * mw + x] = 1;
      }
    }
  }
  return mask;
}

// DBPostProcess.box_score_fast: mean of the probability map under the box (cv2.mean with a fillPoly mask).
export function boxScoreFast(prob, W, H, box) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const [x, y] of box) { minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y); }
  const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
  const xmin = clip(Math.floor(minX), W - 1), xmax = clip(Math.ceil(maxX), W - 1);
  const ymin = clip(Math.floor(minY), H - 1), ymax = clip(Math.ceil(maxY), H - 1);
  const mw = xmax - xmin + 1, mh = ymax - ymin + 1;
  const poly = box.map(([x, y]) => [Math.trunc(Math.fround(x - xmin)), Math.trunc(Math.fround(y - ymin))]);
  const mask = fillPoly(poly, mw, mh);
  let sum = 0, cnt = 0;
  for (let y = 0; y < mh; y++) {
    const row = (ymin + y) * W + xmin;
    for (let x = 0; x < mw; x++) if (mask[y * mw + x]) { sum += prob[row + x]; cnt++; }
  }
  return cnt ? sum / cnt : 0;
}

// ------------------------------------------------------------------ unclip (Clipper round offset)

const cround = (v) => (v < 0 ? Math.trunc(v - 0.5) : Math.trunc(v + 0.5));   // Clipper's Round

// ClipperOffset with JoinType round and EndType closed polygon, arc tolerance 0 (Clipper's default of
// 0.25 px), for one polygon of integer points; returns the offset polygon's points. A port of the
// DoOffset / OffsetPoint / DoRound code in Clipper 6.4, without the final union: for the convex
// quadrilaterals DBPostProcess offsets the polygon is simple, and the union leaves its vertices as they are.
export function offsetRound(path, delta) {
  let hi = path.length - 1;
  while (hi > 0 && path[0][0] === path[hi][0] && path[0][1] === path[hi][1]) hi--;
  let pts = [path[0]];
  for (let i = 1; i <= hi; i++) {
    const l = pts[pts.length - 1];
    if (l[0] !== path[i][0] || l[1] !== path[i][1]) pts.push(path[i]);
  }
  if (pts.length < 3) return [];
  let a = 0;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) a += (pts[j][0] + pts[i][0]) * (pts[j][1] - pts[i][1]);
  if (-a * 0.5 < 0) pts = pts.reverse();
  const n = pts.length;
  const normals = pts.map((p, i) => {
    const q = pts[(i + 1) % n], dx = q[0] - p[0], dy = q[1] - p[1];
    const f = dx || dy ? 1 / Math.sqrt(dx * dx + dy * dy) : 0;
    return [dy * f, -dx * f];
  });
  const y = 0.25;
  let steps = Math.PI / Math.acos(1 - y / Math.abs(delta));
  if (steps > Math.abs(delta) * Math.PI) steps = Math.abs(delta) * Math.PI;
  let sin = Math.sin(2 * Math.PI / steps);
  const cos = Math.cos(2 * Math.PI / steps), stepsPerRad = steps / (2 * Math.PI);
  if (delta < 0) sin = -sin;
  const out = [];
  const at = (p, nrm) => [cround(p[0] + nrm[0] * delta), cround(p[1] + nrm[1] * delta)];
  let k = n - 1;
  for (let j = 0; j < n; j++) {
    const nk = normals[k], nj = normals[j];
    let sinA = nk[0] * nj[1] - nj[0] * nk[1];
    if (Math.abs(sinA * delta) < 1) {
      if (nk[0] * nj[0] + nj[1] * nk[1] > 0) { out.push(at(pts[j], nk)); k = j; continue; }
    } else if (sinA > 1) sinA = 1;
    else if (sinA < -1) sinA = -1;
    if (sinA * delta < 0) {
      out.push(at(pts[j], nk), pts[j], at(pts[j], nj));
    } else {
      const ang = Math.atan2(sinA, nk[0] * nj[0] + nk[1] * nj[1]);
      const st = Math.max(cround(stepsPerRad * Math.abs(ang)), 1);
      let X = nk[0], Y = nk[1];
      for (let i = 0; i < st; i++) {
        out.push([cround(pts[j][0] + X * delta), cround(pts[j][1] + Y * delta)]);
        const X2 = X;
        X = X * cos - sin * Y;
        Y = X2 * sin + Y * cos;
      }
      out.push(at(pts[j], nj));
    }
    k = j;
  }
  return out;
}

// DBPostProcess.unclip for a quadrilateral of float corners: polygon area * unclip_ratio / perimeter is the
// offset distance. pyclipper takes the float corners as integers by truncation toward zero; so does this.
export function unclip(box, ratio) {
  let area = 0, len = 0;
  for (let i = 0, j = box.length - 1; i < box.length; j = i++) {
    area += box[j][0] * box[i][1] - box[i][0] * box[j][1];
    len += Math.hypot(box[i][0] - box[j][0], box[i][1] - box[j][1]);
  }
  area = Math.abs(area) / 2;
  const dist = area * ratio / len;
  return offsetRound(box.map(([x, y]) => [Math.trunc(x), Math.trunc(y)]), dist);
}

// ------------------------------------------------------------------ DBPostProcess

// Banker's rounding, as np.round.
function roundHalfEven(v) {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// prob: Float32Array [H, W] (the padded map). Returns [{ quad, score }] with quad four float [x, y]
// map pixels (top-left, top-right, bottom-right, bottom-left), in the order cv2.findContours lists the contours.
export function dbPostProcess(prob, W, H, post) {
  const { thresh, box_thresh: boxThresh, unclip_ratio: unclipRatio, max_candidates: maxCand, min_size: minSize } =
    { min_size: 3, max_candidates: 1000, ...post };
  const mask = new Uint8Array(W * H);
  for (let i = 0; i < mask.length; i++) mask[i] = prob[i] > thresh ? 1 : 0;
  const boxes = [];
  const contours = [];
  findContours(mask, W, H, (pts) => { contours.push(pts); });
  // cv2.findContours hands back the contours last found first; the candidate cap and the order of the
  // boxes follow that list
  for (let c = contours.length - 1, seen = 0; c >= 0 && seen < maxCand; c--, seen++) {
    const { box: pts, sside } = getMiniBoxes(contours[c]);
    if (!pts || sside < minSize) continue;
    const score = boxScoreFast(prob, W, H, pts);
    if (boxThresh > score) continue;
    const poly = unclip(pts, unclipRatio);
    if (!poly.length) continue;
    const { box, sside: s2 } = getMiniBoxes(poly);
    if (!box || s2 < minSize + 2) continue;
    boxes.push({ quad: box, score });
  }
  return boxes;
}

// Map pixels to page pixels as run/ref_pipeline.py: x * srcW / w1 rounded half to even, clipped to the page.
export function toPage(boxes, w1, h1, srcW, srcH) {
  return boxes.map(({ quad, score }) => ({
    quad: quad.map(([x, y]) => [Math.min(Math.max(roundHalfEven(x * srcW / w1), 0), srcW),
      Math.min(Math.max(roundHalfEven(y * srcH / h1), 0), srcH)]),
    score,
  }));
}

// ------------------------------------------------------------------ boxes to ordered quads

export function orderPointsClockwise(pts) {
  const s = pts.map((p) => p[0] + p[1]);
  const argmin = (a) => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);
  const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
  const iMin = argmin(s), iMax = argmax(s);
  const rest = pts.filter((_, i) => i !== iMin && i !== iMax);
  const diff = rest.map((p) => p[1] - p[0]);
  return [pts[iMin], rest[argmin(diff)], pts[iMax], rest[argmax(diff)]];
}

// filter_tag_det_res: clockwise from the top-left, clipped to the page, boxes of 3 px or less dropped.
export function filterBoxes(boxes, srcW, srcH) {
  const out = [];
  for (const { quad, score } of boxes) {
    const q = orderPointsClockwise(quad).map(([x, y]) =>
      [Math.trunc(Math.min(Math.max(x, 0), srcW - 1)), Math.trunc(Math.min(Math.max(y, 0), srcH - 1))]);
    const d = (a, b) => Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
    if (Math.trunc(d(q[0], q[1])) <= 3 || Math.trunc(d(q[0], q[3])) <= 3) continue;
    out.push({ quad: q, score });
  }
  return out;
}

// sorted_boxes: sort by (y, x) of the top-left corner, then swap neighbours that are within 10 px in y
// and out of order in x. Returns the boxes in reading order.
export function sortedBoxes(boxes) {
  const b = boxes.slice().sort((p, q) => p.quad[0][1] - q.quad[0][1] || p.quad[0][0] - q.quad[0][0]);
  for (let i = 0; i < b.length - 1; i++) {
    for (let j = i; j >= 0; j--) {
      if (Math.abs(b[j + 1].quad[0][1] - b[j].quad[0][1]) < 10 && b[j + 1].quad[0][0] < b[j].quad[0][0]) {
        [b[j], b[j + 1]] = [b[j + 1], b[j]];
      } else break;
    }
  }
  return b;
}

// ------------------------------------------------------------------ crop (get_rotate_crop_image)

// Homography h (3 x 3, row-major, h[8] = 1) that maps each of the four src points to its dst point.
export function homography(src, dst) {
  const A = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [X, Y] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -X * x, -X * y, X], [0, 0, 0, x, y, 1, -Y * x, -Y * y, Y]);
  }
  for (let c = 0; c < 8; c++) {
    let p = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 9; k++) A[r][k] -= f * A[c][k];
    }
  }
  return [...A.map((row, i) => row[8] / row[i]), 1];
}

// Bicubic weights of cv2's INTER_CUBIC (A = -0.75) for the 32 source fractions OpenCV quantizes to: row f
// holds the four tap weights (pixels -1, 0, +1, +2) for a fraction of f / 32.
const CUBIC = (() => {
  const A = -0.75, t = new Float64Array(32 * 4);
  for (let f = 0; f < 32; f++) {
    const x = f / 32;
    t[4 * f] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    t[4 * f + 1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    t[4 * f + 2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    t[4 * f + 3] = 1 - t[4 * f] - t[4 * f + 1] - t[4 * f + 2];
  }
  return t;
})();

// quad: four [x, y] page pixels, clockwise from the top-left. Returns { data: RGB bytes, w, h }.
// cv2.warpPerspective(INTER_CUBIC, BORDER_REPLICATE) with the source position rounded to 1/32 pixel as
// OpenCV does, then np.rot90 (counterclockwise) when the crop is taller than 1.5 times its width.
export function cropQuad(rgba, W, H, quad) {
  const d = (a, b) => Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2);
  const w = Math.trunc(Math.max(d(quad[0], quad[1]), d(quad[2], quad[3])));
  const h = Math.trunc(Math.max(d(quad[0], quad[3]), d(quad[1], quad[2])));
  if (w < 1 || h < 1) return { data: new Uint8Array(3), w: 1, h: 1 };
  const m = homography([[0, 0], [w, 0], [w, h], [0, h]], quad);   // crop pixel -> page position
  const out = new Uint8Array(w * h * 3);
  const px = new Int32Array(4);
  const byte = (v) => (v < 0 ? 0 : v > 255 ? 255 : Math.round(v));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const dn = m[6] * x + m[7] * y + m[8], k = dn ? 32 / dn : 0;
      const X = Math.round((m[0] * x + m[1] * y + m[2]) * k), Y = Math.round((m[3] * x + m[4] * y + m[5]) * k);
      const sx = Math.floor(X / 32), sy = Math.floor(Y / 32);
      const fx = 4 * (X - sx * 32), fy = 4 * (Y - sy * 32);
      for (let i = 0; i < 4; i++) { const xx = sx - 1 + i; px[i] = 4 * (xx < 0 ? 0 : xx > W - 1 ? W - 1 : xx); }   // BORDER_REPLICATE
      let r = 0, g = 0, b = 0;
      for (let j = 0; j < 4; j++) {
        const yy = sy - 1 + j, row = (yy < 0 ? 0 : yy > H - 1 ? H - 1 : yy) * W * 4, wy = CUBIC[fy + j];
        for (let i = 0; i < 4; i++) {
          const wt = CUBIC[fx + i] * wy, p = row + px[i];
          r += wt * rgba[p]; g += wt * rgba[p + 1]; b += wt * rgba[p + 2];
        }
      }
      const o = 3 * (y * w + x);
      out[o] = byte(r); out[o + 1] = byte(g); out[o + 2] = byte(b);
    }
  }
  if (h / w < 1.5) return { data: out, w, h };
  // np.rot90: out[i][j] = in[j][w - 1 - i], shape (w, h)
  const rot = new Uint8Array(out.length);
  for (let i = 0; i < w; i++) for (let j = 0; j < h; j++) {
    const s = 3 * (j * w + (w - 1 - i)), o = 3 * (i * h + j);
    rot[o] = out[s]; rot[o + 1] = out[s + 1]; rot[o + 2] = out[s + 2];
  }
  return { data: rot, w: h, h: w };
}

// ------------------------------------------------------------------ recognizer input

// Width of the padded batch (PaddleOCR: int(48 * max(320 / 48, widest w / h in the batch))).
export function recBatchWidth(crops) {
  let r = REC_MIN_W / REC_H;
  for (const c of crops) r = Math.max(r, c.w / c.h);
  return Math.trunc(REC_H * r);
}

// resize_norm_img for a batch: Float32Array [B, 3, 48, W], channels BGR, (x / 255 - 0.5) / 0.5, each crop
// resized to 48 high keeping its aspect ratio (bilinear, rounded to a byte as cv2.resize does) and
// zero-padded on the right. Returns { x, B, W, widths } (widths: the resized width of each crop).
export function recInput(crops, W = recBatchWidth(crops)) {
  const B = crops.length, plane = REC_H * W;
  const x = new Float32Array(B * 3 * plane);
  const widths = [];
  for (let b = 0; b < B; b++) {
    const { data, w, h } = crops[b];
    const ratio = w / h, rw = Math.ceil(REC_H * ratio) > W ? W : Math.ceil(REC_H * ratio);
    widths.push(rw);
    const sx = w / rw, sy = h / REC_H;
    for (let i = 0; i < REC_H; i++) {
      const fy = Math.min(Math.max((i + 0.5) * sy - 0.5, 0), h - 1), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, h - 1), wy = fy - y0;
      for (let j = 0; j < rw; j++) {
        const fx = Math.min(Math.max((j + 0.5) * sx - 0.5, 0), w - 1), x0 = Math.floor(fx), x1 = Math.min(x0 + 1, w - 1), wx = fx - x0;
        for (let c = 0; c < 3; c++) {   // output channel c is BGR: source RGB channel 2 - c
          const s = 2 - c;
          const top = data[3 * (y0 * w + x0) + s] * (1 - wx) + data[3 * (y0 * w + x1) + s] * wx;
          const bot = data[3 * (y1 * w + x0) + s] * (1 - wx) + data[3 * (y1 * w + x1) + s] * wx;
          const v = Math.round(top * (1 - wy) + bot * wy);
          x[(b * 3 + c) * plane + i * W + j] = (v / 255 - 0.5) / 0.5;
        }
      }
    }
  }
  return { x, B, W, widths };
}

// PaddleOCR's CTCLabelDecode confidence for line b of probs [B, T, V]: the mean over the steps that emit a
// character (best class is not the blank, class 0, and differs from the previous step's) of the best
// probability; 0 when there are none.
export function ctcConfidence(probs, T, V, b) {
  let sum = 0, n = 0, prev = -1;
  for (let t = 0; t < T; t++) {
    const base = (b * T + t) * V;
    let k = 0, best = probs[base];
    for (let v = 1; v < V; v++) if (probs[base + v] > best) { best = probs[base + v]; k = v; }
    if (k !== 0 && k !== prev) { sum += best; n++; }
    prev = k;
  }
  return n ? sum / n : 0;
}

// ------------------------------------------------------------------ pipeline

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

export class TextPipeline {
  // det: object with run(rgba, w, h, longSide) -> { prob, H, W, validH, validW } and post (DetNet)
  // rec: see the top of this file
  // options: longSide (1280), recBatch (6), post (overrides of det.post), minScore (drop lines whose
  //   recognizer score is lower; default 0, keep all)
  constructor({ det, rec, longSide = 1280, recBatch = 6, post = {}, minScore = 0 }) {
    Object.assign(this, { det, rec, longSide, recBatch, post, minScore });
  }

  // Boxes only: [{ box, score }] in reading order, plus the page's RGBA for cropping.
  async detect(image) {
    const t0 = now();
    const img = toRGBA(image);
    const d = await this.det.run(img.data, img.width, img.height, this.longSide);
    const t1 = now();
    const post = { ...this.det.post, ...this.post };
    const raw = dbPostProcess(d.prob, d.W, d.H, post);
    const boxes = sortedBoxes(filterBoxes(toPage(raw, d.validW, d.validH, img.width, img.height), img.width, img.height));
    return { img, boxes: boxes.map(({ quad, score }) => ({ box: quad, score })), timing: { det_ms: t1 - t0, post_ms: now() - t1 } };
  }

  // [{ box, text, score, detScore }] in reading order. score is the recognizer's confidence (PaddleOCR's
  // rec score), detScore the detector's box score. result.timing has the stage times in ms.
  async run(image) {
    const { img, boxes, timing } = await this.detect(image);
    const t0 = now();
    const crops = boxes.map((b) => cropQuad(img.data, img.width, img.height, b.box));
    timing.crop_ms = now() - t0;
    const t1 = now();
    // narrow lines first, so a batch pads little: PaddleOCR sorts by width / height too
    const order = crops.map((_, i) => i).sort((a, b) => crops[a].w / crops[a].h - crops[b].w / crops[b].h);
    const text = new Array(crops.length), conf = new Array(crops.length);
    for (let s = 0; s < order.length; s += this.recBatch) {
      const ids = order.slice(s, s + this.recBatch);
      const { x, B, W } = recInput(ids.map((i) => crops[i]));
      const { T, V, probs } = await this.rec.run(x, B, W);
      const dec = this.rec.decode(probs, T, V, B);
      ids.forEach((id, k) => {
        const e = dec[k];
        text[id] = typeof e === "string" ? e : e.text;
        conf[id] = typeof e !== "string" && e.score != null ? e.score : ctcConfidence(probs, T, V, k);
      });
    }
    timing.rec_ms = now() - t1;
    const out = [];
    boxes.forEach((b, i) => {
      if (conf[i] >= this.minScore) out.push({ box: b.box, text: text[i], score: conf[i], detScore: b.score });
    });
    out.timing = timing;
    return out;
  }
}

// ------------------------------------------------------------------ PaddleOCR-shaped result

// The OCR result of PaddleOCR 3.x (Python `PaddleOCR().predict(img)[0]`, paddlex OCRResult; `res.json["res"]`
// has the same keys), from TextPipeline lines:
//
//   const r = toPaddleResult(await pipe.run(canvas), { width: canvas.width, height: canvas.height });
//   r.rec_texts[i], r.rec_scores[i], r.rec_polys[i], r.rec_boxes[i]
//
//   dt_polys    every detected box, reading order, [[x, y] x 4] page pixels, clockwise from the top-left corner
//               (numpy int16 in Python; plain integer arrays here)
//   rec_texts   string per kept line, same order as dt_polys
//   rec_scores  recognizer confidence per kept line, 0 to 1 (numpy float64 in Python)
//   rec_polys   the polygons of the kept lines (dt_polys minus the lines below text_rec_score_thresh)
//   rec_boxes   [x0, y0, x1, y1] of each rec_polys entry, as paddlex convert_points_to_boxes: min and max of x and y
//
// Options: width, height clip the corners to [0, width] and [0, height] as DBPostProcess does (omit to skip);
// minScore is text_rec_score_thresh (default 0, PaddleOCR's default): lines scoring lower stay in dt_polys and
// leave rec_*. If TextPipeline ran with its own minScore above 0, those lines are already gone and dt_polys
// equals rec_polys. Not produced: input_path, page_index, model_settings, doc_preprocessor_res, text_det_params,
// text_type, text_rec_score_thresh, return_word_box, textline_orientation_angles (README.md lists why).
export function toPaddleResult(lines, { width = Infinity, height = Infinity, minScore = 0 } = {}) {
  const clip = (v, hi) => Math.max(0, Math.min(Math.round(v), hi));
  const dt_polys = lines.map((l) => l.box.map(([x, y]) => [clip(x, width), clip(y, height)]));
  const kept = lines.map((l, i) => i).filter((i) => lines[i].score >= minScore);
  const rec_polys = kept.map((i) => dt_polys[i]);
  const rec_boxes = rec_polys.map((p) => {
    const xs = p.map((q) => q[0]), ys = p.map((q) => q[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  });
  return { dt_polys, rec_texts: kept.map((i) => lines[i].text), rec_scores: kept.map((i) => lines[i].score), rec_polys, rec_boxes };
}
