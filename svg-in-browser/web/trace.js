// In-browser image → SVG, after claude-skills image-to-svg (pipeline.py).
// Quantize with k-means (WGSL compute on WebGPU, plain JS otherwise), clean the
// label map, trace every label's region boundary along pixel edges, simplify,
// and emit the compact format image-to-svg 1.9.0 writes: one scoped class per
// color, relative path data, one evenodd path per color.

const KMEANS_WGSL = /* wgsl */ `
struct Params { n: u32, k: u32 }
@group(0) @binding(0) var<storage, read> px: array<u32>;
@group(0) @binding(1) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> labels: array<u32>;
@group(0) @binding(3) var<storage, read_write> sums: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> p: Params;
var<workgroup> local: array<atomic<u32>, 1024>;  // 256 clusters x (r, g, b, count)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  for (var j = li; j < p.k * 4u; j += 256u) { atomicStore(&local[j], 0u); }
  workgroupBarrier();
  let i = gid.x;
  if (i < p.n) {
    let c = px[i];
    let v = vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u));
    var best = 0u;
    var bd = 1e30;
    for (var k = 0u; k < p.k; k++) {
      let d = centers[k].xyz - v;
      let dd = dot(d, d);
      if (dd < bd) { bd = dd; best = k; }
    }
    labels[i] = best;
    atomicAdd(&local[best * 4u], c & 255u);
    atomicAdd(&local[best * 4u + 1u], (c >> 8u) & 255u);
    atomicAdd(&local[best * 4u + 2u], (c >> 16u) & 255u);
    atomicAdd(&local[best * 4u + 3u], 1u);
  }
  workgroupBarrier();
  for (var j = li; j < p.k * 4u; j += 256u) {
    let s = atomicLoad(&local[j]);
    if (s > 0u) { atomicAdd(&sums[j], s); }
  }
}`;

// ---------- preprocessing ----------

/** Separable 5-tap binomial blur, standing in for the pipeline's Gaussian pre-blur. */
function blur(rgba, w, h) {
  const k = [1, 4, 6, 4, 1];
  const tmp = new Float32Array(w * h * 3);
  const out = new Uint32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let r = 0, g = 0, b = 0, s = 0;
    for (let t = -2; t <= 2; t++) {
      const xx = Math.min(w - 1, Math.max(0, x + t)), o = (y * w + xx) * 4, f = k[t + 2];
      r += rgba[o] * f; g += rgba[o + 1] * f; b += rgba[o + 2] * f; s += f;
    }
    const o = (y * w + x) * 3; tmp[o] = r / s; tmp[o + 1] = g / s; tmp[o + 2] = b / s;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let r = 0, g = 0, b = 0, s = 0;
    for (let t = -2; t <= 2; t++) {
      const yy = Math.min(h - 1, Math.max(0, y + t)), o = (yy * w + x) * 3, f = k[t + 2];
      r += tmp[o] * f; g += tmp[o + 1] * f; b += tmp[o + 2] * f; s += f;
    }
    out[y * w + x] = Math.round(r / s) | (Math.round(g / s) << 8) | (Math.round(b / s) << 16);
  }
  return out;
}

// ---------- k-means ----------

function seedCenters(px, K, rand) {
  // k-means++ on a sample of up to 20k pixels.
  const m = Math.min(px.length, 20000);
  const sample = new Float32Array(m * 3);
  for (let i = 0; i < m; i++) {
    const c = px[Math.floor(rand() * px.length)];
    sample[i * 3] = c & 255; sample[i * 3 + 1] = (c >> 8) & 255; sample[i * 3 + 2] = (c >> 16) & 255;
  }
  const centers = new Float32Array(K * 4);
  const dist = new Float32Array(m).fill(Infinity);
  let pick = Math.floor(rand() * m);
  for (let k = 0; k < K; k++) {
    centers.set(sample.subarray(pick * 3, pick * 3 + 3), k * 4);
    let total = 0;
    for (let i = 0; i < m; i++) {
      const dr = sample[i * 3] - centers[k * 4], dg = sample[i * 3 + 1] - centers[k * 4 + 1], db = sample[i * 3 + 2] - centers[k * 4 + 2];
      dist[i] = Math.min(dist[i], dr * dr + dg * dg + db * db);
      total += dist[i];
    }
    let r = rand() * total;
    for (pick = 0; pick < m - 1 && (r -= dist[pick]) > 0; pick++);
  }
  return centers;
}

function updateCenters(centers, sums, K) {
  let moved = 0;
  for (let k = 0; k < K; k++) {
    const n = sums[k * 4 + 3];
    if (!n) continue;
    for (let c = 0; c < 3; c++) {
      const v = sums[k * 4 + c] / n;
      moved = Math.max(moved, Math.abs(v - centers[k * 4 + c]));
      centers[k * 4 + c] = v;
    }
  }
  return moved;
}

function kmeansCPU(px, centers, K, iters) {
  const labels = new Uint32Array(px.length);
  for (let it = 0; it < iters; it++) {
    const sums = new Float64Array(K * 4);
    for (let i = 0; i < px.length; i++) {
      const c = px[i], r = c & 255, g = (c >> 8) & 255, b = (c >> 16) & 255;
      let best = 0, bd = Infinity;
      for (let k = 0; k < K; k++) {
        const dr = centers[k * 4] - r, dg = centers[k * 4 + 1] - g, db = centers[k * 4 + 2] - b;
        const d = dr * dr + dg * dg + db * db;
        if (d < bd) { bd = d; best = k; }
      }
      labels[i] = best;
      sums[best * 4] += r; sums[best * 4 + 1] += g; sums[best * 4 + 2] += b; sums[best * 4 + 3]++;
    }
    if (updateCenters(centers, sums, K) < 0.5) break;
  }
  return labels;
}

async function kmeansGPU(device, px, centers, K, iters) {
  const n = px.length;
  const usage = GPUBufferUsage;
  const pxBuf = device.createBuffer({ size: n * 4, usage: usage.STORAGE | usage.COPY_DST });
  device.queue.writeBuffer(pxBuf, 0, px);
  const cBuf = device.createBuffer({ size: K * 16, usage: usage.STORAGE | usage.COPY_DST });
  const lBuf = device.createBuffer({ size: n * 4, usage: usage.STORAGE | usage.COPY_SRC });
  const sBuf = device.createBuffer({ size: K * 16, usage: usage.STORAGE | usage.COPY_SRC | usage.COPY_DST });
  const pBuf = device.createBuffer({ size: 8, usage: usage.UNIFORM | usage.COPY_DST });
  device.queue.writeBuffer(pBuf, 0, new Uint32Array([n, K]));
  const sRead = device.createBuffer({ size: K * 16, usage: usage.MAP_READ | usage.COPY_DST });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code: KMEANS_WGSL }), entryPoint: "main" },
  });
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [pxBuf, cBuf, lBuf, sBuf, pBuf].map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  const zeros = new Uint32Array(K * 4);
  for (let it = 0; it < iters; it++) {
    device.queue.writeBuffer(cBuf, 0, centers);
    device.queue.writeBuffer(sBuf, 0, zeros);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();
    enc.copyBufferToBuffer(sBuf, 0, sRead, 0, K * 16);
    device.queue.submit([enc.finish()]);
    await sRead.mapAsync(GPUMapMode.READ);
    const sums = new Uint32Array(sRead.getMappedRange().slice(0));
    sRead.unmap();
    if (updateCenters(centers, sums, K) < 0.5) break;
  }
  // One last assignment against the final centers, then read the labels back.
  device.queue.writeBuffer(cBuf, 0, centers);
  device.queue.writeBuffer(sBuf, 0, zeros);
  const lRead = device.createBuffer({ size: n * 4, usage: usage.MAP_READ | usage.COPY_DST });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(n / 256));
  pass.end();
  enc.copyBufferToBuffer(lBuf, 0, lRead, 0, n * 4);
  device.queue.submit([enc.finish()]);
  await lRead.mapAsync(GPUMapMode.READ);
  const labels = new Uint32Array(lRead.getMappedRange().slice(0));
  lRead.unmap();
  for (const b of [pxBuf, cBuf, lBuf, sBuf, pBuf, sRead, lRead]) b.destroy();
  return labels;
}

// ---------- label cleanup ----------

/** 3x3 majority filter: a pixel takes the most common label around it when that beats its own. */
function majority(labels, w, h, K) {
  const out = new Uint32Array(labels);
  const count = new Uint16Array(K);
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let best = labels[y * w + x], bc = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const l = labels[(y + dy) * w + x + dx];
      if (++count[l] > bc) { bc = count[l]; best = l; }
    }
    if (bc >= 5) out[y * w + x] = best;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) count[labels[(y + dy) * w + x + dx]] = 0;
  }
  return out;
}

/** Merge 4-connected components smaller than minArea pixels into their most common neighbor label. */
function absorbSmall(labels, w, h, minArea) {
  const n = w * h, comp = new Int32Array(n).fill(-1), stack = new Int32Array(n);
  const members = new Int32Array(n);
  for (let s = 0; s < n; s++) {
    if (comp[s] !== -1) continue;
    const L = labels[s];
    let top = 0, size = 0;
    stack[top++] = s; comp[s] = s;
    while (top) {
      const p = stack[--top]; members[size++] = p;
      const x = p % w, y = (p - x) / w;
      if (x > 0 && comp[p - 1] === -1 && labels[p - 1] === L) { comp[p - 1] = s; stack[top++] = p - 1; }
      if (x < w - 1 && comp[p + 1] === -1 && labels[p + 1] === L) { comp[p + 1] = s; stack[top++] = p + 1; }
      if (y > 0 && comp[p - w] === -1 && labels[p - w] === L) { comp[p - w] = s; stack[top++] = p - w; }
      if (y < h - 1 && comp[p + w] === -1 && labels[p + w] === L) { comp[p + w] = s; stack[top++] = p + w; }
    }
    if (size >= minArea) continue;
    const votes = new Map();
    for (let i = 0; i < size; i++) {
      const p = members[i], x = p % w;
      for (const q of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, p - w, p + w]) {
        if (q < 0 || q >= n || labels[q] === L) continue;
        votes.set(labels[q], (votes.get(labels[q]) || 0) + 1);
      }
    }
    let best = L, bv = 0;
    for (const [l, v] of votes) if (v > bv) { bv = v; best = l; }
    for (let i = 0; i < size; i++) labels[members[i]] = best;
  }
  return labels;
}

// ---------- boundary tracing ----------

const DX = [1, 0, -1, 0], DY = [0, 1, 0, -1];  // right, down, left, up (y grows down)

/**
 * Closed boundary loops of every label, along pixel edges with the region on the right,
 * as lists of vertex ids (v = y * (w + 1) + x). Holes come out as loops of the opposite
 * orientation, so one evenodd path per label reproduces the region exactly. At a saddle
 * the trace turns right, which keeps diagonal-only neighbors apart (4-connectivity).
 * Only corners and junctions are kept; both are properties of the vertex, so the two
 * regions on either side of an edge keep the same vertices.
 */
function traceLoops(labels, w, h, K) {
  const V = (w + 1) * (h + 1);
  const byLabel = Array.from({ length: K }, () => []);
  for (let i = 0; i < labels.length; i++) if (labels[i] < K) byLabel[labels[i]].push(i);  // labels >= K are left empty
  const out = Array.from({ length: K }, () => []);
  // Up to two outgoing edges per vertex per label: store direction+1 in two slots.
  const slotA = new Uint8Array(V), slotB = new Uint8Array(V);
  const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? -1 : labels[y * w + x]);
  for (let L = 0; L < K; L++) {
    const touched = [];
    const add = (x, y, d) => {
      const v = y * (w + 1) + x;
      if (!slotA[v]) { slotA[v] = d + 1; touched.push(v); } else slotB[v] = d + 1;
    };
    for (const p of byLabel[L]) {
      const x = p % w, y = (p - x) / w;
      if (at(x, y - 1) !== L) add(x, y, 0);          // top edge, heading right
      if (at(x + 1, y) !== L) add(x + 1, y, 1);      // right edge, heading down
      if (at(x, y + 1) !== L) add(x + 1, y + 1, 2);  // bottom edge, heading left
      if (at(x - 1, y) !== L) add(x, y + 1, 3);      // left edge, heading up
    }
    for (const start of touched) {
      while (slotA[start]) {
        const loop = [];
        let v = start, d = -1;
        for (;;) {
          let a = slotA[v] - 1, b = slotB[v] - 1, take;
          if (a < 0) break;
          if (b >= 0 && d >= 0) take = b === (d + 1) % 4 ? b : a === (d + 1) % 4 ? a : a;
          else take = a;
          if (take === a) { slotA[v] = slotB[v]; slotB[v] = 0; } else slotB[v] = 0;
          const x = v % (w + 1), y = (v - x) / (w + 1);
          if (take !== d || isJunction(labels, w, h, x, y)) loop.push(v);
          d = take;
          v = (y + DY[d]) * (w + 1) + x + DX[d];
        }
        if (loop.length >= 3) out[L].push(loop);
      }
    }
  }
  return out;
}

/**
 * A vertex where three or more regions meet (the image edge counts as one), or a
 * diagonal saddle. Junctions are where shared boundaries begin and end.
 */
function isJunction(labels, w, h, x, y) {
  const at = (xx, yy) => (xx < 0 || yy < 0 || xx >= w || yy >= h ? -1 : labels[yy * w + xx]);
  if ((x === 0 || x === w) && (y === 0 || y === h)) return true;  // image corners stay sharp
  const a = at(x - 1, y - 1), b = at(x, y - 1), c = at(x - 1, y), d = at(x, y);
  if (a === d && b === c) return a !== b;  // diagonal saddle
  let n = 1;
  if (b !== a) n++;
  if (c !== a && c !== b) n++;
  if (d !== a && d !== b && d !== c) n++;
  return n >= 3;
}

/** Douglas–Peucker with fixed endpoints over vertex ids; returns the kept ids. */
function simplifyOpen(ids, W, eps) {
  const n = ids.length;
  if (n <= 2) return ids.slice();
  const X = (v) => v % W, Y = (v) => (v - (v % W)) / W;
  const keep = new Uint8Array(n);
  keep[0] = keep[n - 1] = 1;
  const stack = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const ax = X(ids[a]), ay = Y(ids[a]), bx = X(ids[b]), by = Y(ids[b]);
    const len = Math.hypot(bx - ax, by - ay);
    let worst = -1, wd = eps;
    for (let i = a + 1; i < b; i++) {
      const px = X(ids[i]), py = Y(ids[i]);
      const d = len ? Math.abs((bx - ax) * (ay - py) - (ax - px) * (by - ay)) / len : Math.hypot(px - ax, py - ay);
      if (d > wd) { wd = d; worst = i; }
    }
    if (worst >= 0) { keep[worst] = 1; stack.push([a, worst], [worst, b]); }
  }
  return ids.filter((_, i) => keep[i]);
}

/** Index of the vertex farthest from ids[0], ties to the smaller id, so either traversal picks it. */
function farthest(ids, W) {
  const x0 = ids[0] % W, y0 = (ids[0] - x0) / W;
  let best = -1, bd = -1;
  for (let i = 1; i < ids.length; i++) {
    const x = ids[i] % W, y = (ids[i] - x) / W, d = (x - x0) ** 2 + (y - y0) ** 2;
    if (d > bd || (d === bd && ids[i] < ids[best])) { bd = d; best = i; }
  }
  return best;
}

/**
 * Simplify one arc (open, from junction to junction) through a cache shared by every
 * region, so the region on the other side, which walks the same arc backwards, gets
 * exactly the same points. That is what keeps neighboring paths from pulling apart.
 */
function simplifyArc(arc, W, eps, cache) {
  const n = arc.length;
  if (arc[0] === arc[n - 1]) {
    // Closed on itself: split at the farthest vertex, which both traversals agree on.
    const f = farthest(arc.slice(0, n - 1), W);
    const first = simplifyArc(arc.slice(0, f + 1), W, eps, cache);
    return first.concat(simplifyArc(arc.slice(f), W, eps, cache).slice(1));
  }
  const reversed = arc[0] > arc[n - 1];
  const oriented = reversed ? arc.slice().reverse() : arc;
  const key = oriented[0] * 4294967296 + oriented[1];  // first edge identifies the arc
  let out = cache.get(key);
  if (!out) { out = simplifyOpen(oriented, W, eps); cache.set(key, out); }
  return reversed ? out.slice().reverse() : out;
}

/** Simplify a closed loop of vertex ids by splitting it into arcs at its junctions. */
function simplifyLoop(loop, junction, W, eps, cache) {
  const n = loop.length;
  let cuts = [];
  for (let i = 0; i < n; i++) if (junction(loop[i])) cuts.push(i);
  if (!cuts.length) {
    // A boundary between exactly two regions: anchor at the smallest id, which both share.
    let m = 0;
    for (let i = 1; i < n; i++) if (loop[i] < loop[m]) m = i;
    cuts = [m];
  }
  const out = [];
  for (let c = 0; c < cuts.length; c++) {
    const a = cuts[c], b = cuts[(c + 1) % cuts.length];
    const arc = [];
    for (let i = a; ; i = (i + 1) % n) { arc.push(loop[i]); if (arc.length > 1 && i === b) break; }
    const simple = simplifyArc(arc, W, eps, cache);
    for (let i = 0; i < simple.length - 1; i++) out.push(simple[i]);
  }
  return out;
}

// ---------- emission ----------

const hex = (c) => "#" + [c[0], c[1], c[2]].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");
const shortHex = (h) => (h[1] === h[2] && h[3] === h[4] && h[5] === h[6] ? `#${h[1]}${h[3]}${h[5]}` : h);

function num(v, precision) {
  let t = precision > 0 ? v.toFixed(precision).replace(/\.?0+$/, "") : String(Math.round(v));
  if (t.startsWith("0.")) t = t.slice(1);
  else if (t.startsWith("-0.")) t = "-" + t.slice(2);
  return t === "-0" || t === "" ? "0" : t;
}

function join(values, precision) {
  let out = "", last = "";
  for (const v of values) {
    const t = num(v, precision);
    if (out && !(t[0] === "-" || (t[0] === "." && last.includes(".")))) out += " ";
    out += t; last = t;
  }
  return out;
}

function pathData(loops, scale, precision, sy = scale) {
  const q = 10 ** precision;
  let d = "", cx = 0, cy = 0;
  for (const loop of loops) {
    const pts = [];
    for (let i = 0; i < loop.length; i += 2) {
      const x = Math.round(loop[i] * scale * q), y = Math.round(loop[i + 1] * sy * q);
      if (!pts.length || x !== pts[pts.length - 2] || y !== pts[pts.length - 1]) pts.push(x, y);
    }
    if (pts.length < 6) continue;
    const deltas = [];
    for (let i = 2; i < pts.length; i += 2) deltas.push((pts[i] - pts[i - 2]) / q, (pts[i + 1] - pts[i - 1]) / q);
    d += "m" + join([(pts[0] - cx) / q, (pts[1] - cy) / q], precision) + "l" + join(deltas, precision) + "z";
    cx = pts[0]; cy = pts[1];
  }
  return d;
}

/**
 * Smooth path data for loops of vertex ids: each non-junction vertex becomes a quadratic
 * control point with the curve passing through edge midpoints, and junctions stay
 * corners. A shared arc yields the same curves from either side (a quadratic traced
 * backwards is the same curve), so smoothing keeps neighbors sealed.
 */
function curvedPathData(loops, junction, W, sx, sy, precision) {
  const q = 10 ** precision;
  const P = (v) => [Math.round((v % W) * sx * q), Math.round(((v - (v % W)) / W) * sy * q)];
  const mid = (a, b) => [Math.round((a[0] + b[0]) / 2), Math.round((a[1] + b[1]) / 2)];
  let d = "", cx = 0, cy = 0, last = "";
  const emit = (cmd, ...xy) => {
    const rel = xy.map((v, i) => (v - (i % 2 ? cy : cx)) / q);
    d += (cmd !== last || cmd === "m" ? cmd : rel[0] < 0 ? "" : " ") + join(rel, precision);
    last = cmd === "m" ? "l" : cmd;
    cx = xy[xy.length - 2]; cy = xy[xy.length - 1];
  };
  for (const ids of loops) {
    const n = ids.length;
    const pts = ids.map(P), J = ids.map(junction);
    let start = J.indexOf(true);
    if (start < 0) {
      // No corners: a closed midpoint spline.
      emit("m", ...mid(pts[n - 1], pts[0]));
      for (let i = 0; i < n; i++) emit("q", ...pts[i], ...mid(pts[i], pts[(i + 1) % n]));
    } else {
      emit("m", ...pts[start]);
      for (let k = 1; k <= n; k++) {
        const i = (start + k) % n, prev = pts[(i - 1 + n) % n];
        if (J[i]) { emit("l", ...pts[i]); continue; }
        const a = mid(prev, pts[i]);
        if (a[0] !== cx || a[1] !== cy) emit("l", ...a);
        emit("q", ...pts[i], ...mid(pts[i], pts[(i + 1) % n]));
      }
    }
    d += "z";
    last = "";
  }
  return d;
}

/** Box-average RGBA down to (lw, lh). */
function downsample(data, w, h, lw, lh) {
  if (lw === w && lh === h) return data;
  const out = new Uint8ClampedArray(lw * lh * 4);
  for (let y = 0; y < lh; y++) {
    const y0 = Math.floor((y * h) / lh), y1 = Math.max(y0 + 1, Math.floor(((y + 1) * h) / lh));
    for (let x = 0; x < lw; x++) {
      const x0 = Math.floor((x * w) / lw), x1 = Math.max(x0 + 1, Math.floor(((x + 1) * w) / lw));
      let r = 0, g = 0, b = 0, n = 0;
      for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) {
        const o = (yy * w + xx) * 4; r += data[o]; g += data[o + 1]; b += data[o + 2]; n++;
      }
      const o = (y * lw + x) * 4;
      out[o] = r / n; out[o + 1] = g / n; out[o + 2] = b / n; out[o + 3] = 255;
    }
  }
  return out;
}

/** 4-connected components of a label map: {comp, count}. */
function components(labels, w, h) {
  const n = w * h, comp = new Int32Array(n).fill(-1), stack = new Int32Array(n);
  let count = 0;
  for (let s = 0; s < n; s++) {
    if (comp[s] !== -1) continue;
    const L = labels[s];
    let top = 0;
    stack[top++] = s; comp[s] = count;
    while (top) {
      const p = stack[--top], x = p % w;
      if (x > 0 && comp[p - 1] === -1 && labels[p - 1] === L) { comp[p - 1] = count; stack[top++] = p - 1; }
      if (x < w - 1 && comp[p + 1] === -1 && labels[p + 1] === L) { comp[p + 1] = count; stack[top++] = p + 1; }
      if (p >= w && comp[p - w] === -1 && labels[p - w] === L) { comp[p - w] = count; stack[top++] = p - w; }
      if (p + w < n && comp[p + w] === -1 && labels[p + w] === L) { comp[p + w] = count; stack[top++] = p + w; }
    }
    count++;
  }
  return { comp, count };
}

const DEFAULT_LEVELS = [
  { scale: 1 / 8, K: 8, minArea: 4 },
  { scale: 1 / 4, K: 16, minArea: 6 },
  { scale: 1 / 2, K: 32, minArea: 10 },
  { scale: 1, K: 56, minArea: 24 },
];

/**
 * Coarse-to-fine layers, like a diffusion sampler run backwards from blur: level 0 tiles
 * the whole image with a few smooth blobs; each finer level quantizes the image again
 * at a higher resolution and draws only the regions that lower the error against the
 * source by more than `gain` (mean absolute difference per channel, 0-255) over what
 * is already painted. Every pixel is covered by level 0, so later layers cannot tear.
 *
 * @param {ImageData} image
 * @param {{levels?: {scale: number, K: number, minArea: number}[], gain?: number,
 *          width?: number, precision?: number, eps?: number, curves?: boolean,
 *          reveal?: number, iters?: number, device?: GPUDevice|null, seed?: number}} opts
 *   reveal: seconds per layer for a fade-in animation that plays the levels in order (0 = none)
 * @returns {Promise<{svg: string, stats: object}>}
 */
export async function imageToSvgLayers(image, opts = {}) {
  const { levels = DEFAULT_LEVELS, gain = 4, width = 1000, precision = 1, eps = 1, curves = true,
          reveal = 0, iters = 20, device = null, seed = 1 } = opts;
  const { width: w, height: h, data } = image;
  const t0 = performance.now();
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const height = Math.round(h * (width / w) * 10) / 10;
  const canvas = new Float32Array(w * h * 3).fill(-1);  // what the layers so far paint
  const colors = new Map(), groups = [], perLevel = [];

  for (let li = 0; li < levels.length; li++) {
    const { scale: ls, K, minArea } = levels[li];
    const lw = Math.max(2, Math.round(w * ls)), lh = Math.max(2, Math.round(h * ls));
    const px = blur(downsample(data, w, h, lw, lh), lw, lh);
    const centers = seedCenters(px, K, rand);
    let labels = device ? await kmeansGPU(device, px, centers, K, iters) : kmeansCPU(px, centers, K, iters);
    labels = absorbSmall(majority(labels, lw, lh, K), lw, lh, minArea);

    // Score every region against the full-resolution source and the current canvas.
    const { comp, count } = components(labels, lw, lh);
    const errNow = new Float64Array(count), errNew = new Float64Array(count), size = new Float64Array(count);
    const fx = new Int32Array(w).map((_, x) => Math.min(lw - 1, Math.floor((x * lw) / w)));
    for (let y = 0; y < h; y++) {
      const row = Math.min(lh - 1, Math.floor((y * lh) / h)) * lw;
      for (let x = 0; x < w; x++) {
        const lp = row + fx[x], c = comp[lp], L = labels[lp], i = y * w + x;
        let now = 0, nw = 0;
        for (let ch = 0; ch < 3; ch++) {
          const src = data[i * 4 + ch];
          now += canvas[i * 3] < 0 ? 255 : Math.abs(src - canvas[i * 3 + ch]);
          nw += Math.abs(src - centers[L * 4 + ch]);
        }
        errNow[c] += now / 3; errNew[c] += nw / 3; size[c]++;
      }
    }
    const keep = new Uint8Array(count);
    let kept = 0;
    for (let c = 0; c < count; c++) {
      if ((errNow[c] - errNew[c]) / size[c] > gain) { keep[c] = 1; kept++; }
    }
    // Regions not drawn become label K, which the tracer skips.
    const lv = new Uint32Array(lw * lh);
    for (let p = 0; p < lw * lh; p++) lv[p] = keep[comp[p]] ? labels[p] : K;
    for (let y = 0; y < h; y++) {
      const row = Math.min(lh - 1, Math.floor((y * lh) / h)) * lw;
      for (let x = 0; x < w; x++) {
        const L = lv[row + fx[x]];
        if (L < K) canvas.set(centers.subarray(L * 4, L * 4 + 3), (y * w + x) * 3);
      }
    }

    const loops = traceLoops(lv, lw, lh, K);
    // Separate x and y scales: a rounded level grid must still span the whole viewBox.
    const W = lw + 1, arcs = new Map(), sx = width / lw, sy = height / lh;
    const junction = (v) => isJunction(lv, lw, lh, v % W, (v - (v % W)) / W);
    const area = new Float64Array(K);
    for (const l of lv) if (l < K) area[l]++;
    const order = [...Array(K).keys()].filter((k) => area[k] > 0).sort((a, b) => area[b] - area[a]);
    const paths = [];
    for (const L of order) {
      const simple = loops[L].map((loop) => simplifyLoop(loop, junction, W, eps, arcs)).filter((ids) => ids.length >= 3);
      const d = curves ? curvedPathData(simple, junction, W, sx, sy, precision)
                       : pathData(simple.map((ids) => ids.flatMap((v) => [v % W, (v - (v % W)) / W])), sx, precision, sy);
      if (!d) continue;
      const c = shortHex(hex(centers.subarray(L * 4, L * 4 + 3)));
      if (!colors.has(c)) colors.set(c, `c${colors.size}`);
      paths.push(`<path class="${colors.get(c)}" d="${d}"/>`);
    }
    groups.push(`<g class="L${li}">${paths.join("")}</g>`);
    perLevel.push({ size: `${lw}x${lh}`, K, regions: count, drawn: kept, paths: paths.length });
  }

  const body = groups.join("");
  const scope = "s" + Math.abs(body.split("").reduce((a, ch) => (a * 31 + ch.charCodeAt(0)) | 0, 7)).toString(16).slice(0, 6);
  let css = [...colors].map(([c, k]) => `#${scope} .${k}{fill:${c};stroke:${c}}`).join("");
  if (reveal > 0) {
    css += `@keyframes ${scope}in{from{opacity:0}}` +
      groups.map((_, i) => `#${scope} .L${i}{animation:${scope}in ${num(reveal, 2)}s ${num(i * reveal, 2)}s both}`).join("") +
      `@media (prefers-reduced-motion:reduce){#${scope} g{animation:none}}`;
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" id="${scope}" viewBox="0 0 ${width} ${height}">` +
    `<style>${css}</style>` +
    `<g fill-rule="evenodd" stroke-width="${Math.max(1, Math.round(width / w))}" stroke-linejoin="round">${body}</g></svg>`;
  return {
    svg,
    stats: { backend: device ? "webgpu" : "js", mode: "layers", w, h, bytes: svg.length, levels: perLevel,
             ms: { total: Math.round(performance.now() - t0) } },
  };
}

// ---------- entry point ----------

/**
 * @param {ImageData} image
 * @param {{K?: number, width?: number, precision?: number, minArea?: number, iters?: number,
 *          device?: GPUDevice|null, seed?: number, eps?: number}} opts  eps: simplification
 *          tolerance in source pixels; shared boundaries stay shared at any value
 * @returns {Promise<{svg: string, stats: object}>}
 */
export async function imageToSvg(image, opts = {}) {
  const { K = 56, width = 1000, precision = 1, minArea = 40, iters = 20, device = null, seed = 1, eps = 1 } = opts;
  const { width: w, height: h, data } = image;
  const t = { start: performance.now() };
  let s = seed >>> 0;
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);

  const px = blur(data, w, h);
  t.blur = performance.now();
  const centers = seedCenters(px, K, rand);
  let labels = device ? await kmeansGPU(device, px, centers, K, iters) : kmeansCPU(px, centers, K, iters);
  t.kmeans = performance.now();
  labels = majority(labels, w, h, K);
  labels = absorbSmall(labels, w, h, minArea);
  t.cleanup = performance.now();
  const loops = traceLoops(labels, w, h, K);
  t.trace = performance.now();

  const scale = width / w, height = Math.round(h * scale * 10) / 10;
  const W = w + 1, arcs = new Map();
  const junction = (v) => isJunction(labels, w, h, v % W, (v - (v % W)) / W);
  const area = new Float64Array(K);
  for (const l of labels) area[l]++;
  const order = [...Array(K).keys()].filter((k) => area[k] > 0).sort((a, b) => area[b] - area[a]);
  const colors = new Map(), classOf = new Map(), paths = [];
  for (const L of order) {
    const loopsL = loops[L].map((loop) => simplifyLoop(loop, junction, W, eps, arcs))
      .filter((ids) => ids.length >= 3)
      .map((ids) => ids.flatMap((v) => [v % W, (v - (v % W)) / W]));
    const d = pathData(loopsL, scale, precision);
    if (!d) continue;
    const c = shortHex(hex(centers.subarray(L * 4, L * 4 + 3)));
    if (!colors.has(c)) colors.set(c, `c${colors.size}`);
    paths.push(`<path class="${colors.get(c)}" d="${d}"/>`);
  }
  const scope = "s" + Math.abs(paths.join("").split("").reduce((a, ch) => (a * 31 + ch.charCodeAt(0)) | 0, 7)).toString(16).slice(0, 6);
  const css = [...colors].map(([c, k]) => `#${scope} .${k}{fill:${c};stroke:${c}}`).join("");
  const bg = shortHex(hex(centers.subarray(order[0] * 4, order[0] * 4 + 3)));
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" id="${scope}" viewBox="0 0 ${width} ${height}">` +
    `<style>${css}</style><rect width="${width}" height="${height}" fill="${bg}"/>` +
    `<g fill-rule="evenodd" stroke-width="${Math.max(1, Math.round(scale))}" stroke-linejoin="round">${paths.join("")}</g></svg>`;
  t.emit = performance.now();
  const ms = (a, b) => Math.round(t[b] - t[a]);
  return {
    svg,
    stats: {
      backend: device ? "webgpu" : "js", w, h, K, bytes: svg.length, paths: paths.length,
      loops: loops.reduce((a, l) => a + l.length, 0),
      ms: { blur: ms("start", "blur"), kmeans: ms("blur", "kmeans"), cleanup: ms("kmeans", "cleanup"),
            trace: ms("cleanup", "trace"), emit: ms("trace", "emit"), total: ms("start", "emit") },
    },
  };
}
