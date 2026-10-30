// What counts as a strike. Two detectors behind one interface, frame(...) -> { strikes, view }:
//
// MotionDetector: block matching on the GPU gives each cell of a 16 x 12 grid a velocity. Each candidate shift is
//   scored by zero-mean SAD, so a brightness change (auto-exposure, flicker) does not read as motion. A strike is an
//   impact: something moved fast and then everything around it went still; motion of the whole frame is ignored. mode "any" keeps the original rule (any jump in per-cell luma change) for comparison.
// HandDetector: MediaPipe hand landmarks; a strike is a fingertip that moved fast and stopped, one per hand.
//
// Positions are in source-frame coordinates (0..1, unmirrored); speeds in frame widths per second.

const WGSL = /* wgsl */`
struct MU { w: u32, h: u32, gx: u32, gy: u32, r: i32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var frame: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> cur: array<f32>;
@group(0) @binding(2) var<storage, read> prev: array<f32>;
@group(0) @binding(3) var<storage, read_write> outc: array<vec4f>;   // dx, dy (px/frame), confidence, |change|
@group(0) @binding(4) var<uniform> mu: MU;

// The frame arrives at twice the matching resolution; each luma pixel averages a 4 x 4 texel box around its
// 2 x 2 footprint. Without that low-pass a fine texture moving by a fraction of a pixel aliases into a different
// pattern every frame, and the block match reports large fake shifts.
@compute @workgroup_size(8, 8)
fn luma(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= mu.w || g.y >= mu.h) { return; }
  var a = 0.0;
  for (var dy = -1; dy <= 2; dy++) { for (var dx = -1; dx <= 2; dx++) {
    let x = clamp(i32(g.x) * 2 + dx, 0, i32(mu.w) * 2 - 1); let y = clamp(i32(g.y) * 2 + dy, 0, i32(mu.h) * 2 - 1);
    a += dot(textureLoad(frame, vec2u(u32(x), u32(y)), 0).rgb, vec3f(0.299, 0.587, 0.114));
  } }
  cur[g.y * mu.w + g.x] = a / 16.0;
}
fn P(x: i32, y: i32) -> f32 {
  let xx = clamp(x, 0, i32(mu.w) - 1); let yy = clamp(y, 0, i32(mu.h) - 1);
  return prev[u32(yy) * mu.w + u32(xx)];
}
fn C(x: i32, y: i32) -> f32 { return cur[u32(y) * mu.w + u32(x)]; }
// mean absolute difference between this cell now and the previous frame shifted by (dx, dy), after removing the
// mean difference, so a uniform brightness change costs nothing
fn zsad(ox: i32, oy: i32, cw: i32, ch: i32, dx: i32, dy: i32) -> f32 {
  var md = 0.0;
  for (var y = 0; y < ch; y++) { for (var x = 0; x < cw; x++) { md += C(ox + x, oy + y) - P(ox + x - dx, oy + y - dy); } }
  md /= f32(cw * ch);
  var s = 0.0;
  for (var y = 0; y < ch; y++) { for (var x = 0; x < cw; x++) { s += abs(C(ox + x, oy + y) - P(ox + x - dx, oy + y - dy) - md); } }
  return s / f32(cw * ch);
}
@compute @workgroup_size(4, 4)
fn blockmatch(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= mu.gx || g.y >= mu.gy) { return; }
  let cw = i32(mu.w / mu.gx); let ch = i32(mu.h / mu.gy); let ox = i32(g.x) * cw; let oy = i32(g.y) * ch;
  let n = f32(cw * ch);
  var e = 0.0; var mean = 0.0;
  for (var y = 0; y < ch; y++) { for (var x = 0; x < cw; x++) { let c = C(ox + x, oy + y); e += abs(c - P(ox + x, oy + y)); mean += c; } }
  mean /= n;
  var tex = 0.0;
  for (var y = 0; y < ch; y++) { for (var x = 0; x < cw; x++) { tex += abs(C(ox + x, oy + y) - mean); } }
  tex /= n;
  let s0 = zsad(ox, oy, cw, ch, 0, 0);
  var best = s0; var bx = 0; var by = 0;
  for (var dy = -mu.r; dy <= mu.r; dy++) { for (var dx = -mu.r; dx <= mu.r; dx++) {
    if (dx == 0 && dy == 0) { continue; }
    // a small cost per pixel of shift: on a repeating texture several shifts fit about equally well, and without
    // it a slow drift over stripes reads as a fast jump of a whole period
    let s = zsad(ox, oy, cw, ch, dx, dy) + 0.002 * f32(abs(dx) + abs(dy));
    if (s < best) { best = s; bx = dx; by = dy; }
  } }
  // confidence: how much better the best shift explains the cell than no shift; flat cells cannot be matched
  let conf = select(0.0, clamp((s0 - best) / (s0 + 1e-4), 0.0, 1.0), tex > 0.012);
  outc[g.y * mu.gx + g.x] = vec4f(f32(bx), f32(by), conf, e / n);
}`;

export class MotionDetector {
  static async create(device) {
    const d = new MotionDetector(); d.device = device; d.mode = "impact"; d.sens = 0.55;
    const mod = device.createShaderModule({ code: WGSL });
    const errs = (await mod.getCompilationInfo()).messages.filter((m) => m.type === "error");
    if (errs.length) throw new Error("WGSL: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
    d.layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: {} },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [d.layout] });
    [d.pLuma, d.pMatch] = await Promise.all(["luma", "blockmatch"].map((entryPoint) =>
      device.createComputePipelineAsync({ layout, compute: { module: mod, entryPoint } })));
    d.busy = false; d.k = 0;
    return d;
  }

  // 96 x 72 in the source's orientation, 16 x 12 cells of 6 x 6 px, matched over +-8 px: up to about
  // 2.5 frame widths per second at 30 fps
  setup(srcW, srcH) {
    const land = srcW >= srcH, s = land ? { w: 96, h: 72, gx: 16, gy: 12 } : { w: 72, h: 96, gx: 12, gy: 16 };
    if (this.s && this.s.w === s.w) return;
    const dev = this.device; this.s = s; const n = s.gx * s.gy;
    this.bufs?.forEach((b) => b.destroy());
    this.tex = dev.createTexture({ size: [2 * s.w, 2 * s.h], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    const L = [0, 1].map(() => dev.createBuffer({ size: s.w * s.h * 4, usage: GPUBufferUsage.STORAGE }));
    this.out = dev.createBuffer({ size: n * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.stage = dev.createBuffer({ size: n * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.uni = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(this.uni, 0, new Int32Array([s.w, s.h, s.gx, s.gy, 8, 0, 0, 0]));
    this.bufs = [this.tex, ...L, this.out, this.stage, this.uni];
    // ping-pong: this frame's luma goes into one buffer while the other still holds the previous frame
    this.bg = [0, 1].map((i) => dev.createBindGroup({ layout: this.layout, entries: [
      { binding: 0, resource: this.tex.createView() }, { binding: 1, resource: { buffer: L[i] } }, { binding: 2, resource: { buffer: L[1 - i] } },
      { binding: 3, resource: { buffer: this.out } }, { binding: 4, resource: { buffer: this.uni } }] }));
    this.canvas = new OffscreenCanvas(2 * s.w, 2 * s.h); this.ctx = this.canvas.getContext("2d");
    this.cells = new Float32Array(n * 4); this.speed = new Float32Array(n);
    this.hist = [0, 1, 2].map(() => ({ sp: new Float32Array(n), dx: new Float32Array(n), dy: new Float32Array(n) }));
    this.base = new Float32Array(n); this.last = new Float64Array(n).fill(-1e9);
    this.warm = 0; this.quietUntil = 0; this.tPrev = 0;
  }
  reset() { this.warm = 0; }

  // source: anything drawImage takes (video, canvas). Returns null while the previous frame is still on the GPU.
  async frame(source, srcW, srcH, now) {
    if (this.busy) return null;
    this.busy = true;
    try {
      this.setup(srcW, srcH);
      const s = this.s, dev = this.device;
      this.ctx.drawImage(source, 0, 0, 2 * s.w, 2 * s.h);   // drawImage applies the iOS camera rotation (METHODS.md)
      dev.queue.copyExternalImageToTexture({ source: this.canvas }, { texture: this.tex }, [2 * s.w, 2 * s.h]);
      const enc = dev.createCommandEncoder(), bg = this.bg[this.k++ & 1];
      let p = enc.beginComputePass(); p.setPipeline(this.pLuma); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(s.w / 8), Math.ceil(s.h / 8)); p.end();
      p = enc.beginComputePass(); p.setPipeline(this.pMatch); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.ceil(s.gx / 4), Math.ceil(s.gy / 4)); p.end();
      enc.copyBufferToBuffer(this.out, 0, this.stage, 0, this.cells.byteLength);
      dev.queue.submit([enc.finish()]);
      await this.stage.mapAsync(GPUMapMode.READ);
      this.cells.set(new Float32Array(this.stage.getMappedRange())); this.stage.unmap();
      const dt = this.tPrev ? Math.min(0.2, Math.max(0.005, (now - this.tPrev) / 1000)) : 1 / 30; this.tPrev = now;
      return { strikes: this.mode === "any" ? this.anyMotion(now) : this.impacts(now, dt), view: { kind: this.mode, s, cells: this.cells, speed: this.speed } };
    } finally { this.busy = false; }
  }

  // The original rule: a cell's change jumps above its own running baseline.
  anyMotion(now) {
    const { gx, gy } = this.s, n = gx * gy, e = (i) => this.cells[4 * i + 3], base = this.base, out = [];
    if (this.warm++ < 8) { for (let i = 0; i < n; i++) base[i] = e(i); return out; }
    const thr = 0.09 - 0.075 * this.sens, cand = [];
    for (let i = 0; i < n; i++) {
      const d = e(i) - base[i]; base[i] += (d > thr ? 0.01 : 0.06) * d;
      if (d > thr && now - this.last[i] > 140) cand.push([d, i]);
    }
    cand.sort((a, b) => b[0] - a[0]);
    for (const [d, i] of cand.slice(0, 3)) { this.last[i] = now; out.push(this.at(i, Math.min(1, 0.2 + (d - thr) / 0.15))); }
    return out;
  }

  // An impact: in the last three frames a cell moved at peak speed >= vFast, and now its whole neighbourhood
  // (2 cells around) is still: no cell there moves at half that speed, and none changes brightness much. A cell the
  // moving thing merely left fails this, because the thing is still moving next door, even where the block match
  // there is unsure (the brightness test catches that). A frame where most cells move (camera shake, a pan) mutes
  // strikes for 250 ms.
  impacts(now, dt) {
    const { w, gx, gy } = this.s, n = gx * gy, c = this.cells, sp = this.speed;
    let moving = 0;
    for (let i = 0; i < n; i++) {
      sp[i] = c[4 * i + 2] > 0.3 ? Math.hypot(c[4 * i], c[4 * i + 1]) / w / dt : 0;
      if (sp[i] > 0.25) moving++;
    }
    const H = this.hist, out = [];
    if (this.warm++ < 3) { this.push(); return out; }
    if (moving > 0.45 * n) this.quietUntil = now + 250;
    if (now < this.quietUntil) { for (const h of H) h.sp.fill(0); this.push(); return out; }
    const vFast = 2.0 - 1.4 * this.sens, STILL_E = 0.035, cand = [];
    for (let i = 0; i < n; i++) {
      // fast for at least two of the last three frames: a one-frame spike is the block match failing at the edge
      // of a moving thing (half object, half newly uncovered background), not motion
      let pk = 0, fast = 0, ux = 0, uy = 0;
      for (const h of H) { if (h.sp[i] > pk) { pk = h.sp[i]; ux = h.dx[i]; uy = h.dy[i]; } if (h.sp[i] >= 0.6 * vFast) fast++; }
      if (pk < vFast || fast < 2 || now - this.last[i] < 220) continue;
      const cx = i % gx, cy = (i / gx) | 0;
      let still = true;
      for (let yy = Math.max(0, cy - 2); yy <= Math.min(gy - 1, cy + 2) && still; yy++)
        for (let xx = Math.max(0, cx - 2); xx <= Math.min(gx - 1, cx + 2); xx++) {
          const j = yy * gx + xx;
          if (sp[j] > Math.min(0.5 * pk, 0.5) || c[4 * j + 3] > STILL_E) { still = false; break; }
        }
      // and nothing is moving further along the way it went: that is something leaving, not stopping
      const len = Math.hypot(ux, uy) || 1;
      for (let k = 3; k <= 6 && still; k++) {
        const qx = Math.round(cx + ux / len * k), qy = Math.round(cy + uy / len * k);
        if (qx < 0 || qy < 0 || qx >= gx || qy >= gy) break;
        const j = qy * gx + qx;
        if (sp[j] > 0.5 || c[4 * j + 3] > STILL_E) still = false;
      }
      if (still) cand.push([pk, i]);
    }
    cand.sort((a, b) => b[0] - a[0]);
    const taken = [];
    for (const [pk, i] of cand) {
      const cx = i % gx, cy = (i / gx) | 0;
      if (taken.some(([x, y]) => Math.max(Math.abs(x - cx), Math.abs(y - cy)) < 5)) continue;
      taken.push([cx, cy]); if (taken.length > 2) break;
      // the strike lands where the motion ended: the speed-weighted centre of the cells nearby that moved in the last
      // frame before the stop, moved on by their displacement
      const cw = this.s.w / gx, ch = this.s.h / gy;
      let sx = 0, sy = 0, sw = 0;
      for (let yy = Math.max(0, cy - 3); yy <= Math.min(gy - 1, cy + 3); yy++)
        for (let xx = Math.max(0, cx - 3); xx <= Math.min(gx - 1, cx + 3); xx++) {
          const j = yy * gx + xx, wgt = H[0].sp[j];
          if (wgt < 0.6 * vFast) continue;
          sx += wgt * (xx + 0.5 + H[0].dx[j] / cw); sy += wgt * (yy + 0.5 + H[0].dy[j] / ch); sw += wgt;
        }
      const x = Math.min(0.99, Math.max(0.01, (sw ? sx / sw : cx + 0.5) / gx));
      const y = Math.min(0.99, Math.max(0.01, (sw ? sy / sw : cy + 0.5) / gy));
      out.push({ x, y, vel: Math.min(1, 0.25 + (pk - vFast) / (1.5 * vFast)) });
      for (let yy = Math.max(0, cy - 3); yy <= Math.min(gy - 1, cy + 3); yy++)          // one impact, one strike
        for (let xx = Math.max(0, cx - 3); xx <= Math.min(gx - 1, cx + 3); xx++) { const j = yy * gx + xx; this.last[j] = now; for (const h of H) h.sp[j] = 0; }
    }
    this.push();
    return out;
  }
  push() {
    const h = this.hist.pop(), c = this.cells; h.sp.set(this.speed);
    for (let i = 0; i < h.dx.length; i++) { h.dx[i] = c[4 * i]; h.dy[i] = c[4 * i + 1]; }
    this.hist.unshift(h);
  }
  at(i, vel) { const { gx, gy } = this.s; return { x: (i % gx + 0.5) / gx, y: (((i / gx) | 0) + 0.5) / gy, vel }; }
}

const MP_VER = "1.1.0";   // the version xray-glasses runs on iPhone
const MP = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}`;
const HAND_MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const TIPS = [4, 8, 12, 16, 20];

export class HandDetector {
  static async create() {
    const h = new HandDetector(); h.sens = 0.55; h.track = new Map(); h.lastHit = new Map();
    const mp = await import(`${MP}/vision_bundle.mjs`), files = await mp.FilesetResolver.forVisionTasks(`${MP}/wasm`);
    const opts = (delegate) => ({ baseOptions: { modelAssetPath: HAND_MODEL, delegate }, runningMode: "VIDEO", numHands: 2,
      minHandDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
    try { h.lm = await mp.HandLandmarker.createFromOptions(files, opts("GPU")); h.delegate = "GPU"; }
    catch { h.lm = await mp.HandLandmarker.createFromOptions(files, opts("CPU")); h.delegate = "CPU"; }
    return h;
  }

  // A strike is a fingertip whose speed in the last three frames peaked at >= vFast and has dropped under 30% of
  // that. Per hand only the fastest such fingertip strikes, and the hand then rests 180 ms, so a whole-hand tap is
  // one strike, not five.
  frame(video, now) {
    const r = this.lm.detectForVideo(video, now), aspect = (video.videoHeight || 3) / (video.videoWidth || 4);
    const vFast = 2.4 - 1.6 * this.sens, seen = new Set(), strikes = [], tips = [];
    (r.landmarks || []).forEach((lm, hi) => {
      const hand = r.handedness?.[hi]?.[0]?.categoryName || String(hi);
      let best = null;
      for (const t of TIPS) {
        const p = lm[t], key = hand + t; seen.add(key);
        let tr = this.track.get(key);
        if (!tr) { tr = { x: p.x, y: p.y, t: now, sp: [0, 0, 0] }; this.track.set(key, tr); }
        const dt = Math.max(0.005, (now - tr.t) / 1000), v = Math.hypot(p.x - tr.x, (p.y - tr.y) * aspect) / dt;
        const pk = Math.max(...tr.sp);
        if (pk >= vFast && v < 0.3 * pk && (!best || pk > best.pk)) best = { pk, x: p.x, y: p.y, tr };
        tr.sp.pop(); tr.sp.unshift(v); tr.x = p.x; tr.y = p.y; tr.t = now;
        tips.push({ x: p.x, y: p.y, index: t === 8 });
      }
      if (best && now - (this.lastHit.get(hand) || -1e9) > 180) {
        this.lastHit.set(hand, now);
        for (const t of TIPS) this.track.get(hand + t).sp.fill(0);
        strikes.push({ x: best.x, y: best.y, vel: Math.min(1, 0.25 + (best.pk - vFast) / (1.5 * vFast)) });
      }
    });
    for (const k of [...this.track.keys()]) if (!seen.has(k)) this.track.delete(k);   // a hand that left starts fresh
    return { strikes, view: { kind: "hands", tips } };
  }
  reset() { this.track.clear(); }
}
