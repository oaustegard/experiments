// Eulerian video magnification (Wu et al. 2012, colour variant) as WebGPU passes, plus a float64 reference.
//
// Per frame:  copy    camera texture (rgba8unorm) -> level-0 buffer of vec4<f32>
//             down    5-tap binomial blur and 2x decimation, once per pyramid level
//             temporal per-pixel band-pass at the coarsest level: a two-pole low-pass at fh minus one at fl,
//                     each pole an exponential smoother whose gain comes from the frame's own dt, so dropped or
//                     late camera frames keep the filter's time constants
//             roi     mean colour of the face and background rectangles (level 1) and of the face's filtered signal
//             render  original + amp * bilinear-upsampled filtered level, into the canvas
// Only 48 bytes come back per frame: the three ROI means.

const COPY = /* wgsl */`
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<uniform> d: vec4u;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= d.x || g.y >= d.y) { return; }
  dst[g.y * d.x + g.x] = vec4f(textureLoad(src, vec2u(g.xy), 0).rgb, 0.0);
}`;

const DOWN = /* wgsl */`
struct D { sw: u32, sh: u32, dw: u32, dh: u32 };
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
@group(0) @binding(2) var<uniform> d: D;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= d.dw || g.y >= d.dh) { return; }
  var w = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);
  var s = vec4f(0.0);
  for (var j = 0; j < 5; j++) {
    let y = u32(clamp(i32(g.y * 2u) + j - 2, 0, i32(d.sh) - 1));
    var r = vec4f(0.0);
    for (var k = 0; k < 5; k++) {
      let x = u32(clamp(i32(g.x * 2u) + k - 2, 0, i32(d.sw) - 1));
      r += w[k] * src[y * d.sw + x];
    }
    s += w[j] * r;
  }
  dst[g.y * d.dw + g.x] = s;
}`;

const TEMPORAL = /* wgsl */`
struct T { n: u32, init: u32, kh: f32, kl: f32 };
@group(0) @binding(0) var<storage, read> x: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> st: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outp: array<vec4f>;
@group(0) @binding(3) var<uniform> u: T;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= u.n) { return; }
  let v = x[i];
  var a1 = v; var a2 = v; var b1 = v; var b2 = v;
  if (u.init == 0u) {
    a1 = st[4u * i]; a2 = st[4u * i + 1u]; b1 = st[4u * i + 2u]; b2 = st[4u * i + 3u];
    a1 += u.kh * (v - a1); a2 += u.kh * (a1 - a2);
    b1 += u.kl * (v - b1); b2 += u.kl * (b1 - b2);
  }
  st[4u * i] = a1; st[4u * i + 1u] = a2; st[4u * i + 2u] = b1; st[4u * i + 3u] = b2;
  outp[i] = a2 - b2;
}`;

const ROI = /* wgsl */`
struct R { lw: u32, lh: u32, fw: u32, fh: u32, face: vec4u, bg: vec4u, ff: vec4u };
@group(0) @binding(0) var<storage, read> src: array<vec4f>;
@group(0) @binding(1) var<storage, read> filt: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outp: array<vec4f>;
@group(0) @binding(3) var<uniform> u: R;
var<workgroup> acc: array<vec4f, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  for (var k = 0u; k < 3u; k++) {
    var r = u.face; var w = u.lw;
    if (k == 1u) { r = u.bg; } else if (k == 2u) { r = u.ff; w = u.fw; }
    let rw = r.z - r.x; let n = rw * (r.w - r.y);
    var s = vec4f(0.0);
    for (var i = li; i < n; i += 256u) {
      let j = (r.y + i / rw) * w + r.x + i % rw;
      if (k == 2u) { s += filt[j]; } else { s += src[j]; }
    }
    acc[li] = s;
    workgroupBarrier();
    for (var h = 128u; h > 0u; h >>= 1u) {
      if (li < h) { acc[li] += acc[li + h]; }
      workgroupBarrier();
    }
    if (li == 0u) { outp[k] = acc[0] / f32(max(n, 1u)); }
    workgroupBarrier();
  }
}`;

const COMPOSITE = /* wgsl */`
struct C { w: u32, h: u32, lw: u32, lh: u32, view: u32, p0: u32, amp: f32, p1: f32 };
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> filt: array<vec4f>;
@group(0) @binding(2) var<uniform> c: C;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
fn at(x: i32, y: i32) -> vec3f {
  return filt[u32(clamp(y, 0, i32(c.lh) - 1)) * c.lw + u32(clamp(x, 0, i32(c.lw) - 1))].rgb;
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = vec2u(pos.xy);
  let orig = textureLoad(tex, p, 0).rgb;
  let q = pos.xy * vec2f(f32(c.lw) / f32(c.w), f32(c.lh) / f32(c.h)) - 0.5;
  let q0 = floor(q); let fr = q - q0; let x = i32(q0.x); let y = i32(q0.y);
  let f = mix(mix(at(x, y), at(x + 1, y), fr.x), mix(at(x, y + 1), at(x + 1, y + 1), fr.x), fr.y);
  var o = orig + c.amp * f;
  if (c.view == 1u) { o = vec3f(0.5) + c.amp * f; } else if (c.view == 2u) { o = orig; }
  return vec4f(clamp(o, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

export const DEFAULTS = { levels: 4, amp: 60, fl: 0.7, fh: 3.0 };

export function pyramidDims(w, h, levels) {
  const d = [[w, h]];
  for (let l = 1; l <= levels; l++) d.push([Math.ceil(d[l - 1][0] / 2), Math.ceil(d[l - 1][1] / 2)]);
  return d;
}

// A normalised rectangle [x0, y0, x1, y1] as integer pixel bounds on a w x h grid, at least one pixel.
export function rectPx(r, w, h) {
  const x0 = Math.min(w - 1, Math.max(0, Math.floor(r[0] * w))), y0 = Math.min(h - 1, Math.max(0, Math.floor(r[1] * h)));
  return [x0, y0, Math.min(w, Math.max(x0 + 1, Math.ceil(r[2] * w))), Math.min(h, Math.max(y0 + 1, Math.ceil(r[3] * h)))];
}

// Exponential-smoother gains for one frame of length dt seconds.
export function gains(dt, fl, fh) {
  return { kh: 1 - Math.exp(-2 * Math.PI * fh * dt), kl: 1 - Math.exp(-2 * Math.PI * fl * dt) };
}

export class EvmEngine {
  static async create(device, opts) { const e = new EvmEngine(); await e.init(device, opts); return e; }

  async init(device, { w, h, levels = DEFAULTS.levels, format = "rgba8unorm" }) {
    Object.assign(this, { device, w, h, levels, format });
    this.params = { amp: DEFAULTS.amp, fl: DEFAULTS.fl, fh: DEFAULTS.fh, view: 0 };
    this.face = [0.35, 0.25, 0.65, 0.65]; this.bg = [0.0, 0.1, 0.15, 0.9];
    const dims = this.dims = pyramidDims(w, h, levels), U = GPUBufferUsage;
    const [lw, lh] = dims[levels]; this.lw = lw; this.lh = lh; this.n = lw * lh;
    const buf = (size, usage) => device.createBuffer({ size, usage });
    this.tex = device.createTexture({ size: [w, h], format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    this.bufs = dims.map(([a, b]) => buf(a * b * 16, U.STORAGE | U.COPY_SRC));
    this.state = buf(this.n * 64, U.STORAGE | U.COPY_SRC);
    this.filt = buf(this.n * 16, U.STORAGE | U.COPY_SRC);
    this.roiOut = buf(64, U.STORAGE | U.COPY_SRC);
    this.stages = [];
    const uni = (size) => buf(size, U.UNIFORM | U.COPY_DST);
    const comp = (code) => device.createComputePipelineAsync({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    const cm = device.createShaderModule({ code: COMPOSITE });
    const [pCopy, pDown, pTemp, pRoi, pComp] = await Promise.all([comp(COPY), comp(DOWN), comp(TEMPORAL), comp(ROI),
      device.createRenderPipelineAsync({ layout: "auto", vertex: { module: cm, entryPoint: "vs" },
        fragment: { module: cm, entryPoint: "fs", targets: [{ format }] }, primitive: { topology: "triangle-list" } })]);
    Object.assign(this, { pCopy, pDown, pTemp, pRoi, pComp });
    const bg = (p, entries) => device.createBindGroup({ layout: p.getBindGroupLayout(0),
      entries: entries.map((r, binding) => ({ binding, resource: r instanceof GPUBuffer ? { buffer: r } : r })) });
    const copyU = uni(16); device.queue.writeBuffer(copyU, 0, new Uint32Array([w, h, 0, 0]));
    this.bgCopy = bg(pCopy, [this.tex.createView(), this.bufs[0], copyU]);
    this.bgDown = [];
    for (let l = 1; l <= levels; l++) {
      const u = uni(16); device.queue.writeBuffer(u, 0, new Uint32Array([...dims[l - 1], ...dims[l]]));
      this.bgDown.push(bg(pDown, [this.bufs[l - 1], this.bufs[l], u]));
    }
    this.tempU = uni(16); this.roiU = uni(64); this.compU = uni(32);
    this.bgTemp = bg(pTemp, [this.bufs[levels], this.state, this.filt, this.tempU]);
    this.roiLevel = Math.min(1, levels);
    this.bgRoi = bg(pRoi, [this.bufs[this.roiLevel], this.filt, this.roiOut, this.roiU]);
    this.bgComp = bg(pComp, [this.tex.createView(), this.filt, this.compU]);
    this.lastT = null; this.frames = 0;
  }

  reset() { this.lastT = null; }

  // Uploads one frame taken at time t (seconds), runs every pass and, when `target` (a GPUTextureView of size w x h
  // in this.format) is given, draws the magnified frame into it. Submits before returning; the promise resolves to
  // the ROI means.
  frame(source, t, target) {
    const { device, w, h, levels, params: P } = this, q = device.queue;
    q.copyExternalImageToTexture({ source }, { texture: this.tex }, [w, h]);
    const init = this.lastT === null || t <= this.lastT, dt = init ? 0 : t - this.lastT;
    this.lastT = t; this.frames++;
    const { kh, kl } = gains(dt, P.fl, P.fh), tb = new ArrayBuffer(16);
    new Uint32Array(tb, 0, 2).set([this.n, init ? 1 : 0]); new Float32Array(tb, 8, 2).set([kh, kl]);
    q.writeBuffer(this.tempU, 0, tb);
    const [rw, rh] = this.dims[this.roiLevel];
    q.writeBuffer(this.roiU, 0, new Uint32Array([rw, rh, this.lw, this.lh, ...rectPx(this.face, rw, rh),
      ...rectPx(this.bg, rw, rh), ...rectPx(this.face, this.lw, this.lh)]));
    const cb = new ArrayBuffer(32);
    new Uint32Array(cb, 0, 6).set([w, h, this.lw, this.lh, P.view, 0]); new Float32Array(cb, 24, 2).set([P.amp, 0]);
    q.writeBuffer(this.compU, 0, cb);
    const enc = device.createCommandEncoder(), p = enc.beginComputePass(), wg = (n, s) => Math.ceil(n / s);
    p.setPipeline(this.pCopy); p.setBindGroup(0, this.bgCopy); p.dispatchWorkgroups(wg(w, 8), wg(h, 8));
    p.setPipeline(this.pDown);
    for (let l = 1; l <= levels; l++) { p.setBindGroup(0, this.bgDown[l - 1]); p.dispatchWorkgroups(wg(this.dims[l][0], 8), wg(this.dims[l][1], 8)); }
    p.setPipeline(this.pTemp); p.setBindGroup(0, this.bgTemp); p.dispatchWorkgroups(wg(this.n, 64));
    p.setPipeline(this.pRoi); p.setBindGroup(0, this.bgRoi); p.dispatchWorkgroups(1);
    p.end();
    if (target) {
      const r = enc.beginRenderPass({ colorAttachments: [{ view: target, loadOp: "clear", storeOp: "store", clearValue: [0, 0, 0, 1] }] });
      r.setPipeline(this.pComp); r.setBindGroup(0, this.bgComp); r.draw(3); r.end();
    }
    const stage = this.stages.pop() || device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(this.roiOut, 0, stage, 0, 64);
    q.submit([enc.finish()]);
    return stage.mapAsync(GPUMapMode.READ).then(() => {
      const a = new Float32Array(stage.getMappedRange().slice(0)); stage.unmap(); this.stages.push(stage);
      return { face: [a[0], a[1], a[2]], bg: [a[4], a[5], a[6]], filt: [a[8], a[9], a[10]] };
    });
  }

  // Copies a GPU buffer back (test use): returns Float32Array.
  async read(buffer) {
    const s = this.device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(buffer, 0, s, 0, buffer.size);
    this.device.queue.submit([enc.finish()]); await s.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(s.getMappedRange().slice(0)); s.unmap(); s.destroy(); return a;
  }

  destroy() { this.tex.destroy(); [...this.bufs, this.state, this.filt, this.roiOut, this.tempU, this.roiU, this.compU, ...this.stages].forEach((b) => b.destroy()); }
}

// The same pipeline in float64 JS, one frame at a time. `rgba` is the 8-bit frame the GPU saw.
export class Reference {
  constructor(w, h, levels) {
    Object.assign(this, { w, h, levels, dims: pyramidDims(w, h, levels), lastT: null });
    const [lw, lh] = this.dims[levels]; this.lw = lw; this.lh = lh;
    this.st = new Float64Array(lw * lh * 12);
  }
  step(rgba, t, P, face, bg) {
    const { w, h, levels, dims } = this, W = [0.0625, 0.25, 0.375, 0.25, 0.0625];
    let cur = new Float64Array(w * h * 3);
    for (let i = 0; i < w * h; i++) for (let c = 0; c < 3; c++) cur[i * 3 + c] = rgba[i * 4 + c] / 255;
    const lev = [cur];
    for (let l = 1; l <= levels; l++) {
      const [sw, sh] = dims[l - 1], [dw, dh] = dims[l], out = new Float64Array(dw * dh * 3);
      for (let y = 0; y < dh; y++) for (let x = 0; x < dw; x++) for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let j = 0; j < 5; j++) {
          const yy = Math.min(sh - 1, Math.max(0, 2 * y + j - 2));
          for (let k = 0; k < 5; k++) s += W[j] * W[k] * cur[(yy * sw + Math.min(sw - 1, Math.max(0, 2 * x + k - 2))) * 3 + c];
        }
        out[(y * dw + x) * 3 + c] = s;
      }
      lev.push(cur = out);
    }
    const init = this.lastT === null || t <= this.lastT, { kh, kl } = gains(init ? 0 : t - this.lastT, P.fl, P.fh);
    this.lastT = t;
    const n = this.lw * this.lh, filt = new Float64Array(n * 3), S = this.st;
    for (let i = 0; i < n * 3; i++) {
      const v = cur[i], o = (i / 3 | 0) * 12 + i % 3;   // st layout: per pixel a1, a2, b1, b2 as rgb triples
      if (init) { S[o] = S[o + 3] = S[o + 6] = S[o + 9] = v; }
      else { S[o] += kh * (v - S[o]); S[o + 3] += kh * (S[o] - S[o + 3]); S[o + 6] += kl * (v - S[o + 6]); S[o + 9] += kl * (S[o + 6] - S[o + 9]); }
      filt[i] = S[o + 3] - S[o + 9];
    }
    const rl = Math.min(1, levels), [rw, rh] = dims[rl];
    const mean = (img, iw, r) => { const m = [0, 0, 0]; let k = 0;
      for (let y = r[1]; y < r[3]; y++) for (let x = r[0]; x < r[2]; x++, k++) for (let c = 0; c < 3; c++) m[c] += img[(y * iw + x) * 3 + c];
      return m.map((v) => v / k); };
    const roi = { face: mean(lev[rl], rw, rectPx(face, rw, rh)), bg: mean(lev[rl], rw, rectPx(bg, rw, rh)), filt: mean(filt, this.lw, rectPx(face, this.lw, this.lh)) };
    const comp = new Uint8Array(w * h * 4), { lw, lh } = this;
    const at = (x, y, c) => filt[(Math.min(lh - 1, Math.max(0, y)) * lw + Math.min(lw - 1, Math.max(0, x))) * 3 + c];
    for (let py = 0; py < h; py++) for (let px = 0; px < w; px++) {
      const qx = (px + 0.5) * lw / w - 0.5, qy = (py + 0.5) * lh / h - 0.5, x0 = Math.floor(qx), y0 = Math.floor(qy), fx = qx - x0, fy = qy - y0;
      for (let c = 0; c < 3; c++) {
        const f = (at(x0, y0, c) * (1 - fx) + at(x0 + 1, y0, c) * fx) * (1 - fy) + (at(x0, y0 + 1, c) * (1 - fx) + at(x0 + 1, y0 + 1, c) * fx) * fy;
        const o = P.view === 1 ? 0.5 + P.amp * f : P.view === 2 ? lev[0][(py * w + px) * 3 + c] : lev[0][(py * w + px) * 3 + c] + P.amp * f;
        comp[(py * w + px) * 4 + c] = Math.round(Math.min(1, Math.max(0, o)) * 255);
      }
      comp[(py * w + px) * 4 + 3] = 255;
    }
    return { levels: lev, filt, roi, comp };
  }
}
