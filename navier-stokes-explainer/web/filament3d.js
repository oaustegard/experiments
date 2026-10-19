// Vortex filaments in 3D: closed loops of marker points carried by the
// Biot-Savart velocity of every loop (Rosenhead-Moore core). Inviscid, so it
// shows vortex stretching and nothing else. All-pairs on the GPU, RK2 in time.
//
// Kelvin: circulation round a material tube is conserved, and the tube keeps
// its volume, so stretching it by a factor s raises its vorticity by s. The
// render colours every segment by that factor.

const FIL_WGSL = /* wgsl */ `
struct FP { a: vec4f /* N, M, dt, delta^2 */, b: vec4f /* dl0, tracer count, tracer core^2, 0 */ };
@group(0) @binding(0) var<uniform> P: FP;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
@group(0) @binding(2) var<storage, read> base: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outp: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> stats: array<atomic<u32>, 4>;
@group(0) @binding(5) var<storage, read_write> tr: array<vec4f>;
fn nxt(i: u32, M: u32) -> u32 { return (i / M) * M + (i % M + 1u) % M; }
fn prv(i: u32, M: u32) -> u32 { return (i / M) * M + (i % M + M - 1u) % M; }
const TILE = 64u;
var<workgroup> tp: array<vec4f, 64>;
var<workgroup> tdl: array<vec3f, 64>;
fn velocity(x: vec3f, li: u32, d2: f32) -> vec3f {
  let N = u32(P.a.x); let M = u32(P.a.y);
  var u = vec3f(0.0);
  for (var t0 = 0u; t0 < N; t0 = t0 + TILE) {
    let j = t0 + li;
    if (j < N) {
      tp[li] = pos[j];
      tdl[li] = 0.5 * (pos[nxt(j, M)].xyz - pos[prv(j, M)].xyz);
    } else {
      tp[li] = vec4f(0.0); tdl[li] = vec3f(0.0);
    }
    workgroupBarrier();
    for (var k = 0u; k < TILE; k = k + 1u) {
      let r = x - tp[k].xyz;
      let q = dot(r, r) + d2;
      u = u + tp[k].w * cross(tdl[k], r) / (q * sqrt(q));
    }
    workgroupBarrier();
  }
  return u * 0.0795774715; // 1 / (4 pi)
}
@compute @workgroup_size(64)
fn stage1(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let N = u32(P.a.x); let i = min(g.x, N - 1u);
  let p = pos[i];
  let v = velocity(p.xyz, li, P.a.w);
  if (g.x < N) { outp[i] = vec4f(p.xyz + 0.5 * P.a.z * v, p.w); }
}
@compute @workgroup_size(64)
fn stage2(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let N = u32(P.a.x); let M = u32(P.a.y); let i = min(g.x, N - 1u);
  let v = velocity(pos[i].xyz, li, P.a.w);
  if (g.x < N) {
    let p0 = base[i];
    outp[i] = vec4f(p0.xyz + P.a.z * v, p0.w);
  }
}
// Passive smoke: tracers moved by the filaments' field with a fatter core, so
// they swirl round the tubes instead of being flung off them.
@compute @workgroup_size(64)
fn tracers(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let T = u32(P.b.y); let i = min(g.x, T - 1u);
  let x = tr[i];
  let v = velocity(x.xyz, li, P.b.z);
  if (g.x < T) { tr[i] = vec4f(x.xyz + P.a.z * v, x.w); }
}
@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) g: vec3u) {
  let N = u32(P.a.x); let M = u32(P.a.y);
  if (g.x >= N) { return; }
  let s = 0.5 * length(pos[nxt(g.x, M)].xyz - pos[prv(g.x, M)].xyz) / P.b.x;
  atomicMax(&stats[0], bitcast<u32>(s));
  atomicMax(&stats[1], bitcast<u32>(length(pos[g.x].xyz)));
}
`;

const DRAW_WGSL = /* wgsl */ `
struct R { vp: mat4x4f, a: vec4f /* N, M, thickness px, dl0 */, b: vec4f /* width, height, smax, 0 */ };
@group(0) @binding(0) var<uniform> U: R;
@group(0) @binding(1) var<storage, read> pos: array<vec4f>;
struct VO { @builtin(position) p: vec4f, @location(0) c: vec3f, @location(1) e: f32 };
fn nxt(i: u32, M: u32) -> u32 { return (i / M) * M + (i % M + 1u) % M; }
fn prv(i: u32, M: u32) -> u32 { return (i / M) * M + (i % M + M - 1u) % M; }
fn ramp(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  let c0 = vec3f(0.25, 0.55, 1.0); let c1 = vec3f(0.95, 0.95, 0.95); let c2 = vec3f(1.0, 0.55, 0.15); let c3 = vec3f(1.0, 0.15, 0.25);
  if (x < 0.33) { return mix(c0, c1, x / 0.33); }
  if (x < 0.66) { return mix(c1, c2, (x - 0.33) / 0.33); }
  return mix(c2, c3, (x - 0.66) / 0.34);
}
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  let M = u32(U.a.y);
  let j = nxt(ii, M);
  let A = U.vp * vec4f(pos[ii].xyz, 1.0);
  let B = U.vp * vec4f(pos[j].xyz, 1.0);
  let sa = A.xy / A.w; let sb = B.xy / B.w;
  let res = U.b.xy;
  var dir = (sb - sa) * res;
  if (length(dir) < 1e-6) { dir = vec2f(1.0, 0.0); }
  let n = normalize(vec2f(-dir.y, dir.x)) / res * U.a.z;
  var corner = array<vec2f, 6>(vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, -1.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let cc = corner[vi];
  let P0 = select(A, B, cc.x > 0.5);
  var o: VO;
  o.p = vec4f(P0.xy + n * cc.y * P0.w, P0.z, P0.w);
  // Stretch factor = local segment length / initial length; vorticity scales the same way.
  let s = 0.5 * length(pos[nxt(ii, M)].xyz - pos[prv(ii, M)].xyz) / U.a.w;
  o.c = ramp(log2(max(s, 1e-3)) / log2(max(U.b.z, 1.01)));
  o.e = cc.y;
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let shade = 1.0 - 0.45 * i.e * i.e;
  return vec4f(i.c * shade, 1.0);
}
`;

const SMOKE_WGSL = /* wgsl */ `
struct R { vp: mat4x4f, a: vec4f, b: vec4f /* width, height, smax, size px */ };
@group(0) @binding(0) var<uniform> U: R;
@group(0) @binding(1) var<storage, read> tr: array<vec4f>;
struct VO { @builtin(position) p: vec4f, @location(0) q: vec2f, @location(1) c: vec4f };
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
  var corner = array<vec2f, 6>(vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let t = tr[ii];
  let P = U.vp * vec4f(t.xyz, 1.0);
  let q = corner[vi];
  var o: VO;
  o.p = vec4f(P.xy + q * U.b.w / U.b.xy * P.w, P.z, P.w);
  o.q = q;
  let ring = floor(t.w);
  let bright = fract(t.w);
  let base = select(vec3f(1.0, 0.55, 0.22), vec3f(0.3, 0.75, 1.0), ring > 0.5);
  o.c = vec4f(base * (0.5 + bright), 0.32);
  return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let r2 = dot(i.q, i.q);
  if (r2 > 1.0) { discard; }
  let a = i.c.a * (1.0 - r2);
  return vec4f(i.c.rgb * a, a);
}
`;

function mat4mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far * nf, -1, 0, 0, far * near * nf, 0]);
}
function lookAt(eye, at, up) {
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const norm = (a) => { const l = Math.hypot(...a); return a.map((v) => v / l); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const z = norm(sub(eye, at)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}

// Each ring: centre c, unit normal n (direction of travel for gamma > 0), radius R.
function ringPoints(M, { c, n, R, gamma, wobble = 0, mode = 5, phase = 0 }) {
  const nn = Math.hypot(...n); const N = n.map((v) => v / nn);
  const tmp = Math.abs(N[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const cr = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  let e1 = cr(N, tmp); const l1 = Math.hypot(...e1); e1 = e1.map((v) => v / l1);
  const e2 = cr(N, e1);
  const pts = [];
  for (let k = 0; k < M; k++) {
    const th = (2 * Math.PI * k) / M;
    const r = R * (1 + wobble * Math.cos(mode * th + phase));
    // Points run counter-clockwise about n, so the ring's self-induced motion is along +n for gamma > 0.
    pts.push([c[0] + r * (Math.cos(th) * e1[0] + Math.sin(th) * e2[0]), c[1] + r * (Math.cos(th) * e1[1] + Math.sin(th) * e2[1]), c[2] + r * (Math.cos(th) * e1[2] + Math.sin(th) * e2[2]), gamma]);
  }
  return pts;
}

export const PRESETS = {
  headon: { label: 'Head-on rings', rings: [
    { c: [0, 0, -1.1], n: [0, 0, 1], R: 1, gamma: 1, wobble: 0.02, mode: 6 },
    { c: [0, 0, 1.1], n: [0, 0, -1], R: 1, gamma: 1, wobble: 0.02, mode: 6, phase: 0.6 },
  ], dt: 0.03, tmax: 7 },
  oblique: { label: 'Tilted rings', rings: [
    { c: [-1.15, 0, 0], n: [1, 0, 0.55], R: 1, gamma: 1 },
    { c: [1.15, 0, 0], n: [-1, 0, 0.55], R: 1, gamma: 1 },
  ], dt: 0.03, tmax: 9 },
  leapfrog: { label: 'Leapfrogging rings', rings: [
    { c: [0, 0, -1.6], n: [0, 0, 1], R: 1, gamma: 1 },
    { c: [0, 0, -1.1], n: [0, 0, 1], R: 1, gamma: 1 },
  ], dt: 0.03, tmax: 14 },
};

export class Filaments {
  constructor(device, canvas, { M = 384, T = 8000 } = {}) {
    this.device = device; this.canvas = canvas; this.M = M; this.T = T;
    this.delta = 0.08;
    this.yaw = 0.6; this.pitch = 0.35; this.dist = 6.2; this.userDist = 6.2; this.autoRotate = true; this.showSmoke = true; this.stretchLimit = 4.5;
    this.stats = { maxStretch: 1, time: 0 };
    this._busy = false;
    this.ctx = canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });
    const cm = device.createShaderModule({ code: FIL_WGSL });
    this.p1 = device.createComputePipeline({ layout: 'auto', compute: { module: cm, entryPoint: 'stage1' } });
    this.p2 = device.createComputePipeline({ layout: 'auto', compute: { module: cm, entryPoint: 'stage2' } });
    this.pm = device.createComputePipeline({ layout: 'auto', compute: { module: cm, entryPoint: 'measure' } });
    this.pt = device.createComputePipeline({ layout: 'auto', compute: { module: cm, entryPoint: 'tracers' } });
    const sm = device.createShaderModule({ code: SMOKE_WGSL });
    this.smoke = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: sm, entryPoint: 'vs' },
      fragment: { module: sm, entryPoint: 'fs', targets: [{ format: this.format, blend: { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'always' },
    });
    const dm = device.createShaderModule({ code: DRAW_WGSL });
    this.draw = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: dm, entryPoint: 'vs' },
      fragment: { module: dm, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.U = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.RU = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.statsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.statsRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this._attachOrbit();
  }

  load(name) {
    const pre = PRESETS[name];
    this.preset = name; this.dt = pre.dt; this.tmax = pre.tmax;
    const pts = pre.rings.flatMap((r) => ringPoints(this.M, r));
    this.N = pts.length;
    this.dl0 = (2 * Math.PI * pre.rings[0].R) / this.M;
    const data = new Float32Array(pts.flat());
    const d = this.device;
    const mk = () => d.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.X = mk(); this.Y = mk(); this.MID = mk();
    d.queue.writeBuffer(this.X, 0, data);
    // Smoke: a torus of tracers round each ring's core.
    let seed = 12345;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const tr = new Float32Array(this.T * 4);
    const per = Math.floor(this.T / pre.rings.length);
    for (let k = 0; k < this.T; k++) {
      const ri = Math.min(pre.rings.length - 1, Math.floor(k / per));
      const ring = pre.rings[ri];
      const j = Math.floor(rnd() * this.M);
      const p = pts[ri * this.M + j], q = pts[ri * this.M + (j + 1) % this.M];
      const c = ring.c;
      const rad = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; const rl = Math.hypot(...rad);
      const nn = Math.hypot(...ring.n); const nrm = ring.n.map((v) => v / nn);
      const a = 2 * Math.PI * rnd(), r = 0.2 * Math.sqrt(rnd());
      const f = rnd();
      for (let e = 0; e < 3; e++) tr[k * 4 + e] = p[e] + f * (q[e] - p[e]) + r * (Math.cos(a) * rad[e] / rl + Math.sin(a) * nrm[e]);
      tr[k * 4 + 3] = ri + 0.999 * rnd();
    }
    this.TR = d.createBuffer({ size: tr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.TR, 0, tr);
    this.trGroup = d.createBindGroup({ layout: this.pt.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.U } }, { binding: 1, resource: { buffer: this.MID } }, { binding: 5, resource: { buffer: this.TR } }] });
    this.smokeGroup = d.createBindGroup({ layout: this.smoke.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.RU } }, { binding: 1, resource: { buffer: this.TR } }] });
    const g = (pipe, pos, base, out) => d.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.U } }, { binding: 1, resource: { buffer: pos } },
        // layout 'auto' keeps only the bindings an entry point uses: stage1 never reads base.
        ...(pipe === this.pm ? [{ binding: 4, resource: { buffer: this.statsBuf } }]
          : pipe === this.p1 ? [{ binding: 3, resource: { buffer: out } }]
          : [{ binding: 2, resource: { buffer: base } }, { binding: 3, resource: { buffer: out } }]),
      ],
    });
    this.groups = [
      { s1: g(this.p1, this.X, this.X, this.MID), s2: g(this.p2, this.MID, this.X, this.Y), m: g(this.pm, this.Y), draw: this._drawGroup(this.Y) },
      { s1: g(this.p1, this.Y, this.Y, this.MID), s2: g(this.p2, this.MID, this.Y, this.X), m: g(this.pm, this.X), draw: this._drawGroup(this.X) },
    ];
    this.cur = 0; // groups[cur] advances from the buffer holding the current state
    this._drawNow = this._drawGroup(this.X);
    this.time = 0;
    this.done = false;
    this.gen = (this.gen ?? 0) + 1; // readbacks from before a reload are dropped
    this.stats = { maxStretch: 1, maxR: 0, time: 0 };
    this.history = [];
  }

  _drawGroup(buf) {
    return this.device.createBindGroup({ layout: this.draw.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.RU } }, { binding: 1, resource: { buffer: buf } }] });
  }

  step(n = 1) {
    if (this.done) return;
    const d = this.device;
    d.queue.writeBuffer(this.U, 0, new Float32Array([this.N, this.M, this.dt, this.delta * this.delta, this.dl0, this.T, (2 * this.delta) ** 2, 0]));
    const enc = d.createCommandEncoder();
    const wg = Math.ceil(this.N / 64);
    for (let k = 0; k < n; k++) {
      const G = this.groups[this.cur];
      let p = enc.beginComputePass(); p.setPipeline(this.p1); p.setBindGroup(0, G.s1); p.dispatchWorkgroups(wg); p.end();
      // Tracers step with the midpoint filament positions, which MID holds until stage2 has read them.
      p = enc.beginComputePass(); p.setPipeline(this.pt); p.setBindGroup(0, this.trGroup); p.dispatchWorkgroups(Math.ceil(this.T / 64)); p.end();
      p = enc.beginComputePass(); p.setPipeline(this.p2); p.setBindGroup(0, G.s2); p.dispatchWorkgroups(wg); p.end();
      this._drawNow = G.draw;
      this._lastMeasure = G.m;
      this.cur = 1 - this.cur;
      this.time += this.dt;
    }
    d.queue.submit([enc.finish()]);
    this._steps = (this._steps ?? 0) + n;
    if (this._steps % 10 < n) this.measure();
  }

  measure() {
    if (this._busy || !this._lastMeasure) return;
    this._busy = true;
    const d = this.device;
    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.statsBuf);
    const p = enc.beginComputePass(); p.setPipeline(this.pm); p.setBindGroup(0, this._lastMeasure); p.dispatchWorkgroups(Math.ceil(this.N / 64)); p.end();
    enc.copyBufferToBuffer(this.statsBuf, 0, this.statsRead, 0, 16);
    d.queue.submit([enc.finish()]);
    const t = this.time, gen = this.gen;
    this.statsRead.mapAsync(GPUMapMode.READ).then(() => {
      const f = new Float32Array(this.statsRead.getMappedRange().slice(0));
      this.statsRead.unmap();
      this._busy = false;
      if (gen !== this.gen) return;
      this.stats = { maxStretch: f[0], maxR: f[1], time: t };
      // Past ~5x stretch the marker spacing exceeds the core size and the model is
      // no longer resolved, so the run stops there rather than show numerical breakup.
      if (f[0] > this.stretchLimit || t >= this.tmax) this.done = true;
      this.onStats?.(this.stats);
    }).catch(() => { this._busy = false; });
  }

  render() {
    const d = this.device;
    const w = this.canvas.width, h = this.canvas.height;
    if (!this.depth || this.depth.width !== w || this.depth.height !== h) {
      this.depth?.destroy();
      this.depth = d.createTexture({ size: [w, h], format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    }
    if (this.autoRotate) this.yaw += 0.003;
    // Follow the rings as they spread: keep the farthest point inside the view.
    const want = Math.max(this.userDist, 2.9 * (this.stats.maxR ?? 0));
    this.dist += (want - this.dist) * 0.05;
    const eye = [this.dist * Math.cos(this.pitch) * Math.sin(this.yaw), this.dist * Math.sin(this.pitch), this.dist * Math.cos(this.pitch) * Math.cos(this.yaw)];
    const vp = mat4mul(perspective(0.8, w / h, 0.1, 200), lookAt(eye, [0, 0, 0], [0, 1, 0]));
    const smax = Math.max(4, this.stats.maxStretch);
    const dpr = window.devicePixelRatio || 1;
    d.queue.writeBuffer(this.RU, 0, new Float32Array([...vp, this.N, this.M, 3.2 * dpr, this.dl0, w, h, smax, 2.6 * dpr]));
    const enc = d.createCommandEncoder();
    const rp = enc.beginRenderPass({
      colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0.035, 0.04, 0.06, 1] }],
      depthStencilAttachment: { view: this.depth.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1 },
    });
    if (this.showSmoke) { rp.setPipeline(this.smoke); rp.setBindGroup(0, this.smokeGroup); rp.draw(6, this.T); }
    rp.setPipeline(this.draw); rp.setBindGroup(0, this._drawNow); rp.draw(6, this.N); rp.end();
    d.queue.submit([enc.finish()]);
  }

  _attachOrbit() {
    const el = this.canvas; let last = null;
    el.style.touchAction = 'none';
    el.addEventListener('pointerdown', (e) => { last = [e.clientX, e.clientY]; this.autoRotate = false; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointerup', () => { last = null; });
    el.addEventListener('pointermove', (e) => {
      if (!last) return;
      this.yaw -= (e.clientX - last[0]) * 0.008;
      this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + (e.clientY - last[1]) * 0.008));
      last = [e.clientX, e.clientY];
    });
    el.addEventListener('wheel', (e) => { e.preventDefault(); this.userDist = Math.max(2.5, Math.min(30, this.userDist * Math.exp(e.deltaY * 0.001))); }, { passive: false });
  }
}
