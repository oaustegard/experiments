// A 2D membrane on WebGPU, stepped once per audio sample. Walls come from a mask the page builds from the camera.
//
// The scheme is Bilbao's lossy membrane (Numerical Sound Synthesis, ch. 11): on a grid of spacing h with time step
// k = 1/sr, L the 5-point Laplacian (sum of 4 neighbours - 4u),
//
//   u[n+1] = ( 2u[n] - (1 - a0) u[n-1] + lam2 L u[n] + b (L u[n] - L u[n-1]) ) / (1 + a0)  + force,  then * mask
//
// lam2 = (c k / h)^2, a0 = sigma0 k (frequency-independent loss), b = 2 sigma1 k / h^2 (loss that grows with
// frequency). It is stable for lam2 + 2b <= 1/2. A wall cell has mask 0, so it is held at zero (a clamped edge);
// outside the grid counts as wall too. The previous Laplacian is kept per cell, so a thread needs only its
// neighbours' current displacement, never their history.
//
// Two kernels run the same arithmetic:
//   tile   one workgroup holds the whole grid in workgroup memory and runs every step of a block in one dispatch,
//          two barriers per step. The grid is capped by workgroup memory (4 bytes a cell).
//   multi  one dispatch per step over 8x8 workgroups in storage buffers. Any grid size, a dispatch per sample.
//
// Used by the page (index.html), the bridge benchmark (../bench/job.js) and the checks (../test/verify.mjs).

export const MAXS = 8;   // strikes per block

// A struck membrane's force: a half-sine pulse T samples long starting at sample js, spread over a smooth bump
// of radius r cells. Contact time follows the camera-resonator mallet: a harder strike is shorter and brighter.
export function strikeForce(s, steps) {
  const f = new Float64Array(steps), T = Math.max(2, s.T);
  for (let n = Math.max(0, s.js); n < steps && n < s.js + T; n++) f[n] = s.amp * Math.sin(Math.PI * (n - s.js) / T);
  return f;
}

// Physical settings -> per-sample coefficients. tension in 0..1 maps lam2 from 0.10 to 0.45; t60 is the decay of
// the lowest modes, hf the extra decay at high frequency (b is kept under the stability bound).
export function coeffs({ tension = 1, t60 = 2.5, hf = 0.5, sr = 48000 } = {}) {
  const lam2 = 0.10 + 0.35 * Math.min(1, Math.max(0, tension));
  const a0 = 6.907755 / (t60 * sr);
  const b = Math.min(0.5 - lam2 - 1e-3, 0.02 * Math.min(1, Math.max(0, hf)));
  return { lam2, a0, b };
}

function wgslCommon(nx, ny) {
  return /* wgsl */`
struct P { nx: u32, ny: u32, steps: u32, ns: u32, lam2: f32, a0: f32, b: f32, inv: f32,
           pl: u32, pr: u32, gain: f32, step: u32, sx: array<vec4f, ${MAXS}> };   // sx: x, y, r^2, 0 (cells)
const NX = ${nx}u; const NY = ${ny}u; const N = ${nx * ny}u;
fn bump(i: u32, s: vec4f) -> f32 {
  let dx = f32(i % NX) - s.x; let dy = f32(i / NX) - s.y; let q = (dx * dx + dy * dy) / s.z;
  return select(0.0, (1.0 - q) * (1.0 - q), q < 1.0);
}`;
}

function wgslTile(nx, ny, wg) {
  const cpt = Math.ceil(nx * ny / wg);
  return wgslCommon(nx, ny) + /* wgsl */`
@group(0) @binding(0) var<storage, read_write> state: array<f32>;   // u, prev, lap, each N
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> force: array<f32>;         // MAXS x steps
@group(0) @binding(3) var<storage, read_write> outp: array<f32>;    // 2 x steps, interleaved
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> su: array<f32, N>;
const CPT = ${cpt}u; const WG = ${wg}u;
@compute @workgroup_size(${wg})
fn main(@builtin(local_invocation_index) t: u32) {
  var cu: array<f32, CPT>; var cp: array<f32, CPT>; var cl: array<f32, CPT>; var cm: array<f32, CPT>;
  for (var j = 0u; j < CPT; j++) {
    let i = t + j * WG;
    if (i < N) { cu[j] = state[i]; cp[j] = state[N + i]; cl[j] = state[2u * N + i]; cm[j] = mask[i]; su[i] = cu[j]; }
  }
  for (var n = 0u; n < p.steps; n++) {
    workgroupBarrier();
    var nu: array<f32, CPT>;
    for (var j = 0u; j < CPT; j++) {
      let i = t + j * WG;
      if (i >= N) { break; }
      let x = i % NX; let y = i / NX;
      var l = -4.0 * cu[j];
      if (x > 0u) { l += su[i - 1u]; }
      if (x + 1u < NX) { l += su[i + 1u]; }
      if (y > 0u) { l += su[i - NX]; }
      if (y + 1u < NY) { l += su[i + NX]; }
      var v = (2.0 * cu[j] - (1.0 - p.a0) * cp[j] + p.lam2 * l + p.b * (l - cl[j])) * p.inv;
      for (var s = 0u; s < p.ns; s++) {
        let f = force[s * p.steps + n];
        if (f != 0.0) { v += f * bump(i, p.sx[s]); }
      }
      v *= cm[j];
      if (i == p.pl) { outp[2u * n] = (v - cu[j]) * p.gain; }
      if (i == p.pr) { outp[2u * n + 1u] = (v - cu[j]) * p.gain; }
      cp[j] = cu[j]; cl[j] = l; nu[j] = v;
    }
    workgroupBarrier();
    for (var j = 0u; j < CPT; j++) {
      let i = t + j * WG;
      if (i < N) { cu[j] = nu[j]; su[i] = nu[j]; }
    }
  }
  for (var j = 0u; j < CPT; j++) {
    let i = t + j * WG;
    if (i < N) { state[i] = cu[j]; state[N + i] = cp[j]; state[2u * N + i] = cl[j]; }
  }
}`;
}

// multi: buffers a (u[n], read) and c (u[n-1] in, u[n+1] out) swap each step; lap is per cell, in place.
function wgslMulti(nx, ny) {
  return wgslCommon(nx, ny) + /* wgsl */`
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> c: array<f32>;
@group(0) @binding(2) var<storage, read_write> lap: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read> force: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@group(0) @binding(7) var<uniform> stepi: vec4u;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= NX || g.y >= NY) { return; }
  let i = g.y * NX + g.x; let n = stepi.x; let u = a[i];
  var l = -4.0 * u;
  if (g.x > 0u) { l += a[i - 1u]; }
  if (g.x + 1u < NX) { l += a[i + 1u]; }
  if (g.y > 0u) { l += a[i - NX]; }
  if (g.y + 1u < NY) { l += a[i + NX]; }
  var v = (2.0 * u - (1.0 - p.a0) * c[i] + p.lam2 * l + p.b * (l - lap[i])) * p.inv;
  for (var s = 0u; s < p.ns; s++) {
    let f = force[s * p.steps + n];
    if (f != 0.0) { v += f * bump(i, p.sx[s]); }
  }
  v *= mask[i];
  if (i == p.pl) { outp[2u * n] = (v - u) * p.gain; }
  if (i == p.pr) { outp[2u * n + 1u] = (v - u) * p.gain; }
  lap[i] = l; c[i] = v;
}`;
}

// The largest grid the tile kernel can hold on this device, and the workgroup size it would use.
export function tileLimits(device) {
  const L = device.limits;
  return { maxCells: Math.floor(L.maxComputeWorkgroupStorageSize / 4), wg: Math.min(1024, L.maxComputeInvocationsPerWorkgroup, L.maxComputeWorkgroupSizeX) };
}

export class DrumEngine {
  // opts: nx, ny, block, sr, kernel ('tile' | 'multi' | 'auto'), wg, and coeffs (lam2, a0, b), gain
  static async create(device, opts) {
    const e = new DrumEngine();
    Object.assign(e, { device, nx: opts.nx, ny: opts.ny, B: opts.block || 512, sr: opts.sr || 48000, ms: [] });
    e.N = e.nx * e.ny;
    const lim = tileLimits(device);
    e.kernel = opts.kernel && opts.kernel !== "auto" ? opts.kernel : (e.N <= lim.maxCells ? "tile" : "multi");
    if (e.kernel === "tile" && e.N > lim.maxCells) throw new Error(`grid ${e.nx}x${e.ny} needs ${4 * e.N} bytes of workgroup memory; this GPU has ${4 * lim.maxCells}`);
    e.wg = opts.wg || lim.wg;
    e.set(opts);
    const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, C = GPUBufferUsage.COPY_SRC;
    const buf = (size, usage) => device.createBuffer({ size, usage });
    e.mask = buf(4 * e.N, S | D);
    device.queue.writeBuffer(e.mask, 0, new Float32Array(e.N).fill(1));
    e.force = buf(4 * MAXS * e.B, S | D);
    e.outp = buf(8 * e.B, S | C);
    e.pool = [];   // readback buffers: the page keeps two blocks in flight, and a buffer can have one map pending
    e.uni = buf(48 + 16 * MAXS, GPUBufferUsage.UNIFORM | D);
    if (e.kernel === "tile") {
      e.state = buf(12 * e.N, S | D | C);
      const mod = device.createShaderModule({ code: wgslTile(e.nx, e.ny, e.wg) });
      e.pipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
      e.bg = device.createBindGroup({ layout: e.pipe.getBindGroupLayout(0), entries: [e.state, e.mask, e.force, e.outp, e.uni].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
    } else {
      e.u = [buf(4 * e.N, S | D | C), buf(4 * e.N, S | D | C)]; e.lap = buf(4 * e.N, S | D);
      // one 256-byte slot per step holds its index; a dynamic offset would need an explicit layout, so each
      // step gets its own small bind group instead, built once
      e.steps = buf(256 * e.B, GPUBufferUsage.UNIFORM | D);
      const idx = new Uint32Array(64 * e.B); for (let n = 0; n < e.B; n++) idx[64 * n] = n;
      device.queue.writeBuffer(e.steps, 0, idx);
      const mod = device.createShaderModule({ code: wgslMulti(e.nx, e.ny) });
      e.pipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
      const lay = e.pipe.getBindGroupLayout(0);
      e.bgs = [];
      for (let n = 0; n < e.B; n++) {
        const a = e.u[n & 1], c = e.u[1 - (n & 1)];
        e.bgs.push(device.createBindGroup({ layout: lay, entries: [a, c, e.lap, e.mask, e.force, e.outp, e.uni].map((b, i) => ({ binding: i, resource: { buffer: b } }))
          .concat([{ binding: 7, resource: { buffer: e.steps, offset: 256 * n, size: 16 } }]) }));
      }
      // after B steps (B even) u[n] is back in u[0]; u[n-1] in u[1]
      if (e.B % 2) throw new Error("multi kernel needs an even block");
    }
    e.pick = [0, 0];
    return e;
  }
  set({ lam2, a0, b, gain } = {}) {
    if (lam2 !== undefined) this.lam2 = lam2;
    if (a0 !== undefined) this.a0 = a0;
    if (b !== undefined) this.b = b;
    if (gain !== undefined) this.gain = gain;
    this.lam2 ??= 0.45; this.a0 ??= 6e-5; this.b ??= 0.01; this.gain ??= 1;
  }
  setMask(m) { this.device.queue.writeBuffer(this.mask, 0, m instanceof Float32Array ? m : new Float32Array(m)); }
  setPickups(l, r) { this.pick = [l, r]; }
  clear() {
    const z = new Float32Array(this.kernel === "tile" ? 3 * this.N : this.N);
    if (this.kernel === "tile") this.device.queue.writeBuffer(this.state, 0, z);
    else for (const b of [...this.u, this.lap]) this.device.queue.writeBuffer(b, 0, z);
  }
  // strikes: [{x, y, r, amp, js, T}] in cells and samples. Returns interleaved stereo for B samples;
  // with { field: true } also the displacement u after the block.
  async render(strikes = [], { field = false } = {}) {
    const d = this.device, B = this.B, ss = strikes.slice(0, MAXS);
    const f = new Float32Array(MAXS * B);
    ss.forEach((s, k) => f.set(strikeForce(s, B), k * B));
    d.queue.writeBuffer(this.force, 0, f);
    const u = new ArrayBuffer(48 + 16 * MAXS), U = new Uint32Array(u), F = new Float32Array(u);
    U[0] = this.nx; U[1] = this.ny; U[2] = B; U[3] = ss.length;
    F[4] = this.lam2; F[5] = this.a0; F[6] = this.b; F[7] = 1 / (1 + this.a0);
    U[8] = this.pick[0]; U[9] = this.pick[1]; F[10] = this.gain;
    ss.forEach((s, k) => { F[12 + 4 * k] = s.x; F[13 + 4 * k] = s.y; F[14 + 4 * k] = s.r * s.r; });
    d.queue.writeBuffer(this.uni, 0, u);
    const t0 = performance.now();
    const enc = d.createCommandEncoder(), p = enc.beginComputePass();
    p.setPipeline(this.pipe);
    if (this.kernel === "tile") { p.setBindGroup(0, this.bg); p.dispatchWorkgroups(1); }
    else { const gx = Math.ceil(this.nx / 8), gy = Math.ceil(this.ny / 8); for (let n = 0; n < B; n++) { p.setBindGroup(0, this.bgs[n]); p.dispatchWorkgroups(gx, gy); } }
    p.end();
    const sb = this.pool.pop() || d.createBuffer({ size: 8 * B + 4 * this.N, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(this.outp, 0, sb, 0, 8 * B);
    if (field) enc.copyBufferToBuffer(this.kernel === "tile" ? this.state : this.u[0], 0, sb, 8 * B, 4 * this.N);
    d.queue.submit([enc.finish()]);
    await sb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(sb.getMappedRange(0, 8 * B).slice(0));
    const fld = field ? new Float32Array(sb.getMappedRange(8 * B, 4 * this.N).slice(0)) : null;
    sb.unmap(); this.pool.push(sb);
    this.ms.push(performance.now() - t0); if (this.ms.length > 400) this.ms.splice(0, 200);
    return field ? { out, field: fld } : out;
  }
  destroy() { for (const k of ["mask", "force", "outp", "uni", "state", "lap", "steps"]) this[k]?.destroy(); [...(this.u || []), ...this.pool].forEach((b) => b.destroy()); }
}

// The same scheme in float64, block by block. blocks: [{ strikes, mask? }]. Returns interleaved stereo.
export function reference({ nx, ny, B, lam2, a0, b, gain = 1, pick }, blocks, mask0) {
  const N = nx * ny; let u = new Float64Array(N), up = new Float64Array(N), lp = new Float64Array(N), m = Float64Array.from(mask0 || new Float64Array(N).fill(1));
  const out = new Float64Array(2 * B * blocks.length), inv = 1 / (1 + a0), nu = new Float64Array(N);
  blocks.forEach((blk, bi) => {
    if (blk.mask) m = Float64Array.from(blk.mask);
    const ss = (blk.strikes || []).slice(0, MAXS), fs = ss.map((s) => strikeForce(s, B));
    const bumps = ss.map((s) => { const w = new Float64Array(N); for (let i = 0; i < N; i++) { const dx = (i % nx) - s.x, dy = Math.floor(i / nx) - s.y, q = (dx * dx + dy * dy) / (s.r * s.r); w[i] = q < 1 ? (1 - q) ** 2 : 0; } return w; });
    for (let n = 0; n < B; n++) {
      for (let i = 0; i < N; i++) {
        const x = i % nx, y = (i / nx) | 0;
        let l = -4 * u[i];
        if (x > 0) l += u[i - 1]; if (x + 1 < nx) l += u[i + 1]; if (y > 0) l += u[i - nx]; if (y + 1 < ny) l += u[i + nx];
        let v = (2 * u[i] - (1 - a0) * up[i] + lam2 * l + b * (l - lp[i])) * inv;
        for (let k = 0; k < ss.length; k++) if (fs[k][n]) v += fs[k][n] * bumps[k][i];
        nu[i] = v * m[i]; lp[i] = l;
      }
      const o = 2 * (bi * B + n);
      out[o] = (nu[pick[0]] - u[pick[0]]) * gain; out[o + 1] = (nu[pick[1]] - u[pick[1]]) * gain;
      const t = up; up = u; u = t; u.set(nu);
    }
  });
  return out;
}

// Mode frequencies the scheme should ring at in an nx x ny rectangle (walls just outside the grid), lossless:
// the discrete dispersion relation sin^2(w/2) = lam2 (sin^2(p pi / 2(nx+1)) + sin^2(q pi / 2(ny+1))), against
// the continuous membrane's f = (sqrt(lam2) sr / 2) sqrt((p/(nx+1))^2 + (q/(ny+1))^2).
export function rectModes(nx, ny, lam2, sr, count = 12) {
  const list = [];
  for (let p = 1; p <= 8; p++) for (let q = 1; q <= 8; q++) {
    const sx = Math.sin(p * Math.PI / (2 * (nx + 1))), sy = Math.sin(q * Math.PI / (2 * (ny + 1)));
    const w = 2 * Math.asin(Math.sqrt(lam2 * (sx * sx + sy * sy)));
    list.push({ p, q, fd: w * sr / (2 * Math.PI), fc: Math.sqrt(lam2) * sr / 2 * Math.hypot(p / (nx + 1), q / (ny + 1)) });
  }
  return list.sort((a, c) => a.fd - c.fd).slice(0, count);
}

// GPU block stream against reference() on a random mask with random strikes. Returns error in dB re peak.
export async function selfTest(device, { nx = 40, ny = 30, block = 256, blocks = 6, kernel = "auto", sr = 48000, seed = 3 } = {}) {
  let a = seed >>> 0; const rnd = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const N = nx * ny, mk = () => { const m = new Float32Array(N).fill(1); for (let i = 0; i < N; i++) if (rnd() < 0.06) m[i] = 0; for (let y = 0; y < ny; y++) m[y * nx + (nx >> 1)] = y % 9 < 6 ? 0 : 1; return m; };
  const c = { ...coeffs({ tension: 1, t60: 1.5, hf: 0.6, sr }), gain: 1 };
  const e = await DrumEngine.create(device, { nx, ny, block, sr, kernel, ...c });
  const pick = [3 + 5 * nx, nx - 4 + (ny - 5) * nx]; e.setPickups(...pick);
  const m0 = mk(); e.setMask(m0);
  const plan = [];
  for (let k = 0; k < blocks; k++) {
    const strikes = [];
    if (k % 2 === 0) for (let s = 0; s < 3; s++) strikes.push({ x: rnd() * nx, y: rnd() * ny, r: 1.5 + 2 * rnd(), amp: 0.3, js: Math.floor(rnd() * block), T: 8 + Math.floor(40 * rnd()) });
    plan.push({ strikes, mask: k === 3 ? mk() : null });
  }
  const gpu = [];
  for (const blk of plan) { if (blk.mask) e.setMask(blk.mask); gpu.push(...await e.render(blk.strikes)); }
  const kern = e.kernel; e.destroy();
  const ref = reference({ nx, ny, B: block, ...c, pick }, plan, m0);
  let pk = 0, se = 0; ref.forEach((v, i) => { pk = Math.max(pk, Math.abs(v)); se += (gpu[i] - v) ** 2; });
  return { kernel: kern, nx, ny, block, blocks, peak: pk, rmsDb: 20 * Math.log10(Math.sqrt(se / ref.length) / pk) };
}
