// refract.js: WebGPU performance-cliff sweeps with verified kernels. ES module, no dependencies.
// Runs in a browser (index.html) and in Node on Dawn (run/check.mjs); it touches no DOM.
//
//   const ctx = await openDevice(navigator.gpu);
//   const res = await SWEEPS.gemv.run(ctx, { quick: true, onProgress });
//   const profile = makeProfile(ctx, { gemv: res });
//
// Every config is compiled (getCompilationInfo + an error scope), run once on seeded data, and
// compared with a JS reference before it is timed. A config that fails the check is reported as
// 'wrong' and can never be a winner.

export const VERSION = 'refractometer/1';

// WebGPU's default limits: what a device created without requiredLimits (onnxruntime-web's, for
// one) gets. Configs above them are flagged needsLimits and a second winner is kept without them.
export const DEFAULT_LIMITS = {
  maxComputeWorkgroupStorageSize: 16384,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxComputeWorkgroupSizeY: 256,
  maxStorageBufferBindingSize: 134217728,
  maxBufferSize: 268435456,
};

const U = () => globalThis.GPUBufferUsage;
const now = () => globalThis.performance.now();

// ------------------------------------------------------------------ device

export function adapterInfo(adapter) {
  const i = adapter.info || {};
  const o = {};
  for (const k of ['vendor', 'architecture', 'device', 'description', 'subgroupMinSize', 'subgroupMaxSize', 'isFallbackAdapter']) {
    if (i[k] !== undefined) o[k] = i[k];
  }
  return o;
}

// The profile key. Chrome often reports an empty device string, so the key keeps empty parts as
// '-' and tuned.js falls back to vendor/architecture.
export function profileKey(info) {
  const part = (s) => (String(s ?? '').trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '') || '-');
  return `${part(info.vendor)}/${part(info.architecture)}/${part(info.device)}`;
}

function plainLimits(L) {
  const o = {};
  for (const k in L) if (typeof L[k] === 'number') o[k] = L[k];
  return o;
}

export async function openDevice(gpu, opts = {}) {
  if (!gpu) throw new Error('WebGPU is not available in this browser (navigator.gpu is missing)');
  const adapter = await gpu.requestAdapter({ powerPreference: opts.powerPreference ?? 'high-performance' });
  if (!adapter) throw new Error('requestAdapter() returned null: no usable WebGPU adapter');
  if (!adapter.info && adapter.requestAdapterInfo) adapter.info = await adapter.requestAdapterInfo();
  const info = adapterInfo(adapter);
  const features = ['timestamp-query', 'subgroups', 'shader-f16'].filter((f) => adapter.features.has(f));
  const AL = adapter.limits;
  const requiredLimits = {};
  for (const k of Object.keys(DEFAULT_LIMITS)) if (AL[k] !== undefined) requiredLimits[k] = AL[k];
  const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits });
  const lost = { reason: null };
  device.lost?.then((l) => { lost.reason = `${l.reason}: ${l.message}`; });
  const uncaptured = [];
  device.addEventListener?.('uncapturederror', (e) => uncaptured.push(e.error?.message ?? String(e)));
  return {
    gpu, adapter, device, info, key: profileKey(info), features,
    allFeatures: [...adapter.features].sort(),
    limits: plainLimits(device.limits),
    timing: features.includes('timestamp-query') && !opts.forceWall ? 'timestamp' : 'wall',
    lost, uncaptured, pipelines: new Map(),
  };
}

// Does a config need more than WebGPU's default limits?
export function needsLimits(wgStorageBytes, invocations, sizeX = invocations) {
  return wgStorageBytes > DEFAULT_LIMITS.maxComputeWorkgroupStorageSize
    || invocations > DEFAULT_LIMITS.maxComputeInvocationsPerWorkgroup
    || sizeX > DEFAULT_LIMITS.maxComputeWorkgroupSizeX;
}

function fits(ctx, wgStorageBytes, invocations, sizeX = invocations, sizeY = 1) {
  const L = ctx.limits;
  return wgStorageBytes <= L.maxComputeWorkgroupStorageSize
    && invocations <= L.maxComputeInvocationsPerWorkgroup
    && sizeX <= L.maxComputeWorkgroupSizeX && sizeY <= (L.maxComputeWorkgroupSizeY ?? 256);
}

export class KernelError extends Error {
  constructor(label, message) { super(`${label}: ${message}`); this.label = label; }
}

// Compile and build a pipeline, surfacing every compiler message and validation error.
export async function pipeline(ctx, code, label) {
  const hit = ctx.pipelines.get(code);
  if (hit) return hit;
  const dev = ctx.device;
  dev.pushErrorScope('validation');
  const module = dev.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errs = info.messages.filter((m) => m.type === 'error').map((m) => `${m.lineNum}:${m.linePos} ${m.message}`);
  let p = null;
  if (!errs.length) {
    try {
      const desc = { layout: 'auto', compute: { module, entryPoint: 'main' }, label };
      // ctx.syncPipelines: an escape hatch for hosts whose async pipeline path misbehaves.
      p = ctx.syncPipelines ? dev.createComputePipeline(desc) : await dev.createComputePipelineAsync(desc);
    } catch (e) { errs.push(String(e.message ?? e)); }
  }
  const scoped = await dev.popErrorScope();
  if (scoped) errs.push(scoped.message);
  if (errs.length) throw new KernelError(label, errs.join(' | '));
  ctx.pipelines.set(code, p);
  return p;
}

// ------------------------------------------------------------------ buffers

function makeBuffer(ctx, data, usage, label) {
  const size = Math.max(16, Math.ceil(data.byteLength / 16) * 16);
  const b = ctx.device.createBuffer({ size, usage: usage | U().COPY_DST | U().COPY_SRC, mappedAtCreation: true, label });
  new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  b.unmap();
  return b;
}

function emptyBuffer(ctx, bytes, label) {
  return ctx.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: U().STORAGE | U().COPY_SRC | U().COPY_DST, label });
}

async function readBuffer(ctx, buf, bytes, offset = 0) {
  const rb = ctx.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: U().MAP_READ | U().COPY_DST });
  const enc = ctx.device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, offset, rb, 0, Math.ceil(bytes / 4) * 4);
  ctx.device.queue.submit([enc.finish()]);
  await rb.mapAsync(globalThis.GPUMapMode.READ);
  const out = rb.getMappedRange().slice(0);
  rb.unmap(); rb.destroy();
  return out;
}

function bindGroup(ctx, p, buffers) {
  return ctx.device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: buffers.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })),
  });
}

// One dispatch of the kernel; used for verification.
async function runOnce(ctx, p, bg, grid, clear = []) {
  const enc = ctx.device.createCommandEncoder();
  for (const b of clear) enc.clearBuffer(b);
  const pass = enc.beginComputePass();
  pass.setPipeline(p); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...grid);
  pass.end();
  ctx.device.queue.submit([enc.finish()]);
  await ctx.device.queue.onSubmittedWorkDone();
}

// Seeded PRNG (mulberry32), so every device verifies against the same data.
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Small integers 0..6 from a hash of the index: sums of them stay exact in f32.
const hash7 = (i) => (Math.imul(i, 2654435761) >>> 0) % 7;

// ------------------------------------------------------------------ timing

function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function stats(samples) {
  const s = [...samples].sort((a, b) => a - b);
  return { median: quantile(s, 0.5), p10: quantile(s, 0.1), p90: quantile(s, 0.9), n: s.length };
}

// Time one dispatch of (p, bg, grid). Each sample is one compute pass holding `inner` back-to-back
// dispatches, with inner calibrated so a pass lasts about targetMs; the per-dispatch time is the
// pass time / inner. Timestamp mode reads beginning/end-of-pass timestamps (Chrome quantizes them
// to 100 us unless its developer flag is on, hence the long passes); wall mode times
// submit -> onSubmittedWorkDone with performance.now(). Returns microseconds per dispatch.
export async function measure(ctx, p, bg, grid, opts = {}) {
  const dev = ctx.device;
  const reps = opts.reps ?? 25, warmup = opts.warmup ?? 3;
  const targetMs = opts.targetMs ?? (ctx.timing === 'timestamp' ? 2 : 4);
  const record = (pass, n) => {
    pass.setPipeline(p); pass.setBindGroup(0, bg);
    for (let i = 0; i < n; i++) pass.dispatchWorkgroups(...grid);
  };
  const wall = async (n) => {
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass(); record(pass, n); pass.end();
    const cb = enc.finish();
    const t0 = now();
    dev.queue.submit([cb]);
    await dev.queue.onSubmittedWorkDone();
    return now() - t0;
  };
  // Calibrate with wall time: the slope between 1 and 8 dispatches removes submit overhead.
  await wall(1);
  const t1 = await wall(1), t8 = await wall(8);
  const per = Math.max((t8 - t1) / 7, t8 / 64, 1e-4);
  const inner = Math.max(1, Math.min(opts.maxInner ?? 512, Math.round(targetMs / per)));
  for (let i = 0; i < warmup; i++) await wall(inner);

  if (ctx.timing === 'timestamp') {
    const qs = dev.createQuerySet({ type: 'timestamp', count: 2 * reps });
    const res = dev.createBuffer({ size: 16 * reps, usage: U().QUERY_RESOLVE | U().COPY_SRC });
    const rb = dev.createBuffer({ size: 16 * reps, usage: U().MAP_READ | U().COPY_DST });
    const enc = dev.createCommandEncoder();
    for (let r = 0; r < reps; r++) {
      const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * r, endOfPassWriteIndex: 2 * r + 1 } });
      record(pass, inner); pass.end();
    }
    enc.resolveQuerySet(qs, 0, 2 * reps, res, 0);
    enc.copyBufferToBuffer(res, 0, rb, 0, 16 * reps);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(globalThis.GPUMapMode.READ);
    const t = new BigInt64Array(rb.getMappedRange().slice(0));
    rb.unmap(); rb.destroy(); res.destroy(); qs.destroy();
    const raw = [];
    for (let r = 0; r < reps; r++) raw.push(t[2 * r + 1] - t[2 * r]);
    const good = raw.filter((d) => d > 0n);
    if (good.length >= Math.max(3, reps / 2)) {
      const us = good.map((d) => Number(d) / 1000 / inner);
      const quantized = good.every((d) => d % 100000n === 0n);
      return { ...stats(us), inner, mode: 'timestamp', quantized };
    }
    // Some implementations return zeros; fall through to wall time.
  }
  const us = [];
  for (let r = 0; r < reps; r++) us.push((await wall(inner)) * 1000 / inner);
  return { ...stats(us), inner, mode: 'wall', quantized: false };
}

// ------------------------------------------------------------------ analysis

// Upper envelope of y over x (best config at each x), then cliffs: an x where the envelope falls
// below `drop` times the best seen at smaller x. Higher y is better everywhere in this file.
export function envelope(points) {
  const by = new Map();
  for (const p of points) if (p.ok && Number.isFinite(p.y)) {
    const b = by.get(p.x);
    if (!b || p.y > b.y) by.set(p.x, p);
  }
  return [...by.values()].sort((a, b) => a.x - b.x);
}

export function cliffs(points, drop = 0.75) {
  const env = envelope(points);
  const out = [];
  let best = -Infinity;
  for (let i = 0; i < env.length; i++) {
    if (i > 0 && env[i].y < drop * best && env[i - 1].y >= drop * best) {
      out.push({ x: env[i].x, from: env[i - 1].x, ratio: env[i].y / best });
    }
    best = Math.max(best, env[i].y);
  }
  return out;
}

function winners(points) {
  const ok = points.filter((p) => p.ok && Number.isFinite(p.y));
  const best = ok.reduce((a, p) => (!a || p.y > a.y ? p : a), null);
  const bestDefault = ok.filter((p) => !p.needsLimits).reduce((a, p) => (!a || p.y > a.y ? p : a), null);
  return { best, bestDefault };
}

const fmt = (x, d = 3) => Number(x.toPrecision(d));

// Verify then time one config. `build` returns { code, label, buffers, grid, check(ctx) -> {ok,
// maxErr, detail}, clear }. Failures are recorded, never thrown.
async function trial(ctx, cfg, build, opts, metric) {
  const pt = { config: cfg.config, x: cfg.x, series: cfg.series, needsLimits: !!cfg.needsLimits, ok: false };
  let k;
  try {
    k = await build();
    const p = await pipeline(ctx, k.code, k.label);
    const bg = bindGroup(ctx, p, k.bind);
    await runOnce(ctx, p, bg, k.grid, k.clear ?? []);
    const v = await k.check();
    pt.verify = { ok: v.ok, maxErr: v.maxErr, tol: v.tol };
    if (!v.ok) { pt.status = 'wrong'; pt.error = v.detail ?? 'output does not match the JS reference'; return pt; }
    if (opts.verifyOnly) { pt.status = 'verified'; return pt; }
    const t = await measure(ctx, p, bg, k.grid, opts);
    pt.time = { median_us: fmt(t.median, 4), p10_us: fmt(t.p10, 4), p90_us: fmt(t.p90, 4), inner: t.inner, mode: t.mode, quantized: t.quantized };
    pt.y = fmt(metric(t.median), 4);
    pt.ok = true; pt.status = 'ok';
  } catch (e) {
    pt.status = e instanceof KernelError ? 'invalid' : 'error';
    pt.error = String(e.message ?? e);
  } finally {
    k?.dispose?.();
  }
  return pt;
}

// Element-wise compare with an absolute tolerance per element.
function compare(got, ref, tol) {
  let maxErr = 0, bad = -1;
  for (let i = 0; i < ref.length; i++) {
    const e = Math.abs(got[i] - ref[i]);
    if (!(e <= (typeof tol === 'number' ? tol : tol[i]))) { if (bad < 0) bad = i; }
    if (e > maxErr || Number.isNaN(e)) maxErr = Number.isNaN(e) ? Infinity : e;
  }
  return { ok: bad < 0, maxErr, tol: typeof tol === 'number' ? tol : 'per-element', detail: bad < 0 ? null : `first mismatch at ${bad}: got ${got[bad]}, want ${ref[bad]}` };
}

// ================================================================== kernels (WGSL generators)

export const WGSL = {
  // a) bandwidth: grid-stride vec4 copy.
  copy: ({ n4, wg = 256 }) => /* wgsl */ `
const N4: u32 = ${n4}u;
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
  let T = nw.x * ${wg}u;
  for (var i = g.x; i < N4; i += T) { dst[i] = src[i]; }
}`,

  // a) bandwidth: grid-stride vec4 read, one partial sum per workgroup.
  readReduce: ({ n4, wg = 256 }) => /* wgsl */ `
const N4: u32 = ${n4}u;
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> part: array<f32>;
var<workgroup> red: array<f32, ${wg}>;
@compute @workgroup_size(${wg})
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let T = nw.x * ${wg}u;
  var acc = vec4<f32>(0.0);
  for (var i = g.x; i < N4; i += T) { acc += src[i]; }
  red[l] = acc.x + acc.y + acc.z + acc.w;
  workgroupBarrier();
  for (var s = ${wg / 2}u; s > 0u; s >>= 1u) {
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
  if (l == 0u) { part[wid.x] = red[0]; }
}`,

  // b/d) reduction of N floats: each workgroup reduces WG*IPT contiguous values (thread l reads
  // l, l+WG, ... so loads coalesce), then a tree in workgroup memory ('tree') or subgroupAdd plus a
  // slot per subgroup ('subgroup'). Workgroups are indexed 2-D past 32768.
  reduce: ({ n, wg, ipt, variant = 'tree' }) => {
    const groups = n / (wg * ipt);
    const sub = variant === 'subgroup';
    return /* wgsl */ `${sub ? 'enable subgroups;\n' : ''}
const N: u32 = ${n}u;
const WG: u32 = ${wg}u;
const IPT: u32 = ${ipt}u;
const GROUPS: u32 = ${groups}u;
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<f32>;
${sub ? `var<workgroup> slots: array<f32, ${wg}>;\nvar<workgroup> cnt: atomic<u32>;` : `var<workgroup> red: array<f32, ${wg}>;`}
@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32${sub ? ',\n        @builtin(subgroup_invocation_id) sid: u32' : ''}) {
  let w = wid.x + wid.y * 32768u;
  if (w >= GROUPS) { return; }
  let base = w * WG * IPT;
  var acc = 0.0;
  for (var k = 0u; k < IPT; k++) { acc += src[base + l + k * WG]; }
${sub ? `  let s = subgroupAdd(acc);
  if (sid == 0u) { slots[atomicAdd(&cnt, 1u)] = s; }
  workgroupBarrier();
  if (l == 0u) {
    let m = atomicLoad(&cnt);
    var t = 0.0;
    for (var j = 0u; j < m; j++) { t += slots[j]; }
    part[w] = t;
  }` : `  red[l] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s >>= 1u) {
    if (l < s) { red[l] += red[l + s]; }
    workgroupBarrier();
  }
  if (l == 0u) { part[w] = red[0]; }`}
}`;
  },

  // b) q4 GEMV / skinny matmul in onnxruntime's MatMulNBits layout (4-bit, block 32, low nibble
  // first, f32 scales, packed 4-bit zero points), the layout eg2-wgsl-engine reads. Y[M,N] = X W^T.
  // Workgroup = C columns x S k-splits; it covers R rows of X, dequantizing each weight once for
  // all R rows. eg2-wgsl-engine's matmul is C=32, S=8, R=8.
  gemvQ4: ({ N, K, M, C, S, R }) => {
    const nb = K / 32, bps = nb / S, zrow = Math.ceil(nb / 2);
    return /* wgsl */ `
const N: u32 = ${N}u;
const K: u32 = ${K}u;
const M: u32 = ${M}u;
const C: u32 = ${C}u;
const S: u32 = ${S}u;
const R: u32 = ${R}u;
const NB: u32 = ${nb}u;
const BPS: u32 = ${bps}u;
const ZROW: u32 = ${zrow}u;
@group(0) @binding(0) var<storage, read> Q: array<u32>;
@group(0) @binding(1) var<storage, read> SC: array<f32>;
@group(0) @binding(2) var<storage, read> ZP: array<u32>;
@group(0) @binding(3) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, ${C * S * R}>;
fn zbyte(i: u32) -> u32 { return (ZP[i >> 2u] >> ((i & 3u) * 8u)) & 255u; }
@compute @workgroup_size(${C * S})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) l: u32) {
  let nl = l % C; let ks = l / C;
  let n = wid.x * C + nl;
  let r0 = wid.y * R;
  let K4 = K / 4u;
  var rr: array<u32, ${R}>;
  for (var t = 0u; t < R; t++) { rr[t] = min(r0 + t, M - 1u) * K4; }
  var acc: array<f32, ${R}>;
  for (var t = 0u; t < R; t++) { acc[t] = 0.0; }
  for (var b = ks * BPS; b < (ks + 1u) * BPS; b++) {
    let sc = SC[n * NB + b];
    let z = f32((zbyte(n * ZROW + b / 2u) >> (4u * (b % 2u))) & 15u);
    var part: array<f32, ${R}>;
    for (var t = 0u; t < R; t++) { part[t] = 0.0; }
    for (var w = 0u; w < 4u; w++) {
      let word = Q[n * (K / 8u) + b * 4u + w];
      let lo = vec4<f32>(f32(word & 15u), f32((word >> 4u) & 15u), f32((word >> 8u) & 15u), f32((word >> 12u) & 15u)) - z;
      let hi = vec4<f32>(f32((word >> 16u) & 15u), f32((word >> 20u) & 15u), f32((word >> 24u) & 15u), f32(word >> 28u)) - z;
      let k4 = b * 8u + w * 2u;
      for (var t = 0u; t < R; t++) {
        part[t] += dot(lo, X[rr[t] + k4]) + dot(hi, X[rr[t] + k4 + 1u]);
      }
    }
    for (var t = 0u; t < R; t++) { acc[t] += sc * part[t]; }
  }
  for (var t = 0u; t < R; t++) { red[(ks * R + t) * C + nl] = acc[t]; }
  workgroupBarrier();
  for (var o = l; o < C * R; o += C * S) {
    let c = o % C; let t = o / C;
    var s = 0.0;
    for (var j = 0u; j < S; j++) { s += red[(j * R + t) * C + c]; }
    let row = r0 + t;
    if (row < M) { Y[row * N + wid.x * C + c] = s; }
  }
}`;
  },

  // b) the corpus scan of embeddinggemma2-webgpu-tail (2-bit codes, 48 words per vector stored
  // word-major, a 3072-entry LUT per query). Knobs: workgroup size, LUT in workgroup memory or read
  // from storage, and V vectors per thread. The tail's SCAN is WG=256, lut='wg', V=1.
  scan2: ({ n, q, wg, lut, v }) => /* wgsl */ `
const NV: u32 = ${n}u;
const WG: u32 = ${wg}u;
const V: u32 = ${v}u;
const STRIDE: u32 = 3073u;
@group(0) @binding(0) var<storage, read> wds: array<u32>;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read> lut: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;
${lut === 'wg' ? 'var<workgroup> lsh: array<f32, 3072>;' : ''}
@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
  let q = wg.y;
  let lb = q * STRIDE;
${lut === 'wg' ? `  for (var t = tid; t < 3072u; t += WG) { lsh[t] = lut[lb + t]; }
  workgroupBarrier();` : ''}
  for (var vv = 0u; vv < V; vv++) {
    let i = wg.x * WG * V + vv * WG + tid;
    if (i < NV) {
      var acc: f32 = 0.0;
      for (var w = 0u; w < 48u; w++) {
        let word = wds[w * NV + i];
        for (var bb = 0u; bb < 4u; bb++) {
          for (var m = 0u; m < 4u; m++) {
            let code = (word >> (8u * bb + 6u - 2u * m)) & 3u;
            acc += ${lut === 'wg' ? 'lsh[(w * 16u + bb * 4u + m) * 4u + code]' : 'lut[lb + (w * 16u + bb * 4u + m) * 4u + code]'};
          }
        }
      }
      scores[q * NV + i] = aux[i * 4u] * acc + aux[i * 4u + 1u] + lut[lb + 3072u];
    }
  }
}`,

  // c/e) tiled matmul C[M,N] = A[M,K] B[K,N], row-major. Workgroup (WX, WY) computes a TM x TN tile;
  // thread (x, y) owns rows y + i*WY and columns x + j*WX. shared=true stages TM x TK of A and
  // TK x TN of B in workgroup memory per k-step; shared=false reads global memory directly.
  // type 'f32' | 'f16' (storage and multiply) and acc 'f32' | 'f16'.
  matmul: ({ M, N, K, TM, TN, TK, WX, WY, shared, type = 'f32', acc = 'f32' }) => {
    const RM = TM / WY, RN = TN / WX, T = WX * WY, h = type === 'f16' || acc === 'f16';
    const cvt = (e) => (type === acc ? e : `${acc}(${e})`);
    const body = shared ? `
  for (var k0 = 0u; k0 < K; k0 += TK) {
    for (var i = li; i < TM * TK; i += T) { As[i] = A[(row0 + i / TK) * K + k0 + i % TK]; }
    for (var i = li; i < TK * TN; i += T) { Bs[i] = B[(k0 + i / TN) * N + col0 + i % TN]; }
    workgroupBarrier();
    for (var kk = 0u; kk < TK; kk++) {
      var b: array<${acc}, RN>;
      for (var j = 0u; j < RN; j++) { b[j] = ${cvt('Bs[kk * TN + lid.x + j * WX]')}; }
      for (var i = 0u; i < RM; i++) {
        let a = ${cvt('As[(lid.y + i * WY) * TK + kk]')};
        for (var j = 0u; j < RN; j++) { c[i * RN + j] += a * b[j]; }
      }
    }
    workgroupBarrier();
  }` : `
  for (var k = 0u; k < K; k++) {
    var b: array<${acc}, RN>;
    for (var j = 0u; j < RN; j++) { b[j] = ${cvt('B[k * N + col0 + lid.x + j * WX]')}; }
    for (var i = 0u; i < RM; i++) {
      let a = ${cvt('A[(row0 + lid.y + i * WY) * K + k]')};
      for (var j = 0u; j < RN; j++) { c[i * RN + j] += a * b[j]; }
    }
  }`;
    return /* wgsl */ `${h ? 'enable f16;\n' : ''}
const M: u32 = ${M}u;
const N: u32 = ${N}u;
const K: u32 = ${K}u;
const TM: u32 = ${TM}u;
const TN: u32 = ${TN}u;
const TK: u32 = ${TK}u;
const WX: u32 = ${WX}u;
const WY: u32 = ${WY}u;
const RM: u32 = ${RM}u;
const RN: u32 = ${RN}u;
const T: u32 = ${T}u;
@group(0) @binding(0) var<storage, read> A: array<${type}>;
@group(0) @binding(1) var<storage, read> B: array<${type}>;
@group(0) @binding(2) var<storage, read_write> Cm: array<f32>;
${shared ? `var<workgroup> As: array<${type}, ${TM * TK}>;\nvar<workgroup> Bs: array<${type}, ${TK * TN}>;` : ''}
@compute @workgroup_size(${WX}, ${WY})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let row0 = wid.y * TM; let col0 = wid.x * TN;
  var c: array<${acc}, ${RM * RN}>;
  for (var i = 0u; i < RM * RN; i++) { c[i] = ${acc}(0.0); }${body}
  for (var i = 0u; i < RM; i++) {
    for (var j = 0u; j < RN; j++) {
      Cm[(row0 + lid.y + i * WY) * N + col0 + lid.x + j * WX] = f32(c[i * RN + j]);
    }
  }
}`;
  },
};

// ================================================================== sweeps

// ---------- a) bandwidth

async function sweepBandwidth(ctx, opts = {}) {
  const say = opts.onProgress ?? (() => {});
  const cap = Math.min(opts.maxBytes ?? (opts.quick ? 32 << 20 : 128 << 20), ctx.limits.maxStorageBufferBindingSize, ctx.limits.maxBufferSize);
  const sizes = [];
  for (let b = 16 << 10; b <= cap; b *= 2) sizes.push(b);
  const maxN4 = sizes[sizes.length - 1] / 16;
  const host = new Float32Array(maxN4 * 4);
  for (let i = 0; i < host.length; i++) host[i] = hash7(i);
  const src = makeBuffer(ctx, host, U().STORAGE, 'bw-src');
  const dst = emptyBuffer(ctx, maxN4 * 16, 'bw-dst');
  const part = emptyBuffer(ctx, 4096 * 4, 'bw-part');
  const points = [];
  const VERIFY_FULL = 32 << 20;
  try {
    for (const bytes of sizes) {
      const n4 = bytes / 16;
      const groups = Math.min(Math.ceil(n4 / 256), 4096);
      const T = groups * 256;
      // copy
      say(`bandwidth: copy ${human(bytes)}`);
      points.push(await trial(ctx, { config: { kernel: 'copy', bytes }, x: bytes, series: 'copy' }, async () => ({
        code: WGSL.copy({ n4 }), label: `copy-${bytes}`, grid: [groups],
        bind: [{ buffer: src, size: bytes }, { buffer: dst, size: bytes }], clear: [dst],
        check: async () => {
          // Full readback up to 32 MiB; above that, four 1 MiB windows (start, 1/3, 2/3, end).
          const wins = bytes <= VERIFY_FULL ? [[0, bytes]] : [0, 1 / 3, 2 / 3, 1].map((f) => [Math.floor((bytes - (1 << 20)) * f / 16) * 16, 1 << 20]);
          let worst = { ok: true, maxErr: 0, tol: 0 };
          for (const [off, len] of wins) {
            const got = new Float32Array(await readBuffer(ctx, dst, len, off));
            const ref = host.subarray(off / 4, off / 4 + len / 4);
            const c = compare(got, ref, 0);
            if (!c.ok) return c;
            worst.maxErr = Math.max(worst.maxErr, c.maxErr);
          }
          return worst;
        },
      }), opts, (us) => (2 * bytes) / (us * 1e3)));
      // read-reduce
      say(`bandwidth: read-reduce ${human(bytes)}`);
      points.push(await trial(ctx, { config: { kernel: 'read-reduce', bytes }, x: bytes, series: 'read-reduce' }, async () => ({
        code: WGSL.readReduce({ n4 }), label: `readreduce-${bytes}`, grid: [groups],
        bind: [{ buffer: src, size: bytes }, part], clear: [part],
        check: async () => {
          const ref = new Float64Array(groups);
          for (let j = 0; j < n4; j++) {
            const w = Math.floor((j % T) / 256);
            ref[w] += host[4 * j] + host[4 * j + 1] + host[4 * j + 2] + host[4 * j + 3];
          }
          return compare(new Float32Array(await readBuffer(ctx, part, groups * 4)), ref, 0);
        },
      }), opts, (us) => bytes / (us * 1e3)));
    }
  } finally { src.destroy(); dst.destroy(); part.destroy(); }
  const out = { id: 'bandwidth', unit: 'GB/s', xKind: 'bytes', points, cliffs: {} , best: {} };
  for (const s of ['copy', 'read-reduce']) {
    const ps = points.filter((p) => p.series === s);
    out.cliffs[s] = cliffs(ps, 0.8);
    const w = winners(ps).best;
    if (w) out.best[`bandwidth_${s}`] = { bytes: w.x, gbps: w.y };
  }
  return out;
}

// ---------- b) reduction workgroup sweep, d) subgroup vs workgroup memory

const REDUCE_N = 1 << 22;
let reduceHost = null;
function reduceData() {
  if (!reduceHost) { reduceHost = new Float32Array(REDUCE_N); for (let i = 0; i < REDUCE_N; i++) reduceHost[i] = hash7(i * 3 + 1); }
  return reduceHost;
}

async function reduceSweep(ctx, opts, variants, id, ipts) {
  const say = opts.onProgress ?? (() => {});
  const host = reduceData();
  const src = makeBuffer(ctx, host, U().STORAGE, 'red-src');
  const maxParts = REDUCE_N / 32;
  const part = emptyBuffer(ctx, maxParts * 4, 'red-part');
  const wgs = (opts.quick ? [64, 256, 1024] : [32, 64, 128, 256, 512, 1024]).filter((w) => fits(ctx, w * 4 + 4, w));
  const points = [];
  const ref = new Map();
  try {
    for (const variant of variants) for (const ipt of ipts) for (const wg of wgs) {
      const groups = REDUCE_N / (wg * ipt);
      const grid = [Math.min(groups, 32768), Math.ceil(groups / 32768)];
      say(`${id}: ${variant} wg=${wg} ipt=${ipt}`);
      points.push(await trial(ctx, {
        config: { variant, wg, ipt }, x: wg, series: variants.length > 1 ? `${variant} ipt=${ipt}` : `ipt=${ipt}`,
        needsLimits: needsLimits(wg * 4 + 4, wg),
      }, async () => ({
        code: WGSL.reduce({ n: REDUCE_N, wg, ipt, variant }), label: `reduce-${variant}-${wg}-${ipt}`, grid,
        bind: [src, part], clear: [part],
        check: async () => {
          const chunk = wg * ipt;
          if (!ref.has(chunk)) {
            const r = new Float64Array(groups);
            for (let i = 0; i < REDUCE_N; i++) r[(i / chunk) | 0] += host[i];
            ref.set(chunk, r);
          }
          return compare(new Float32Array(await readBuffer(ctx, part, groups * 4)), ref.get(chunk), 0);
        },
      }), opts, (us) => (REDUCE_N * 4) / (us * 1e3)));
    }
  } finally { src.destroy(); part.destroy(); }
  return points;
}

async function sweepReduce(ctx, opts = {}) {
  const points = await reduceSweep(ctx, opts, ['tree'], 'reduce', opts.quick ? [1, 8] : [1, 2, 4, 8, 16]);
  const w = winners(points);
  return {
    id: 'reduce', unit: 'GB/s', xKind: 'wg', points, cliffs: { all: cliffs(points) },
    best: w.best ? { [`reduce_f32:${REDUCE_N}`]: { ...w.best.config, gbps: w.best.y, ...(w.bestDefault && w.bestDefault !== w.best ? { default_limits: { ...w.bestDefault.config, gbps: w.bestDefault.y } } : {}) } } : {},
  };
}

async function sweepSubgroup(ctx, opts = {}) {
  if (!ctx.features.includes('subgroups')) return { id: 'subgroup', skipped: 'adapter has no "subgroups" feature', points: [] };
  const points = await reduceSweep(ctx, opts, ['tree', 'subgroup'], 'subgroup', opts.quick ? [4] : [1, 4, 16]);
  const best = {};
  const verdict = {};
  for (const v of ['tree', 'subgroup']) {
    const w = winners(points.filter((p) => p.config.variant === v)).best;
    if (w) { best[`reduce_f32_${v}:${REDUCE_N}`] = { ...w.config, gbps: w.y }; verdict[v] = w.y; }
  }
  if (verdict.tree && verdict.subgroup) verdict.subgroup_over_tree = fmt(verdict.subgroup / verdict.tree, 3);
  return { id: 'subgroup', unit: 'GB/s', xKind: 'wg', points, cliffs: {}, best, verdict };
}

// ---------- b) q4 GEMV over the EG2 shapes

// The 4-layer EG2 query encoder (eg2-wgsl-engine manifest.json): hidden 512, q 1024, k/v 512,
// MLP 2048; M = tokens in a typical 10-20 token query, padded to 16. The 512->768 embedding
// projection runs once per pooled sequence, so it is a true GEMV (M = 1).
export const EG2_SHAPES = [
  { name: 'gate_up', N: 2048, K: 512, M: 16 },
  { name: 'q_proj', N: 1024, K: 512, M: 16 },
  { name: 'down', N: 512, K: 2048, M: 16 },
  { name: 'emb_proj', N: 768, K: 512, M: 1 },
];

function q4Data(N, K, M, seed) {
  const r = rng(seed);
  const nb = K / 32, zrow = Math.ceil(nb / 2);
  const q = new Uint32Array(N * K / 8);
  for (let i = 0; i < q.length; i++) q[i] = (r() * 4294967296) >>> 0;
  const sc = new Float32Array(N * nb);
  for (let i = 0; i < sc.length; i++) sc[i] = 0.002 + 0.02 * r();
  const zb = new Uint8Array(Math.ceil(N * zrow / 4) * 4);
  for (let i = 0; i < N * zrow; i++) zb[i] = (r() * 256) | 0;
  const x = new Float32Array(M * K);
  for (let i = 0; i < x.length; i++) x[i] = r() * 2 - 1;
  // Reference in f64, plus sum |w x| per output for the tolerance.
  const y = new Float64Array(M * N), mag = new Float64Array(M * N);
  const w = new Float64Array(K);
  for (let n = 0; n < N; n++) {
    for (let b = 0; b < nb; b++) {
      const z = (zb[n * zrow + (b >> 1)] >> (4 * (b & 1))) & 15;
      const s = sc[n * nb + b];
      for (let wi = 0; wi < 4; wi++) {
        const word = q[n * (K / 8) + b * 4 + wi];
        for (let e = 0; e < 8; e++) w[b * 32 + wi * 8 + e] = (((word >>> (4 * e)) & 15) - z) * s;
      }
    }
    for (let m = 0; m < M; m++) {
      let a = 0, g = 0;
      for (let k = 0; k < K; k++) { const t = w[k] * x[m * K + k]; a += t; g += Math.abs(t); }
      y[m * N + n] = a; mag[m * N + n] = g;
    }
  }
  return { q, sc, zp: new Uint32Array(zb.buffer), x, y, mag };
}

function gemvConfigs(ctx, shape, quick) {
  const out = [];
  const Cs = quick ? [16, 32, 64] : [8, 16, 32, 64, 128];
  const Ss = quick ? [2, 4, 8] : [1, 2, 4, 8, 16];
  const Rs = shape.M === 1 ? [1] : (quick ? [4, 8, 16] : [1, 2, 4, 8, 16]);
  const nb = shape.K / 32;
  for (const C of Cs) for (const S of Ss) for (const R of Rs) {
    if (shape.N % C || nb % S || R > shape.M) continue;
    const inv = C * S, wgmem = C * S * R * 4;
    if (!fits(ctx, wgmem, inv)) continue;
    out.push({ C, S, R, inv, wgmem, needsLimits: needsLimits(wgmem, inv) });
  }
  return out;
}

async function sweepGemv(ctx, opts = {}) {
  const say = opts.onProgress ?? (() => {});
  const shapes = opts.shapes ?? EG2_SHAPES;
  const points = [];
  const best = {};
  const perShape = {};
  for (const [si, sh] of shapes.entries()) {
    const d = q4Data(sh.N, sh.K, sh.M, 1000 + si);
    const bufs = [
      makeBuffer(ctx, d.q, U().STORAGE, 'q'), makeBuffer(ctx, d.sc, U().STORAGE, 'sc'),
      makeBuffer(ctx, d.zp, U().STORAGE, 'zp'), makeBuffer(ctx, d.x, U().STORAGE, 'x'),
    ];
    const yb = emptyBuffer(ctx, sh.M * sh.N * 4, 'y');
    const tol = Float64Array.from(d.mag, (g) => 2e-5 * g + 1e-6);
    const bytes = d.q.byteLength + d.sc.byteLength + d.zp.byteLength + d.x.byteLength + sh.M * sh.N * 4;
    const tag = `gemv_q4:${sh.N}x${sh.K}:m${sh.M}`;
    const ps = [];
    try {
      for (const c of gemvConfigs(ctx, sh, opts.quick)) {
        say(`gemv ${sh.name} ${sh.N}x${sh.K} M=${sh.M}: C=${c.C} S=${c.S} R=${c.R}`);
        ps.push(await trial(ctx, {
          config: { shape: sh.name, N: sh.N, K: sh.K, M: sh.M, C: c.C, S: c.S, R: c.R }, x: c.inv,
          series: `${sh.name} R=${c.R}`, needsLimits: c.needsLimits,
        }, async () => ({
          code: WGSL.gemvQ4({ ...sh, ...c }), label: `${tag}-C${c.C}-S${c.S}-R${c.R}`,
          grid: [sh.N / c.C, Math.ceil(sh.M / c.R)], bind: [...bufs, yb], clear: [yb],
          check: async () => compare(new Float32Array(await readBuffer(ctx, yb, sh.M * sh.N * 4)), d.y, tol),
        }), opts, (us) => bytes / (us * 1e3)));
      }
    } finally { for (const b of [...bufs, yb]) b.destroy(); }
    for (const p of ps) p.shape = sh.name;
    points.push(...ps);
    const w = winners(ps);
    const eng = ps.find((p) => p.ok && p.config.C === 32 && p.config.S === 8 && p.config.R === Math.min(8, sh.M));
    if (w.best) {
      best[tag] = {
        C: w.best.config.C, S: w.best.config.S, R: w.best.config.R, us: w.best.time.median_us, gbps: w.best.y,
        ...(w.bestDefault && w.bestDefault !== w.best ? { default_limits: { C: w.bestDefault.config.C, S: w.bestDefault.config.S, R: w.bestDefault.config.R, us: w.bestDefault.time.median_us } } : {}),
        ...(eng ? { engine_config_us: eng.time.median_us, speedup_vs_engine: fmt(eng.time.median_us / w.best.time.median_us, 3) } : {}),
      };
    }
    perShape[sh.name] = cliffs(ps);
  }
  return { id: 'gemv', unit: 'GB/s', xKind: 'inv', points, cliffs: perShape, best };
}

// ---------- b) corpus scan (embeddinggemma2-webgpu-tail SCAN)

const SCAN_N = 65536, SCAN_Q = 8;
function scanData() {
  const r = rng(77);
  const wds = new Uint32Array(48 * SCAN_N);
  for (let i = 0; i < wds.length; i++) wds[i] = (r() * 4294967296) >>> 0;
  const aux = new Float32Array(SCAN_N * 4);
  for (let i = 0; i < SCAN_N; i++) { aux[4 * i] = 0.5 + r(); aux[4 * i + 1] = r() - 0.5; }
  const lut = new Float32Array(SCAN_Q * 3073);
  for (let i = 0; i < lut.length; i++) lut[i] = r() * 2 - 1;
  // Reference via per-(query, byte position, byte value) tables: 192 lookups per vector.
  const ref = new Float64Array(SCAN_Q * SCAN_N), tol = new Float64Array(SCAN_Q * SCAN_N);
  const tab = new Float64Array(192 * 256), atab = new Float64Array(192 * 256);
  for (let q = 0; q < SCAN_Q; q++) {
    const lb = q * 3073;
    for (let p = 0; p < 192; p++) {
      const w = p >> 2, bb = p & 3;
      for (let v = 0; v < 256; v++) {
        let s = 0, a = 0;
        for (let m = 0; m < 4; m++) {
          const code = (v >> (6 - 2 * m)) & 3;
          const e = lut[lb + (w * 16 + bb * 4 + m) * 4 + code];
          s += e; a += Math.abs(e);
        }
        tab[p * 256 + v] = s; atab[p * 256 + v] = a;
      }
    }
    for (let i = 0; i < SCAN_N; i++) {
      let s = 0, a = 0;
      for (let w = 0; w < 48; w++) {
        const word = wds[w * SCAN_N + i];
        for (let bb = 0; bb < 4; bb++) { const v = (word >>> (8 * bb)) & 255; s += tab[(w * 4 + bb) * 256 + v]; a += atab[(w * 4 + bb) * 256 + v]; }
      }
      ref[q * SCAN_N + i] = aux[4 * i] * s + aux[4 * i + 1] + lut[lb + 3072];
      tol[q * SCAN_N + i] = 4e-6 * aux[4 * i] * a + 1e-5;
    }
  }
  return { wds, aux, lut, ref, tol };
}

async function sweepScan(ctx, opts = {}) {
  const say = opts.onProgress ?? (() => {});
  say('scan: building reference');
  const d = scanData();
  const bufs = [makeBuffer(ctx, d.wds, U().STORAGE, 'wds'), makeBuffer(ctx, d.aux, U().STORAGE, 'aux'), makeBuffer(ctx, d.lut, U().STORAGE, 'lut')];
  const sb = emptyBuffer(ctx, SCAN_Q * SCAN_N * 4, 'scores');
  const bytes = SCAN_Q * SCAN_N * 192;
  const points = [];
  const wgs = (opts.quick ? [64, 256, 1024] : [32, 64, 128, 256, 512, 1024]);
  try {
    for (const lut of ['wg', 'storage']) for (const v of (opts.quick ? [1, 4] : [1, 2, 4])) for (const wg of wgs) {
      const wgmem = lut === 'wg' ? 3072 * 4 : 0;
      if (!fits(ctx, wgmem, wg)) continue;
      say(`scan: lut=${lut} V=${v} wg=${wg}`);
      points.push(await trial(ctx, {
        config: { wg, lut, v }, x: wg, series: `lut=${lut} V=${v}`, needsLimits: needsLimits(wgmem, wg),
      }, async () => ({
        code: WGSL.scan2({ n: SCAN_N, q: SCAN_Q, wg, lut, v }), label: `scan-${lut}-${v}-${wg}`,
        grid: [Math.ceil(SCAN_N / (wg * v)), SCAN_Q], bind: [...bufs, sb], clear: [sb],
        check: async () => compare(new Float32Array(await readBuffer(ctx, sb, SCAN_Q * SCAN_N * 4)), d.ref, d.tol),
      }), opts, (us) => bytes / (us * 1e3)));
    }
  } finally { for (const b of [...bufs, sb]) b.destroy(); }
  const w = winners(points);
  const tail = points.find((p) => p.ok && p.config.wg === 256 && p.config.lut === 'wg' && p.config.v === 1);
  return {
    id: 'scan', unit: 'GB/s', xKind: 'wg', points, cliffs: { all: cliffs(points) },
    best: w.best ? { [`scan_2bit:${SCAN_N}x${SCAN_Q}`]: {
      ...w.best.config, us: w.best.time.median_us, gbps: w.best.y,
      ...(w.bestDefault && w.bestDefault !== w.best ? { default_limits: { ...w.bestDefault.config, us: w.bestDefault.time.median_us } } : {}),
      ...(tail ? { tail_config_us: tail.time.median_us, speedup_vs_tail: fmt(tail.time.median_us / w.best.time.median_us, 3) } : {}),
    } } : {},
  };
}

// ---------- c) tiled matmul, e) f16

export const MATMUL_SHAPE = { M: 128, N: 2048, K: 512 }; // gate/up projection at the 128-token cap

function matmulData(sh, f16) {
  // Multiples of 1/8 in [-1, 1]: exact in f16, products exact, f32 sums of K=512 exact.
  const r = rng(f16 ? 5 : 4);
  const A = new Float32Array(sh.M * sh.K), B = new Float32Array(sh.K * sh.N);
  for (let i = 0; i < A.length; i++) A[i] = Math.round(r() * 16 - 8) / 8;
  for (let i = 0; i < B.length; i++) B[i] = Math.round(r() * 16 - 8) / 8;
  const C = new Float64Array(sh.M * sh.N), mag = new Float64Array(sh.M * sh.N);
  const row = new Float64Array(sh.N), rowa = new Float64Array(sh.N);
  for (let m = 0; m < sh.M; m++) {
    row.fill(0); rowa.fill(0);
    for (let k = 0; k < sh.K; k++) {
      const a = A[m * sh.K + k];
      if (a === 0) continue;
      const off = k * sh.N;
      for (let n = 0; n < sh.N; n++) { const t = a * B[off + n]; row[n] += t; rowa[n] += Math.abs(t); }
    }
    C.set(row, m * sh.N); mag.set(rowa, m * sh.N);
  }
  return { A, B, C, mag };
}

export function toHalf(f32) {
  const out = new Uint16Array(f32.length);
  const fb = new Float32Array(1), ub = new Uint32Array(fb.buffer);
  for (let i = 0; i < f32.length; i++) {
    fb[0] = f32[i];
    const x = ub[0], sign = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 0xff) - 127 + 15, m = x & 0x7fffff;
    if (e >= 31) { out[i] = sign | 0x7c00; continue; }
    if (e <= 0) {
      if (e < -10) { out[i] = sign; continue; }
      m |= 0x800000;
      const s = 14 - e;
      let h = m >> s;
      if ((m >> (s - 1)) & 1 && ((m & ((1 << (s - 1)) - 1)) || (h & 1))) h++;
      out[i] = sign | h; continue;
    }
    let h = (e << 10) | (m >> 13);
    if ((m & 0x1000) && ((m & 0xfff) || (h & 1))) h++;
    out[i] = sign | h;
  }
  return out;
}

export function matmulConfigs(ctx, quick) {
  const out = [];
  const tiles = quick ? [[32, 32], [64, 64], [64, 128], [128, 128]] : [[16, 16], [32, 32], [32, 64], [64, 32], [64, 64], [64, 128], [128, 64], [128, 128]];
  const tks = quick ? [8, 32] : [8, 16, 32];
  const wgs = quick ? [[8, 8], [16, 16]] : [[8, 8], [16, 8], [8, 16], [16, 16], [32, 8]];
  for (const [TM, TN] of tiles) for (const [WX, WY] of wgs) {
    if (TN % WX || TM % WY) continue;
    const RM = TM / WY, RN = TN / WX;
    if (RM * RN > 64) continue;
    const inv = WX * WY;
    for (const shared of [true, false]) for (const TK of shared ? tks : [0]) {
      const wgmem = shared ? (TM * TK + TK * TN) * 4 : 0;
      if (!fits(ctx, wgmem, inv, WX, WY)) continue;
      out.push({ TM, TN, TK, WX, WY, shared, inv, wgmem, regs: RM * RN, needsLimits: needsLimits(wgmem, inv, WX) });
    }
  }
  return out;
}

async function matmulRun(ctx, opts, cfgs, variant, say) {
  const sh = opts.matmulShape ?? MATMUL_SHAPE;
  const f16 = variant.type === 'f16';
  const d = matmulData(sh, f16);
  const A = makeBuffer(ctx, f16 ? toHalf(d.A) : d.A, U().STORAGE, 'A');
  const B = makeBuffer(ctx, f16 ? toHalf(d.B) : d.B, U().STORAGE, 'B');
  const Cb = emptyBuffer(ctx, sh.M * sh.N * 4, 'C');
  const exact = variant.acc === 'f32';
  const tol = exact ? Float64Array.from(d.mag, (g) => 1e-6 * g) : Float64Array.from(d.mag, (g) => 2e-2 * g + 1e-3);
  const flops = 2 * sh.M * sh.N * sh.K;
  const points = [];
  try {
    for (const c of cfgs) {
      if (sh.M % c.TM || sh.N % c.TN || (c.shared && sh.K % c.TK)) continue;
      const vname = variant.type === 'f32' ? 'f32' : `f16 (acc ${variant.acc})`;
      say(`matmul ${vname}: ${c.TM}x${c.TN}${c.shared ? ` k${c.TK}` : ' direct'} wg ${c.WX}x${c.WY}`);
      points.push(await trial(ctx, {
        config: { TM: c.TM, TN: c.TN, TK: c.TK, WX: c.WX, WY: c.WY, shared: c.shared, type: variant.type, acc: variant.acc },
        x: c.wgmem, series: variant.series ?? `${c.inv} threads`, needsLimits: c.needsLimits,
      }, async () => ({
        code: WGSL.matmul({ ...sh, ...c, ...variant }),
        label: `matmul-${variant.type}-${variant.acc}-${c.TM}x${c.TN}x${c.TK}-${c.WX}x${c.WY}-${c.shared ? 's' : 'g'}`,
        grid: [sh.N / c.TN, sh.M / c.TM], bind: [A, B, Cb], clear: [Cb],
        check: async () => compare(new Float32Array(await readBuffer(ctx, Cb, sh.M * sh.N * 4)), d.C, tol),
      }), opts, (us) => flops / (us * 1e3)));
    }
  } finally { A.destroy(); B.destroy(); Cb.destroy(); }
  return points;
}

const mmKey = (c) => `${c.TM}x${c.TN}x${c.TK}-${c.WX}x${c.WY}-${c.shared ? 's' : 'g'}`;

async function sweepMatmul(ctx, opts = {}) {
  const say = opts.onProgress ?? (() => {});
  const sh = opts.matmulShape ?? MATMUL_SHAPE;
  const points = await matmulRun(ctx, opts, matmulConfigs(ctx, opts.quick), { type: 'f32', acc: 'f32' }, say);
  for (const p of points) p.series = p.config.shared ? `${p.config.WX * p.config.WY} threads` : 'no workgroup memory';
  const w = winners(points);
  const pick = (p) => ({ TM: p.config.TM, TN: p.config.TN, TK: p.config.TK, WX: p.config.WX, WY: p.config.WY, shared: p.config.shared, gflops: p.y, us: p.time.median_us });
  return {
    id: 'matmul', unit: 'GFLOP/s', xKind: 'wgmem', points, cliffs: { shared: cliffs(points.filter((p) => p.config.shared), 0.75) },
    best: w.best ? { [`matmul_f32:${sh.M}x${sh.N}x${sh.K}`]: { ...pick(w.best), ...(w.bestDefault && w.bestDefault !== w.best ? { default_limits: pick(w.bestDefault) } : {}) } } : {},
  };
}

async function sweepF16(ctx, opts = {}, matmulResult = null) {
  if (!ctx.features.includes('shader-f16')) return { id: 'f16', skipped: 'adapter has no "shader-f16" feature', points: [] };
  const say = opts.onProgress ?? (() => {});
  const sh = opts.matmulShape ?? MATMUL_SHAPE;
  // The top shared-memory configs from (c) if it ran, else a fixed set. f16 tiles take half the
  // workgroup memory, so the same tile shapes fit.
  let cfgs = matmulConfigs(ctx, false).filter((c) => c.shared);
  if (matmulResult?.points?.length) {
    const top = matmulResult.points.filter((p) => p.ok && p.config.shared).sort((a, b) => b.y - a.y).slice(0, opts.quick ? 3 : 5).map((p) => mmKey(p.config));
    cfgs = cfgs.filter((c) => top.includes(mmKey(c)));
  } else {
    const fixed = ['32x32x16-8x8-s', '64x64x16-16x16-s', '64x64x8-8x8-s', '64x128x16-16x16-s'];
    cfgs = cfgs.filter((c) => fixed.includes(mmKey(c)));
  }
  const points = [];
  for (const variant of [{ type: 'f32', acc: 'f32', series: 'f32' }, { type: 'f16', acc: 'f32', series: 'f16 data, f32 accumulate' }, { type: 'f16', acc: 'f16', series: 'f16 data, f16 accumulate' }]) {
    const ps = await matmulRun(ctx, opts, cfgs, variant, say);
    points.push(...ps);
  }
  const keys = [...new Set(points.map((p) => mmKey(p.config)))];
  for (const p of points) { p.x = keys.indexOf(mmKey(p.config)); p.label = mmKey(p.config); }
  const best = {};
  const verdict = {};
  for (const s of ['f32', 'f16 data, f32 accumulate', 'f16 data, f16 accumulate']) {
    const w = winners(points.filter((p) => p.series === s)).best;
    if (w) verdict[s] = w.y;
    if (w && s !== 'f32') best[`matmul_f16${s.endsWith('f16 accumulate') ? '' : '_acc32'}:${sh.M}x${sh.N}x${sh.K}`] = { TM: w.config.TM, TN: w.config.TN, TK: w.config.TK, WX: w.config.WX, WY: w.config.WY, shared: true, gflops: w.y };
  }
  if (verdict.f32) for (const s of Object.keys(verdict)) if (s !== 'f32') verdict[`${s} / f32`] = fmt(verdict[s] / verdict.f32, 3);
  return { id: 'f16', unit: 'GFLOP/s', xKind: 'cat', categories: keys, points, cliffs: {}, best, verdict };
}

// ------------------------------------------------------------------ registry

export const SWEEPS = {
  bandwidth: { title: 'a) Memory bandwidth vs buffer size', run: sweepBandwidth },
  reduce: { title: 'b) Reduction: workgroup size x items per thread', run: sweepReduce },
  gemv: { title: 'b) q4 GEMV on the EG2 encoder shapes', run: sweepGemv },
  scan: { title: 'b) 2-bit corpus scan (the tail\'s SCAN)', run: sweepScan },
  matmul: { title: 'c) Tiled matmul: tile x workgroup x workgroup memory', run: sweepMatmul },
  subgroup: { title: 'd) Subgroup vs workgroup-memory reduction', run: sweepSubgroup },
  f16: { title: 'e) f16 vs f32 matmul', run: sweepF16 },
};

export function makeProfile(ctx, results, extra = {}) {
  const best = {};
  for (const r of Object.values(results)) Object.assign(best, r?.best ?? {});
  const slim = {};
  for (const [k, r] of Object.entries(results)) {
    if (!r) continue;
    slim[k] = {
      skipped: r.skipped, unit: r.unit, cliffs: r.cliffs, verdict: r.verdict, categories: r.categories, best: r.best,
      points: (r.points ?? []).map((p) => ({ x: p.x, series: p.series, label: p.label, config: p.config, status: p.status, y: p.y, time: p.time, needsLimits: p.needsLimits || undefined, error: p.error, maxErr: p.verify?.maxErr })),
    };
  }
  return {
    schema: VERSION, key: ctx.key, date: new Date().toISOString(),
    adapter: { info: ctx.info, features: ctx.allFeatures, limits: ctx.limits },
    userAgent: globalThis.navigator?.userAgent ?? `node ${globalThis.process?.version ?? ''}`,
    timing: ctx.timing, ...extra, best, sweeps: slim,
  };
}

export function human(bytes) {
  if (bytes >= 1 << 30) return `${+(bytes / (1 << 30)).toFixed(2)} GiB`;
  if (bytes >= 1 << 20) return `${+(bytes / (1 << 20)).toFixed(2)} MiB`;
  return `${+(bytes / 1024).toFixed(1)} KiB`;
}

// Every WGSL variant the full (non-quick) sweeps can generate on this device, for a compile check.
export function allKernels(ctx) {
  const out = [];
  const L = ctx.limits;
  for (let b = 16 << 10; b <= Math.min(L.maxStorageBufferBindingSize, 256 << 20); b *= 2) {
    out.push({ label: `copy-${b}`, code: WGSL.copy({ n4: b / 16 }) }, { label: `readreduce-${b}`, code: WGSL.readReduce({ n4: b / 16 }) });
  }
  for (const variant of ['tree', ...(ctx.features.includes('subgroups') ? ['subgroup'] : [])]) {
    for (const wg of [32, 64, 128, 256, 512, 1024]) for (const ipt of [1, 2, 4, 8, 16]) {
      if (fits(ctx, wg * 4 + 4, wg)) out.push({ label: `reduce-${variant}-${wg}-${ipt}`, code: WGSL.reduce({ n: REDUCE_N, wg, ipt, variant }) });
    }
  }
  for (const sh of EG2_SHAPES) for (const c of gemvConfigs(ctx, sh, false)) out.push({ label: `gemv-${sh.name}-C${c.C}-S${c.S}-R${c.R}`, code: WGSL.gemvQ4({ ...sh, ...c }) });
  for (const lut of ['wg', 'storage']) for (const v of [1, 2, 4]) for (const wg of [32, 64, 128, 256, 512, 1024]) {
    if (fits(ctx, lut === 'wg' ? 12288 : 0, wg)) out.push({ label: `scan-${lut}-${v}-${wg}`, code: WGSL.scan2({ n: SCAN_N, q: SCAN_Q, wg, lut, v }) });
  }
  const variants = [{ type: 'f32', acc: 'f32' }, ...(ctx.features.includes('shader-f16') ? [{ type: 'f16', acc: 'f32' }, { type: 'f16', acc: 'f16' }] : [])];
  for (const c of matmulConfigs(ctx, false)) for (const v of variants) {
    out.push({ label: `matmul-${v.type}-${v.acc}-${mmKey(c)}`, code: WGSL.matmul({ ...MATMUL_SHAPE, ...c, ...v }) });
  }
  return out;
}
