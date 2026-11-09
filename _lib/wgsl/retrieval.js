// Retrieval ops for WebGPU (kernel library step 6): exact k-nearest neighbours by dot product, a
// multi-pass top-k over score rows, an exclusive prefix sum and an LSD radix sort. Unlike kernels.js
// these are not SeqEngine steps: each is WGSL plus, where it takes several dispatches, a planner that
// returns the dispatch list [pipeline, bindGroup, workgroups] for the caller to record (the bridge
// jobs put timestamp writes around them). Sources: embeddinggemma2-webgpu-tail/web/tail.js (top-k)
// and gpu-bridge/jobs/{knn,knngraph,scan,sort}; the moved shaders are byte-identical to them.

// Compile `code` and one compute pipeline per entry point (layout "auto"); a WGSL error names its line.
export async function compile(dev, code, entries = ["main"]) {
  const m = dev.createShaderModule({ code });
  const errs = (await m.getCompilationInfo()).messages.filter((x) => x.type === "error");
  if (errs.length) throw new Error("WGSL: " + errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("; "));
  return Promise.all(entries.map((e) => dev.createComputePipelineAsync({ layout: "auto", compute: { module: m, entryPoint: e } })));
}

const STORAGE = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const uniform = (dev, words) => {
  const u = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(u, 0, new Uint32Array([...words, 0, 0, 0, 0].slice(0, 4)));
  return u;
};
// bufs: buffers for bindings 0, 1, ... in order, or [binding, buffer] pairs where an entry point skips one.
const bind = (dev, pipe, bufs) => dev.createBindGroup({ layout: pipe.getBindGroupLayout(0),
  entries: bufs.map((b, i) => (Array.isArray(b) ? { binding: b[0], resource: { buffer: b[1] } } : { binding: i, resource: { buffer: b } })) });

// ---- k nearest neighbours ------------------------------------------------------------------------
// One fused kernel: each workgroup takes QB queries, streams every doc once (vec4 loads), keeps a
// per-thread top-K per query, then merges the WG thread lists with a tree argmax. Bindings: docs
// [n, D] (f32, or f16 with f16: true), queries [nq, D] f32, outIdx and outScore [nq, K], uniform
// P { n, q0 } (q0: first query row of this dispatch). excludeSelf skips doc j == query row, for a
// kNN graph where queries and docs are the same buffer. Dispatch nq / QB workgroups.
export const knnWGSL = ({ D, K = 10, QB = 4, WG = 256, f16 = false, excludeSelf = false }) => /* wgsl */ `${f16 ? "enable f16;" : ""}
const K: u32 = ${K}u; const QB: u32 = ${QB}u; const WG: u32 = ${WG}u; const DV: u32 = ${D / 4}u;
struct P { n: u32, q0: u32 }
@group(0) @binding(0) var<storage, read> docs: array<${f16 ? "vec4<f16>" : "vec4f"}>;
@group(0) @binding(1) var<storage, read> qs: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> outIdx: array<u32>;
@group(0) @binding(3) var<storage, read_write> outScore: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> qsh: array<vec4f, ${QB * D / 4}>;
var<workgroup> cs: array<f32, ${WG * K}>;
var<workgroup> ci: array<u32, ${WG * K}>;
var<workgroup> rv: array<f32, ${WG}>;
var<workgroup> rt: array<u32, ${WG}>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) w: vec3u) {
  let q0 = p.q0 + w.x * QB;
  for (var i = t; i < QB * DV; i += WG) { qsh[i] = qs[q0 * DV + i]; }
  workgroupBarrier();
  var bs: array<f32, ${QB * K}>;
  var bi: array<u32, ${QB * K}>;
  for (var i = 0u; i < QB * K; i++) { bs[i] = -1e30; bi[i] = 0u; }
  for (var j = t; j < p.n; j += WG) {
    var acc = vec4f(0.0);
    for (var d = 0u; d < DV; d++) {
      let x = vec4f(docs[j * DV + d]);
      acc += vec4f(dot(x, qsh[d]), dot(x, qsh[DV + d]), dot(x, qsh[2u * DV + d]), dot(x, qsh[3u * DV + d]));
    }
    for (var q = 0u; q < QB; q++) {${excludeSelf ? `
      if (j == q0 + q) { continue; }   // a point is not its own neighbour` : ""}
      let s = acc[q];
      let base = q * K;
      if (s > bs[base + K - 1u]) {
        var pos = K - 1u;
        loop {
          if (pos == 0u) { break; }
          if (bs[base + pos - 1u] >= s) { break; }
          bs[base + pos] = bs[base + pos - 1u]; bi[base + pos] = bi[base + pos - 1u];
          pos = pos - 1u;
        }
        bs[base + pos] = s; bi[base + pos] = j;
      }
    }
  }
  for (var q = 0u; q < QB; q++) {
    for (var k = 0u; k < K; k++) { cs[t * K + k] = bs[q * K + k]; ci[t * K + k] = bi[q * K + k]; }
    var h = 0u;
    workgroupBarrier();
    for (var k = 0u; k < K; k++) {
      if (h < K) { rv[t] = cs[t * K + h]; } else { rv[t] = -1e30; }
      rt[t] = t;
      workgroupBarrier();
      for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if (t < s && rv[t + s] > rv[t]) { rv[t] = rv[t + s]; rt[t] = rt[t + s]; }
        workgroupBarrier();
      }
      let win = rt[0];
      workgroupBarrier();
      if (t == win) {
        outScore[(q0 + q) * K + k] = cs[t * K + h];
        outIdx[(q0 + q) * K + k] = ci[t * K + h];
        h = h + 1u;
      }
    }
    workgroupBarrier();
  }
}`;

// ---- top-k over score rows -----------------------------------------------------------------------
// From tail.js (RemexTail.topk drives the passes): each pass keeps the best k of every chunk of
// TOPK_CHUNK pairs, so k <= 256 shrinks the input at least 4x per pass.
export const TOPK_CHUNK = 1024;
export const TOPK = /* wgsl */ `
struct TopP {
  n: u32,          // valid entries per query in the input
  in_stride: u32,  // input row stride per query
  out_stride: u32, // output row stride per query (= chunks * k)
  k: u32,
  implicit: u32,   // 1: index = position in the row (pass 1)
  pad0: u32,
  pad1: u32,
  pad2: u32,
}
@group(0) @binding(0) var<uniform> tp: TopP;
@group(0) @binding(1) var<storage, read> sin: array<f32>;
@group(0) @binding(2) var<storage, read> iin: array<u32>;
@group(0) @binding(3) var<storage, read_write> sout: array<f32>;
@group(0) @binding(4) var<storage, read_write> iout: array<u32>;

const CH: u32 = 1024u;
var<workgroup> ks: array<f32, 1024>;
var<workgroup> ki: array<u32, 1024>;

fn before(sa: f32, ia: u32, sb: f32, ib: u32) -> bool {
  return sa > sb || (sa == sb && ia < ib);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let q = wg.y;
  let base = wg.x * CH;
  for (var t: u32 = tid; t < CH; t += 256u) {
    let p = base + t;
    if (p < tp.n) {
      ks[t] = sin[q * tp.in_stride + p];
      if (tp.implicit != 0u) {
        ki[t] = p;
      } else {
        ki[t] = iin[q * tp.in_stride + p];
      }
    } else {
      ks[t] = bitcast<f32>(0xff7fffffu);   // -FLT_MAX
      ki[t] = 0xffffffffu;
    }
  }
  workgroupBarrier();

  for (var size: u32 = 2u; size <= CH; size = size << 1u) {
    for (var stride: u32 = size >> 1u; stride > 0u; stride = stride >> 1u) {
      for (var h: u32 = 0u; h < 2u; h += 1u) {
        let pr = tid + 256u * h;                       // pair 0..511
        let i = 2u * stride * (pr / stride) + pr % stride;
        let j = i + stride;
        let si = ks[i]; let ii = ki[i];
        let sj = ks[j]; let ij = ki[j];
        // blocks with (i & size) == 0 end best-first, the others worst-first; the last size is all best-first
        let want_first = (i & size) == 0u;
        if (before(sj, ij, si, ii) == want_first) {
          ks[i] = sj; ki[i] = ij;
          ks[j] = si; ki[j] = ii;
        }
      }
      workgroupBarrier();
    }
  }

  for (var t: u32 = tid; t < tp.k; t += 256u) {
    let o = q * tp.out_stride + wg.x * tp.k + t;
    sout[o] = ks[t];
    iout[o] = ki[t];
  }
}
`;

// Plan top-k of nq score rows (row q at scores[q * inStride], n valid): passes until one chunk is left.
// Returns { dispatches, scores, idx (the final [nq, k] buffers), temps (intermediates to free) }.
// Rows of fewer than k entries pad with -FLT_MAX and index 0xffffffff. pipe: compile(dev, TOPK).
export function planTopk(dev, pipe, scores, { nq, n, inStride = n, k }) {
  if (k < 1 || k > 256) throw new Error(`topk: k=${k} outside 1..256`);
  const dispatches = [], temps = [];
  let sIn = scores, iIn = scores, m = n, stride = inStride, implicit = 1, sOut, iOut;
  for (;;) {
    const chunks = Math.max(1, Math.ceil(m / TOPK_CHUNK));
    if (chunks > 65535 || nq > 65535) throw new Error("topk: grid exceeds 65535");
    const outStride = chunks * k;
    sOut = dev.createBuffer({ size: Math.max(16, nq * outStride * 4), usage: STORAGE() });
    iOut = dev.createBuffer({ size: Math.max(16, nq * outStride * 4), usage: STORAGE() });
    const u = dev.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(u, 0, new Uint32Array([m, stride, outStride, k, implicit, 0, 0, 0]));
    temps.push(u);
    dispatches.push([pipe, bind(dev, pipe, [u, sIn, iIn, sOut, iOut]), [chunks, nq, 1]]);
    if (chunks === 1) break;
    if (sIn !== scores) temps.push(sIn, iIn);
    sIn = sOut; iIn = iOut; m = outStride; stride = outStride; implicit = 0;
  }
  if (sIn !== scores) temps.push(sIn, iIn);
  return { dispatches, scores: sOut, idx: iOut, temps };
}

// ---- exclusive prefix sum (Blelloch, 512 per workgroup, recursive over block sums) -------------------
// Entry points scanBlocks and addOffsets; bindings data (u32, scanned in place), sums, uniform P { n }.
export const SCAN = /* wgsl */ `
struct P { n: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> sums: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> tmp: array<u32, 512>;
@compute @workgroup_size(256) fn scanBlocks(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) w: vec3u) {
  let a = w.x * 512u + 2u * t;
  tmp[2u * t] = select(0u, data[a], a < p.n);
  tmp[2u * t + 1u] = select(0u, data[a + 1u], a + 1u < p.n);
  var off = 1u;
  for (var d = 256u; d > 0u; d = d >> 1u) {
    workgroupBarrier();
    if (t < d) { let ai = off * (2u * t + 1u) - 1u; let bi = off * (2u * t + 2u) - 1u; tmp[bi] = tmp[bi] + tmp[ai]; }
    off = off * 2u;
  }
  workgroupBarrier();
  if (t == 0u) { sums[w.x] = tmp[511]; tmp[511] = 0u; }
  for (var d = 1u; d < 512u; d = d * 2u) {
    off = off >> 1u;
    workgroupBarrier();
    if (t < d) { let ai = off * (2u * t + 1u) - 1u; let bi = off * (2u * t + 2u) - 1u; let v = tmp[ai]; tmp[ai] = tmp[bi]; tmp[bi] = tmp[bi] + v; }
  }
  workgroupBarrier();
  if (a < p.n) { data[a] = tmp[2u * t]; }
  if (a + 1u < p.n) { data[a + 1u] = tmp[2u * t + 1u]; }
}
@compute @workgroup_size(256) fn addOffsets(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) w: vec3u) {
  let off = sums[w.x];
  let a = w.x * 512u + t;
  if (a < p.n) { data[a] = data[a] + off; }
  if (a + 256u < p.n) { data[a + 256u] = data[a + 256u] + off; }
}`;

// Plan a scan of n u32 values in `data` in place: returns { levels, dispatches }, the dispatches in
// order (scanBlocks per level, then addOffsets back down). Buffers it creates are in levels[].sums.
export async function scanPipelines(dev) { const [scan, add] = await compile(dev, SCAN, ["scanBlocks", "addOffsets"]); return { scan, add }; }
export function planScan(dev, { scan, add }, data, n) {
  const levels = [];
  let len = n, cur = data;
  while (true) {
    const blocks = Math.ceil(len / 512);
    if (blocks > 65535) throw new Error("scan: n too large for a 1-D dispatch");
    const sums = dev.createBuffer({ size: Math.max(16, blocks * 4), usage: STORAGE() }), uni = uniform(dev, [len]);
    const bufs = [cur, sums, uni];
    levels.push({ len, blocks, sums, uni, bgScan: bind(dev, scan, bufs), bgAdd: bind(dev, add, bufs) });
    if (blocks === 1) break;
    cur = sums; len = blocks;
  }
  const dispatches = levels.map((l) => [scan, l.bgScan, l.blocks]);
  for (let k = levels.length - 2; k >= 0; k--) dispatches.push([add, levels[k].bgAdd, levels[k].blocks]);
  return { levels, dispatches };
}

// ---- LSD radix sort of u32 keys, 8 passes of 4 bits -------------------------------------------------
// Per pass: per-block digit histogram, exclusive scan of the digit-major histogram (SCAN above), stable
// scatter. Entry points histogram and scatter; bindings src, dst, hist, uniform P { n, shift, nb, len }.
// scatter uses 17,408 bytes of workgroup memory: request maxComputeWorkgroupStorageSize >= that
// (WebGPU's default is 16,384).
export const radixWGSL = (BLOCK = 1024) => /* wgsl */ `
struct P { n: u32, shift: u32, nb: u32, len: u32 }
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read_write> hist: array<u32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> cnt: array<atomic<u32>, 16>;
@compute @workgroup_size(256) fn histogram(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) w: vec3u) {
  if (t < 16u) { atomicStore(&cnt[t], 0u); }
  workgroupBarrier();
  for (var k = 0u; k < 4u; k++) {
    let i = w.x * ${BLOCK}u + t * 4u + k;
    if (i < p.n) { atomicAdd(&cnt[(src[i] >> p.shift) & 15u], 1u); }
  }
  workgroupBarrier();
  if (t < 16u) { hist[t * p.nb + w.x] = atomicLoad(&cnt[t]); }
}
var<workgroup> loc: array<u32, 4096>;
var<workgroup> part: array<u32, 256>;
@compute @workgroup_size(256) fn scatter(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) w: vec3u) {
  var c: array<u32, 16>;
  var ks: array<u32, 4>;
  for (var d = 0u; d < 16u; d++) { c[d] = 0u; }
  for (var k = 0u; k < 4u; k++) {
    let i = w.x * ${BLOCK}u + t * 4u + k;
    if (i < p.n) { ks[k] = src[i]; c[(ks[k] >> p.shift) & 15u] += 1u; }
  }
  for (var d = 0u; d < 16u; d++) { loc[d * 256u + t] = c[d]; }
  workgroupBarrier();
  // exclusive scan of loc (digit-major, 4096 entries): 16 per thread, then a scan of the 256 partials
  var s = 0u;
  for (var j = 0u; j < 16u; j++) { let v = loc[t * 16u + j]; loc[t * 16u + j] = s; s += v; }
  part[t] = s;
  workgroupBarrier();
  for (var off = 1u; off < 256u; off = off * 2u) {
    var v = 0u;
    if (t >= off) { v = part[t - off]; }
    workgroupBarrier();
    part[t] = part[t] + v;
    workgroupBarrier();
  }
  let excl = part[t] - s;
  for (var j = 0u; j < 16u; j++) { loc[t * 16u + j] = loc[t * 16u + j] + excl; }
  workgroupBarrier();
  // this thread's start within each digit's run in the block; private, so no thread writes what another reads
  for (var d = 0u; d < 16u; d++) { c[d] = hist[d * p.nb + w.x] + loc[d * 256u + t] - loc[d * 256u]; }
  for (var k = 0u; k < 4u; k++) {
    let i = w.x * ${BLOCK}u + t * 4u + k;
    if (i < p.n) {
      let d = (ks[k] >> p.shift) & 15u;
      dst[c[d]] = ks[k];
      c[d] = c[d] + 1u;
    }
  }
}`;

// Plan a sort of n keys in A (B is scratch of the same size; 8 passes end back in A): the dispatch list.
export async function sortPipelines(dev, block = 1024) {
  const [hist, scatter] = await compile(dev, radixWGSL(block), ["histogram", "scatter"]);
  return { hist, scatter, block, ...(await scanPipelines(dev)) };
}
export function planSort(dev, P, A, B, n) {
  const nb = Math.ceil(n / P.block);
  if (nb > 65535) throw new Error("sort: n too large");
  const hist = dev.createBuffer({ size: 16 * nb * 4, usage: STORAGE() });
  const sc = planScan(dev, P, hist, 16 * nb), dispatches = [];
  for (let pass = 0; pass < 8; pass++) {
    const u = uniform(dev, [n, pass * 4, nb, 0]), [s, d] = pass % 2 === 0 ? [A, B] : [B, A];
    dispatches.push([P.hist, bind(dev, P.hist, [[0, s], [2, hist], [3, u]]), nb]);   // histogram does not use dst
    dispatches.push(...sc.dispatches);
    dispatches.push([P.scatter, bind(dev, P.scatter, [s, d, hist, u]), nb]);
  }
  return { nb, hist, scanLevels: sc.levels, dispatches };
}

// Record a dispatch list into one compute pass each (what the jobs do) or all in one pass.
export function record(enc, dispatches) {
  for (const [pipe, bg, groups] of dispatches) {
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...[groups].flat()); pass.end();
  }
}
