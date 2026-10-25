// Exact top-10 nearest neighbours by dot product: Q queries against N docs of 256 dims.
// One fused kernel: each workgroup takes 4 queries, streams every doc once (vec4 loads),
// keeps a per-thread top-10 per query, then merges the 256 thread lists with a tree argmax.
const { docs, queries } = ctx.inputs;
const D = 256, K = 10, QB = 4, WG = 256;
const N = docs.length / D, Q = queries.length / D;
if (Q % QB) throw new Error(`query count ${Q} must be a multiple of ${QB}`);
const dev = await ctx.gpu();
const wgsl = `
const K: u32 = ${K}u; const QB: u32 = ${QB}u; const WG: u32 = ${WG}u; const DV: u32 = ${D / 4}u;
struct P { n: u32, nq: u32 }
@group(0) @binding(0) var<storage, read> docs: array<vec4f>;
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
  let q0 = w.x * QB;
  for (var i = t; i < QB * DV; i += WG) { qsh[i] = qs[q0 * DV + i]; }
  workgroupBarrier();
  var bs: array<f32, ${QB * K}>;
  var bi: array<u32, ${QB * K}>;
  for (var i = 0u; i < QB * K; i++) { bs[i] = -1e30; bi[i] = 0u; }
  for (var j = t; j < p.n; j += WG) {
    var acc = vec4f(0.0);
    for (var d = 0u; d < DV; d++) {
      let x = docs[j * DV + d];
      acc += vec4f(dot(x, qsh[d]), dot(x, qsh[DV + d]), dot(x, qsh[2u * DV + d]), dot(x, qsh[3u * DV + d]));
    }
    for (var q = 0u; q < QB; q++) {
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
const module = dev.createShaderModule({ code: wgsl });
const errs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
if (errs.length) throw new Error("WGSL: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
const pipe = await dev.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
const buf = (size, usage) => dev.createBuffer({ size, usage });
const bDocs = buf(docs.byteLength, S | CD), bQ = buf(queries.byteLength, S | CD);
const bIdx = buf(Q * K * 4, S | CS), bScore = buf(Q * K * 4, S | CS), bP = buf(16, GPUBufferUsage.UNIFORM | CD);
dev.queue.writeBuffer(bDocs, 0, docs); dev.queue.writeBuffer(bQ, 0, queries); dev.queue.writeBuffer(bP, 0, new Uint32Array([N, Q, 0, 0]));
const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [bDocs, bQ, bIdx, bScore, bP].map((buffer, binding) => ({ binding, resource: { buffer } })) });
const ITERS = 6, hasTs = dev.features.has("timestamp-query");
const qset = hasTs ? dev.createQuerySet({ type: "timestamp", count: 2 * ITERS }) : null;
const wall = [];
for (let it = 0; it < ITERS; it++) {
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass(hasTs ? { timestampWrites: { querySet: qset, beginningOfPassWriteIndex: 2 * it, endOfPassWriteIndex: 2 * it + 1 } } : {});
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(Q / QB); pass.end();
  const t0 = performance.now(); dev.queue.submit([enc.finish()]); await dev.queue.onSubmittedWorkDone(); wall.push(performance.now() - t0);
}
const enc = dev.createCommandEncoder();
const rIdx = buf(Q * K * 4, GPUBufferUsage.MAP_READ | CD), rScore = buf(Q * K * 4, GPUBufferUsage.MAP_READ | CD);
enc.copyBufferToBuffer(bIdx, 0, rIdx, 0, Q * K * 4); enc.copyBufferToBuffer(bScore, 0, rScore, 0, Q * K * 4);
let rTs = null;
if (hasTs) {
  const res = buf(16 * ITERS, GPUBufferUsage.QUERY_RESOLVE | CS); rTs = buf(16 * ITERS, GPUBufferUsage.MAP_READ | CD);
  enc.resolveQuerySet(qset, 0, 2 * ITERS, res, 0); enc.copyBufferToBuffer(res, 0, rTs, 0, 16 * ITERS);
}
dev.queue.submit([enc.finish()]);
await Promise.all([rIdx.mapAsync(GPUMapMode.READ), rScore.mapAsync(GPUMapMode.READ), rTs?.mapAsync(GPUMapMode.READ)]);
const idx = new Uint32Array(rIdx.getMappedRange().slice(0)), score = new Float32Array(rScore.getMappedRange().slice(0));
let gpuMs = null;
if (rTs) { const ts = new BigInt64Array(rTs.getMappedRange().slice(0)); gpuMs = []; for (let i = 0; i < ITERS; i++) gpuMs.push(Number(ts[2 * i + 1] - ts[2 * i]) / 1e6); }
[bDocs, bQ, bIdx, bScore, bP, rIdx, rScore, rTs].forEach((b) => b?.destroy());
const med = (a) => { const s = a.slice(1).sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
ctx.log(`N=${N} Q=${Q} wall median ${med(wall).toFixed(2)} ms` + (gpuMs ? `, gpu median ${med(gpuMs).toFixed(3)} ms` : ", no timestamp-query"));
return { idx, score, wallMs: wall, gpuMs, gflop: (2 * N * Q * D) / 1e9 };
