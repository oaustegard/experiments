// Exact k-nearest-neighbour graph: for each of n points, its 10 nearest others by dot product, D=256.
// Points come from a counter hash, so nothing is sent and numpy can rebuild them bit for bit.
// Batches of 1024 queries per dispatch keep each GPU submission short (mobile watchdogs).
const n = ctx.inputs.n, D = 256, K = 10, QB = 4, WG = 256, BATCH = Math.min(1024, n);
if (n % BATCH || BATCH % QB) throw new Error(`n must be a multiple of ${BATCH}`);
const hash = (i) => { let h = Math.imul(i, 0x9E3779B1) >>> 0; h ^= h >>> 16; h = Math.imul(h, 0x85EBCA6B) >>> 0; h ^= h >>> 13; h = Math.imul(h, 0xC2B2AE35) >>> 0; h ^= h >>> 16; return h >>> 0; };
const tg = performance.now();
const pts = new Float32Array(n * D);
for (let i = 0; i < n * D; i++) pts[i] = (hash(i + 777) & 0xFFFFFF) / 16777216 - 0.5;   // exact in float32
const genMs = performance.now() - tg;
const dev = await ctx.gpu();
const wgsl = `
const K: u32 = ${K}u; const QB: u32 = ${QB}u; const WG: u32 = ${WG}u; const DV: u32 = ${D / 4}u;
struct P { n: u32, q0: u32 }
@group(0) @binding(0) var<storage, read> pts: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outIdx: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> qsh: array<vec4f, ${QB * D / 4}>;
var<workgroup> cs: array<f32, ${WG * K}>;
var<workgroup> ci: array<u32, ${WG * K}>;
var<workgroup> rv: array<f32, ${WG}>;
var<workgroup> rt: array<u32, ${WG}>;
@compute @workgroup_size(${WG})
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) w: vec3u) {
  let q0 = p.q0 + w.x * QB;
  for (var i = t; i < QB * DV; i += WG) { qsh[i] = pts[q0 * DV + i]; }
  workgroupBarrier();
  var bs: array<f32, ${QB * K}>;
  var bi: array<u32, ${QB * K}>;
  for (var i = 0u; i < QB * K; i++) { bs[i] = -1e30; bi[i] = 0u; }
  for (var j = t; j < p.n; j += WG) {
    var acc = vec4f(0.0);
    for (var d = 0u; d < DV; d++) {
      let x = pts[j * DV + d];
      acc += vec4f(dot(x, qsh[d]), dot(x, qsh[DV + d]), dot(x, qsh[2u * DV + d]), dot(x, qsh[3u * DV + d]));
    }
    for (var q = 0u; q < QB; q++) {
      if (j == q0 + q) { continue; }   // a point is not its own neighbour
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
      if (t == win) { outIdx[(q0 + q) * K + k] = ci[t * K + h]; h = h + 1u; }
    }
    workgroupBarrier();
  }
}`;
const module = dev.createShaderModule({ code: wgsl });
const errs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
if (errs.length) throw new Error("WGSL: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
const pipe = await dev.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
const bPts = dev.createBuffer({ size: pts.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
const bIdx = dev.createBuffer({ size: n * K * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
dev.queue.writeBuffer(bPts, 0, pts);
const batches = n / BATCH, hasTs = dev.features.has("timestamp-query");
const qset = hasTs ? dev.createQuerySet({ type: "timestamp", count: 2 * batches }) : null;
const wall = [];
const tAll = performance.now();
for (let b = 0; b < batches; b++) {
  const u = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(u, 0, new Uint32Array([n, b * BATCH, 0, 0]));
  const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [bPts, bIdx, u].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass(hasTs ? { timestampWrites: { querySet: qset, beginningOfPassWriteIndex: 2 * b, endOfPassWriteIndex: 2 * b + 1 } } : {});
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(BATCH / QB); pass.end();
  const t0 = performance.now(); dev.queue.submit([enc.finish()]); await dev.queue.onSubmittedWorkDone(); wall.push(performance.now() - t0);
  u.destroy();
  if (b % 16 === 15) ctx.log(`batch ${b + 1}/${batches}, ${((performance.now() - tAll) / 1000).toFixed(1)} s`);
}
const totalWallMs = performance.now() - tAll;
const enc = dev.createCommandEncoder();
const rIdx = dev.createBuffer({ size: n * K * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
enc.copyBufferToBuffer(bIdx, 0, rIdx, 0, n * K * 4);
let rTs = null;
if (hasTs) { const res = dev.createBuffer({ size: 16 * batches, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }); rTs = dev.createBuffer({ size: 16 * batches, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); enc.resolveQuerySet(qset, 0, 2 * batches, res, 0); enc.copyBufferToBuffer(res, 0, rTs, 0, 16 * batches); }
dev.queue.submit([enc.finish()]);
await rIdx.mapAsync(GPUMapMode.READ); const idx = new Uint32Array(rIdx.getMappedRange().slice(0));
let gpuMs = null;
if (rTs) { await rTs.mapAsync(GPUMapMode.READ); const ts = new BigInt64Array(rTs.getMappedRange().slice(0)); gpuMs = []; for (let b = 0; b < batches; b++) gpuMs.push(Number(ts[2 * b + 1] - ts[2 * b]) / 1e6); }
[bPts, bIdx, rIdx, rTs].forEach((x) => x?.destroy());
const gpuTotal = gpuMs ? gpuMs.reduce((a, b) => a + b, 0) : null, flop = 2 * n * n * D;
ctx.log(`n=${n}: ${batches} batches, wall ${(totalWallMs / 1000).toFixed(2)} s` + (gpuTotal ? `, gpu ${(gpuTotal / 1000).toFixed(2)} s = ${(flop / (gpuTotal / 1000) / 1e9).toFixed(0)} GFLOP/s` : "") + `, point generation ${genMs.toFixed(0)} ms`);
return { n, idx, gpuMs, gpuTotalMs: gpuTotal, wallMs: wall, totalWallMs, gflop: flop / 1e9, genMs };
