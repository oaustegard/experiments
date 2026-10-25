// Exclusive prefix sum of n u32 values on the GPU (Blelloch, 512 per workgroup, recursive over block sums).
// Inputs are generated from a counter hash so numpy can rebuild them; the job verifies itself in full
// against a JS scan and returns a checksum and samples for an independent numpy check.
const n = ctx.inputs.n, SAMPLES = 64;
const hash = (i) => { let h = Math.imul(i, 0x9E3779B1) >>> 0; h ^= h >>> 16; h = Math.imul(h, 0x85EBCA6B) >>> 0; h ^= h >>> 13; return h >>> 0; };
const data = new Uint32Array(n); for (let i = 0; i < n; i++) data[i] = hash(i) & 15;
const dev = await ctx.gpu();
const wgsl = `
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
const module = dev.createShaderModule({ code: wgsl });
const errs = (await module.getCompilationInfo()).messages.filter((m) => m.type === "error");
if (errs.length) throw new Error("WGSL: " + errs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
const pScan = await dev.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "scanBlocks" } });
const pAdd = await dev.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "addOffsets" } });
const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
// Plan the levels: level k scans len_k values into ceil(len_k/512) block sums.
const levels = []; let len = n;
const bData = dev.createBuffer({ size: Math.max(16, n * 4), usage: ST });
let cur = bData;
while (true) {
  const blocks = Math.ceil(len / 512);
  const sums = dev.createBuffer({ size: Math.max(16, blocks * 4), usage: ST });
  const uni = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(uni, 0, new Uint32Array([len, 0, 0, 0]));
  const entries = [cur, sums, uni].map((buffer, binding) => ({ binding, resource: { buffer } }));
  levels.push({ len, blocks, sums, uni, bgScan: dev.createBindGroup({ layout: pScan.getBindGroupLayout(0), entries }), bgAdd: dev.createBindGroup({ layout: pAdd.getBindGroupLayout(0), entries }) });
  if (blocks === 1) break;
  cur = sums; len = blocks;
}
if (levels.some((l) => l.blocks > 65535)) throw new Error("n too large for 1-D dispatch");
const hasTs = dev.features.has("timestamp-query"), ITERS = 6;
const qset = hasTs ? dev.createQuerySet({ type: "timestamp", count: 2 * ITERS }) : null;
const gpuNs = [], wall = [];
for (let it = 0; it < ITERS; it++) {
  dev.queue.writeBuffer(bData, 0, data);
  const enc = dev.createCommandEncoder();
  const passes = [];
  for (const l of levels) passes.push([pScan, l.bgScan, l.blocks]);
  for (let k = levels.length - 2; k >= 0; k--) passes.push([pAdd, levels[k].bgAdd, levels[k].blocks]);
  passes.forEach(([pipe, bg, wgs], i) => {
    const tw = hasTs && (i === 0 || i === passes.length - 1) ? { timestampWrites: { querySet: qset, ...(i === 0 ? { beginningOfPassWriteIndex: 2 * it } : {}), ...(i === passes.length - 1 ? { endOfPassWriteIndex: 2 * it + 1 } : {}) } } : {};
    const pass = enc.beginComputePass(tw); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgs); pass.end();
  });
  const t0 = performance.now(); dev.queue.submit([enc.finish()]); await dev.queue.onSubmittedWorkDone(); wall.push(performance.now() - t0);
}
const rd = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
let rts = null;
const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(bData, 0, rd, 0, n * 4);
if (hasTs) { const res = dev.createBuffer({ size: 16 * ITERS, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }); rts = dev.createBuffer({ size: 16 * ITERS, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); enc.resolveQuerySet(qset, 0, 2 * ITERS, res, 0); enc.copyBufferToBuffer(res, 0, rts, 0, 16 * ITERS); }
dev.queue.submit([enc.finish()]);
await rd.mapAsync(GPUMapMode.READ); const out = new Uint32Array(rd.getMappedRange().slice(0)); rd.unmap();
let gpuMs = null; if (rts) { await rts.mapAsync(GPUMapMode.READ); const ts = new BigInt64Array(rts.getMappedRange().slice(0)); gpuMs = []; for (let i = 0; i < ITERS; i++) gpuMs.push(Number(ts[2 * i + 1] - ts[2 * i]) / 1e6); }
// full check against a JS exclusive scan
const t0 = performance.now(); let acc = 0, bad = 0, firstBad = -1;
for (let i = 0; i < n; i++) { if (out[i] !== acc) { bad++; if (firstBad < 0) firstBad = i; } acc = (acc + data[i]) >>> 0; }
const jsMs = performance.now() - t0;
let checksum = 0; for (let i = 0; i < n; i++) checksum = (checksum + out[i]) >>> 0;
const sampleIdx = Array.from({ length: SAMPLES }, (_, k) => Math.floor((k * (n - 1)) / (SAMPLES - 1)));
const med = (a) => { const s = a.slice(1).sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
ctx.log(`n=${n} levels=${levels.length} mismatches=${bad}` + (gpuMs ? ` gpu median ${med(gpuMs).toFixed(3)} ms` : "") + ` js check ${jsMs.toFixed(0)} ms`);
return { n, mismatches: bad, firstBad, checksum, total: acc, sampleIdx, samples: sampleIdx.map((i) => out[i]), gpuMs, wallMs: wall, jsScanMs: jsMs, levels: levels.length };
