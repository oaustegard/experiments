// LSD radix sort of n u32 keys on the GPU: 8 passes of 4 bits. Each pass: per-block digit histogram,
// exclusive scan of the digit-major histogram (same Blelloch scan as jobs/scan), stable scatter.
// Keys come from a counter hash so numpy can rebuild them. Verified in full against a JS sort.
const n = ctx.inputs.n, BLOCK = 1024, SAMPLES = 64;
const hash = (i) => { let h = Math.imul(i, 0x9E3779B1) >>> 0; h ^= h >>> 16; h = Math.imul(h, 0x85EBCA6B) >>> 0; h ^= h >>> 13; h = Math.imul(h, 0xC2B2AE35) >>> 0; h ^= h >>> 16; return h >>> 0; };
const keys = new Uint32Array(n); for (let i = 0; i < n; i++) keys[i] = hash(i + 12345);
const nb = Math.ceil(n / BLOCK);
if (nb > 65535) throw new Error("n too large");
const dev = await ctx.gpu();
const wgsl = `
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
// the scan from jobs/scan, over the histogram
const scanWgsl = `
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
async function compile(code, entries) {
  const m = dev.createShaderModule({ code });
  const errs = (await m.getCompilationInfo()).messages.filter((x) => x.type === "error");
  if (errs.length) throw new Error("WGSL: " + errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("; "));
  return Promise.all(entries.map((e) => dev.createComputePipelineAsync({ layout: "auto", compute: { module: m, entryPoint: e } })));
}
const [pHist, pScatter] = await compile(wgsl, ["histogram", "scatter"]);
const [pScan, pAdd] = await compile(scanWgsl, ["scanBlocks", "addOffsets"]);
const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, UN = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const A = dev.createBuffer({ size: n * 4, usage: ST }), B = dev.createBuffer({ size: n * 4, usage: ST });
const hist = dev.createBuffer({ size: 16 * nb * 4, usage: ST });
// scan levels over hist (16*nb entries)
const levels = []; let len = 16 * nb, cur = hist;
while (true) {
  const blocks = Math.ceil(len / 512), sums = dev.createBuffer({ size: Math.max(16, blocks * 4), usage: ST }), uni = dev.createBuffer({ size: 16, usage: UN });
  dev.queue.writeBuffer(uni, 0, new Uint32Array([len, 0, 0, 0]));
  const entries = [cur, sums, uni].map((buffer, binding) => ({ binding, resource: { buffer } }));
  levels.push({ blocks, bgScan: dev.createBindGroup({ layout: pScan.getBindGroupLayout(0), entries }), bgAdd: dev.createBindGroup({ layout: pAdd.getBindGroupLayout(0), entries }) });
  if (blocks === 1) break; cur = sums; len = blocks;
}
const passUni = [], bgs = [];
for (let pass = 0; pass < 8; pass++) {
  const u = dev.createBuffer({ size: 16, usage: UN }); dev.queue.writeBuffer(u, 0, new Uint32Array([n, pass * 4, nb, 0])); passUni.push(u);
  const [s, d] = pass % 2 === 0 ? [A, B] : [B, A];
  const e = [s, d, hist, u].map((buffer, binding) => ({ binding, resource: { buffer } }));
  bgs.push({ hist: dev.createBindGroup({ layout: pHist.getBindGroupLayout(0), entries: e.filter((x) => x.binding !== 1) }), scatter: dev.createBindGroup({ layout: pScatter.getBindGroupLayout(0), entries: e }) });
}
const hasTs = dev.features.has("timestamp-query"), ITERS = 4;
const qset = hasTs ? dev.createQuerySet({ type: "timestamp", count: 2 * ITERS }) : null;
const wall = [];
for (let it = 0; it < ITERS; it++) {
  dev.queue.writeBuffer(A, 0, keys);
  const enc = dev.createCommandEncoder(), seq = [];
  for (let pass = 0; pass < 8; pass++) {
    seq.push([pHist, bgs[pass].hist, nb]);
    for (const l of levels) seq.push([pScan, l.bgScan, l.blocks]);
    for (let k = levels.length - 2; k >= 0; k--) seq.push([pAdd, levels[k].bgAdd, levels[k].blocks]);
    seq.push([pScatter, bgs[pass].scatter, nb]);
  }
  seq.forEach(([pipe, bg, wgs], i) => {
    const tw = {};
    if (hasTs && i === 0) tw.beginningOfPassWriteIndex = 2 * it;
    if (hasTs && i === seq.length - 1) tw.endOfPassWriteIndex = 2 * it + 1;
    const pass = enc.beginComputePass(Object.keys(tw).length ? { timestampWrites: { querySet: qset, ...tw } } : {});
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(wgs); pass.end();
  });
  const t0 = performance.now(); dev.queue.submit([enc.finish()]); await dev.queue.onSubmittedWorkDone(); wall.push(performance.now() - t0);
}
// 8 passes end back in A
const rd = dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(A, 0, rd, 0, n * 4);
let rts = null;
if (hasTs) { const res = dev.createBuffer({ size: 16 * ITERS, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }); rts = dev.createBuffer({ size: 16 * ITERS, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); enc.resolveQuerySet(qset, 0, 2 * ITERS, res, 0); enc.copyBufferToBuffer(res, 0, rts, 0, 16 * ITERS); }
dev.queue.submit([enc.finish()]);
await rd.mapAsync(GPUMapMode.READ); const out = new Uint32Array(rd.getMappedRange().slice(0)); rd.unmap();
let gpuMs = null; if (rts) { await rts.mapAsync(GPUMapMode.READ); const ts = new BigInt64Array(rts.getMappedRange().slice(0)); gpuMs = []; for (let i = 0; i < ITERS; i++) gpuMs.push(Number(ts[2 * i + 1] - ts[2 * i]) / 1e6); }
const t0 = performance.now(); const ref = keys.slice().sort(); const jsSortMs = performance.now() - t0;
let bad = 0, firstBad = -1; for (let i = 0; i < n; i++) if (out[i] !== ref[i]) { bad++; if (firstBad < 0) firstBad = i; }
let checksum = 0; for (let i = 0; i < n; i++) checksum = (checksum + Math.imul(out[i], (i % 1000) + 1)) >>> 0;
const sampleIdx = Array.from({ length: SAMPLES }, (_, k) => Math.floor((k * (n - 1)) / (SAMPLES - 1)));
const med = (a) => { const s = a.slice(1).sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
ctx.log(`n=${n} blocks=${nb} scan levels=${levels.length} mismatches=${bad}` + (gpuMs ? ` gpu median ${med(gpuMs).toFixed(3)} ms` : "") + ` js sort ${jsSortMs.toFixed(0)} ms`);
return { n, mismatches: bad, firstBad, checksum, sampleIdx, samples: sampleIdx.map((i) => out[i]), gpuMs, wallMs: wall, jsSortMs };
