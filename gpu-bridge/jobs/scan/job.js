// Exclusive prefix sum of n u32 values on the GPU (Blelloch, 512 per workgroup, recursive over block sums).
// Inputs are generated from a counter hash so numpy can rebuild them; the job verifies itself in full
// against a JS scan and returns a checksum and samples for an independent numpy check. The kernels are
// _lib/wgsl/retrieval.js SCAN / planScan at inputs.sha (default main).
const n = ctx.inputs.n, SAMPLES = 64;
const hash = (i) => { let h = Math.imul(i, 0x9E3779B1) >>> 0; h ^= h >>> 16; h = Math.imul(h, 0x85EBCA6B) >>> 0; h ^= h >>> 13; return h >>> 0; };
const data = new Uint32Array(n); for (let i = 0; i < n; i++) data[i] = hash(i) & 15;
const dev = await ctx.gpu();
const LIB = await ctx.importModule(`https://cdn.jsdelivr.net/gh/oaustegard/experiments@${ctx.inputs.sha ?? "main"}/_lib/wgsl/retrieval.js`);
const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const bData = dev.createBuffer({ size: Math.max(16, n * 4), usage: ST });
// level k scans len_k values into ceil(len_k/512) block sums (planScan in _lib/wgsl/retrieval.js)
const { levels, dispatches } = LIB.planScan(dev, await LIB.scanPipelines(dev), bData, n);
const hasTs = dev.features.has("timestamp-query"), ITERS = 6;
const qset = hasTs ? dev.createQuerySet({ type: "timestamp", count: 2 * ITERS }) : null;
const gpuNs = [], wall = [];
for (let it = 0; it < ITERS; it++) {
  dev.queue.writeBuffer(bData, 0, data);
  const enc = dev.createCommandEncoder();
  const passes = dispatches;
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
