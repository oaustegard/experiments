// LSD radix sort of n u32 keys on the GPU: 8 passes of 4 bits. Each pass: per-block digit histogram,
// exclusive scan of the digit-major histogram (same Blelloch scan as jobs/scan), stable scatter.
// Keys come from a counter hash so numpy can rebuild them. Verified in full against a JS sort. The
// kernels are _lib/wgsl/retrieval.js radixWGSL / planSort at inputs.sha (default main).
const n = ctx.inputs.n, BLOCK = 1024, SAMPLES = 64;
const hash = (i) => { let h = Math.imul(i, 0x9E3779B1) >>> 0; h ^= h >>> 16; h = Math.imul(h, 0x85EBCA6B) >>> 0; h ^= h >>> 13; h = Math.imul(h, 0xC2B2AE35) >>> 0; h ^= h >>> 16; return h >>> 0; };
const keys = new Uint32Array(n); for (let i = 0; i < n; i++) keys[i] = hash(i + 12345);
const nb = Math.ceil(n / BLOCK);
if (nb > 65535) throw new Error("n too large");
const dev = await ctx.gpu();
const LIB = await ctx.importModule(`https://cdn.jsdelivr.net/gh/oaustegard/experiments@${ctx.inputs.sha ?? "main"}/_lib/wgsl/retrieval.js`);
const ST = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
const A = dev.createBuffer({ size: n * 4, usage: ST }), B = dev.createBuffer({ size: n * 4, usage: ST });
// histogram, scan levels over hist (16*nb entries), scatter, x 8 passes (planSort in _lib/wgsl/retrieval.js)
const plan = LIB.planSort(dev, await LIB.sortPipelines(dev, BLOCK), A, B, n), levels = plan.scanLevels;
const hasTs = dev.features.has("timestamp-query"), ITERS = 4;
const qset = hasTs ? dev.createQuerySet({ type: "timestamp", count: 2 * ITERS }) : null;
const wall = [];
for (let it = 0; it < ITERS; it++) {
  dev.queue.writeBuffer(A, 0, keys);
  const enc = dev.createCommandEncoder(), seq = plan.dispatches;
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
