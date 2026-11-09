// Exact k-nearest-neighbour graph: for each of n points, its 10 nearest others by dot product, D=256.
// Points come from a counter hash, so nothing is sent and numpy can rebuild them bit for bit.
// Batches of 1024 queries per dispatch keep each GPU submission short (mobile watchdogs). The kernel
// is _lib/wgsl/retrieval.js knnWGSL (excludeSelf) at inputs.sha (default main).
const n = ctx.inputs.n, D = 256, K = 10, QB = 4, WG = 256, BATCH = Math.min(1024, n);
if (n % BATCH || BATCH % QB) throw new Error(`n must be a multiple of ${BATCH}`);
const hash = (i) => { let h = Math.imul(i, 0x9E3779B1) >>> 0; h ^= h >>> 16; h = Math.imul(h, 0x85EBCA6B) >>> 0; h ^= h >>> 13; h = Math.imul(h, 0xC2B2AE35) >>> 0; h ^= h >>> 16; return h >>> 0; };
const tg = performance.now();
const pts = new Float32Array(n * D);
for (let i = 0; i < n * D; i++) pts[i] = (hash(i + 777) & 0xFFFFFF) / 16777216 - 0.5;   // exact in float32
const genMs = performance.now() - tg;
const dev = await ctx.gpu();
const LIB = await ctx.importModule(`https://cdn.jsdelivr.net/gh/oaustegard/experiments@${ctx.inputs.sha ?? "main"}/_lib/wgsl/retrieval.js`);
const [pipe] = await LIB.compile(dev, LIB.knnWGSL({ D, K, QB, WG, excludeSelf: true }));
const bPts = dev.createBuffer({ size: pts.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
const bIdx = dev.createBuffer({ size: n * K * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const bScore = dev.createBuffer({ size: n * K * 4, usage: GPUBufferUsage.STORAGE });   // the shared kernel also writes scores
dev.queue.writeBuffer(bPts, 0, pts);
const batches = n / BATCH, hasTs = dev.features.has("timestamp-query");
const qset = hasTs ? dev.createQuerySet({ type: "timestamp", count: 2 * batches }) : null;
const wall = [];
const tAll = performance.now();
for (let b = 0; b < batches; b++) {
  const u = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(u, 0, new Uint32Array([n, b * BATCH, 0, 0]));
  const bg = dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [bPts, bPts, bIdx, bScore, u].map((buffer, binding) => ({ binding, resource: { buffer } })) });
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
[bPts, bIdx, bScore, rIdx, rTs].forEach((x) => x?.destroy());
const gpuTotal = gpuMs ? gpuMs.reduce((a, b) => a + b, 0) : null, flop = 2 * n * n * D;
ctx.log(`n=${n}: ${batches} batches, wall ${(totalWallMs / 1000).toFixed(2)} s` + (gpuTotal ? `, gpu ${(gpuTotal / 1000).toFixed(2)} s = ${(flop / (gpuTotal / 1000) / 1e9).toFixed(0)} GFLOP/s` : "") + `, point generation ${genMs.toFixed(0)} ms`);
return { n, idx, gpuMs, gpuTotalMs: gpuTotal, wallMs: wall, totalWallMs, gflop: flop / 1e9, genMs };
