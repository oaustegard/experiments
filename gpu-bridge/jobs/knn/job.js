// Exact top-10 nearest neighbours by dot product: Q queries against N docs of 256 dims.
// One fused kernel: each workgroup takes 4 queries, streams every doc once (vec4 loads),
// keeps a per-thread top-10 per query, then merges the 256 thread lists with a tree argmax.
// Options (inputs): docDtype "f32" (default) or "f16" (docs as Uint16Array halves, needs shader-f16);
// cache "store" keeps the docs buffer on the GPU in ctx.cache for later jobs, "use" runs against it
// with only queries sent. Defaults reproduce knn-1. The kernel is _lib/wgsl/retrieval.js knnWGSL at
// inputs.sha (default main).
const { docs, queries, docDtype = "f32", cache = null } = ctx.inputs;
const D = 256, K = 10, QB = 4, WG = 256, KEY = "knn-docs-" + docDtype;
const Q = queries.length / D;
if (Q % QB) throw new Error(`query count ${Q} must be a multiple of ${QB}`);
const dev = await ctx.gpu();
if (docDtype === "f16" && !dev.features.has("shader-f16")) throw new Error("this GPU has no shader-f16");
const cached = cache === "use" ? ctx.cache.get(KEY) : null;
if (cache === "use" && !cached) throw new Error(`no ${KEY} in the worker cache: run a job with cache "store" first in this tab`);
const N = cached ? cached.N : docs.length / D;
const LIB = await ctx.importModule(`https://cdn.jsdelivr.net/gh/oaustegard/experiments@${ctx.inputs.sha ?? "main"}/_lib/wgsl/retrieval.js`);
const [pipe] = await LIB.compile(dev, LIB.knnWGSL({ D, K, QB, WG, f16: docDtype === "f16" }));
const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
const buf = (size, usage) => dev.createBuffer({ size, usage });
const bDocs = cached ? cached.buf : buf(docs.byteLength, S | CD), bQ = buf(queries.byteLength, S | CD);
const tUp0 = performance.now();
const bIdx = buf(Q * K * 4, S | CS), bScore = buf(Q * K * 4, S | CS), bP = buf(16, GPUBufferUsage.UNIFORM | CD);
if (!cached) dev.queue.writeBuffer(bDocs, 0, docs);
dev.queue.writeBuffer(bQ, 0, queries); dev.queue.writeBuffer(bP, 0, new Uint32Array([N, 0, 0, 0]));
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
if (cache === "store") ctx.cache.set(KEY, { buf: bDocs, N }); else if (!cached) bDocs.destroy();
[bQ, bIdx, bScore, bP, rIdx, rScore, rTs].forEach((b) => b?.destroy());
const med = (a) => { const s = a.slice(1).sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
ctx.log(`${docDtype}${cached ? " cached" : ""} N=${N} Q=${Q} wall median ${med(wall).toFixed(2)} ms` + (gpuMs ? `, gpu median ${med(gpuMs).toFixed(3)} ms` : ", no timestamp-query"));
return { idx, score, wallMs: wall, gpuMs, gflop: (2 * N * Q * D) / 1e9, docDtype, cache, N, Q };
