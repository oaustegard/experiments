// GPU Bridge worker runtime: holds the GPU device and a cache across jobs.
// The page fetches this file's text and starts it as a Worker (or prepends it to a job when
// importScripts of blob URLs is refused). A job is the BODY of `async function (ctx)`.
"use strict";
let adapter = null, device = null;
const cache = new Map();
const TYPES = { f32: Float32Array, f16: Uint16Array, u32: Uint32Array, i32: Int32Array, u8: Uint8Array, i8: Int8Array, u16: Uint16Array, i16: Int16Array, f64: Float64Array };
async function gpu() {
  if (device) return device;
  if (!self.navigator || !navigator.gpu) throw new Error("WebGPU is not available in this browser's workers");
  adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter");
  const want = ["shader-f16", "timestamp-query", "subgroups"].filter((f) => adapter.features.has(f));
  const L = adapter.limits, requiredLimits = {};
  for (const k of ["maxBufferSize", "maxStorageBufferBindingSize", "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupSizeX", "maxComputeWorkgroupSizeY", "maxComputeWorkgroupSizeZ", "maxComputeWorkgroupsPerDimension", "maxStorageBuffersPerShaderStage"]) if (k in L) requiredLimits[k] = L[k];
  device = await adapter.requestDevice({ requiredFeatures: want, requiredLimits });
  device.lost.then((i) => { postMessage({ t: "lost", s: i.message }); device = null; cache.clear(); });
  return device;
}
async function describe() {
  const d = await gpu(), info = adapter.info || {};
  const L = {}; for (const k of ["maxBufferSize", "maxStorageBufferBindingSize", "maxComputeWorkgroupStorageSize", "maxComputeInvocationsPerWorkgroup", "maxComputeWorkgroupsPerDimension"]) L[k] = d.limits[k];
  return { vendor: info.vendor || "", architecture: info.architecture || "", device: info.device || "", description: info.description || "", features: [...d.features].sort(), limits: L };
}
const pad4 = (n) => Math.max(4, Math.ceil(n / 4) * 4);
// One compute dispatch with group(0) bindings. Returns the read-back arrays and per-iteration wall times.
async function compute({ wgsl, entry = "main", bindings = [], dispatch = [1, 1, 1], read, iterations = 1, constants }) {
  const d = await gpu();
  const key = "pipe:" + entry + ":" + JSON.stringify(constants || {}) + ":" + wgsl;
  let pipe = cache.get(key);
  if (!pipe) {
    const module = d.createShaderModule({ code: wgsl });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === "error");
    if (errs.length) throw new Error("WGSL: " + errs.map((m) => m.lineNum + ":" + m.linePos + " " + m.message).join("; "));
    pipe = await d.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: entry, constants } });
    cache.set(key, pipe);
  }
  const bufs = [], kinds = [];
  bindings.forEach((b, i) => {
    let arr = null, size, usage, ctor = Float32Array, uniform = false;
    if (ArrayBuffer.isView(b)) { arr = b; ctor = b.constructor; }
    else if (b && b.uniform) { arr = b.uniform; ctor = arr.constructor; uniform = true; }
    else if (b && b.size != null) { ctor = TYPES[b.type || "f32"] || Float32Array; size = b.size * ctor.BYTES_PER_ELEMENT; }
    else throw new Error("binding " + i + ": pass a typed array, {uniform: typedArray} or {size, type}");
    size = uniform ? Math.ceil(arr.byteLength / 16) * 16 : pad4(arr ? arr.byteLength : size);
    usage = uniform ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const buf = d.createBuffer({ size, usage });
    if (arr) d.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    bufs.push(buf); kinds.push({ ctor, n: arr ? arr.length : b.size, uniform, sized: !arr });
  });
  const bg = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
  const times = [];
  for (let it = 0; it < Math.max(1, iterations); it++) {
    const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(...dispatch); pass.end();
    const t0 = performance.now(); d.queue.submit([enc.finish()]); await d.queue.onSubmittedWorkDone(); times.push(performance.now() - t0);
  }
  const which = read || kinds.map((k, i) => k.sized ? i : -1).filter((i) => i >= 0);
  const enc = d.createCommandEncoder(), stage = {};
  for (const i of which) { stage[i] = d.createBuffer({ size: bufs[i].size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); enc.copyBufferToBuffer(bufs[i], 0, stage[i], 0, bufs[i].size); }
  d.queue.submit([enc.finish()]);
  const out = {};
  for (const i of which) { await stage[i].mapAsync(GPUMapMode.READ); const k = kinds[i]; out[i] = new k.ctor(stage[i].getMappedRange().slice(0, k.n * k.ctor.BYTES_PER_ELEMENT)); stage[i].unmap(); stage[i].destroy(); }
  bufs.forEach((b) => b.destroy());
  const sorted = times.slice(iterations > 1 ? 1 : 0).sort((a, b) => a - b);
  return { out, ms: times, medianMs: sorted[Math.floor(sorted.length / 2)] };
}
self.__jobs = {};
onmessage = async (e) => {
  const m = e.data;
  if (m.t === "probe") {
    try { importScripts(URL.createObjectURL(new Blob(["self.__probe = 1;"], { type: "text/javascript" }))); postMessage({ t: "probe", ok: self.__probe === 1 }); }
    catch (err) { postMessage({ t: "probe", ok: false, s: String(err) }); }
    return;
  }
  if (m.t === "describe") { try { postMessage({ t: "describe", info: await describe() }); } catch (err) { postMessage({ t: "describe", err: String(err.message || err) }); } return; }
  if (m.t !== "run") return;
  const id = m.id;
  const log = (...a) => postMessage({ t: "log", id, s: a.map((x) => typeof x === "string" ? x : (() => { try { return JSON.stringify(x); } catch { return String(x); } })()).join(" ") });
  try {
    if (m.src) importScripts(URL.createObjectURL(new Blob([m.src], { type: "text/javascript" })));
    const fn = self.__jobs[id]; delete self.__jobs[id];
    if (typeof fn !== "function") throw new Error("job did not load");
    const ctx = { inputs: m.inputs, gpu, compute, cache, log, describe, types: TYPES, get device() { return device; } };
    const t0 = performance.now();
    const result = await fn(ctx);
    postMessage({ t: "done", id, result, runMs: performance.now() - t0 });
  } catch (err) {
    postMessage({ t: "error", id, s: (err && (err.stack || err.message)) || String(err) });
  }
};
