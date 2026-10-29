// Modal bell synthesis: sum of damped complex exponentials, rendered sample-exact at 96 kHz.
// Inputs: modes f64 [N x 6] (omega rad/sample, sigma 1/sample, gainL re/im, gainR re/im),
// strike_mode u32, strike_on u32 (sample), strike_amp f64 (sorted by mode, then onset), meta {n_samples}.
// Pass 1 (JS, float64): each mode's complex state at every 4096-sample block start, keeping only
// modes still above -130 dB. Pass 2 (GPU): each sample sums Im(z * e^{s j}) over the block's live
// modes, with e^{s j} = hi[j >> 6] * lo[j & 63] from float64-built tables, so no sin/exp runs on the GPU.
const f64 = (a) => a instanceof Float64Array ? a : new Float64Array(a.buffer, a.byteOffset, a.byteLength / 8);
const I = ctx.inputs, modes = f64(I.modes), sMode = I.strike_mode, sOn = I.strike_on, sAmp = f64(I.strike_amp);
const meta = I.meta, nS = meta.n_samples, N = modes.length / 6, B = 4096, nB = Math.ceil(nS / B);
const t0 = performance.now();

// Per-mode strike ranges.
const first = new Int32Array(N + 1).fill(-1);
for (let k = sMode.length - 1; k >= 0; k--) first[sMode[k]] = k;
const ptr = new Int32Array(N), end = new Int32Array(N);
{ let k = 0; for (let m = 0; m < N; m++) { ptr[m] = k; while (k < sMode.length && sMode[k] === m) k++; end[m] = k; } }

let gmax = 0, amax = 0;
for (let m = 0; m < N; m++) gmax = Math.max(gmax, Math.hypot(modes[6*m+2], modes[6*m+3]), Math.hypot(modes[6*m+4], modes[6*m+5]));
for (const a of sAmp) amax = Math.max(amax, Math.abs(a));
const thr = 3e-7 * amax * gmax;   // -130 dB re the loudest single mode

let cap = 1 << 20, ent = new Float32Array(cap * 4), entM = new Uint32Array(cap * 2), nE = 0;
const offs = new Uint32Array(nB + 1);
const zr = new Float64Array(N), zi = new Float64Array(N), EBr = new Float64Array(N), EBi = new Float64Array(N);
for (let m = 0; m < N; m++) { const w = modes[6*m], s = modes[6*m+1], e = Math.exp(-s * B); EBr[m] = e * Math.cos(w * B); EBi[m] = e * Math.sin(w * B); }
const push = (m, a, b, js) => {
  if (nE === cap) { cap *= 2; const e2 = new Float32Array(cap * 4); e2.set(ent); ent = e2; const m2 = new Uint32Array(cap * 2); m2.set(entM); entM = m2; }
  const lr = modes[6*m+2], li = modes[6*m+3], rr = modes[6*m+4], ri = modes[6*m+5];
  ent[4*nE] = a*lr - b*li; ent[4*nE+1] = a*li + b*lr; ent[4*nE+2] = a*rr - b*ri; ent[4*nE+3] = a*ri + b*rr;
  entM[2*nE] = m; entM[2*nE+1] = js; nE++;
};
for (let b = 0; b < nB; b++) {
  offs[b] = nE;
  const b0 = b * B, b1 = b0 + B;
  for (let m = 0; m < N; m++) {
    let a = zr[m], c = zi[m];
    if (Math.hypot(a, c) * gmax > thr) push(m, a, c, 0);
    // state at the next block start: decay this one, then add strikes landing inside this block
    let nr = a * EBr[m] - c * EBi[m], ni = a * EBi[m] + c * EBr[m];
    const w = modes[6*m], s = modes[6*m+1];
    while (ptr[m] < end[m] && sOn[ptr[m]] < b1) {
      const k = ptr[m]++, js = sOn[k] - b0, A = sAmp[k];
      const g = A * Math.exp(s * js);                   // c * e^{-s js}, defined at j = 0
      push(m, g * Math.cos(w * js), -g * Math.sin(w * js), js);
      const d = B - js, h = A * Math.exp(-s * d);
      nr += h * Math.cos(w * d); ni += h * Math.sin(w * d);
    }
    zr[m] = nr; zi[m] = ni;
  }
  if ((b & 31) === 0) ctx.progress(0.3 * b / nB, `mode states: block ${b}/${nB}`);
}
offs[nB] = nE;
const lo = new Float32Array(N * 128), hi = new Float32Array(N * 128);
for (let m = 0; m < N; m++) {
  const w = modes[6*m], s = modes[6*m+1];
  for (let j = 0; j < 64; j++) {
    let e = Math.exp(-s * j); lo[128*m + 2*j] = e * Math.cos(w * j); lo[128*m + 2*j + 1] = e * Math.sin(w * j);
    const J = 64 * j; e = Math.exp(-s * J); hi[128*m + 2*j] = e * Math.cos(w * J); hi[128*m + 2*j + 1] = e * Math.sin(w * J);
  }
}
const prepMs = performance.now() - t0;
ctx.log(`modes ${N}, strikes ${sMode.length}, entries ${nE} (${(nE * 24 / 2**20).toFixed(1)} MiB), prep ${prepMs.toFixed(0)} ms`);

const wgsl = `
struct U { nS: u32, gx: u32, wg0: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> zz: array<vec4f>;
@group(0) @binding(1) var<storage, read> em: array<vec2u>;
@group(0) @binding(2) var<storage, read> offs: array<u32>;
@group(0) @binding(3) var<storage, read> tlo: array<vec2f>;
@group(0) @binding(4) var<storage, read> thi: array<vec2f>;
@group(0) @binding(5) var<storage, read_write> outp: array<vec2f>;
@group(0) @binding(6) var<uniform> u: U;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let w = u.wg0 + wg.x + wg.y * u.gx;
  if (w * 64u >= u.nS) { return; }
  let n = w * 64u + lid.x;
  let b = n / ${B}u; let j = n % ${B}u; let h = j >> 6u;
  var aL = 0.0; var aR = 0.0;
  let e1 = offs[b + 1u];
  for (var e = offs[b]; e < e1; e++) {
    let m = em[e]; let z = zz[e];
    let H = thi[m.x * 64u + h]; let L = tlo[m.x * 64u + lid.x];
    let wr = H.x * L.x - H.y * L.y; let wi = H.x * L.y + H.y * L.x;
    let on = select(0.0, 1.0, j >= m.y);
    aL += on * (z.x * wi + z.y * wr);
    aR += on * (z.z * wi + z.w * wr);
  }
  if (n < u.nS) { outp[n] = vec2f(aL, aR); }
}`;
const d = await ctx.gpu();
const mod = d.createShaderModule({ code: wgsl });
const errs = (await mod.getCompilationInfo()).messages.filter((m) => m.type === "error");
if (errs.length) throw new Error("WGSL: " + errs.map((m) => m.lineNum + ":" + m.message).join("; "));
const pipe = await d.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: "main" } });
const mk = (arr, usage = GPUBufferUsage.STORAGE) => { const buf = d.createBuffer({ size: Math.max(16, Math.ceil(arr.byteLength / 16) * 16), usage: usage | GPUBufferUsage.COPY_DST }); d.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength); return buf; };
const bufs = [mk(ent.subarray(0, 4 * nE)), mk(entM.subarray(0, 2 * nE)), mk(offs), mk(lo), mk(hi)];
const outB = d.createBuffer({ size: nS * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const uni = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const bg = d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [...bufs, outB, uni].map((buffer, binding) => ({ binding, resource: { buffer } })) });
const nWG = Math.ceil(nS / 64), CH = 64 * 16, gx = Math.min(CH, 65535);   // 16 blocks (0.68 s of audio) per submit
await d.queue.onSubmittedWorkDone();
const tg = performance.now(); const chunkMs = [];
for (let wg0 = 0; wg0 < nWG; wg0 += CH) {
  const cnt = Math.min(CH, nWG - wg0);
  d.queue.writeBuffer(uni, 0, new Uint32Array([nS, gx, wg0, 0]));
  const enc = d.createCommandEncoder(), p = enc.beginComputePass();
  p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(Math.min(cnt, gx), Math.ceil(cnt / gx)); p.end();
  const tc = performance.now(); d.queue.submit([enc.finish()]); await d.queue.onSubmittedWorkDone(); chunkMs.push(+(performance.now() - tc).toFixed(1));
  ctx.progress(0.3 + 0.7 * (wg0 + cnt) / nWG, `GPU render: ${((wg0 + cnt) * 64 / meta.sr).toFixed(1)} of ${(nS / meta.sr).toFixed(1)} s`);
}
const gpuMs = performance.now() - tg;
const st = d.createBuffer({ size: nS * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const enc = d.createCommandEncoder(); enc.copyBufferToBuffer(outB, 0, st, 0, nS * 8); d.queue.submit([enc.finish()]);
await st.mapAsync(GPUMapMode.READ); const audio = new Float32Array(st.getMappedRange().slice(0)); st.unmap();
[...bufs, outB, uni, st].forEach((x) => x.destroy());
let modeSamples = 0; for (let b = 0; b < nB; b++) modeSamples += (offs[b + 1] - offs[b]) * Math.min(B, nS - b * B);
ctx.log(`GPU ${gpuMs.toFixed(0)} ms for ${(nS / meta.sr).toFixed(2)} s stereo at ${meta.sr} Hz; ${(modeSamples / 1e9).toFixed(2)} G mode-samples`);
return { audio, prepMs, gpuMs, chunkMs, entries: nE, modeSamples, realtimeX: (nS / meta.sr) / (gpuMs / 1000), adapter: await ctx.describe() };
