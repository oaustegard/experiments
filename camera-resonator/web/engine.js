// Streaming modal synthesis on WebGPU: thousands of damped modes, one audio block per GPU submit.
//
// Every mode m is a damped complex exponential with s = -sigma + i*omega (per sample). Its complex state z[m]
// lives on the GPU; a block's sample j is Im(g * (z e^{s j} + sum_k c_k e^{s (j - js_k)} [j >= js_k])), summed
// over modes, where strike k lands at sample js_k with excitation c_k. e^{s t} comes from two float64-built
// tables, hi[t >> 6] * lo[t & 63], so the GPU evaluates no sin or exp (WGSL only promises sin to 2^-11, about
// -66 dB per mode). Strike excitations are computed in float64 JS, which is cheap: a few strikes per block.
//
// Used by the page (index.html), the bridge benchmark (../bench/job.js) and the checks (../test/verify.mjs).

const MAXS = 16;   // strikes per block
const TAU = 2 * Math.PI;

export function rng(seed) {   // mulberry32
  let a = seed >>> 0;
  const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  u.normal = () => Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(TAU * u());
  u.range = (a0, b0) => a0 + (b0 - a0) * u();
  return u;
}

// Magnitude spectrum of a half-sine force pulse of duration tau: how a mallet of that contact time excites f.
export function contact(f, tau) {
  const x = 2 * f * tau, d = 1 - x * x;
  return Math.abs(d) < 1e-6 ? Math.PI / 4 : Math.abs(Math.cos(Math.PI * f * tau) / d);
}

// A mode set: per mode omega, sigma (per sample), complex pickup gains; excite(strike) -> Float64Array of c.
function modeSet(n) {
  return { n, w: new Float64Array(n), sig: new Float64Array(n), gl: new Float64Array(2 * n), gr: new Float64Array(2 * n), f: new Float64Array(n), amp: new Float64Array(n) };
}

// Simply supported rectangular plate shaped like the camera frame (x across, y down, both 0..1). The strike lands
// where the motion was, so position picks which modes ring: sin(m pi x) sin(n pi y). Two contact pickups give stereo.
// The plate's stiffness is chosen so exactly `count` modes fall between 30 Hz and fmax.
export function plate(count, sr, { aspect = 4 / 3, fmax = 16000, t60 = 7, seed = 1 } = {}) {
  const a = aspect, b = 1, F = Math.min(fmax, 0.45 * sr);
  let C = Math.PI * a * b * F / (4 * count), list = [];
  for (let tries = 0; tries < 8; tries++) {          // the area estimate misses edge effects; rescale until enough
    list = [];
    const M = Math.ceil(a * Math.sqrt(F / C)), N = Math.ceil(b * Math.sqrt(F / C));
    for (let m = 1; m <= M; m++) for (let k = 1; k <= N; k++) {
      const f = C * ((m / a) ** 2 + (k / b) ** 2);
      if (f >= 30 && f <= F) list.push([f, m, k]);
    }
    if (list.length >= count) break;
    C *= 0.8 * list.length / count;
  }
  list.sort((p, q) => p[0] - q[0]); list.length = count;
  const S = modeSet(count), mm = new Float64Array(count), nn = new Float64Array(count), r = rng(seed);
  const pick = [[0.27, 0.62], [0.71, 0.33]];
  let e2 = 0;
  list.forEach(([f, m, k], i) => {
    const fj = f * (1 + 0.0015 * r.normal());        // a real sheet is not perfectly uniform
    S.f[i] = fj; S.w[i] = TAU * fj / sr;
    S.sig[i] = 6.907755 / (sr * t60 / (1 + fj / 2000 + (fj / 6000) ** 2));
    S.gl[2 * i] = Math.sin(m * Math.PI * pick[0][0]) * Math.sin(k * Math.PI * pick[0][1]);
    S.gr[2 * i] = Math.sin(m * Math.PI * pick[1][0]) * Math.sin(k * Math.PI * pick[1][1]);
    S.amp[i] = (1 + fj / 3000) ** -0.5; mm[i] = m * Math.PI; nn[i] = k * Math.PI;
    e2 += S.amp[i] ** 2 / 16;
  });
  const G = 0.12 / Math.sqrt(e2);
  S.excite = ({ x, y, vel, tau }) => {
    const c = new Float64Array(count);
    for (let i = 0; i < count; i++) c[i] = G * vel * S.amp[i] * Math.sin(mm[i] * x) * Math.sin(nn[i] * y) * contact(S.f[i], tau);
    return c;
  };
  S.label = `plate, ${count} modes, 30 Hz–${(F / 1000).toFixed(0)} kHz`;
  return S;
}

// Bell partials relative to the strike pitch: ratio, nodal meridians, amplitude, T60 factor (as webgpu-bells).
const PARTIALS = [[0.5, 2, 0.55, 1], [1, 2, 0.45, 0.62], [1.19, 3, 0.6, 0.48], [1.5, 3, 0.22, 0.34], [2, 4, 1, 0.32],
  [2.51, 4, 0.3, 0.22], [2.67, 5, 0.28, 0.2], [3.01, 5, 0.45, 0.17], [3.36, 6, 0.18, 0.14], [4.02, 6, 0.32, 0.12]];
[4.55, 5.02, 5.37, 6.04, 6.62, 7.13, 8.06, 8.85, 9.47, 10.4, 11.3, 12.2, 13.4, 14.5, 15.9, 17.1, 18.6, 20.2, 21.9, 23.7]
  .forEach((r, k) => PARTIALS.push([r, 7 + (k >> 1), 0.26 * (4 / r) ** 0.6, 0.11 * (4.5 / r) ** 0.9]));

// A row of bells across the frame: the column of the motion picks the bell, its height the strike angle around
// the bell's rim, so the same bell sounds different struck high or low. Each partial is a split doublet.
export function bells(sr, { notes = [62, 64, 66, 69, 71, 74, 76, 78], transients = 300, seed = 7 } = {}) {
  const r = rng(seed), rows = [], nb = notes.length, F = Math.min(20000, 0.45 * sr);
  notes.forEach((midi, bi) => {
    const f0 = 440 * 2 ** ((midi - 69) / 12), size = 293.66 / f0, tHum = 22 * size ** 0.8, loud = size ** 0.5;
    const pan = -0.8 + 1.6 * bi / (nb - 1), itd = pan * 0.0006, gl = Math.sqrt(0.5 * (1 - pan)), gr = Math.sqrt(0.5 * (1 + pan));
    const thL = 0.87, thR = 0.93;
    const add = (f, t60, n, th0, ph, amp) => rows.push({ bi, f, t60, n, th0, ph, amp, gl, gr, thL, thR, itd });
    for (const [ratio, n, amp, tf] of PARTIALS) {
      const f = f0 * ratio * (1 + 0.003 * r.normal());
      if (f > F) continue;
      const split = f * r.range(2e-4, 1.4e-3), th0 = r() * Math.PI, t60 = tHum * tf * r.range(0.9, 1.1);
      add(f - split / 2, t60, n, th0, 0, loud * amp);
      add(f + split / 2, t60, n, th0, Math.PI / 2, loud * amp);
    }
    for (let k = 0; k < transients; k++) {
      const f = Math.min(F, 2200 * (19000 / 2200) ** Math.sqrt(r()));
      add(f, r.range(0.04, 0.35) * (3000 / f) ** 0.5, 6 + Math.floor(r() * 34), r() * Math.PI, r() < 0.5 ? 0 : Math.PI / 2,
        loud * 0.05 * Math.sqrt(-2 * Math.log(1 - r())) * (4000 / f) ** 0.3);
    }
  });
  const n = rows.length, S = modeSet(n), bell = new Int32Array(n);
  const perBell = new Float64Array(nb);
  rows.forEach((q, i) => {
    const w = TAU * q.f / sr;
    S.f[i] = q.f; S.w[i] = w; S.sig[i] = 6.907755 / (q.t60 * sr); S.amp[i] = q.amp; bell[i] = q.bi;
    const pl = Math.cos(q.n * (q.thL - q.th0) - q.ph), pr = Math.cos(q.n * (q.thR - q.th0) - q.ph), d = TAU * q.f * q.itd / 2;
    S.gl[2 * i] = q.gl * pl * Math.cos(d); S.gl[2 * i + 1] = q.gl * pl * Math.sin(d);
    S.gr[2 * i] = q.gr * pr * Math.cos(-d); S.gr[2 * i + 1] = q.gr * pr * Math.sin(-d);
    perBell[q.bi] += q.amp ** 2 * 0.25;
  });
  const G = 0.2 / Math.sqrt(Math.max(...perBell));
  S.excite = ({ x, y, vel, tau }) => {
    const c = new Float64Array(n), b = Math.min(nb - 1, Math.max(0, Math.floor(x * nb))), th = Math.PI * y;
    for (let i = 0; i < n; i++) if (bell[i] === b) {
      const q = rows[i];
      c[i] = G * vel * q.amp * Math.cos(q.n * (th - q.th0) - q.ph) * contact(q.f, tau);
    }
    return c;
  };
  S.bellCount = nb;
  S.label = `${nb} bells, ${n} modes`;
  return S;
}

const WGSL = /* wgsl */`
struct U { n: u32, nStr: u32, parts: u32, B: u32, H: u32, chunk: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> gain: array<vec4f>;          // gL.re, gL.im, gR.re, gR.im
@group(0) @binding(2) var<storage, read> lo: array<vec2f>;            // e^{s t}, t = 0..63
@group(0) @binding(3) var<storage, read> hi: array<vec2f>;            // e^{s 64 k}, k = 0..B/64
@group(0) @binding(4) var<storage, read_write> z: array<vec2f>;       // state at the block start
@group(0) @binding(5) var<storage, read> cb: array<f32>;              // strike excitations, m * 16 + k
@group(0) @binding(6) var<storage, read> js: array<u32, 16>;          // strike sample offsets
@group(0) @binding(7) var<storage, read_write> part: array<vec2f>;    // partial sums, p * B + j
@group(0) @binding(8) var<storage, read_write> outp: array<vec2f>;

fn cexp(m: u32, t: u32) -> vec2f {
  let h = hi[m * u.H + (t >> 6u)]; let l = lo[m * 64u + (t & 63u)];
  return vec2f(h.x * l.x - h.y * l.y, h.x * l.y + h.y * l.x);
}

// One thread per sample, one workgroup row per slice of modes; reduce() adds the slices.
@compute @workgroup_size(64)
fn synth(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let j = wg.x * 64u + lid.x; let p = wg.y;
  let m0 = p * u.chunk; let m1 = min(u.n, m0 + u.chunk);
  var aL = 0.0; var aR = 0.0;
  for (var m = m0; m < m1; m++) {
    let zz = z[m];
    if (u.nStr == 0u && abs(zz.x) + abs(zz.y) < 1e-10) { continue; }   // silent mode; uniform across the group
    let e = cexp(m, j);
    var wr = zz.x * e.x - zz.y * e.y; var wi = zz.x * e.y + zz.y * e.x;
    for (var k = 0u; k < u.nStr; k++) {
      let s = js[k];
      if (j >= s) { let c = cb[m * 16u + k]; let e2 = cexp(m, j - s); wr += c * e2.x; wi += c * e2.y; }
    }
    let g = gain[m];
    aL += g.x * wi + g.y * wr; aR += g.z * wi + g.w * wr;            // Im(g * w)
  }
  part[p * u.B + j] = vec2f(aL, aR);
}

// Advance each mode's state to the next block start, adding this block's strikes.
@compute @workgroup_size(64)
fn advance(@builtin(global_invocation_id) gid: vec3u) {
  let m = gid.x; if (m >= u.n) { return; }
  let zz = z[m]; let hb = hi[m * u.H + u.H - 1u];                      // e^{s B}
  var r = vec2f(zz.x * hb.x - zz.y * hb.y, zz.x * hb.y + zz.y * hb.x);
  for (var k = 0u; k < u.nStr; k++) { r += cb[m * 16u + k] * cexp(m, u.B - js[k]); }
  z[m] = r;
}

@compute @workgroup_size(64)
fn reduce(@builtin(global_invocation_id) gid: vec3u) {
  let j = gid.x; if (j >= u.B) { return; }
  var a = vec2f(0.0);
  for (var p = 0u; p < u.parts; p++) { a += part[p * u.B + j]; }
  outp[j] = a;
}`;

export class ModalEngine {
  static async create(device, S, { block = 512, parts = 0 } = {}) {
    if (block % 64) throw new Error("block must be a multiple of 64");
    const e = new ModalEngine(); e.device = device; e.S = S; e.B = block; e.H = block / 64 + 1;
    const n = S.n;
    e.parts = parts || Math.max(1, Math.min(32, Math.ceil(n / 256)));
    e.chunk = Math.ceil(n / e.parts);
    const lo = new Float32Array(n * 128), hi = new Float32Array(n * 2 * e.H), gain = new Float32Array(n * 4);
    for (let m = 0; m < n; m++) {
      const w = S.w[m], s = S.sig[m];
      for (let t = 0; t < 64; t++) { const a = Math.exp(-s * t); lo[128 * m + 2 * t] = a * Math.cos(w * t); lo[128 * m + 2 * t + 1] = a * Math.sin(w * t); }
      for (let k = 0; k < e.H; k++) { const t = 64 * k, a = Math.exp(-s * t); hi[2 * (m * e.H + k)] = a * Math.cos(w * t); hi[2 * (m * e.H + k) + 1] = a * Math.sin(w * t); }
      gain.set([S.gl[2 * m], S.gl[2 * m + 1], S.gr[2 * m], S.gr[2 * m + 1]], 4 * m);
    }
    const mod = device.createShaderModule({ code: WGSL });
    const errs = (await mod.getCompilationInfo()).messages.filter((x) => x.type === "error");
    if (errs.length) throw new Error("WGSL: " + errs.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("; "));
    const S_ = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, R = GPUBufferUsage.COPY_SRC;
    const ro = { type: "read-only-storage" }, rw = { type: "storage" };
    const layout = device.createBindGroupLayout({ entries: [{ type: "uniform" }, ro, ro, ro, rw, ro, ro, rw, rw]
      .map((buffer, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer })) });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const mk = (entryPoint) => device.createComputePipelineAsync({ layout: pl, compute: { module: mod, entryPoint } });
    [e.pSynth, e.pAdvance, e.pReduce] = await Promise.all([mk("synth"), mk("advance"), mk("reduce")]);
    const buf = (size, usage, data) => { const b = device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage }); if (data) device.queue.writeBuffer(b, 0, data); return b; };
    e.uni = buf(32, GPUBufferUsage.UNIFORM | D);
    e.bufs = [e.uni, buf(gain.byteLength, S_ | D, gain), buf(lo.byteLength, S_ | D, lo), buf(hi.byteLength, S_ | D, hi),
      e.zb = buf(n * 8, S_ | D), e.cbb = buf(n * MAXS * 4, S_ | D), e.jsb = buf(MAXS * 4, S_ | D),
      buf(e.parts * block * 8, S_), e.outb = buf(block * 8, S_ | R)];
    e.bg = device.createBindGroup({ layout, entries: e.bufs.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    e.cb = new Float32Array(n * MAXS);
    e.stage = []; e.ms = [];
    return e;
  }

  reset() { this.device.queue.writeBuffer(this.zb, 0, new Float32Array(this.S.n * 2)); }

  // Render the next block. strikes: [{x, y, vel, tau, js}] (js = sample offset in this block, default 0).
  // Calls may overlap: the GPU queue keeps blocks in order and each call gets its own readback buffer.
  async render(strikes = []) {
    const d = this.device, n = this.S.n, B = this.B, ns = Math.min(MAXS, strikes.length);
    if (ns) {
      const jsArr = new Uint32Array(MAXS);
      for (let k = 0; k < ns; k++) {
        const c = this.S.excite(strikes[k]);
        for (let m = 0; m < n; m++) this.cb[m * MAXS + k] = c[m];
        jsArr[k] = Math.min(B - 1, Math.max(0, strikes[k].js | 0));
      }
      d.queue.writeBuffer(this.cbb, 0, this.cb);
      d.queue.writeBuffer(this.jsb, 0, jsArr);
    }
    d.queue.writeBuffer(this.uni, 0, new Uint32Array([n, ns, this.parts, B, this.H, this.chunk, 0, 0]));
    const st = this.stage.pop() || d.createBuffer({ size: B * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = d.createCommandEncoder();
    const pass = (pipe, x, y = 1) => { const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, this.bg); p.dispatchWorkgroups(x, y); p.end(); };
    pass(this.pSynth, B / 64, this.parts);
    pass(this.pAdvance, Math.ceil(n / 64));
    pass(this.pReduce, B / 64);
    enc.copyBufferToBuffer(this.outb, 0, st, 0, B * 8);
    const t0 = performance.now();
    d.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); this.stage.push(st);
    this.ms.push(performance.now() - t0); if (this.ms.length > 200) this.ms.shift();
    return out;
  }

  destroy() { this.bufs.forEach((b) => b.destroy()); this.stage.forEach((b) => b.destroy()); }
}

// float64 reference of the same block stream: blocks is a list of strike lists. Returns interleaved stereo.
export function reference(S, blocks, B) {
  const n = S.n, zr = new Float64Array(n), zi = new Float64Array(n), out = new Float64Array(blocks.length * B * 2);
  blocks.forEach((strikes, b) => {
    const cs = strikes.map((s) => ({ c: S.excite(s), js: Math.min(B - 1, Math.max(0, s.js | 0)) }));
    for (let m = 0; m < n; m++) {
      const w = S.w[m], sg = S.sig[m], glr = S.gl[2 * m], gli = S.gl[2 * m + 1], grr = S.gr[2 * m], gri = S.gr[2 * m + 1];
      if (zr[m] === 0 && zi[m] === 0 && cs.every((q) => q.c[m] === 0)) continue;
      for (let j = 0; j < B; j++) {
        let a = Math.exp(-sg * j), wr = a * (zr[m] * Math.cos(w * j) - zi[m] * Math.sin(w * j)), wi = a * (zr[m] * Math.sin(w * j) + zi[m] * Math.cos(w * j));
        for (const q of cs) if (j >= q.js && q.c[m]) { const t = j - q.js, e = q.c[m] * Math.exp(-sg * t); wr += e * Math.cos(w * t); wi += e * Math.sin(w * t); }
        const o = 2 * (b * B + j);
        out[o] += glr * wi + gli * wr; out[o + 1] += grr * wi + gri * wr;
      }
      let a = Math.exp(-sg * B), nr = a * (zr[m] * Math.cos(w * B) - zi[m] * Math.sin(w * B)), ni = a * (zr[m] * Math.sin(w * B) + zi[m] * Math.cos(w * B));
      for (const q of cs) { const t = B - q.js, e = q.c[m] * Math.exp(-sg * t); nr += e * Math.cos(w * t); ni += e * Math.sin(w * t); }
      zr[m] = nr; zi[m] = ni;
    }
  });
  return out;
}

// GPU stream against the float64 reference: a plate, random strikes. Returns the error re the peak.
export async function selfTest(device, { modes = 1500, blocks = 24, block = 512, sr = 48000, seed = 3 } = {}) {
  const S = plate(modes, sr), r = rng(seed), plan = [];
  for (let b = 0; b < blocks; b++) {
    const k = b === 0 ? 2 : r() < 0.3 ? 1 + Math.floor(r() * 3) : 0, ss = [];
    for (let i = 0; i < k; i++) ss.push({ x: r(), y: r(), vel: r.range(0.3, 1), tau: r.range(2e-4, 1.5e-3), js: Math.floor(r() * block) });
    plan.push(ss);
  }
  const e = await ModalEngine.create(device, S, { block });
  const gpu = new Float32Array(blocks * block * 2);
  for (let b = 0; b < blocks; b++) gpu.set(await e.render(plan[b]), b * block * 2);
  e.destroy();
  const ref = reference(S, plan, block);
  let peak = 0, se = 0, mx = 0;
  for (let i = 0; i < ref.length; i++) { peak = Math.max(peak, Math.abs(ref[i])); const d = gpu[i] - ref[i]; se += d * d; mx = Math.max(mx, Math.abs(d)); }
  return { modes, blocks, block, peak, maxErr: mx, rmsDb: 20 * Math.log10(Math.sqrt(se / ref.length) / peak) };
}
