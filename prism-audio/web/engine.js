// PrismEngine — STFT multiband processor in one WGSL compute dispatch.
// Works in browsers and in Node (webgpu package). Assigns globalThis.PrismEngine.
(function () {
const N = 1024, HOP = 256, NB = 5;
const OLA = 0, ENV = 1024, FRZ = 1040, STATE_LEN = 1040 + 520;
const STATS = 16; // floats at the head of the out buffer

const COMPUTE_WGSL = /* wgsl */`
const N: u32 = ${N}u;
const HOP: u32 = ${HOP}u;
const NB: u32 = ${NB}u;
const HALF: u32 = ${N / 2}u;
const OLA: u32 = ${OLA}u;
const ENV: u32 = ${ENV}u;
const FRZ: u32 = ${FRZ}u;
const STATS: u32 = ${STATS}u;
const TWO_PI: f32 = 6.283185307179586;
const SLOPE: f32 = 6.0;          // crossover steepness per octave (logistic)
const OLA_NORM: f32 = 1.5;       // sum of hann^2 at 75% overlap

struct Params {
  sr: f32, nFrames: u32, frameCounter: u32, rowBase: u32,
  texH: u32, texW: u32, wet: f32, pad0: u32,
  xover: vec4<f32>,
  b0: array<vec4<f32>, 5>,   // gainDb, thrDb, ratio, gateDb
  b1: array<vec4<f32>, 5>,   // attCoef, relCoef, freeze(0/1), mute(0/1)
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> win: array<f32>;
@group(0) @binding(2) var<storage, read_write> outb: array<f32>;
@group(0) @binding(3) var<storage, read_write> st: array<f32>;
@group(0) @binding(4) var wf: texture_storage_2d<rgba8unorm, write>;

var<workgroup> re: array<f32, ${N}>;
var<workgroup> im: array<f32, ${N}>;
var<workgroup> red: array<f32, ${NB * 256}>;
var<workgroup> bandGain: array<f32, ${NB}>;

fn LOG10(x: f32) -> f32 { return log2(x) * 0.30102999566; }

fn hann(i: u32) -> f32 {
  return 0.5 - 0.5 * cos(TWO_PI * f32(i) / f32(N));
}

fn weights(freq: f32) -> array<f32, 5> {
  let lf = log2(freq + 1.0);
  var s: array<f32, 4>;
  for (var c = 0u; c < 4u; c++) {
    s[c] = 1.0 / (1.0 + exp(-(lf - log2(P.xover[c])) * SLOPE));
  }
  var w: array<f32, 5>;
  w[0] = 1.0 - s[0];
  w[1] = s[0] - s[1];
  w[2] = s[1] - s[2];
  w[3] = s[2] - s[3];
  w[4] = s[3];
  return w;
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

// In-place radix-2 DIT FFT on re/im. sgn = +1 forward, -1 inverse (unscaled).
fn fft(lid: u32, sgn: f32) {
  workgroupBarrier();
  for (var q = 0u; q < 4u; q++) {
    let i = lid + q * 256u;
    let j = reverseBits(i) >> 22u;
    if (i < j) {
      let tr = re[i]; let ti = im[i];
      re[i] = re[j]; im[i] = im[j];
      re[j] = tr; im[j] = ti;
    }
  }
  workgroupBarrier();
  for (var s = 1u; s <= 10u; s++) {
    let m = 1u << s;
    let hlf = m >> 1u;
    for (var q = 0u; q < 2u; q++) {
      let k = lid + q * 256u;
      let pos = k % hlf;
      let i = (k / hlf) * m + pos;
      let j = i + hlf;
      let ang = -sgn * TWO_PI * f32(pos) / f32(m);
      let wr = cos(ang); let wi = sin(ang);
      let tr = wr * re[j] - wi * im[j];
      let ti = wr * im[j] + wi * re[j];
      re[j] = re[i] - tr; im[j] = im[i] - ti;
      re[i] = re[i] + tr; im[i] = im[i] + ti;
    }
    workgroupBarrier();
  }
}

fn bandColor(b: u32) -> vec3<f32> {
  switch b {
    case 0u: { return vec3<f32>(1.0, 0.18, 0.15); }
    case 1u: { return vec3<f32>(1.0, 0.62, 0.1); }
    case 2u: { return vec3<f32>(0.35, 0.95, 0.3); }
    case 3u: { return vec3<f32>(0.2, 0.55, 1.0); }
    default: { return vec3<f32>(0.68, 0.32, 1.0); }
  }
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lid: u32) {
  let binHz = P.sr / f32(N);
  let refE = 3.0 * f32(N) * f32(N) / 32.0;   // energy of a full-scale sine, positive bins
  for (var f = 0u; f < P.nFrames; f++) {
    // 1. windowed frame
    for (var q = 0u; q < 4u; q++) {
      let i = lid + q * 256u;
      re[i] = win[f * HOP + i] * hann(i);
      im[i] = 0.0;
    }
    fft(lid, 1.0);

    // 2. band energies
    for (var b = 0u; b < NB; b++) { red[b * 256u + lid] = 0.0; }
    for (var q = 0u; q < 3u; q++) {
      var k = lid + q * 256u;
      if (q == 2u) { if (lid != 0u) { continue; } k = HALF; }
      let m2 = re[k] * re[k] + im[k] * im[k];
      let w = weights(f32(k) * binHz);
      for (var b = 0u; b < NB; b++) { red[b * 256u + lid] += w[b] * m2; }
    }
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
      if (lid < stride) {
        for (var b = 0u; b < NB; b++) { red[b * 256u + lid] += red[b * 256u + lid + stride]; }
      }
      workgroupBarrier();
    }

    // 3. per-band dynamics (one thread per band)
    if (lid < NB) {
      let b = lid;
      let level = 10.0 * LOG10(red[b * 256u] / refE + 1e-12);
      var env = st[ENV + b];
      let coef = select(P.b1[b].y, P.b1[b].x, level > env);
      env = coef * env + (1.0 - coef) * level;
      st[ENV + b] = env;
      let thr = P.b0[b].y; let ratio = max(P.b0[b].z, 1.0);
      let gr = max(0.0, env - thr) * (1.0 - 1.0 / ratio);
      var gDb = P.b0[b].x - gr;
      if (env < P.b0[b].w) { gDb -= 60.0; }
      if (P.b1[b].w > 0.5) { gDb = -120.0; }
      bandGain[b] = pow(10.0, gDb / 20.0);
      outb[b] = level;
      outb[5u + b] = gr;
      outb[10u + b] = env;
    }
    workgroupBarrier();

    // 4. per-bin gain + spectral freeze
    for (var q = 0u; q < 3u; q++) {
      var k = lid + q * 256u;
      if (q == 2u) { if (lid != 0u) { continue; } k = HALF; }
      let w = weights(f32(k) * binHz);
      var g = 0.0; var F = 0.0;
      for (var b = 0u; b < NB; b++) { g += w[b] * bandGain[b]; F += w[b] * P.b1[b].z; }
      var r = re[k]; var i2 = im[k];
      let mag = sqrt(r * r + i2 * i2);
      if (F < 0.01) { st[FRZ + k] = mag; }
      if (F > 0.0) {
        let fm = st[FRZ + k] * 1.85;   // offsets the energy lost to random-phase overlap-add
        let ph = f32(pcg(k * 9781u + (P.frameCounter + f) * 6271u) & 0xffffu) / 65536.0 * TWO_PI;
        r = mix(r, fm * cos(ph), F);
        i2 = mix(i2, fm * sin(ph), F);
      }
      if (k == 0u || k == HALF) { i2 = 0.0; }
      re[k] = r * g; im[k] = i2 * g;
    }
    workgroupBarrier();

    // 5. Hermitian mirror for a real inverse
    for (var q = 0u; q < 2u; q++) {
      let k = lid + q * 256u;
      if (k > 0u && k < HALF) { re[N - k] = re[k]; im[N - k] = -im[k]; }
    }
    workgroupBarrier();

    // 6. waterfall row (log-frequency columns, colored by band)
    let row = (P.rowBase + f) % P.texH;
    for (var q = 0u; q < 2u; q++) {
      let x = lid + q * 256u;
      if (x < P.texW) {
        let fr = 30.0 * pow((P.sr * 0.5) / 30.0, f32(x) / f32(P.texW));
        let kb = clamp(fr / binHz, 0.0, f32(HALF) - 1.0);
        let k0 = u32(kb); let t = kb - f32(k0);
        let m0 = sqrt(re[k0] * re[k0] + im[k0] * im[k0]);
        let m1 = sqrt(re[k0 + 1u] * re[k0 + 1u] + im[k0 + 1u] * im[k0 + 1u]);
        let m = mix(m0, m1, t);
        let db = 20.0 * LOG10(m / (f32(N) * 0.25) + 1e-9);
        let inten = clamp((db + 90.0) / 84.0, 0.0, 1.0);
        let w = weights(fr);
        var c = vec3<f32>(0.0);
        for (var b = 0u; b < NB; b++) { c += w[b] * bandColor(b); }
        textureStore(wf, vec2<i32>(i32(x), i32(row)), vec4<f32>(c * inten * inten, 1.0));
      }
    }

    // 7. inverse FFT, synthesis window, overlap-add
    fft(lid, -1.0);
    for (var q = 0u; q < 4u; q++) {
      let i = lid + q * 256u;
      st[OLA + i] += re[i] / f32(N) * hann(i) / OLA_NORM;
    }
    storageBarrier();
    workgroupBarrier();
    // lid < 256 == HOP: one output sample per thread
    let dry = win[f * HOP + lid];
    outb[STATS + f * HOP + lid] = mix(dry, st[OLA + lid], P.wet);
    var tmp: array<f32, 4>;
    for (var q = 0u; q < 4u; q++) {
      let i = lid + q * 256u;
      tmp[q] = select(0.0, st[OLA + i + HOP], i + HOP < N);
    }
    storageBarrier();
    workgroupBarrier();
    for (var q = 0u; q < 4u; q++) { st[OLA + lid + q * 256u] = tmp[q]; }
    storageBarrier();
    workgroupBarrier();
  }
}
`;

const RENDER_WGSL = /* wgsl */`
struct R { head: u32, texH: u32, texW: u32, pad: u32, sr: f32, p1: f32, p2: f32, p3: f32, xover: vec4<f32> };
@group(0) @binding(0) var<uniform> U: R;
@group(0) @binding(1) var tex: texture_2d<f32>;
struct VO { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: VO;
  o.pos = vec4<f32>(p[i], 0.0, 1.0);
  o.uv = vec2<f32>((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5);
  return o;
}
@fragment fn fs(v: VO) -> @location(0) vec4<f32> {
  let px = min(u32(v.uv.x * f32(U.texW)), U.texW - 1u);
  let py = min(u32(v.uv.y * f32(U.texH)), U.texH - 1u);
  let row = (U.head + 1u + py) % U.texH;
  var c = textureLoad(tex, vec2<i32>(i32(px), i32(row)), 0).rgb;
  // crossover guides
  for (var k = 0u; k < 4u; k++) {
    let x = log(U.xover[k] / 30.0) / log((U.sr * 0.5) / 30.0);
    let d = abs(v.uv.x - x) * f32(U.texW);
    c += vec3<f32>(0.22) * (1.0 - smoothstep(0.0, 1.2, d));
  }
  // fade oldest rows
  c *= mix(0.35, 1.0, v.uv.y);
  return vec4<f32>(c, 1.0);
}
`;

class PrismEngine {
  static get N() { return N; }
  static get HOP() { return HOP; }
  static get NB() { return NB; }
  static get computeWGSL() { return COMPUTE_WGSL; }
  static get renderWGSL() { return RENDER_WGSL; }

  constructor(device, { sampleRate = 48000, maxFrames = 16, texW = 512, texH = 256 } = {}) {
    this.device = device;
    this.sr = sampleRate;
    this.maxFrames = maxFrames;
    this.texW = texW; this.texH = texH;
    this.frameCounter = 0; this.rowBase = 0;
    this.history = new Float32Array(N - HOP);
    const U = GPUBufferUsage;
    this.paramBuf = device.createBuffer({ size: 208, usage: U.UNIFORM | U.COPY_DST });
    this.winBuf = device.createBuffer({ size: (N - HOP + maxFrames * HOP) * 4, usage: U.STORAGE | U.COPY_DST });
    this.outBytes = (STATS + maxFrames * HOP) * 4;
    this.outBuf = device.createBuffer({ size: this.outBytes, usage: U.STORAGE | U.COPY_SRC });
    this.stateBuf = device.createBuffer({ size: STATE_LEN * 4, usage: U.STORAGE | U.COPY_DST });
    const st = new Float32Array(STATE_LEN); st.fill(-100, ENV, ENV + NB);
    device.queue.writeBuffer(this.stateBuf, 0, st);
    this.tex = device.createTexture({
      size: [texW, texH], format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.module = device.createShaderModule({ code: COMPUTE_WGSL });
    this.pipeline = device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint: 'main' } });
    this.bind = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramBuf } },
        { binding: 1, resource: { buffer: this.winBuf } },
        { binding: 2, resource: { buffer: this.outBuf } },
        { binding: 3, resource: { buffer: this.stateBuf } },
        { binding: 4, resource: this.tex.createView() },
      ],
    });
    this.pool = [];
    this.setParams({});
  }

  // params: { wet, xover:[4], attackMs, releaseMs, bands:[{gainDb,thrDb,ratio,gateDb,freeze,mute,solo}] }
  setParams(p) {
    const d = {
      wet: 1, xover: [120, 500, 2000, 6000], attackMs: 10, releaseMs: 150,
      bands: Array.from({ length: NB }, () => ({ gainDb: 0, thrDb: 0, ratio: 1, gateDb: -120, freeze: 0, mute: 0, solo: 0 })),
    };
    this.params = Object.assign(d, this.params || {}, p);
  }

  _writeParams(nFrames) {
    const p = this.params;
    const buf = new ArrayBuffer(208), f = new Float32Array(buf), u = new Uint32Array(buf);
    f[0] = this.sr; u[1] = nFrames; u[2] = this.frameCounter; u[3] = this.rowBase;
    u[4] = this.texH; u[5] = this.texW; f[6] = p.wet; u[7] = 0;
    for (let i = 0; i < 4; i++) f[8 + i] = p.xover[i];
    const hopSec = HOP / this.sr;
    const att = Math.exp(-hopSec / Math.max(p.attackMs / 1000, 1e-4));
    const rel = Math.exp(-hopSec / Math.max(p.releaseMs / 1000, 1e-4));
    const anySolo = p.bands.some(b => b.solo);
    p.bands.forEach((b, i) => {
      f[12 + i * 4] = b.gainDb; f[13 + i * 4] = b.thrDb; f[14 + i * 4] = b.ratio; f[15 + i * 4] = b.gateDb;
      const mute = b.mute || (anySolo && !b.solo) ? 1 : 0;
      f[32 + i * 4] = att; f[33 + i * 4] = rel; f[34 + i * 4] = b.freeze ? 1 : 0; f[35 + i * 4] = mute;
    });
    this.device.queue.writeBuffer(this.paramBuf, 0, buf);
  }

  // samples: Float32Array whose length is a multiple of HOP (<= maxFrames*HOP).
  // Returns processed samples of the same length; latency is N-HOP samples.
  async process(samples) {
    const nFrames = samples.length / HOP;
    if (!Number.isInteger(nFrames) || nFrames < 1 || nFrames > this.maxFrames) throw new Error('bad block length ' + samples.length);
    const winData = new Float32Array(N - HOP + samples.length);
    winData.set(this.history, 0); winData.set(samples, N - HOP);
    this.history = winData.slice(winData.length - (N - HOP));
    const dev = this.device;
    dev.queue.writeBuffer(this.winBuf, 0, winData);
    this._writeParams(nFrames);
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.bind); pass.dispatchWorkgroups(1);
    pass.end();
    const bytes = (STATS + samples.length) * 4;
    const staging = this.pool.pop() || dev.createBuffer({ size: this.outBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    enc.copyBufferToBuffer(this.outBuf, 0, staging, 0, bytes);
    dev.queue.submit([enc.finish()]);
    this.frameCounter += nFrames;
    this.rowBase = (this.rowBase + nFrames) % this.texH;
    this.head = (this.rowBase + this.texH - 1) % this.texH;
    await staging.mapAsync(GPUMapMode.READ, 0, bytes);
    const all = new Float32Array(staging.getMappedRange(0, bytes).slice(0));
    staging.unmap();
    this.pool.push(staging);
    return { out: all.subarray(STATS), stats: all.subarray(0, STATS) };
  }

  // Draw the waterfall into a configured canvas context.
  initRender(context, format) {
    const dev = this.device;
    this.ctx = context;
    this.rUni = dev.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const mod = dev.createShaderModule({ code: RENDER_WGSL });
    this.rPipe = dev.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.rBind = dev.createBindGroup({
      layout: this.rPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.rUni } }, { binding: 1, resource: this.tex.createView() }],
    });
  }

  render() {
    if (!this.rPipe) return;
    const b = new ArrayBuffer(48), u = new Uint32Array(b), f = new Float32Array(b);
    u[0] = this.head ?? 0; u[1] = this.texH; u[2] = this.texW; f[4] = this.sr;
    for (let i = 0; i < 4; i++) f[8 + i] = this.params.xover[i];
    this.device.queue.writeBuffer(this.rUni, 0, b);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.rPipe); pass.setBindGroup(0, this.rBind); pass.draw(3);
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }
}
globalThis.PrismEngine = PrismEngine;
})();
