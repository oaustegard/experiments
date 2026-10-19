// 2D incompressible Navier-Stokes on a uniform grid, every pass a WebGPU compute shader.
// Stable-fluids splitting (Stam 1999): advect -> diffuse -> force -> project.
// Each term can be switched off, which is what chapter 1 is about.

const WG = 8;

const SIM_WGSL = /* wgsl */ `
struct Params {
  a: vec4f,      // W, H, dt, nu*dt
  b: vec4f,      // bc (0 box, 1 periodic, 2 channel), dye decay, inflow U, time
  splat: vec4f,  // x, y, vx, vy (grid units)
  splatC: vec4f, // r, g, b, radius
  obst: vec4f,   // cx, cy, r, enabled
  c: vec4f,      // splatOn, forceMode, forceAmp, forceK
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var srcA: texture_2d<f32>;
@group(0) @binding(2) var srcB: texture_2d<f32>;
@group(0) @binding(3) var srcC: texture_2d<f32>;
@group(0) @binding(4) var srcD: texture_2d<f32>;
@group(0) @binding(5) var dst: texture_storage_2d<rgba32float, write>;
@group(0) @binding(6) var mask: texture_2d<f32>; // x > 0.5 marks a solid cell

fn dims() -> vec2i { return vec2i(i32(P.a.x), i32(P.a.y)); }
fn bc() -> u32 { return u32(P.b.x); }
fn wrapc(c: vec2i) -> vec2i {
  let s = dims();
  if (bc() == 1u) { return (c % s + s) % s; }
  return clamp(c, vec2i(0), s - 1);
}
fn outside(c: vec2i) -> bool {
  let s = dims();
  return bc() != 1u && (c.x < 0 || c.y < 0 || c.x >= s.x || c.y >= s.y);
}
fn solid(c: vec2i) -> bool {
  return P.obst.w > 0.5 && textureLoad(mask, clamp(c, vec2i(0), dims() - 1), 0).x > 0.5;
}
fn ldA(c: vec2i) -> vec4f { return textureLoad(srcA, wrapc(c), 0); }
fn ldB(c: vec2i) -> vec4f { return textureLoad(srcB, wrapc(c), 0); }
fn bilerpA(p: vec2f) -> vec4f {
  let q = p - 0.5; let i = vec2i(floor(q)); let f = q - floor(q);
  return mix(mix(ldA(i), ldA(i + vec2i(1, 0)), f.x), mix(ldA(i + vec2i(0, 1)), ldA(i + vec2i(1, 1)), f.x), f.y);
}
fn bilerpB(p: vec2f) -> vec4f {
  let q = p - 0.5; let i = vec2i(floor(q)); let f = q - floor(q);
  return mix(mix(ldB(i), ldB(i + vec2i(1, 0)), f.x), mix(ldB(i + vec2i(0, 1)), ldB(i + vec2i(1, 1)), f.x), f.y);
}
fn inb(g: vec3u) -> bool { let s = dims(); return i32(g.x) < s.x && i32(g.y) < s.y; }

// Backtrace with a midpoint (RK2) step through the velocity in srcB.
fn backtrace(x: vec2f, dt: f32) -> vec2f {
  let v1 = bilerpB(x).xy;
  let v2 = bilerpB(x - 0.5 * dt * v1).xy;
  return x - dt * v2;
}

@compute @workgroup_size(${WG}, ${WG})
fn advect(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let x = vec2f(c) + 0.5;
  textureStore(dst, c, bilerpA(backtrace(x, P.a.z)));
}
@compute @workgroup_size(${WG}, ${WG})
fn advectBack(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let x = vec2f(c) + 0.5;
  textureStore(dst, c, bilerpA(backtrace(x, -P.a.z)));
}
// MacCormack correction: srcA = phi, srcB = velocity, srcC = phi_hat, srcD = phi_bar.
@compute @workgroup_size(${WG}, ${WG})
fn maccormack(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let x = vec2f(c) + 0.5;
  let hat = textureLoad(srcC, c, 0);
  let bar = textureLoad(srcD, c, 0);
  var r = hat + 0.5 * (ldA(c) - bar);
  let q = backtrace(x, P.a.z) - 0.5; let i = vec2i(floor(q));
  let s0 = ldA(i); let s1 = ldA(i + vec2i(1, 0)); let s2 = ldA(i + vec2i(0, 1)); let s3 = ldA(i + vec2i(1, 1));
  r = clamp(r, min(min(s0, s1), min(s2, s3)), max(max(s0, s1), max(s2, s3)));
  textureStore(dst, c, r);
}
@compute @workgroup_size(${WG}, ${WG})
fn advectDye(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let x = vec2f(c) + 0.5;
  textureStore(dst, c, bilerpA(backtrace(x, P.a.z)) * P.b.y);
}
// Implicit viscosity, one Jacobi sweep: srcA = current iterate, srcB = right-hand side.
@compute @workgroup_size(${WG}, ${WG})
fn diffuse(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let a = P.a.w;
  let s = ldA(c + vec2i(1, 0)) + ldA(c - vec2i(1, 0)) + ldA(c + vec2i(0, 1)) + ldA(c - vec2i(0, 1));
  textureStore(dst, c, (ldB(c) + a * s) / (1.0 + 4.0 * a));
}
fn kolmogorov(x: vec2f) -> vec2f {
  let s = vec2f(dims()); let k = P.c.w; let t = P.b.w;
  let fx = sin(6.2831853 * k * x.y / s.y);
  let wob = 0.15 * sin(6.2831853 * x.x / s.x + 0.7 * t) ;
  return P.c.z * vec2f(fx, wob * cos(6.2831853 * 2.0 * x.y / s.y));
}
@compute @workgroup_size(${WG}, ${WG})
fn force(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let x = vec2f(c) + 0.5; let s = dims();
  var u = ldA(c);
  if (P.c.x > 0.5) {
    let d = x - P.splat.xy;
    u = vec4f(u.xy + P.splat.zw * exp(-dot(d, d) / (P.splatC.w * P.splatC.w)), u.zw);
  }
  if (u32(P.c.y) == 1u) { u = vec4f(u.xy + P.a.z * kolmogorov(x), u.zw); }
  if (bc() == 0u) {
    if (c.x == 0 || c.x == s.x - 1) { u.x = 0.0; }
    if (c.y == 0 || c.y == s.y - 1) { u.y = 0.0; }
  }
  if (bc() == 2u) {
    if (c.x == 0) { u = vec4f(P.b.z, 0.0, 0.0, 0.0); }
    if (c.y == 0 || c.y == s.y - 1) { u.y = 0.0; }
  }
  if (solid(c)) { u = vec4f(0.0); }
  textureStore(dst, c, u);
}
@compute @workgroup_size(${WG}, ${WG})
fn splatDye(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let x = vec2f(c) + 0.5; let s = dims();
  var d = ldA(c);
  if (P.c.x > 0.5) {
    let r = x - P.splat.xy;
    d = d + vec4f(P.splatC.rgb, 0.0) * exp(-dot(r, r) / (P.splatC.w * P.splatC.w));
  }
  if (bc() == 2u && c.x < 2) {
    // Streaklines: a dye band every 12 cells across the inlet.
    let band = (c.y / 6) % 2;
    let hue = f32(c.y) / f32(s.y);
    if (band == 0) { d = vec4f(0.35 + 0.65 * hue, 0.55, 1.0 - 0.6 * hue, 1.0); } else { d = vec4f(0.0); }
  }
  if (solid(c)) { d = vec4f(0.0); }
  textureStore(dst, c, d);
}
// Neighbour velocity as the divergence and gradient stencils see it.
fn velN(c: vec2i) -> vec2f {
  let s = dims();
  if (bc() == 2u) {
    if (c.x < 0) { return vec2f(P.b.z, 0.0); }
    if (c.x >= s.x) { return ldA(vec2i(s.x - 1, c.y)).xy; }
  }
  if (outside(c) || solid(c)) { return vec2f(0.0); }
  return ldA(c).xy;
}
@compute @workgroup_size(${WG}, ${WG})
fn divergence(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy);
  let dv = 0.5 * ((velN(c + vec2i(1, 0)).x - velN(c - vec2i(1, 0)).x) + (velN(c + vec2i(0, 1)).y - velN(c - vec2i(0, 1)).y));
  textureStore(dst, c, vec4f(dv, 0.0, 0.0, 0.0));
}
fn pN(c: vec2i, pc: f32) -> f32 {
  let s = dims();
  if (bc() == 2u && c.x >= s.x) { return 0.0; }
  if (outside(c) || solid(c)) { return pc; }
  return ldA(c).x;
}
// Pressure Poisson, one Jacobi sweep: srcA = p, srcB = divergence.
@compute @workgroup_size(${WG}, ${WG})
fn jacobi(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let pc = ldA(c).x;
  let s = pN(c + vec2i(1, 0), pc) + pN(c - vec2i(1, 0), pc) + pN(c + vec2i(0, 1), pc) + pN(c - vec2i(0, 1), pc);
  textureStore(dst, c, vec4f((s - ldB(c).x) * 0.25, 0.0, 0.0, 0.0));
}
fn pB(c: vec2i, pc: f32) -> f32 {
  let s = dims();
  if (bc() == 2u && c.x >= s.x) { return 0.0; }
  if (outside(c) || solid(c)) { return pc; }
  return ldB(c).x;
}
// u -= grad p: srcA = velocity, srcB = pressure.
@compute @workgroup_size(${WG}, ${WG})
fn subtract(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  let c = vec2i(g.xy); let pc = ldB(c).x;
  let gp = 0.5 * vec2f(pB(c + vec2i(1, 0), pc) - pB(c - vec2i(1, 0), pc), pB(c + vec2i(0, 1), pc) - pB(c - vec2i(0, 1), pc));
  var u = ldA(c);
  u = vec4f(u.xy - gp, u.zw);
  if (solid(c)) { u = vec4f(0.0); }
  textureStore(dst, c, u);
}
@compute @workgroup_size(${WG}, ${WG})
fn copyTex(@builtin(global_invocation_id) g: vec3u) {
  if (!inb(g)) { return; }
  textureStore(dst, vec2i(g.xy), ldA(vec2i(g.xy)));
}
`;

// Max |vorticity| and kinetic energy, reduced with atomics. Positive f32 bit
// patterns order like u32, so atomicMax on the bits is a float max.
const STATS_WGSL = /* wgsl */ `
struct Params { a: vec4f, b: vec4f, splat: vec4f, splatC: vec4f, obst: vec4f, c: vec4f };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var vel: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<atomic<u32>, 4>;
fn dims() -> vec2i { return vec2i(i32(P.a.x), i32(P.a.y)); }
fn ld(c: vec2i) -> vec2f {
  let s = dims();
  if (u32(P.b.x) == 1u) { return textureLoad(vel, (c % s + s) % s, 0).xy; }
  return textureLoad(vel, clamp(c, vec2i(0), s - 1), 0).xy;
}
var<workgroup> wmax: array<f32, 64>;
var<workgroup> wsum: array<f32, 64>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let s = dims(); let c = vec2i(g.xy);
  var w = 0.0; var e = 0.0;
  if (c.x < s.x && c.y < s.y) {
    // Skip the outermost ring of cells: the wall stencil there is one-sided noise.
    let edge = u32(P.b.x) != 1u && (c.x < 2 || c.y < 2 || c.x > s.x - 3 || c.y > s.y - 3);
    if (!edge) {
      w = abs(0.5 * ((ld(c + vec2i(1, 0)).y - ld(c - vec2i(1, 0)).y) - (ld(c + vec2i(0, 1)).x - ld(c - vec2i(0, 1)).x)));
    }
    let u = ld(c); e = 0.5 * dot(u, u);
  }
  wmax[li] = w; wsum[li] = e;
  workgroupBarrier();
  for (var k = 32u; k > 0u; k = k >> 1u) {
    if (li < k) { wmax[li] = max(wmax[li], wmax[li + k]); wsum[li] = wsum[li] + wsum[li + k]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    atomicMax(&out[0], bitcast<u32>(wmax[0]));
    atomicAdd(&out[1], u32(wsum[0] * 64.0));
  }
}
`;

const DRAW_WGSL = /* wgsl */ `
struct D { a: vec4f /* W, H, mode, gain */, obst: vec4f, b: vec4f /* bc, 0,0,0 */ };
@group(0) @binding(0) var<uniform> U: D;
@group(0) @binding(1) var dye: texture_2d<f32>;
@group(0) @binding(2) var vel: texture_2d<f32>;
@group(0) @binding(3) var mask: texture_2d<f32>;
struct VO { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: VO; o.pos = vec4f(p[i], 0.0, 1.0); o.uv = p[i] * 0.5 + 0.5; return o;
}
fn dims() -> vec2i { return vec2i(i32(U.a.x), i32(U.a.y)); }
fn wr(c: vec2i) -> vec2i {
  let s = dims();
  if (u32(U.b.x) == 1u) { return (c % s + s) % s; }
  return clamp(c, vec2i(0), s - 1);
}
fn V(c: vec2i) -> vec2f { return textureLoad(vel, wr(c), 0).xy; }
fn vort(c: vec2i) -> f32 { return 0.5 * ((V(c + vec2i(1, 0)).y - V(c - vec2i(1, 0)).y) - (V(c + vec2i(0, 1)).x - V(c - vec2i(0, 1)).x)); }
fn divg(c: vec2i) -> f32 { return 0.5 * ((V(c + vec2i(1, 0)).x - V(c - vec2i(1, 0)).x) + (V(c + vec2i(0, 1)).y - V(c - vec2i(0, 1)).y)); }
fn field(c: vec2i, mode: u32) -> vec4f {
  if (mode == 0u) { return textureLoad(dye, wr(c), 0); }
  if (mode == 1u) { return vec4f(vort(c)); }
  if (mode == 2u) { return vec4f(length(V(c))); }
  return vec4f(divg(c));
}
fn sampleF(p: vec2f, mode: u32) -> vec4f {
  let q = p - 0.5; let i = vec2i(floor(q)); let f = q - floor(q);
  return mix(mix(field(i, mode), field(i + vec2i(1, 0), mode), f.x), mix(field(i + vec2i(0, 1), mode), field(i + vec2i(1, 1), mode), f.x), f.y);
}
fn inferno(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  let c0 = vec3f(0.0002, 0.0016, -0.0194); let c1 = vec3f(0.1065, 0.5639, 3.9327);
  let c2 = vec3f(11.6024, -3.9728, -15.9423); let c3 = vec3f(-41.7039, 17.4363, 44.3541);
  let c4 = vec3f(77.1629, -33.4023, -81.8073); let c5 = vec3f(-71.3194, 32.6260, 73.2095);
  let c6 = vec3f(25.1311, -12.2426, -23.0703);
  return clamp(c0 + x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + x * c6))))), vec3f(0.0), vec3f(1.0));
}
fn diverging(t: f32) -> vec3f {
  let x = clamp(t, -1.0, 1.0); let a = sqrt(abs(x));
  let bg = vec3f(0.035, 0.04, 0.06);
  if (x > 0.0) { return mix(bg, mix(vec3f(0.95, 0.35, 0.15), vec3f(1.0, 0.92, 0.6), a * a), a); }
  return mix(bg, mix(vec3f(0.15, 0.45, 0.95), vec3f(0.7, 0.95, 1.0), a * a), a);
}
@fragment fn fs(i: VO) -> @location(0) vec4f {
  let s = vec2f(dims());
  let p = vec2f(i.uv.x, 1.0 - i.uv.y) * s;
  let mode = u32(U.a.z); let gain = U.a.w;
  if (U.obst.w > 0.5 && textureLoad(mask, clamp(vec2i(floor(p)), vec2i(0), dims() - 1), 0).x > 0.5) { return vec4f(0.32, 0.34, 0.38, 1.0); }
  let v = sampleF(p, mode);
  var col: vec3f;
  if (mode == 0u) { col = vec3f(0.035, 0.04, 0.06) + (1.0 - exp(-v.rgb * gain)); }
  else if (mode == 1u) { col = diverging(v.x * gain); }
  else if (mode == 2u) { col = inferno(v.x * gain); }
  else { col = diverging(v.x * gain); }
  return vec4f(col, 1.0);
}
`;

export class Fluid2D {
  constructor(device, canvas, opts = {}) {
    this.device = device;
    this.canvas = canvas;
    this.W = opts.W ?? 256;
    this.H = opts.H ?? 256;
    this.bc = opts.bc ?? 0;
    this.dt = opts.dt ?? 0.5;
    this.nu = opts.nu ?? 0.0;
    this.dyeDecay = opts.dyeDecay ?? 0.998;
    this.inflow = opts.inflow ?? 0;
    this.obst = opts.obst ?? [0, 0, 0, 0];
    this.pressureIters = opts.pressureIters ?? 40;
    this.diffuseIters = opts.diffuseIters ?? 20;
    this.maccormack = opts.maccormack ?? false;
    this.forceMode = opts.forceMode ?? 0;
    this.forceAmp = opts.forceAmp ?? 0;
    this.forceK = opts.forceK ?? 4;
    this.mode = opts.mode ?? 0;
    this.gain = opts.gain ?? 1.5;
    this.terms = { advection: true, pressure: true, viscosity: true, forcing: true };
    this.time = 0;
    this.splatQueue = [];
    this.stats = { maxVort: 0, energy: 0, frame: 0 };
    this._statsBusy = false;
    this._init();
  }

  _tex(label) {
    return this.device.createTexture({
      label, size: [this.W, this.H], format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    });
  }

  _init() {
    const d = this.device;
    this.vel = [this._tex('vel0'), this._tex('vel1')];
    this.dye = [this._tex('dye0'), this._tex('dye1')];
    this.prs = [this._tex('p0'), this._tex('p1')];
    this.div = this._tex('div');
    this.t1 = this._tex('t1');
    this.t2 = this._tex('t2');
    this.vrhs = this._tex('vrhs');
    this.mask = this._tex('mask');
    this.params = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.layout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } })),
        { binding: 5, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32float' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    const mod = d.createShaderModule({ code: SIM_WGSL });
    const pl = d.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    this.pipes = {};
    for (const e of ['advect', 'advectBack', 'maccormack', 'advectDye', 'diffuse', 'force', 'splatDye', 'divergence', 'jacobi', 'subtract', 'copyTex']) {
      this.pipes[e] = d.createComputePipeline({ layout: pl, compute: { module: mod, entryPoint: e } });
    }
    this.bgCache = new Map();

    this.statsBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.statsRead = d.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.statsPipe = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: STATS_WGSL }), entryPoint: 'main' } });

    this.ctx = this.canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: d, format: this.format, alphaMode: 'opaque' });
    const dm = d.createShaderModule({ code: DRAW_WGSL });
    this.drawPipe = d.createRenderPipeline({
      layout: 'auto',
      vertex: { module: dm, entryPoint: 'vs' },
      fragment: { module: dm, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.drawU = d.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.vi = 0; this.di = 0;
  }

  _bg(a, b, c, dd, dst) {
    const key = [a, b, c, dd, dst].map((t) => t.label).join('|');
    let bg = this.bgCache.get(key);
    if (!bg) {
      bg = this.device.createBindGroup({
        layout: this.layout,
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: a.createView() }, { binding: 2, resource: b.createView() },
          { binding: 3, resource: c.createView() }, { binding: 4, resource: dd.createView() },
          { binding: 5, resource: dst.createView() },
          { binding: 6, resource: this.mask.createView() },
        ],
      });
      this.bgCache.set(key, bg);
    }
    return bg;
  }

  _pass(enc, entry, a, b, dst, c = null, dd = null) {
    const filler = [this.t1, this.t2, this.div, this.vrhs, this.prs[0], this.prs[1]].find((t) => t !== dst);
    const p = enc.beginComputePass();
    p.setPipeline(this.pipes[entry]);
    p.setBindGroup(0, this._bg(a, b, c ?? filler, dd ?? filler, dst));
    p.dispatchWorkgroups(Math.ceil(this.W / WG), Math.ceil(this.H / WG));
    p.end();
  }

  // Solid cells: a W*H array, nonzero where solid. Enables the obstacle.
  setMask(solid) {
    const m = new Float32Array(this.W * this.H * 4);
    for (let i = 0; i < this.W * this.H; i++) m[i * 4] = solid[i] ? 1 : 0;
    this.device.queue.writeTexture({ texture: this.mask }, m, { bytesPerRow: this.W * 16 }, [this.W, this.H]);
    this.solid = solid;
    this.obst = [0, 0, 0, 1];
  }

  // Upload an initial velocity (and optional dye) field: Float32Array of W*H*4.
  setFields(vel, dye) {
    const q = this.device.queue;
    const lay = { bytesPerRow: this.W * 16 };
    for (const t of this.vel) q.writeTexture({ texture: t }, vel, lay, [this.W, this.H]);
    for (const t of this.prs) q.writeTexture({ texture: t }, new Float32Array(this.W * this.H * 4), lay, [this.W, this.H]);
    if (dye) for (const t of this.dye) q.writeTexture({ texture: t }, dye, lay, [this.W, this.H]);
    this.time = 0;
  }

  clear() {
    const z = new Float32Array(this.W * this.H * 4);
    if (this.bc === 2) for (let i = 0; i < this.W * this.H; i++) if (!this.solid?.[i]) z[i * 4] = this.inflow;
    this.setFields(z, new Float32Array(this.W * this.H * 4));
  }

  splat(x, y, vx, vy, color, radius = 6) {
    this.splatQueue.push({ x, y, vx, vy, color, radius });
  }

  _writeParams(splat) {
    const s = splat ?? { x: 0, y: 0, vx: 0, vy: 0, color: [0, 0, 0], radius: 1 };
    const t = this.terms;
    this.device.queue.writeBuffer(this.params, 0, new Float32Array([
      this.W, this.H, this.dt, this.nu * this.dt,
      this.bc, this.dyeDecay, this.inflow, this.time,
      s.x, s.y, t.forcing ? s.vx : 0, t.forcing ? s.vy : 0,
      ...s.color, s.radius,
      ...this.obst,
      splat ? 1 : 0, t.forcing ? this.forceMode : 0, this.forceAmp, this.forceK,
    ]));
  }

  // One time step. Splats queued since the last step are applied in this step's
  // force pass; only the first is used so every pass sees one uniform write.
  step() {
    const d = this.device;
    const splat = this.splatQueue.shift() ?? null;
    if (this.splatQueue.length > 4) this.splatQueue.length = 4;
    this._writeParams(splat);
    const enc = d.createCommandEncoder();
    const V = this.vel;
    let vi = this.vi;
    const t = this.terms;

    if (t.advection) {
      if (this.maccormack) {
        this._pass(enc, 'advect', V[vi], V[vi], this.t1);
        this._pass(enc, 'advectBack', this.t1, V[vi], this.t2);
        this._pass(enc, 'maccormack', V[vi], V[vi], V[1 - vi], this.t1, this.t2);
      } else {
        this._pass(enc, 'advect', V[vi], V[vi], V[1 - vi]);
      }
      vi = 1 - vi;
    }
    if (t.viscosity && this.nu > 0) {
      this._pass(enc, 'copyTex', V[vi], V[vi], this.vrhs);
      // Jacobi for (1 - a*Laplacian) contracts by 4a/(1+4a) per sweep, so small a needs few sweeps.
      const iters = Math.min(this.diffuseIters, 2 + Math.ceil(6 * this.nu * this.dt));
      for (let k = 0; k < iters; k++) {
        this._pass(enc, 'diffuse', V[vi], this.vrhs, V[1 - vi]);
        vi = 1 - vi;
      }
    }
    this._pass(enc, 'force', V[vi], V[vi], V[1 - vi]);
    vi = 1 - vi;
    if (t.pressure) {
      this._pass(enc, 'divergence', V[vi], V[vi], this.div);
      let pi = 0;
      for (let k = 0; k < this.pressureIters; k++) {
        this._pass(enc, 'jacobi', this.prs[pi], this.div, this.prs[1 - pi]);
        pi = 1 - pi;
      }
      this._pass(enc, 'subtract', V[vi], this.prs[pi], V[1 - vi]);
      vi = 1 - vi;
      // Warm-start the next solve from this one.
      if (pi !== 0) this._pass(enc, 'copyTex', this.prs[1], this.prs[1], this.prs[0]);
    }
    this._pass(enc, 'advectDye', this.dye[this.di], V[vi], this.dye[1 - this.di]);
    this.di = 1 - this.di;
    this._pass(enc, 'splatDye', this.dye[this.di], V[vi], this.dye[1 - this.di]);
    this.di = 1 - this.di;
    this.vi = vi;
    d.queue.submit([enc.finish()]);
    this.time += this.dt;
    this.stats.frame++;
  }

  // Kick off an asynchronous read of max|omega| and energy; results land in this.stats.
  measure() {
    if (this._statsBusy) return;
    this._statsBusy = true;
    const d = this.device;
    const enc = d.createCommandEncoder();
    enc.clearBuffer(this.statsBuf);
    const bg = d.createBindGroup({
      layout: this.statsPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: this.vel[this.vi].createView() }, { binding: 2, resource: { buffer: this.statsBuf } }],
    });
    const p = enc.beginComputePass();
    p.setPipeline(this.statsPipe); p.setBindGroup(0, bg);
    p.dispatchWorkgroups(Math.ceil(this.W / 8), Math.ceil(this.H / 8)); p.end();
    enc.copyBufferToBuffer(this.statsBuf, 0, this.statsRead, 0, 16);
    d.queue.submit([enc.finish()]);
    const t = this.time;
    this.statsRead.mapAsync(GPUMapMode.READ).then(() => {
      const u = new Uint32Array(this.statsRead.getMappedRange().slice(0));
      this.statsRead.unmap();
      const f = new Float32Array(u.buffer);
      this.stats.maxVort = f[0];
      this.stats.energy = u[1] / 64 / (this.W * this.H);
      this.stats.time = t;
      this._statsBusy = false;
      this.onStats?.(this.stats);
    }).catch(() => { this._statsBusy = false; });
  }

  // Read the whole velocity field back (for the energy spectrum). Resolves to Float32Array W*H*4.
  async readVelocity() {
    const d = this.device;
    const bytes = this.W * this.H * 16;
    if (!this._velRead || this._velRead.size !== bytes) {
      this._velRead = d.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    if (this._velBusy) return null;
    this._velBusy = true;
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: this.vel[this.vi] }, { buffer: this._velRead, bytesPerRow: this.W * 16 }, [this.W, this.H]);
    d.queue.submit([enc.finish()]);
    await this._velRead.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this._velRead.getMappedRange().slice(0));
    this._velRead.unmap();
    this._velBusy = false;
    return out;
  }

  render() {
    const d = this.device;
    d.queue.writeBuffer(this.drawU, 0, new Float32Array([this.W, this.H, this.mode, this.gain, ...this.obst, this.bc, 0, 0, 0]));
    const enc = d.createCommandEncoder();
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    rp.setPipeline(this.drawPipe);
    // One cached bind group per (dye, velocity) parity pair.
    rp.setBindGroup(0, this._drawGroup());
    rp.draw(3); rp.end();
    d.queue.submit([enc.finish()]);
  }

  _drawGroup() {
    const key = `${this.di}${this.vi}`;
    this._dg ??= new Map();
    let g = this._dg.get(key);
    if (!g) {
      g = this.device.createBindGroup({
        layout: this.drawPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: this.drawU } }, { binding: 1, resource: this.dye[this.di].createView() }, { binding: 2, resource: this.vel[this.vi].createView() }, { binding: 3, resource: this.mask.createView() }],
      });
      this._dg.set(key, g);
    }
    return g;
  }

  // Map a pointer event to grid coordinates.
  toGrid(ev) {
    const r = this.canvas.getBoundingClientRect();
    return [((ev.clientX - r.left) / r.width) * this.W, ((ev.clientY - r.top) / r.height) * this.H];
  }

  attachPointer({ color = null, radius = 6, strength = 1 } = {}) {
    let last = null;
    let hue = Math.random();
    const el = this.canvas;
    el.style.touchAction = 'none';
    el.addEventListener('pointerdown', (e) => { last = this.toGrid(e); el.setPointerCapture(e.pointerId); hue = (hue + 0.17) % 1; });
    el.addEventListener('pointerup', () => { last = null; });
    el.addEventListener('pointercancel', () => { last = null; });
    el.addEventListener('pointermove', (e) => {
      if (!last) return;
      const p = this.toGrid(e);
      const vx = (p[0] - last[0]) * strength, vy = (p[1] - last[1]) * strength;
      const c = color ?? hsl(hue, 0.85, 0.55);
      this.splat(p[0], p[1], vx, vy, c, radius);
      last = p;
    });
  }
}

export function hsl(h, s, l) {
  const f = (n) => {
    const k = (n + h * 12) % 12;
    return l - s * Math.min(l, 1 - l) * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

// Random Gaussian vortex blobs, with dye coloured by sign. Returns {vel, dye}.
export function vortexBlobs(W, H, n, { radius = 10, strength = 1.2, seed = 1, periodic = false } = {}) {
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const blobs = [];
  for (let i = 0; i < n; i++) {
    blobs.push({ x: (0.1 + 0.8 * rnd()) * W, y: (0.1 + 0.8 * rnd()) * H, g: (rnd() < 0.5 ? -1 : 1) * strength * (0.6 + 0.8 * rnd()), r: radius * (0.7 + 0.6 * rnd()) });
  }
  const vel = new Float32Array(W * H * 4);
  const dye = new Float32Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let u = 0, v = 0, w = 0;
      for (const b of blobs) {
        let dx = x + 0.5 - b.x, dy = y + 0.5 - b.y;
        if (periodic) { dx -= W * Math.round(dx / W); dy -= H * Math.round(dy / H); }
        const r2 = dx * dx + dy * dy;
        const e = Math.exp(-r2 / (b.r * b.r));
        // Lamb-Oseen-like vortex: azimuthal speed g*r*e/r0, peak vorticity ~ 2g/r0.
        u += -b.g * dy * e / b.r;
        v += b.g * dx * e / b.r;
        w += b.g * e;
      }
      const i = (y * W + x) * 4;
      vel[i] = u; vel[i + 1] = v;
      dye[i] = Math.max(0, w) * 0.9; dye[i + 1] = Math.abs(w) * 0.25; dye[i + 2] = Math.max(0, -w) * 0.9;
    }
  }
  return { vel, dye };
}
