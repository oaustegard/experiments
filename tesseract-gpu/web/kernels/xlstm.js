// x-direction LSTM kernel for the tesseract-gpu engine (the sequential part).
//
//   recordXLstm(gpu, { x, w, s, tanhT, logT, out, state, lines, ni, ns, reverse, intMode })
//
// x       f32 [sum T_i, ni]   packed input; line i occupies rows t_off_i .. t_off_i + T_i - 1
// w       f32 [4*ns, ni+ns+1] gate rows in block order CI (tanh), GI, GF1, GO (logistic);
//                             columns: input, previous output, bias
// s       f32 [4*ns]          int8 models: per-row scale (dot = (sum q*src + q_bias*127) * s);
//                             pass null for float models
// tanhT   f32 [4096]          Tesseract's tanh table (x = i/256), linear interpolation as ref._lut
// logT    f32 [4096]          Tesseract's logistic table
// out     f32 [sum T_i, ns]   output, written at each position as it is computed
// state   f32 [nlines, ns]    scratch cell state, zeroed by the first dispatch
// lines   [{t_off, T}, ...]   lines sorted longest first (manifest `lines` works directly)
// reverse processes x = T_i-1-s at step s (RTLReversed) instead of x = s
// intMode int8 model: src and out are integers; out is quantised before it is written and fed back
//
// One dispatch per step, all recorded into the caller's command encoder (no submit). Lines still
// running at step s are a prefix of the list, so each step covers nact = #{T_i > s} lines.
//
// Workgroup = 64 threads, one hidden unit each, for a tile of LT = 4 active lines. The
// workgroup stages the lines' src vectors in workgroup memory in chunks of CK columns, and each
// weight row is read once per workgroup per step and reused for all LT lines.

const HT = 64; // hidden units per workgroup (workgroup size)
const LT = 4;  // active lines per workgroup
const CK = 64; // src columns staged per chunk

const ZERO = /* wgsl */ `
@group(0) @binding(0) var<uniform> z: vec4<u32>;
@group(0) @binding(1) var<storage, read_write> st: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x < z.x) { st[g.x] = 0.0; }
}`;

const STEP = /* wgsl */ `
struct P {
  step: u32,
  nact: u32,
  ni: u32,
  ns: u32,
  k: u32,
  int_mode: u32,
  reverse: u32,
  nlines: u32,
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> xin: array<f32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> scl: array<f32>;
@group(0) @binding(4) var<storage, read> tanh_t: array<f32>;
@group(0) @binding(5) var<storage, read> logi_t: array<f32>;
@group(0) @binding(6) var<storage, read_write> yout: array<f32>;
@group(0) @binding(7) var<storage, read_write> st: array<f32>;
@group(0) @binding(8) var<storage, read> lines: array<u32>;

const HT: u32 = ${HT}u;
const LT: u32 = ${LT}u;
const CK: u32 = ${CK}u;

var<workgroup> srcs: array<f32, ${LT * CK}>;

// Tesseract's table lookup (ref._lut): |x| indexed at 256 per unit, clamped at 4095/256.
fn lut_tanh(x: f32) -> f32 {
  let a = abs(x);
  var v: f32 = 1.0;
  if (a < 4095.0 / 256.0) {
    let xs = a * 256.0;
    let idx = min(u32(xs), 4094u);
    let t0 = tanh_t[idx];
    let t1 = tanh_t[idx + 1u];
    v = t0 + (t1 - t0) * (xs - f32(idx));
  }
  if (x < 0.0) { return -v; }
  return v;
}

fn lut_logi(x: f32) -> f32 {
  let a = abs(x);
  var v: f32 = 1.0;
  if (a < 4095.0 / 256.0) {
    let xs = a * 256.0;
    let idx = min(u32(xs), 4094u);
    let t0 = logi_t[idx];
    let t1 = logi_t[idx + 1u];
    v = t0 + (t1 - t0) * (xs - f32(idx));
  }
  if (x < 0.0) { return 1.0 - v; }
  return v;
}

@compute @workgroup_size(${HT})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let j = wid.x * HT + tid;      // hidden unit (may exceed ns in the last tile)
  let l0 = wid.y * LT;           // first active line of this tile
  let ni = p.ni;
  let ns = p.ns;
  let kn = p.k;
  let is_int = p.int_mode == 1u;
  let rev = p.reverse == 1u;
  let bias_val: f32 = select(1.0, 127.0, is_int);

  var acc: array<f32, ${4 * LT}>;
  for (var i = 0u; i < ${4 * LT}u; i++) { acc[i] = 0.0; }

  for (var c = 0u; c < kn; c += CK) {
    // Stage src[li][kk] = src vector of line l0+li at column c+kk: input, previous output, bias.
    for (var i = tid; i < LT * CK; i += HT) {
      let li = i / CK;
      let k = c + (i % CK);
      let line = l0 + li;
      var v: f32 = 0.0;
      if (line < p.nact && k < kn) {
        let t_off = lines[2u * line];
        let t_len = lines[2u * line + 1u];
        let pos = select(t_off + p.step, t_off + t_len - 1u - p.step, rev);
        if (k < ni) {
          v = xin[pos * ni + k];
        } else if (k < ni + ns) {
          if (p.step > 0u) {
            let ppos = select(t_off + p.step - 1u, t_off + t_len - p.step, rev);
            v = yout[ppos * ns + (k - ni)];
          }
        } else {
          v = bias_val;
        }
      }
      srcs[i] = v;
    }
    workgroupBarrier();

    if (j < ns) {
      for (var kk = 0u; kk < CK; kk++) {
        let k = c + kk;
        if (k < kn) {
          let w0 = wt[j * kn + k];
          let w1 = wt[(ns + j) * kn + k];
          let w2 = wt[(2u * ns + j) * kn + k];
          let w3 = wt[(3u * ns + j) * kn + k];
          for (var li = 0u; li < LT; li++) {
            let sv = srcs[li * CK + kk];
            acc[li] += w0 * sv;
            acc[LT + li] += w1 * sv;
            acc[2u * LT + li] += w2 * sv;
            acc[3u * LT + li] += w3 * sv;
          }
        }
      }
    }
    workgroupBarrier();
  }

  if (j < ns) {
    for (var li = 0u; li < LT; li++) {
      let line = l0 + li;
      if (line < p.nact) {
        var a0 = acc[li];
        var a1 = acc[LT + li];
        var a2 = acc[2u * LT + li];
        var a3 = acc[3u * LT + li];
        if (is_int) {
          a0 *= scl[j];
          a1 *= scl[ns + j];
          a2 *= scl[2u * ns + j];
          a3 *= scl[3u * ns + j];
        }
        let ci = lut_tanh(a0);
        let gi = lut_logi(a1);
        let gf = lut_logi(a2);
        let go = lut_logi(a3);
        let sidx = line * ns + j;
        let sn = clamp(st[sidx] * gf + ci * gi, -100.0, 100.0);
        st[sidx] = sn;
        var o = lut_tanh(sn) * go;
        if (is_int) {
          let q = clamp(o, -1.0, 1.0) * 127.0;
          o = sign(q) * floor(abs(q) + 0.5);
        }
        let t_off = lines[2u * line];
        let t_len = lines[2u * line + 1u];
        let pos = select(t_off + p.step, t_off + t_len - 1u - p.step, rev);
        yout[pos * ns + j] = o;
      }
    }
  }
}`;

// Accepts [{t_off, T}, ...] (the manifest's lines) or [[t_off, T], ...].
function normaliseLines(lines) {
  return lines.map((l) => (Array.isArray(l) ? { t_off: l[0], T: l[1] } : { t_off: l.t_off, T: l.T }));
}

export function recordXLstm(gpu, { x, w, s, tanhT, logT, out, state, lines, ni, ns, reverse = false, intMode = false }) {
  const L = normaliseLines(lines);
  const nlines = L.length;
  if (nlines === 0) return;
  for (let i = 1; i < nlines; i++) {
    if (L[i].T > L[i - 1].T) throw new Error("xlstm: lines must be sorted longest first");
  }
  const K = ni + ns + 1;
  const lineBuf = gpu.upload(new Uint32Array(L.flatMap((l) => [l.t_off, l.T])));
  const scaleBuf = s || gpu.buffer(16);

  // Zero the cell state of every line.
  const zero = gpu.uniform(new Uint32Array([nlines * ns, 0, 0, 0]));
  gpu.dispatch(ZERO, [zero, state], [Math.ceil((nlines * ns) / 64)]);

  const maxT = L[0].T;
  for (let step = 0; step < maxT; step++) {
    let nact = 0;
    while (nact < nlines && L[nact].T > step) nact++;
    const u = gpu.uniform(new Uint32Array([step, nact, ni, ns, K, intMode ? 1 : 0, reverse ? 1 : 0, nlines]));
    gpu.dispatch(STEP, [u, x, w, scaleBuf, tanhT, logT, out, state, lineBuf],
      [Math.ceil(ns / HT), Math.ceil(nact / LT), 1]);
  }
}
