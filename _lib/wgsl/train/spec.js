// The training kernels' spec format and its shared WGSL helpers (kernel library, _lib/wgsl/README.md).
//
// A kernel spec is { u, b, wg, pre, body }: u the uniform names in order (suffix ":f" for f32, else
// u32), b the storage bindings [name, "r" | "rw", element type] from binding 1 up, wg the workgroup
// size, pre any helper functions or workgroup arrays, body the code after `let gi = ...` (the
// workgroup's linear index over a 2-D grid). wgsl(name, spec) assembles the shader; uniforms are one
// struct at binding 0, read with a dynamic offset so one buffer holds every dispatch's values.
// Moved from ettin-wgsl-train/web/train.js (step 3); the shaders are byte-for-byte the same.

export const SMAX = 1024;   // longest sequence: the attention kernels' workgroup arrays

// Abramowitz & Stegun 7.1.26 erf (|error| < 1.5e-7): exact-erf GELU as in PyTorch, and its derivative.
export const GELU = /* wgsl */ `
fn erf_(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return sign(x) * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.7071067811865476)); }
fn gelu_d(x: f32) -> f32 { return 0.5 * (1.0 + erf_(x * 0.7071067811865476)) + x * exp(-0.5 * x * x) * 0.3989422804014327; }
`;

export const REDUCE = (n) => /* wgsl */ `
var<workgroup> red: array<f32, ${n}>;
fn wsum(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) {
    if (t < s) { red[t] = red[t] + red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn wmax(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) {
    if (t < s) { red[t] = max(red[t], red[t + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`;

export const ENTRY = (wg) => `@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let gi = wid.x + wid.y * nwg.x;`;

export function wgsl(name, k) {
  const fields = k.u.map((s) => { const [n, t] = s.split(":"); return `  ${n}: ${t === "f" ? "f32" : "u32"},`; });
  const st = `struct Uni {\n${fields.join("\n")}\n}\n@group(0) @binding(0) var<uniform> p: Uni;\n`;
  const binds = k.b.map(([n, m, t], i) => `@group(0) @binding(${i + 1}) var<storage, ${m === "rw" ? "read_write" : "read"}> ${n}: array<${t}>;`).join("\n");
  return `// ${name}\n${st}${binds}\n${k.pre || ""}\n${ENTRY(k.wg)}\n${k.body}\n}\n`;
}

export function grid(n) {
  if (n <= 65535) return [n, 1, 1];
  return [65535, Math.ceil(n / 65535), 1];
}
