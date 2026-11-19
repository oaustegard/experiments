// Runner smoke test: copy a fixture through a trivial kernel and compare.
import { Gpu, loadCase, compare } from "../gpu.js";
const COPY = `
@group(0) @binding(0) var<uniform> n: vec4<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x < n.x) { b[g.x] = a[g.x]; }
}`;
export default async function run({ cases }) {
  const gpu = await Gpu.create();
  const res = [];
  for (const base of cases) {
    const { man, f32 } = await loadCase(base);
    const x = await f32("in");
    const a = gpu.upload(x), b = gpu.buffer(x.byteLength);
    gpu.dispatch(COPY, [gpu.uniform(new Uint32Array([x.length, 0, 0, 0])), a, b], [Math.ceil(x.length / 64)]);
    res.push({ case: man.case, ...compare(await gpu.read(b, Float32Array, x.byteLength), x) });
  }
  return { ok: res.every((r) => r.bad === 0), adapter: (await Gpu.create()).info, res };
}
