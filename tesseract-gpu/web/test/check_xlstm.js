// Parity of the x-direction LSTM kernel (web/kernels/xlstm.js) with the NumPy reference.
// Per case, three layers, each from its fixture input and compared with its fixture output:
//   x1 = forward(ylstm, w_x1), x2 = reverse(x1, w_x2), x3 = forward(x2, w_x3).
// Float cases: max |diff| <= 1e-4. Int case: no entry off by more than 1, at most 0.1% off by 1.
import { Gpu, loadCase } from "../gpu.js";
import { recordXLstm } from "../kernels/xlstm.js";

const LAYERS = [
  { name: "x1", input: "ylstm", w: "w_x1", s: "s_x1", out: "x1", reverse: false },
  { name: "x2", input: "x1", w: "w_x2", s: "s_x2", out: "x2", reverse: true },
  { name: "x3", input: "x2", w: "w_x3", s: "s_x3", out: "x3", reverse: false },
];

export default async function run({ cases }) {
  const gpu = await Gpu.create();
  const res = [];
  for (const base of cases) {
    const { man, f32 } = await loadCase(base);
    const intMode = !!man.int_mode;
    const tanhT = gpu.upload(await f32("tanh"));
    const logT = gpu.upload(await f32("logistic"));
    const nT = man.total.t;
    for (const L of LAYERS) {
      const [rows, K] = man.files[L.w].shape;
      const ns = rows / 4;
      const ni = K - ns - 1;
      const xin = gpu.upload(await f32(L.input));
      const wb = gpu.upload(await f32(L.w));
      const sb = intMode ? gpu.upload(await f32(L.s)) : null;
      const out = gpu.buffer(nT * ns * 4);
      const state = gpu.buffer(man.lines.length * ns * 4);
      const t0 = performance.now();
      recordXLstm(gpu, { x: xin, w: wb, s: sb, tanhT, logT, out, state, lines: man.lines, ni, ns,
        reverse: L.reverse, intMode });
      const got = await gpu.read(out, Float32Array, nT * ns * 4);
      const ms = performance.now() - t0;
      const want = await f32(L.out);
      let maxDiff = 0, over = 0, off1 = 0;
      for (let i = 0; i < want.length; i++) {
        const d = Math.abs(got[i] - want[i]);
        if (d > maxDiff) maxDiff = d;
        if (intMode) { if (d > 1) over++; else if (d > 0) off1++; }
        else if (d > 1e-4) over++;
      }
      const pass = intMode ? over === 0 && off1 <= 0.001 * want.length : maxDiff <= 1e-4;
      res.push({ case: man.case, layer: L.name, ns, ni, maxDiff, bad: over, off1, n: want.length,
        ms: Math.round(ms * 10) / 10, pass });
    }
  }
  return { ok: res.every((r) => r.pass), res };
}
