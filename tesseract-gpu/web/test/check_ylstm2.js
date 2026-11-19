// Parity of the y-summarising LSTM kernel with the NumPy reference (tessgpu/ref.py) on the
// packed fixtures from run/make_fixtures.py. Float cases: max |diff| <= 1e-4. Int case: no
// entry off by more than 1, and at most 0.1% of entries off by exactly 1.
import { Gpu, loadCase, compare } from "../gpu.js";
import { recordYLstm } from "../kernels/ylstm2.js";
import { interleaveGates } from "../kernels/xlstm2.js";

export default async function run({ cases }) {
  const gpu = await Gpu.create();
  const res = [];
  for (const base of cases) {
    const { man, f32 } = await loadCase(base);
    const pool = await f32("pool");
    const want = await f32("ylstm");
    const ns = man.ns.y;
    const intMode = !!man.int_mode;
    const wHost = await f32("w_y");
    const sHost = intMode ? await f32("s_y") : null;
    const out = gpu.buffer(want.byteLength);
    recordYLstm(gpu, {
      pool: gpu.upload(pool),
      w: gpu.upload(interleaveGates(wHost, ns, 16 + ns + 1)),
      s: sHost ? gpu.upload(sHost) : null,
      tanhT: gpu.upload(await f32("tanh")),
      logT: gpu.upload(await f32("logistic")),
      out,
      lines: man.lines,
      H: man.H,
      ns,
      intMode,
    });
    const got = await gpu.read(out, Float32Array, want.byteLength);
    const c = compare(got, want, intMode ? 1 : 1e-4);
    let off1 = 0;
    for (let i = 0; i < want.length; i++) if (Math.abs(got[i] - want[i]) > 0) off1++;
    const frac1 = want.length ? off1 / want.length : 0;
    const ok = intMode
      ? c.bad === 0 && frac1 <= 0.001
      : c.maxDiff <= 1e-4;
    res.push({
      case: man.case,
      int_mode: intMode,
      ok,
      maxDiff: c.maxDiff,
      maxRef: c.maxRef,
      bad: intMode ? c.bad : compare(got, want, 1e-4).bad,
      off_by_any: off1,
      frac_off: frac1,
      n: c.n,
    });
  }
  return { ok: res.every((r) => r.ok), res };
}
