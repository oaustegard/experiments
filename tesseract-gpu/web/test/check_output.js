// Output layer parity: softmax probs and per-timestep argmax against the packed fixtures.
//
//   node web/test/run.mjs test/check_output.js best_eng fast_eng best_grc
//
// Pass bar: max|probs - fixture| <= 1e-5 in every case, identical argmax at every timestep,
// and the reported best probability within 1e-5 of the fixture's max.
import { Gpu, loadCase, compare } from "../gpu.js";
import { recordOutput } from "../kernels/output.js";

const TOL = 1e-5;

export default async function run({ cases }) {
  const gpu = await Gpu.create();
  const res = [];
  for (const base of cases) {
    const { man, f32 } = await loadCase(base);
    const T = man.total.t, ni = man.ns.x3, C = man.C;
    const x = await f32("x3");
    const w = await f32("w_out");
    const s = man.int_mode ? await f32("s_out") : null;
    const want = await f32("probs");
    if (x.length !== T * ni) throw new Error(`${man.case}: x3 has ${x.length}, want ${T * ni}`);
    if (w.length !== C * (ni + 1)) throw new Error(`${man.case}: w_out has ${w.length}`);
    if (want.length !== T * C) throw new Error(`${man.case}: probs has ${want.length}`);

    // Expected argmax and its probability, first index on ties (numpy order).
    const wantBest = new Uint32Array(T);
    const wantBp = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      let bi = 0, bv = -Infinity;
      for (let c = 0; c < C; c++) {
        const p = want[t * C + c];
        if (p > bv) { bv = p; bi = c; }
      }
      wantBest[t] = bi;
      wantBp[t] = bv;
    }

    const out = gpu.buffer(T * C * 4);
    const best = gpu.buffer(T * 4);
    const bestp = gpu.buffer(T * 4);
    recordOutput(gpu, {
      x: gpu.upload(x), w: gpu.upload(w), s: s ? gpu.upload(s) : null,
      out, best, bestp, T, ni, C, intMode: man.int_mode,
    });
    gpu.submit();
    const probs = await gpu.read(out, Float32Array, T * C * 4);
    const bests = await gpu.read(best, Uint32Array, T * 4);
    const bps = await gpu.read(bestp, Float32Array, T * 4);

    const cmp = compare(probs, want, TOL);
    let mismatches = 0, bestpDiff = 0;
    for (let t = 0; t < T; t++) {
      if (bests[t] !== wantBest[t]) mismatches++;
      bestpDiff = Math.max(bestpDiff, Math.abs(bps[t] - wantBp[t]));
    }
    res.push({
      case: man.case, T, C, maxDiff: cmp.maxDiff, badProbs: cmp.bad,
      mismatches, bestpDiff,
    });
  }
  const ok = res.every((r) => r.maxDiff <= TOL && r.badProbs === 0 && r.mismatches === 0 && r.bestpDiff <= TOL);
  return { ok, res };
}
