// Browser-side parity check for the detector: WGSL forward (through DetNet.run, from raw RGBA
// pixels) against onnxruntime's probability maps in data/fixtures/det_<page>/<model>.ort.f32.
// Run by run_det.mjs, which serves web/ at /web/ and data/fixtures/ at /fix/.
import { Gpu } from "../../gpu.js";
import { DetNet } from "../engine.js";

const THRESH = 0.2;
const TOL_MAX = 2e-3;
const TOL_MISMATCH = 1e-3;

async function bytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

export default async function run({ models, pages }) {
  const gpu = await Gpu.create();
  const out = { adapter: gpu.info, models: {} };
  let ok = true;
  for (const m of models) {
    const det = await DetNet.load(gpu, `/web/det/models/${m}/`);
    await det.run(new Uint8Array(96 * 96 * 4).fill(128), 96, 96, 256); // compile pipelines
    out.models[m] = {};
    for (const p of pages) {
      const meta = JSON.parse(new TextDecoder().decode(await bytes(`/fix/det_${p}/meta.json`)));
      const rgba = await bytes(`/fix/det_${p}/rgba.bin`);
      const ref = new Float32Array((await bytes(`/fix/det_${p}/${m}.ort.f32`)).buffer);
      const t0 = performance.now();
      const r = await det.run(rgba, meta.w, meta.h, meta.long_side);
      const ms = performance.now() - t0;
      if (r.prob.length !== ref.length) throw new Error(`${m} ${p}: ${r.prob.length} values, ref ${ref.length}`);
      let maxAbs = 0, mis = 0;
      for (let i = 0; i < ref.length; i++) {
        const d = Math.abs(r.prob[i] - ref[i]);
        if (d > maxAbs) maxAbs = d;
        if ((r.prob[i] > THRESH) !== (ref[i] > THRESH)) mis++;
      }
      const mismatch = mis / ref.length;
      const pass = maxAbs <= TOL_MAX && mismatch <= TOL_MISMATCH;
      ok = ok && pass;
      out.models[m][p] = { H: r.H, W: r.W, wgsl_max_abs: maxAbs, wgsl_mismatch_share: mismatch,
        swiftshader_ms: Math.round(ms), pass };
      console.log(`${m} ${p} ${r.H}x${r.W} max|d|=${maxAbs.toExponential(2)} mismatch=${mismatch.toExponential(2)} ${Math.round(ms)} ms ${pass ? "ok" : "FAIL"}`);
    }
  }
  return { ok, ...out };
}
