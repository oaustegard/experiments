// Browser-side parity check for the recognizer, run by run_rec.mjs (which serves ppocr-pack/web/ at /web/).
// For each model and variant, every fixture line goes through RecNet.run one line per call and is
// compared with the reference probabilities that run/ref_rec.py wrote next to it:
//   packed  web/rec/models/<m>/      vs <line>/<m>.ref.f32  (NumPy, same dequantised weights)
//   f32     web/rec/models/<m>-f32/  vs <line>/<m>.ort.f32  (onnxruntime on the fp32 ONNX)
// The packed variant also runs the 4-line batch (zero-padded to the widest) in one call.
import { RecNet, requestDevice } from "../engine.js";

const FIX = "/web/rec/test/fixtures";

async function f32(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Float32Array(await r.arrayBuffer());
}

function maxAbs(a, b) {
  if (a.length !== b.length) throw new Error(`length ${a.length} vs reference ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m || d !== d) m = d !== d ? Infinity : d; }
  return m;
}

export default async function run({ models, lines, batch, variants, pack = "" }) {
  const device = await requestDevice();
  let lost = null;
  device.lost.then((e) => { lost = e.message; console.log(`device lost: ${e.message}`); });
  const info = device.adapterInfo || {};
  const out = { adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description }, models: {} };
  for (const m of models) {
    out.models[m] = {};
    for (const v of variants) {
      const dir = v === "packed" ? (pack ? `${m}-${pack}` : m) : `${m}-f32`;
      const refTag = v === "packed" ? "ref" : "ort";
      const t0 = performance.now();
      const rec = await RecNet.load(device, `/web/rec/models/${dir}/`);
      await rec.run(new Float32Array(3 * 48 * 64), 1, 64);   // compile the pipelines
      const res = { load_and_warm_ms: Math.round(performance.now() - t0), lines: {} };
      for (const e of lines) {
        const x = await f32(`${FIX}/${e.id}/x.f32`);
        const t1 = performance.now();
        const r = await rec.run(x, 1, e.W);
        const ms = performance.now() - t1;
        const ref = await f32(`${FIX}/${e.id}/${v === "packed" ? dir : m}.${refTag}.f32`);
        const text = rec.decode(r)[0], refText = rec.decode(ref, r.T)[0];
        const d = maxAbs(r.probs, ref);
        res.lines[e.id] = { T: r.T, max_abs: d, text, ref_text: refText, same_text: text === refText, swiftshader_ms: Math.round(ms) };
        console.log(`${m} ${v} ${e.id} W=${e.W} T=${r.T} max|d|=${d.toExponential(2)} ${text === refText ? "same" : "DIFF"} ${Math.round(ms)} ms ${JSON.stringify(text)}`);
        if (lost) throw new Error(`device lost: ${lost}`);
      }
      if (v === "packed" && batch) {
        const x = await f32(`${FIX}/${batch.id}/x.f32`);
        const t1 = performance.now();
        const r = await rec.run(x, batch.B, batch.W);
        const ms = performance.now() - t1;
        const ref = await f32(`${FIX}/${batch.id}/${dir}.ref.f32`);
        const texts = rec.decode(r), refTexts = rec.decode(ref, r.T);
        res.batch = { B: batch.B, W: batch.W, T: r.T, max_abs: maxAbs(r.probs, ref), texts, ref_texts: refTexts,
          same_text: texts.every((t, i) => t === refTexts[i]), swiftshader_ms: Math.round(ms) };
        console.log(`${m} ${v} batch B=${batch.B} W=${batch.W} max|d|=${res.batch.max_abs.toExponential(2)} ${res.batch.same_text ? "same" : "DIFF"}`);
      }
      rec.destroy();
      res.dir = dir;
      out.models[m][v] = res;
    }
  }
  return out;
}
