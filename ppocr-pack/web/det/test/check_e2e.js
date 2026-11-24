// In-page end-to-end run: web/pipeline.js with the real WGSL detector (web/det) and recognizer (web/rec)
// on one fixture page (rgba.bin + meta.json from run/ref_det.py). Run by run_e2e.mjs.
import { Gpu } from "../gpu.js";
import { DetNet } from "../engine.js";
import { RecNet } from "../../rec/engine.js";
import { TextPipeline } from "../../pipeline.js";

async function bytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

export default async function run({ model, page }) {
  const gpu = await Gpu.create();
  const t0 = performance.now();
  const det = await DetNet.load(gpu, `/web/det/models/${model}/`);
  const rec = await RecNet.load(gpu, `/web/rec/models/${model}/`);
  const loadMs = performance.now() - t0;
  const meta = JSON.parse(new TextDecoder().decode(await bytes(`/fix/det_${page}/meta.json`)));
  const rgba = await bytes(`/fix/det_${page}/rgba.bin`);
  const pipe = new TextPipeline({ det, rec });
  const lines = await pipe.run({ data: rgba, width: meta.w, height: meta.h });
  return { adapter: gpu.info, loadMs: Math.round(loadMs), timing: lines.timing,
    lines: lines.map((l) => ({ text: l.text, score: +l.score.toFixed(3), detScore: +l.detScore.toFixed(3), box: l.box })) };
}
