// web/pipeline.js against run/ref_pipeline.py, in Node (no GPU): both read the same probability map
// (fixtures/det_<page>/<model>.ref.f32). Per page and model it checks
//   raw boxes     DBPostProcess output in page pixels: same count, corners within 1 px, score difference
//   final boxes   after filtering and sorting: same count, same order, corners within 1 px
//   crops         size equal; pixel difference against cv2.warpPerspective(INTER_CUBIC)
//   recognizer input  recInput() on the Python crops against resize_norm_img (first 4 crops)
//   TextPipeline  run end to end with a stub detector (the same map) and a stub recognizer; every line
//                 must get its own crop back through the batching
// and merges the numbers into results/pipeline_parity.json (key js_vs_python). Exit 0 when all pass.
//
//   python3 run/ref_det.py && python3 run/ref_pipeline.py && node web/det/test/check_pipeline.mjs [tiny] [small] [page ...]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as P from "../../pipeline.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "../../..");
const FIX = path.join(HERE, "data/fixtures");
const RESULTS = path.join(ROOT, "results/pipeline_parity.json");
const args = process.argv.slice(2);
const models = args.filter((a) => a === "tiny" || a === "small");
const wantModels = models.length ? models : ["tiny", "small"];
const pages = args.filter((a) => a !== "tiny" && a !== "small");
const allPages = fs.readdirSync(FIX).filter((d) => d.startsWith("det_")).map((d) => d.slice(4)).sort();
const wantPages = pages.length ? pages : allPages;

const TOL_PX = 1, TOL_REC = 0.02;
const f32 = (f) => { const b = fs.readFileSync(f); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };

// largest corner distance (max of |dx|, |dy|) between two quads, both put in clockwise order first
function quadDev(a, b) {
  const A = P.orderPointsClockwise(a), B = P.orderPointsClockwise(b);
  let d = 0;
  for (let i = 0; i < 4; i++) d = Math.max(d, Math.abs(A[i][0] - B[i][0]), Math.abs(A[i][1] - B[i][1]));
  return d;
}

// ctcConfidence on a hand-made [B=2, T=6, V=4] map: line 0 reads classes 1,1,0,2,2,3 (steps emitting a
// character: 0, 3, 5), line 1 is all blank
{
  const T = 6, V = 4, probs = new Float32Array(2 * T * V);
  const set = (b, t, k, p) => { for (let v = 0; v < V; v++) probs[(b * T + t) * V + v] = v === k ? p : (1 - p) / 3; };
  [[1, 0.5], [1, 0.75], [0, 0.9], [2, 0.4], [2, 0.5], [3, 1]].forEach(([k, p], t) => set(0, t, k, p));
  for (let t = 0; t < T; t++) set(1, t, 0, 0.9);
  const a = P.ctcConfidence(probs, T, V, 0), b = P.ctcConfidence(probs, T, V, 1);
  if (Math.abs(a - (0.5 + 0.4 + 1) / 3) > 1e-6 || b !== 0) { console.log("ctcConfidence FAIL", a, b); process.exit(1); }
}

const out = {};
let ok = true;
for (const model of wantModels) {
  out[model] = {};
  for (const page of wantPages) {
    const dir = path.join(FIX, `det_${page}`);
    const ref = JSON.parse(fs.readFileSync(path.join(dir, `${model}.pipe.json`), "utf8"));
    const prob = f32(path.join(dir, `${model}.ref.f32`));
    const rgba = new Uint8Array(fs.readFileSync(path.join(dir, "rgba.bin")));
    const rec = {};

    // raw boxes
    const t0 = performance.now();
    const raw = P.toPage(P.dbPostProcess(prob, ref.W, ref.H, ref.post), ref.w1, ref.h1, ref.w, ref.h);
    rec.post_ms = Math.round(performance.now() - t0);
    rec.raw_count = [raw.length, ref.raw_count];
    let rawDev = 0, scoreDev = 0;
    if (raw.length === ref.raw_count) {
      raw.forEach((b, i) => {
        rawDev = Math.max(rawDev, quadDev(b.quad, ref.raw[i]));
        scoreDev = Math.max(scoreDev, Math.abs(b.score - ref.raw_scores[i]));
      });
    }
    rec.raw_max_corner_px = rawDev;
    rec.raw_max_score_diff = scoreDev;

    // final boxes
    const final = P.sortedBoxes(P.filterBoxes(raw, ref.w, ref.h));
    rec.boxes = [final.length, ref.boxes.length];
    let boxDev = 0;
    if (final.length === ref.boxes.length) final.forEach((b, i) => { boxDev = Math.max(boxDev, quadDev(b.quad, ref.boxes[i])); });
    rec.max_corner_px = boxDev;
    const countsOk = raw.length === ref.raw_count && final.length === ref.boxes.length;

    // crops
    const refCrops = fs.readFileSync(path.join(dir, `${model}.crops.bin`));
    let off = 0, sizeBad = 0, sumAbs = 0, nPix = 0, maxAbs = 0, worstMean = 0, jsCrops = [];
    if (countsOk) {
      final.forEach((b, i) => {
        const c = P.cropQuad(rgba, ref.w, ref.h, b.quad);
        jsCrops.push(c);
        const r = ref.crops[i], n = 3 * r.w * r.h;
        if (c.w !== r.w || c.h !== r.h) { sizeBad++; off += n; return; }
        let s = 0;
        for (let k = 0; k < n; k++) { const d = Math.abs(c.data[k] - refCrops[off + k]); s += d; if (d > maxAbs) maxAbs = d; }
        sumAbs += s; nPix += n; worstMean = Math.max(worstMean, s / n);
        off += n;
      });
    }
    rec.crop_size_mismatch = sizeBad;
    rec.crop_mean_abs = nPix ? sumAbs / nPix : null;
    rec.crop_worst_mean_abs = worstMean;
    rec.crop_max_abs = maxAbs;

    // recognizer input on the Python crops (so a crop difference does not mask a preprocessing one)
    const recin = fs.existsSync(path.join(dir, `${model}.recin.f32`)) ? f32(path.join(dir, `${model}.recin.f32`)) : new Float32Array(0);
    let recDev = 0, recWidthBad = 0, at = 0, recOff = 0;
    for (let i = 0; i < ref.crops.length && at < recin.length; i++) {
      const r = ref.crops[i];
      const n = 3 * r.w * r.h, data = new Uint8Array(refCrops.buffer, refCrops.byteOffset + recOff, n);
      recOff += n;
      const { x, W, widths } = P.recInput([{ data, w: r.w, h: r.h }]);
      if (W !== r.W || widths[0] !== r.resized_w) recWidthBad++;
      const len = 3 * 48 * r.W;
      if (x.length === len) for (let k = 0; k < len; k++) recDev = Math.max(recDev, Math.abs(x[k] - recin[at + k]));
      at += len;
    }
    rec.rec_input_max_abs = recDev;
    rec.rec_width_mismatch = recWidthBad;

    // TextPipeline end to end, stub detector and recognizer
    let pipeOk = null;
    if (countsOk && final.length) {
      const det = { post: ref.post, run: async () => ({ prob, H: ref.H, W: ref.W, validH: ref.h1, validW: ref.w1 }) };
      // run() hands the recognizer x [B, 3, 48, W]; the stub answers with the sum of each item's first plane
      const recog = {
        calls: [],
        async run(x, B, W) {
          this.calls.push([B, W]);
          const plane = 3 * 48 * W, probs = new Float32Array(B);
          for (let b = 0; b < B; b++) { let s = 0; for (let k = 0; k < 48 * W; k++) s += x[b * plane + k]; probs[b] = s; }
          return { T: 1, V: 1, probs };
        },
        decode(probs, T, V, B) { return Array.from({ length: B }, (_, b) => ({ text: probs[b].toFixed(2), score: 0.9 })); },
      };
      // every line must come back with its own crop, whatever the batch size; the expected text is
      // the sum computed from batches formed here (narrow lines first, slices of the batch size)
      pipeOk = true;
      for (const batch of [1, 6]) {
        recog.calls.length = 0;
        const pipe = new P.TextPipeline({ det, rec: recog, recBatch: batch });
        const lines = await pipe.run({ data: rgba, width: ref.w, height: ref.h });
        if (lines.length !== final.length || recog.calls.some(([B]) => B > batch)) pipeOk = false;
        const ids = jsCrops.map((_, i) => i).sort((p, q) => jsCrops[p].w / jsCrops[p].h - jsCrops[q].w / jsCrops[q].h);
        const expect = new Array(jsCrops.length);
        for (let s0 = 0; s0 < ids.length; s0 += batch) {
          const grp = ids.slice(s0, s0 + batch);
          const { x, W } = P.recInput(grp.map((i) => jsCrops[i]));
          grp.forEach((id, k) => { let t = 0; for (let j = 0; j < 48 * W; j++) t += x[k * 3 * 48 * W + j]; expect[id] = Math.fround(t).toFixed(2); });
        }
        lines.forEach((ln, i) => {
          if (ln.text !== expect[i]) pipeOk = false;
          if (JSON.stringify(ln.box) !== JSON.stringify(final[i].quad) || ln.score !== 0.9 || ln.detScore !== final[i].score) pipeOk = false;
        });
        rec[`pipeline_batches_${batch}`] = recog.calls.length;
      }
    }
    rec.pipeline_ok = pipeOk;

    rec.pass = countsOk && rawDev <= TOL_PX && boxDev <= TOL_PX && sizeBad === 0 && recWidthBad === 0 &&
      recDev <= TOL_REC && (pipeOk === null || pipeOk);
    ok = ok && rec.pass;
    out[model][page] = rec;
    console.log(`${model.padEnd(5)} ${page.padEnd(16)} raw ${rec.raw_count.join("/")} boxes ${rec.boxes.join("/")} ` +
      `corner ${boxDev}px (raw ${rawDev}px) score ${scoreDev.toExponential(1)} crop mean ${rec.crop_mean_abs?.toFixed(2)} ` +
      `max ${maxAbs} rec ${recDev.toExponential(1)} pipeline ${pipeOk} ${rec.post_ms} ms ${rec.pass ? "ok" : "FAIL"}`);
  }
}
const res = fs.existsSync(RESULTS) ? JSON.parse(fs.readFileSync(RESULTS, "utf8")) : {};
res.js_vs_python = { ...(res.js_vs_python || {}) };
for (const m of Object.keys(out)) res.js_vs_python[m] = { ...(res.js_vs_python[m] || {}), ...out[m] };
res.js_tolerance = { corner_px: TOL_PX, rec_input_max_abs: TOL_REC };
fs.writeFileSync(RESULTS, JSON.stringify(res, null, 1) + "\n");
console.log(ok ? "pipeline parity pass" : "pipeline parity FAIL");
process.exit(ok ? 0 : 1);
