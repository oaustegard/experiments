// Measure the design choices on the six Kodak images (data/*.ppm from run/prep_images.py).
// Textures come from the bridge bank: class choice uses the seeds 0-1 table (results/table_train.json)
// and the collage is filled with the held-out seed-2 tiles, so a choice is judged on a texture the
// table never saw.
//   node run/eval.mjs [K ...]   ->  results/eval.json, results/eval_rows.json, out/*.ppm
import fs from "node:fs";
import * as C from "../web/collage.js";
import * as M from "./metrics.mjs";
import { readPPM, writePPM } from "./ppm.mjs";

const Ks = process.argv.slice(2).map(Number).filter(Boolean);
const KS = Ks.length ? Ks : [6, 12];
const meta = JSON.parse(fs.readFileSync("bank/meta.json", "utf8"));
const bin = fs.readFileSync("bank/bank.bin");
const table = JSON.parse(fs.readFileSync("results/table_train.json", "utf8"));
const { S, seeds } = meta;
const EVAL_SEED = 2;
const tileRGB = (c, s) => new Uint8Array(bin.buffer, bin.byteOffset + (c * seeds + s) * S * S * 3, S * S * 3);
const prepCache = new Map();
const prep = (c, sc) => { const k = `${c}:${sc}`; if (!prepCache.has(k)) prepCache.set(k, C.prepTile(tileRGB(c, EVAL_SEED), S, sc)); return prepCache.get(k); };

const DEF = { mode: "structure", distinct: false, tint: "region", feather: 2, align: true, strength: 8, scales: [1, 2], seed: 1 };
const ARMS = {
  "structure": {},
  "colour": { mode: "colour" },
  "random": { mode: "random", reps: [1, 2, 3] },
  "fixed:marbled": { mode: "fixed:marbled" },
  "structure+distinct": { distinct: true },
  "tint:none structure": { tint: "none" },
  "tint:none colour": { tint: "none", mode: "colour" },
  "tint:local": { tint: "local" },
  "feather:0": { feather: 0 },
  "feather:6": { feather: 6 },
  "align:off": { align: false },
  "scale:1": { scales: [1] },
  "scale:2": { scales: [2] },
  "strength:0": { strength: 0 },
  "strength:16": { strength: 16 },
  "gain:uncapped": { maxGainL: 1e9, maxGainAB: 1e9 },
};
const SAVE = new Set(["structure", "colour", "random", "tint:local", "tint:none colour", "feather:0", "solid"]);
fs.mkdirSync("out", { recursive: true });

const rows = [];
const files = fs.readdirSync("data").filter((f) => f.endsWith(".ppm") && (!process.env.ONLY || f.startsWith(process.env.ONLY))).sort();
const t00 = performance.now();
for (const f of files) {
  const name = f.replace(".ppm", "");
  const { W, H, rgba } = readPPM(`data/${f}`);
  const N = W * H, lab = C.rgbToLab(rgba, N, 4);
  for (const K of KS) {
    let t = performance.now();
    const seg = C.segment(lab, W, H, { K });
    const tSeg = performance.now() - t; t = performance.now();
    const rdesc = C.regionDescriptors(lab, seg);
    const tDesc = performance.now() - t;
    const regions = rdesc.map((d, k) => ({ desc: d, stats: seg.stats[k], frac: seg.stats[k].frac }));
    const band = M.boundaryBand(seg.labels, W, H, 3);
    const rB = Math.round(Math.max(W, H) / 100);
    const score = (out) => ({
      dE: M.deltaEBlur(lab, out, W, H, rB), ssim: M.ssimL(lab, out, W, H), gmsd: M.gmsd(lab, out, W, H),
      gmsdEdge: M.gmsd(lab, out, W, H, band), dEreg: M.regionColourError(lab, out, seg), ...M.structure(lab, out, seg, table.norm) });
    // solid-colour baseline
    const solid = C.renderSolid(seg);
    rows.push({ image: name, K, arm: "solid", segK: seg.K, ...score(solid) });
    writePPM(`out/${name}-K${K}-solid.ppm`, C.labToRgba(solid, N), W, H);
    for (const [arm, o] of Object.entries(ARMS)) {
      const reps = o.reps || [DEF.seed];
      for (const seed of reps) {
        const opt = { ...DEF, ...o, seed };
        const pick = C.chooseClasses(regions, table, opt);
        const tex = pick.map((p) => prep(p.cls, p.scale));
        t = performance.now();
        const out = C.render(lab, seg, tex, { ...opt, scales: pick.map((p) => p.scale), rdesc });
        const tRender = performance.now() - t;
        const sc = score(out);
        rows.push({ image: name, K, arm, rep: seed, segK: seg.K, classes: pick.map((p) => table.names[p.cls]),
          scalesUsed: pick.map((p) => p.scale), tSeg, tDesc, tRender, ...sc });
        if (SAVE.has(arm) && seed === reps[0]) writePPM(`out/${name}-K${K}-${arm.replace(/[: ]/g, "_")}.ppm`, C.labToRgba(out, N), W, H);
      }
    }
    console.log(`${name} K=${K} done, ${((performance.now() - t00) / 1000).toFixed(0)} s`);
  }
}
// aggregate: mean over images (and random reps), per K and arm
const agg = {};
const keys = ["dE", "dEreg", "ssim", "gmsd", "gmsdEdge", "dist", "orientErr", "tRender"];
for (const r of rows) {
  const a = (agg[`K${r.K}`] ??= {})[r.arm] ??= { n: 0 };
  a.n++;
  for (const k of keys) if (Number.isFinite(r[k])) { a[k] = (a[k] || 0) + r[k]; a[`n_${k}`] = (a[`n_${k}`] || 0) + 1; }
}
for (const K of Object.values(agg)) for (const a of Object.values(K)) {
  for (const k of keys) if (a[`n_${k}`]) { a[k] = +(a[k] / a[`n_${k}`]).toFixed(4); delete a[`n_${k}`]; }
}
fs.writeFileSync("results/eval_rows.json", JSON.stringify(rows));
fs.writeFileSync("results/eval.json", JSON.stringify({ images: files.length, Ks: KS, evalSeed: EVAL_SEED, defaults: DEF, arms: agg }, null, 1));
console.log(JSON.stringify(agg, null, 1));
