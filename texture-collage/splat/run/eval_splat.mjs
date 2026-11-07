// Measure the splat collage on the six Kodak pictures (../data/*.ppm from ../run/prep_images.py), the
// same fixture as the pixel collage. Classes are chosen with the seeds 0-1 splat table and filled with
// held-out seed-2 splats; every collage is rendered at 768 px by evalCPU, the shader's CPU mirror.
//   node run/eval_splat.mjs [K ...] -> results/eval.json, results/eval_rows.json, results/geometry.json, out/*.ppm
import fs from "node:fs";
import zlib from "node:zlib";
import * as C from "../../web/collage.js";
import * as M from "../../run/metrics.mjs";
import { readPPM, writePPM } from "../../run/ppm.mjs";
import * as S from "../splatcollage.js";

const KS = process.argv.slice(2).map(Number).filter(Boolean);
const Ks = KS.length ? KS : [6, 12];
const table = JSON.parse(fs.readFileSync("results/table_train.json", "utf8"));
const bin = fs.readFileSync("bank/splats.bin"), all = new Float32Array(bin.buffer, bin.byteOffset, bin.length / 4);
const NK = 512, SEEDS = 3, EVAL_SEED = 2;
const texCache = new Map();
function tex(cls, scale) {
  const key = `${cls}:${scale}`;
  if (!texCache.has(key)) {
    const sp = all.subarray((cls * SEEDS + EVAL_SEED) * NK * 12, (cls * SEEDS + EVAL_SEED + 1) * NK * 12);
    const d = S.describeSplatTile(sp, NK, scale);
    texCache.set(key, { splats: sp, stats: d.stats, theta: d.theta, gcoh: d.gcoh });
  }
  return texCache.get(key);
}
const DEF = { LW: 192, mode: "distance", tint: "region", align: true, blend: 0 };
const ARMS = {
  "distance LW192": {},
  "distance LW96": { LW: 96 },
  "spline LW192": { mode: "spline" },
  "contour LW192": { mode: "contour" },
  "nearest LW192": { mode: "nearest" },
  "tint:none": { tint: "none" },
  "align:off": { align: false },
};
const SAVE = new Set(["distance LW192", "nearest LW192", "spline LW192"]);
fs.mkdirSync("out", { recursive: true });
const files = fs.readdirSync("../data").filter((f) => f.endsWith(".ppm") && (!process.env.ONLY || f.startsWith(process.env.ONLY))).sort();
const rows = [], geom = [];
const t00 = performance.now();
for (const f of files) {
  const name = f.replace(".ppm", "");
  const { W, H, rgba } = readPPM(`../data/${f}`);
  const N = W * H, lab = C.rgbToLab(rgba, N, 4);
  for (const K of Ks) {
    const seg = C.segment(lab, W, H, { K });
    const rdesc = C.regionDescriptors(lab, seg);
    const regions = rdesc.map((d, k) => ({ desc: d, stats: seg.stats[k], frac: seg.stats[k].frac }));
    const pick = C.chooseClasses(regions, table, { mode: "structure", exclude: ["freckled"] });
    const band = M.boundaryBand(seg.labels, W, H, 3);
    const rB = Math.round(Math.max(W, H) / 100);
    // geometry: how well each label map reproduces the full-resolution regions, and the baked file's size
    for (const LW of [64, 96, 128, 192, 256]) { const lm = S.labelMap(seg, LW); for (const mode of ["nearest", "contour", "spline", "distance"]) {
      const sc = S.makeScene({ W0: W, H0: H, lm, regions: pick.map(() => ({ splats: new Float32Array(NK * 12), phi: 0, scale: 1, off: [0, 0] })), K: NK, mode });
      let wrong = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (S.dominantLabel(sc, x + 0.5, y + 0.5) !== seg.labels[y * W + x]) wrong++;
      const g = { image: name, K, LW, mode, LH: sc.LH, wrongFrac: wrong / N };
      if (mode === "spline" || mode === "distance") {
        const src = S.bakeCollage(S.buildScene({ lab, seg, rdesc, pick, tex: pick.map((p) => tex(p.cls, p.scale)), K: NK, LW, mode }));
        g.bytes = Buffer.byteLength(src); g.gzip = zlib.gzipSync(src, { level: 9 }).length;
        g.labelWords = sc.packed.words.length; g.bits = sc.packed.bits;
      }
      geom.push(g);
    } }
    for (const [arm, o] of Object.entries(ARMS)) {
      const opt = { ...DEF, ...o };
      const t = performance.now();
      const scene = S.buildScene({ lab, seg, rdesc, pick, tex: pick.map((p) => tex(p.cls, p.scale)), K: NK, ...opt });
      const rgbaOut = S.renderCPU(scene, W, H);
      const ms = performance.now() - t;
      const out = C.rgbToLab(rgbaOut, N, 4);
      rows.push({ image: name, K, arm, segK: seg.K, classes: pick.map((p) => table.names[p.cls]), renderMs: ms,
        dE: M.deltaEBlur(lab, out, W, H, rB), dEreg: M.regionColourError(lab, out, seg), ssim: M.ssimL(lab, out, W, H),
        gmsd: M.gmsd(lab, out, W, H), gmsdEdge: M.gmsd(lab, out, W, H, band), ...M.structure(lab, out, seg, table.norm) });
      if (SAVE.has(arm)) writePPM(`out/${name}-K${K}-${arm.replace(/[: ]/g, "_")}.ppm`, rgbaOut, W, H);
    }
    console.log(`${name} K=${K} done, ${((performance.now() - t00) / 1000).toFixed(0)} s`);
  }
}
const keys = ["dE", "dEreg", "ssim", "gmsd", "gmsdEdge", "dist", "orientErr", "renderMs"];
const agg = {};
for (const r of rows) {
  const a = agg[r.arm] ??= { n: 0 };
  a.n++;
  for (const k of keys) if (Number.isFinite(r[k])) { a[k] = (a[k] || 0) + r[k]; a[`_${k}`] = (a[`_${k}`] || 0) + 1; }
}
for (const a of Object.values(agg)) for (const k of keys) if (a[`_${k}`]) { a[k] = +(a[k] / a[`_${k}`]).toFixed(4); delete a[`_${k}`]; }
const gagg = {};
for (const g of geom) {
  const key = `${g.mode} LW${g.LW}`, a = gagg[key] ??= { n: 0, wrongFrac: 0 };
  a.n++; a.wrongFrac += g.wrongFrac;
  if (g.bytes) for (const K of [6, 12]) if (g.K === K) { (a[`K${K}`] ??= { bytes: 0, gzip: 0, n: 0 }); a[`K${K}`].bytes += g.bytes; a[`K${K}`].gzip += g.gzip; a[`K${K}`].n++; }
}
for (const a of Object.values(gagg)) {
  a.wrongFrac = +(a.wrongFrac / a.n).toFixed(5);
  for (const K of [6, 12]) if (a[`K${K}`]) { const b = a[`K${K}`]; a[`K${K}`] = { bytes: Math.round(b.bytes / b.n), gzip: Math.round(b.gzip / b.n) }; }
}
fs.writeFileSync("results/eval_rows.json", JSON.stringify(rows));
fs.writeFileSync("results/geometry_rows.json", JSON.stringify(geom));
fs.writeFileSync("results/eval.json", JSON.stringify({ images: files.length, Ks, evalSeed: EVAL_SEED, defaults: DEF, arms: agg, geometry: gagg }, null, 1));
console.log(JSON.stringify({ arms: agg, geometry: gagg }, null, 1));
