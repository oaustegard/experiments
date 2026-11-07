// Class descriptor table for the splat bank (bridge job tc-splatbank-1), as web/classes.json is for the
// pixel bank: each tile painted on the CPU at 128 px and described at scales 1 and 2.
//   node run/splat_table.mjs -> classes.json (all seeds), results/table_train.json (seeds 0-1),
//   results/descriptor_separability.json, results/bank_sheet.ppm (seed 0 of each class)
import fs from "node:fs";
import { FEATS } from "../../web/collage.js";
import { describeSplatTile } from "../splatcollage.js";
const names = JSON.parse(fs.readFileSync("../web/classes.json", "utf8")).names;
const bin = fs.readFileSync("bank/splats.bin"), all = new Float32Array(bin.buffer, bin.byteOffset, bin.length / 4);
const C = 47, SEEDS = 3, K = 512, SC = [1, 2];
const t0 = performance.now();
const tiles = [];
for (let c = 0; c < C; c++) {
  tiles.push([]);
  for (let s = 0; s < SEEDS; s++) {
    const sp = all.subarray((c * SEEDS + s) * K * 12, (c * SEEDS + s + 1) * K * 12);
    const per = {};
    let base;
    for (const sc of SC) { const d = describeSplatTile(sp, K, sc); per[sc] = { f: d.f }; if (sc === 1) base = d; }
    tiles[c].push({ mean: base.stats.mean, per, rgb: base.rgb });
  }
}
console.log("painted and described", C * SEEDS, "tiles in", ((performance.now() - t0) / 1000).toFixed(1), "s");
function table(use) {
  const classes = names.map((n, c) => {
    const ts = use.map((s) => tiles[c][s]);
    const per = {};
    for (const sc of SC) per[sc] = { f: FEATS.map((_, i) => ts.reduce((a, t) => a + t.per[sc].f[i], 0) / ts.length) };
    return { name: n, mean: [0, 1, 2].map((k) => ts.reduce((a, t) => a + t.mean[k], 0) / ts.length), per };
  });
  const norm = FEATS.map((_, i) => {
    const v = classes.flatMap((c) => SC.map((sc) => c.per[sc].f[i])).filter(Number.isFinite);
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length) || 1;
  });
  return { names, S: 128, K, feats: FEATS, scales: SC, seeds: use, norm, classes };
}
const r4 = (x) => JSON.parse(JSON.stringify(x, (k, v) => (typeof v === "number" ? +v.toFixed(4) : v)));
fs.writeFileSync("classes.json", JSON.stringify(r4(table([0, 1, 2]))));
const tt = table([0, 1]);
fs.writeFileSync("results/table_train.json", JSON.stringify(r4(tt)));
let h1 = 0, h5 = 0;
for (let c = 0; c < C; c++) {
  const f = tiles[c][2].per[1].f;
  const d = tt.classes.map((k, j) => [j, FEATS.reduce((a, _, i) => a + (Number.isFinite(f[i]) ? ((f[i] - k.per[1].f[i]) / tt.norm[i]) ** 2 : 0), 0)]).sort((a, b) => a[1] - b[1]);
  const rank = d.findIndex(([j]) => j === c);
  h1 += rank === 0; h5 += rank < 5;
}
fs.writeFileSync("results/descriptor_separability.json", JSON.stringify({ top1: h1 / C, top5: h5 / C, n: C, chance_top1: 1 / C }, null, 1));
console.log("separability top1", h1 / C, "top5", h5 / C);
// contact sheet: seed 0, 8 per row
const S = 128, cols = 8, rows = Math.ceil(C / cols), W = cols * S, H = rows * S, img = Buffer.alloc(W * H * 3);
for (let c = 0; c < C; c++) {
  const rgb = tiles[c][0].rgb, x0 = (c % cols) * S, y0 = Math.floor(c / cols) * S;
  for (let y = 0; y < S; y++) img.set(rgb.subarray(y * S * 3, (y + 1) * S * 3), ((y0 + y) * W + x0) * 3);
}
fs.writeFileSync("results/bank_sheet.ppm", Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`), img]));
