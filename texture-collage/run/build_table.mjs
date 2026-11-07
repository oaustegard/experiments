// Build the class descriptor table from the texture bank (bridge job tc-bank-1).
//   node run/build_table.mjs bank/bank.bin  ->  web/classes.json (all seeds) and results/table_train.json (seeds 0-1)
// bank.bin: uint8, [class][seed][S][S][3]; bank/meta.json gives names, S, seeds.
import fs from "node:fs";
import { describeTile, FEATS } from "../web/collage.js";
const bin = fs.readFileSync(process.argv[2] || "bank/bank.bin");
const meta = JSON.parse(fs.readFileSync("bank/meta.json", "utf8"));
const { names, S, seeds } = meta, C = names.length, SC = [1, 2];
const tiles = [];
for (let c = 0; c < C; c++) {
  tiles.push([]);
  for (let s = 0; s < seeds; s++) {
    const o = (c * seeds + s) * S * S * 3;
    tiles[c].push(describeTile(new Uint8Array(bin.buffer, bin.byteOffset + o, S * S * 3), S, SC));
  }
}
function table(useSeeds) {
  const classes = names.map((n, c) => {
    const ts = useSeeds.map((s) => tiles[c][s]);
    const mean = [0, 1, 2].map((k) => ts.reduce((a, t) => a + t.mean[k], 0) / ts.length);
    const per = {};
    for (const sc of SC) per[sc] = { f: FEATS.map((_, i) => ts.reduce((a, t) => a + t.per[sc].f[i], 0) / ts.length) };
    return { name: n, mean, per };
  });
  // per-feature spread across all class x scale means: the distance unit
  const norm = FEATS.map((_, i) => {
    const v = classes.flatMap((c) => SC.map((sc) => c.per[sc].f[i]));
    const m = v.reduce((a, b) => a + b, 0) / v.length;
    return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length) || 1;
  });
  return { names, S, feats: FEATS, scales: SC, seeds: useSeeds, norm, classes };
}
const r = (x) => JSON.parse(JSON.stringify(x, (k, v) => (typeof v === "number" ? +v.toFixed(4) : v)));
fs.writeFileSync("web/classes.json", JSON.stringify(r(table([...Array(seeds).keys()]))));
fs.writeFileSync("results/table_train.json", JSON.stringify(r(table([0, 1])), null, 0));
// how well does one seed's descriptor identify its class among the 47 (seeds 2-3 against a seeds 0-1 table)?
const tt = table([0, 1]);
let hit1 = 0, hit5 = 0, n = 0;
for (let c = 0; c < C; c++) for (const s of [2, 3].filter((x) => x < seeds)) {
  const f = tiles[c][s].per[1].f;
  const d = tt.classes.map((k, j) => [j, FEATS.reduce((a, _, i) => a + ((f[i] - k.per[1].f[i]) / tt.norm[i]) ** 2, 0)]).sort((a, b) => a[1] - b[1]);
  const rank = d.findIndex(([j]) => j === c);
  hit1 += rank === 0; hit5 += rank < 5; n++;
}
const sep = { top1: hit1 / n, top5: hit5 / n, n, chance_top1: 1 / C, chance_top5: 5 / C };
fs.writeFileSync("results/descriptor_separability.json", JSON.stringify(sep, null, 1));
console.log("separability", sep);
