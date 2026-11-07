// The page's stored textures: seed 0 of each class from the splat bank, with the painted tile's Lab
// stats, grain orientation and coherence (what recolouring and alignment need).
//   node run/page_bank.mjs -> bank.bin (f32, 47 x 512 x 12), bank_meta.json
import fs from "node:fs";
import { describeSplatTile } from "../splatcollage.js";
const bin = fs.readFileSync("bank/splats.bin"), all = new Float32Array(bin.buffer, bin.byteOffset, bin.length / 4);
const C = 47, SEEDS = 3, K = 512, out = new Float32Array(C * K * 12), meta = [];
for (let c = 0; c < C; c++) {
  const sp = all.subarray(c * SEEDS * K * 12, (c * SEEDS + 1) * K * 12);
  out.set(sp, c * K * 12);
  const d = describeSplatTile(sp, K, 1);
  meta.push({ stats: { mean: d.stats.mean.map((v) => +v.toFixed(3)), std: d.stats.std.map((v) => +v.toFixed(3)) },
    theta: +d.theta.toFixed(4), gcoh: +d.gcoh.toFixed(4) });
}
fs.writeFileSync("bank.bin", Buffer.from(out.buffer));
fs.writeFileSync("bank_meta.json", JSON.stringify({ K, seed: 2000, job: "tc-splatbank-1", classes: meta }));
console.log("ok", out.byteLength);
