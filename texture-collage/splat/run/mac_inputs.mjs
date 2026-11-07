// Inputs for bridge job tc-splat-time: the parrots (kodim23) cut into 6, 12 and 24 regions with stored
// textures, as the page builds them, written as one JSON per region count (baked WGSL + storage buffers).
//   node run/mac_inputs.mjs OUTDIR
import fs from "node:fs";
import * as C from "../../web/collage.js";
import { readPPM } from "../../run/ppm.mjs";
import * as S from "../splatcollage.js";
const out = process.argv[2];
const table = JSON.parse(fs.readFileSync("classes.json", "utf8")), meta = JSON.parse(fs.readFileSync("bank_meta.json", "utf8"));
const bank = new Float32Array(fs.readFileSync("bank.bin").buffer.slice(0));
const { W, H, rgba } = readPPM("../data/kodim23.ppm");
const lab = C.rgbToLab(rgba, W * H, 4);
for (const K of [6, 12, 24]) {
  const seg = C.segment(lab, W, H, { K }), rdesc = C.regionDescriptors(lab, seg);
  const pick = C.chooseClasses(rdesc.map((d, k) => ({ desc: d, stats: seg.stats[k], frac: seg.stats[k].frac })), table, { exclude: ["freckled"] });
  const tex = pick.map((p) => ({ splats: bank.subarray(p.cls * 512 * 12, (p.cls + 1) * 512 * 12), ...meta.classes[p.cls] }));
  const sc = S.buildScene({ lab, seg, rdesc, pick, tex, K: 512, LW: 192 });
  const b = S.sceneBuffers(sc);
  fs.writeFileSync(`${out}/scene${K}.json`, JSON.stringify({ K: seg.K, W0: W, H0: H, baked: S.bakeCollage(sc), storage: S.storageShader(),
    sp: Array.from(b.sp), labels: Array.from(b.labels), xf: Array.from(b.xf), uni: Array.from(new Uint32Array(S.sceneUniforms(sc, {}))) }));
  console.log(K, seg.K, fs.statSync(`${out}/scene${K}.json`).size);
}
