// results/figure.jpg: top row input | pixel collage | splat collage (kodim23, 12 regions, eval outputs);
// bottom row an 8x zoom of the same 96 x 64 reference-pixel window: the pixel collage enlarged, and the
// splat collage re-evaluated at the zoomed resolution by the shader's CPU mirror.
import fs from "node:fs";
import * as C from "../../web/collage.js";
import { readPPM, writePPM } from "../../run/ppm.mjs";
import * as S from "../splatcollage.js";
const table = JSON.parse(fs.readFileSync("results/table_train.json", "utf8"));
const bin = fs.readFileSync("bank/splats.bin"), all = new Float32Array(bin.buffer, bin.byteOffset, bin.length / 4);
const { W, H, rgba } = readPPM("../data/kodim23.ppm");
const lab = C.rgbToLab(rgba, W * H, 4);
const seg = C.segment(lab, W, H, { K: 12 }), rdesc = C.regionDescriptors(lab, seg);
const pick = C.chooseClasses(rdesc.map((d, k) => ({ desc: d, stats: seg.stats[k], frac: seg.stats[k].frac })), table, { exclude: ["freckled"] });
const tex = pick.map((p) => { const sp = all.subarray((p.cls * 3 + 2) * 512 * 12, (p.cls * 3 + 3) * 512 * 12); return { splats: sp, ...S.describeSplatTile(sp, 512, p.scale) }; });
const sc = S.buildScene({ lab, seg, rdesc, pick, tex, K: 512, LW: 192 });
const x0 = 300, y0 = 110, w = 96, h = 64, f = 8, out = new Uint8ClampedArray(4 * w * f * h * f), c = new Float32Array(3);
for (let y = 0; y < h * f; y++) for (let x = 0; x < w * f; x++) {
  S.evalCPU(sc, x0 + (x + 0.5) / f, y0 + (y + 0.5) / f, 1 / f, c);
  const o = 4 * (y * w * f + x); out[o] = c[0] * 255; out[o + 1] = c[1] * 255; out[o + 2] = c[2] * 255; out[o + 3] = 255;
}
writePPM("out/zoom_splat.ppm", out, w * f, h * f);
fs.writeFileSync("out/zoom_window.json", JSON.stringify({ x0, y0, w, h, f }));
