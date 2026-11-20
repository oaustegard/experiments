// lineprep.js against the Python tight crop + flat-field (run/check_lineprep.py fixtures).
// Pass: same output size on every line, and at most 0.5% of pixels differing by more than 2 levels
// (float32 here against float64 in scipy, so a pixel at a rounding edge can land one level off).
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { cleanLine } from "../lineprep.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "../../data/fixtures/lineprep");
const meta = JSON.parse(readFileSync(join(root, "index.json"), "utf8"));
let fails = 0, worst = 0;
for (const m of meta) {
  const data = new Uint8Array(readFileSync(join(root, `${m.name}.in.u8`)));
  const want = new Uint8Array(readFileSync(join(root, `${m.name}.out.u8`)));
  const got = cleanLine({ data, width: m.w, height: m.h });
  if (got.width !== m.ow || got.height !== m.oh) { fails++; console.log(`FAIL ${m.name} size ${got.width}x${got.height} want ${m.ow}x${m.oh}`); continue; }
  let off = 0, maxd = 0;
  for (let i = 0; i < want.length; i++) { const d = Math.abs(got.data[i] - want[i]); if (d > 2) off++; if (d > maxd) maxd = d; }
  const frac = off / want.length; worst = Math.max(worst, frac);
  if (frac > 0.005) { fails++; console.log(`FAIL ${m.name} ${(100 * frac).toFixed(2)}% pixels off by >2 (max ${maxd})`); }
}
console.log(`${meta.length - fails}/${meta.length} lines match; worst ${(100 * worst).toFixed(3)}% pixels off by >2`);
process.exit(fails ? 1 : 0);
