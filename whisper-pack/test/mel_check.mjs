import { readFileSync } from "node:fs";
import { logMel, melFilters } from "../web/mel.js";
const dir = new URL("./fixtures/", import.meta.url);
const f32 = (n) => { const b = readFileSync(new URL(n, dir)); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const f64 = (n) => { const b = readFileSync(new URL(n, dir)); return new Float64Array(b.buffer, b.byteOffset, b.byteLength / 8); };
let fail = false;
for (const n of [80, 128]) {
  const hf = f64(`filters${n}.f64`), ours = melFilters(n); // hf [freq, mel]
  let d = 0; for (let m = 0; m < n; m++) for (let k = 0; k < 201; k++) d = Math.max(d, Math.abs(hf[k * n + m] - ours[m * 201 + k]));
  console.log(`melFilters ${n}: maxdiff ${d.toExponential(2)}`); if (d > 1e-6) fail = true;
}
for (const c of ["utt0", "utt1", "utt2", "long35", "silence_sine"]) {
  const pcm = f32(`${c}.f32`);
  for (const n of [80, 128]) {
    const ref = f32(`${c}.mel${n}.f32`);
    const t0 = performance.now(); const o = logMel(pcm, n); const ms = performance.now() - t0;
    let d = 0; for (let i = 0; i < ref.length; i++) d = Math.max(d, Math.abs(ref[i] - o[i]));
    console.log(`${c.padEnd(13)} mel${n}  maxdiff ${d.toExponential(2)}  ${ms.toFixed(0)} ms`);
    if (!(d <= 2e-4)) fail = true;
  }
}
process.exit(fail ? 1 : 0);
