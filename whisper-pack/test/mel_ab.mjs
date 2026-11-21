// old vs new logMel: bit-identical output and timing. usage: node test/mel_ab.mjs <path to the old mel.js>
import { readFileSync } from "node:fs";
const NEW = await import(new URL("../web/mel.js", import.meta.url));
const OLD = await import(process.argv[2]);
const F = new URL("fixtures/", import.meta.url).pathname;
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4); };
const clips = { utt7s: f32(F + "engine/utt0.f32"), clip28: f32(F + "engine/clip28.f32"), clip32: f32(F + "ts/clip3.f32"),
  silence: new Float32Array(16000), one: new Float32Array(1).fill(0.3), exact30: f32(F + "ts/clip3.f32").subarray(0, 480000),
  edge: f32(F + "ts/clip3.f32").subarray(0, 479900) };
let bad = 0;
for (const [name, pcm] of Object.entries(clips)) for (const n of [80, 128]) {
  OLD.logMel(pcm, n); NEW.logMel(pcm, n);   // warm
  let t = performance.now(); const a = OLD.logMel(pcm, n); const to = performance.now() - t;
  t = performance.now(); const b = NEW.logMel(pcm, n); const tn = performance.now() - t;
  let diff = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
  if (diff) bad++;
  console.log(`${name.padEnd(8)} mel${n}  differing values ${diff}  old ${to.toFixed(0)} ms  new ${tn.toFixed(0)} ms`);
}
process.exit(bad ? 1 : 0);
