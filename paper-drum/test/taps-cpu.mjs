// What the tap detector costs the audio thread: 20 s of noise through TapDetector, with and without the echo
// canceller, in Node on this container's CPU (V8, as Chrome's worklet). Writes ../results/taps-cpu.json.
import { TapDetector } from "../web/taps.js";
import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url)), sr = 48000, secs = 20;
const mic = new Float32Array(128).map(() => 0.01 * (Math.random() - 0.5)), out = new Float32Array(128).map(() => 0.1 * (Math.random() - 0.5));
const res = { secs };
for (const aecMs of [0, 80, 120]) {
  const d = new TapDetector(sr, { aecMs }), t0 = performance.now();
  for (let c = 0; c < secs * sr; c += 128) d.process(mic, out);
  const ms = performance.now() - t0;
  res[`aec${aecMs}`] = { msPerAudioSecond: +(ms / secs).toFixed(2), coreShare: +(ms / secs / 1000).toFixed(4) };
}
fs.writeFileSync(path.join(HERE, "..", "results", "taps-cpu.json"), JSON.stringify(res, null, 1));
console.log(JSON.stringify(res));
