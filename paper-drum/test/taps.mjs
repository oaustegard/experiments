// The tap detector (web/taps.js) on synthetic microphone audio, closed loop: every detected tap starts a drum strike
// that is played (the `out` the detector sees) and comes back into the mic through a simulated speaker: a soft
// clipper (a small speaker's distortion, which no linear canceller can remove), a direct path and two reflections,
// delayed. The page's start-up noise burst plays first, as on the page. Taps are short noise bursts with a low thump, at random times and loudness, over background noise.
//   node test/taps.mjs            writes ../results/taps.json
// Scored: a tap counts as found when an onset lands within 15 ms after it starts; any other onset is a false one.
import { TapDetector, trainingNoise } from "../web/taps.js";
import fs from "node:fs"; import path from "node:path"; import { fileURLToPath } from "node:url";
const HERE = path.dirname(fileURLToPath(import.meta.url));
const sr = 48000;
let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

function scenario({ name, secs = 30, noiseDb = -60, tapDb = [-36, -14], gap = [0.18, 0.9], echoGain = 0.2, echoMs = 28, toldMs = echoMs, drumMs = 35, ring = 0.8, aecMs = 120, drive = 1.5, train = true }) {
  const N = secs * sr, mic = new Float32Array(N), taps = [];
  for (let i = 0; i < N; i++) mic[i] = 10 ** (noiseDb / 20) * gauss();
  for (let t = 1.0 * sr; t < N - sr; t += Math.round((gap[0] + (gap[1] - gap[0]) * rnd()) * sr)) {
    const a = 10 ** ((tapDb[0] + (tapDb[1] - tapDb[0]) * rnd()) / 20), f0 = 120 + 80 * rnd();
    for (let k = 0; k < 0.012 * sr; k++) mic[t + k] += a * Math.exp(-k / (0.0025 * sr)) * (0.6 * gauss() + 0.8 * Math.sin(2 * Math.PI * f0 * k / sr));
    taps.push({ n: t, db: 20 * Math.log10(a) });
  }
  // closed loop: play a strike drumMs after each detected onset, echo it into the mic echoMs later
  const det = new TapDetector(sr, { echoMs: toldMs, aecMs }), out = new Float32Array(N), found = [];
  if (train) { const tn = trainingNoise(sr); for (let i = 0; i < tn.length / 2; i++) out[i] = tn[2 * i]; }
  const strike = (n0, amp) => {
    const modes = [[1, 1], [1.59, 0.6], [2.14, 0.45], [2.3, 0.35], [2.65, 0.3], [2.92, 0.2]], f1 = 150 + 250 * rnd();
    for (let k = 0; k < ring * sr && n0 + k < N; k++) { let v = 0; for (const [r, g] of modes) v += g * Math.sin(2 * Math.PI * f1 * r * k / sr) * Math.exp(-k * r / (0.25 * ring * sr)); out[n0 + k] += amp * v / 2.9 * Math.min(1, k / 24); }
  };
  for (let c = 0; c < N; c += 128) {
    const m = mic.subarray(c, c + 128), o = out.subarray(c, c + 128);
    const de = Math.round(echoMs * sr / 1000), sp = (v) => Math.tanh(drive * v) / drive;   // echo: the speaker, then the room
    for (let i = 0; i < m.length; i++) for (const [dt, g] of [[0, 1], [61, 0.35], [173, -0.2]]) { const j = c + i - de - dt; if (j >= 0) m[i] += g * echoGain * sp(out[j]); }
    for (const t of det.process(m, o)) {
      found.push(t);
      det.learnFrom(t.n);
      strike(t.n + Math.round(drumMs * sr / 1000), 0.25 + 0.5 * Math.min(1, (10 * Math.log10(t.level) - 9) / 21));
    }
  }
  let hit = 0, lat = []; const used = new Set();
  for (const tp of taps) { const f = found.find((x) => x.n >= tp.n && x.n - tp.n < 0.015 * sr && !used.has(x)); if (f) { hit++; used.add(f); lat.push((f.n - tp.n) / sr * 1000); } }
  const missed = taps.filter((tp) => !found.some((x) => x.n >= tp.n && x.n - tp.n < 0.015 * sr)).map((t) => +t.db.toFixed(1));
  lat.sort((a, b) => a - b);
  return { name, taps: taps.length, found: hit, recall: +(hit / taps.length).toFixed(3), falseOnsets: found.length - hit, missedTapDb: missed.slice(0, 12),
    onsetLatencyMsMedian: +(lat[lat.length >> 1] ?? NaN).toFixed(2), cpl: +det.cpl.toFixed(3), erleDbWhilePlaying: +det.erleDb().toFixed(1) };
}

// The drum plays at up to 0.75 peak; echoGain is the speaker-to-mic amplitude ratio, so 0.2 puts the drum's echo
// at about -16 dBFS in the mic, level with a firm tap, and 1.0 at about -2 dBFS, louder than any tap.
const runs = [
  scenario({ name: "typical: taps -36..-14 dBFS, drum echo up to -16 dBFS" }),
  scenario({ name: "fast taps (every 0.16-0.3 s)", gap: [0.16, 0.3] }),
  scenario({ name: "noisy room (-45 dBFS noise)", noiseDb: -45 }),
  scenario({ name: "echo delay misjudged (actual 60 ms, detector told 28)", echoMs: 60, toldMs: 28 }),
  scenario({ name: "loud speaker: drum echo up to -2 dBFS", echoGain: 1 }),
  scenario({ name: "loud speaker, long ring (2.5 s)", echoGain: 1, ring: 2.5 }),
  scenario({ name: "loud speaker, no echo cancelling", echoGain: 1, aecMs: 0 }),
  scenario({ name: "loud speaker, no start-up noise", echoGain: 1, train: false }),
];
// Gate: typical, fast, misjudged delay and loud speaker. The noisy room is reported, not gated: taps within ~10 dB
// of the room's noise are below what an energy detector can find, and the sensitivity slider is the remedy.
const gated = [0, 1, 3, 4].map((i) => runs[i]);
const res = { runs, gated: gated.map((r) => r.name), pass: gated.every((r) => r.recall >= 0.9 && r.falseOnsets <= 2) };
fs.writeFileSync(path.join(HERE, "..", "results", "taps.json"), JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));
process.exit(res.pass ? 0 : 1);
