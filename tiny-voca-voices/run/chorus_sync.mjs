// How far apart are the chorus voices? Sings the default score's parts separately (tinyvoca.js chorus() returns
// them) and, in 1 s windows, finds the lag that best aligns each part's loudness envelope with the lead's.
//   NODE_PATH=$(npm root -g) node run/chorus_sync.mjs [BASE_URL] [SEED]
import { createRequire } from "node:module";
import { join } from "node:path";
const pw = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const base = process.argv[2] || "http://localhost:8765/";
const seed = process.argv[3] || "12345";
const browser = await pw.chromium.launch({ proxy: { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } });
const page = await browser.newPage();
await page.goto(base + "?mode=sing&engine=page&seed=" + seed);
await page.waitForFunction(() => ready === true, null, { timeout: 120000 });
const r = await page.evaluate(async () => {
  const tts = await TinyVoca.load();
  const c = SONG.find((x) => x.id === "voices"); c.input.value = 4; c.show();
  const parts = chorusParts(4).map((p) => ({ ...p, voice: tts.voiceFromSettings(p.settings) }));
  const { voices, ...rest } = song();
  const out = await tts.chorus({ lyrics: document.getElementById("lyrics").value, notes: document.getElementById("notes").value, ...rest }, parts);
  const env = (a) => {                       // 10 ms RMS envelope
    const hop = 240, e = new Float32Array(Math.floor(a.length / hop));
    for (let i = 0; i < e.length; i++) { let s = 0; for (let j = 0; j < hop; j++) s += a[i * hop + j] ** 2; e[i] = Math.sqrt(s / hop); }
    return e;
  };
  window.__parts = out.parts.map((a) => Array.from(a)); window.__delays = parts.map((p) => p.delayMs);
  const lead = env(out.parts[0]);
  return out.parts.slice(1).map((p, k) => {
    const e = env(p), lags = [];
    for (let w = 0; w + 100 <= Math.min(lead.length, e.length); w += 100) {
      let best = 0, bestLag = 0;
      for (let lag = -15; lag <= 15; lag++) {
        let s = 0;
        for (let i = w; i < w + 100; i++) { const j = i + lag; if (j >= 0 && j < e.length) s += lead[i] * e[j]; }
        if (s > best) { best = s; bestLag = lag; }
      }
      lags.push(bestLag * 10);
    }
    const abs = lags.map(Math.abs).sort((a, b) => a - b);
    return { part: k + 1, delayMs: Math.round(parts[k + 1].delayMs), medianAbsLagMs: abs[Math.floor(abs.length / 2)], maxAbsLagMs: abs[abs.length - 1], lengths: [out.parts[0].length, p.length] };
  });
});
console.log(JSON.stringify(r));
if (process.argv[4]) {                       // also save each part for scripts that look at voicing onsets
  const { writeFileSync, mkdirSync } = await import("node:fs");
  mkdirSync(process.argv[4], { recursive: true });
  const parts = await page.evaluate(() => window.__parts);
  const delays = await page.evaluate(() => window.__delays);
  parts.forEach((a, k) => {
    const buf = Buffer.alloc(a.length * 4);
    a.forEach((x, i) => buf.writeFloatLE(x, i * 4));
    writeFileSync(`${process.argv[4]}/part${k}.f32`, buf);
  });
  writeFileSync(`${process.argv[4]}/delays.json`, JSON.stringify(delays));
}
await browser.close();
