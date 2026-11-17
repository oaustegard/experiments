// Do all chorus parts get the same frames per phoneme? Records each part's timing plan (frames per phoneme per
// line) through chorus()'s onPlan and compares every part with the lead. Optional second arg "own" makes each part
// time itself (the behaviour before the shared plan) for comparison.
//   NODE_PATH=$(npm root -g) node run/chorus_plan.mjs [BASE_URL] [own]
import { createRequire } from "node:module";
import { join } from "node:path";
const pw = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const base = process.argv[2] || "http://localhost:8765/";
const own = process.argv[3] === "own";
const browser = await pw.chromium.launch({ proxy: { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } });
const page = await browser.newPage();
await page.goto(base + "?mode=sing&engine=page&seed=12345");
await page.waitForFunction(() => ready === true, null, { timeout: 120000 });
const r = await page.evaluate(async (own) => {
  const tts = await TinyVoca.load();
  const c = SONG.find((x) => x.id === "voices"); c.input.value = 4; c.show();
  const parts = chorusParts(4).map((p) => ({ ...p, voice: tts.voiceFromSettings(p.settings) }));
  const { voices, ...rest } = song();
  const score = { lyrics: document.getElementById("lyrics").value, notes: document.getElementById("notes").value, ...rest };
  const plans = parts.map(() => []);
  if (own) {
    for (let i = 0; i < parts.length; i++) await tts.sing(score, parts[i].voice, { onPlan: (line, f) => { plans[i][line] = f; } });
  } else {
    await tts.chorus(score, parts, { onPlan: (part, line, f) => { plans[part][line] = f; } });
  }
  const fps = 24000 / 256;
  return plans.slice(1).map((plan, k) => {
    let tokens = 0, differ = 0, worstMs = 0;
    plan.forEach((frames, line) => {
      let a = 0, b = 0;
      frames.forEach((f, t) => {                 // compare cumulative start times of each phoneme
        tokens++;
        const lag = Math.abs(a - b) * 1000 / fps;
        if (lag > 0) differ++;
        worstMs = Math.max(worstMs, lag);
        a += plans[0][line][t]; b += f;
      });
    });
    return { part: k + 1, phonemes: tokens, phonemesStartingAtADifferentTime: differ, worstLagMs: Math.round(worstMs) };
  });
}, own);
console.log((own ? "each part times itself: " : "shared plan (lead's timing): ") + JSON.stringify(r));
await browser.close();
