// Write the tuner's presets as voice files into web/voices/, with the model URL on austegard.com.
//   NODE_PATH=$(npm root -g) node run/make_preset_voices.mjs [TUNER_URL]
import { createRequire } from "node:module";
import { join } from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const url = process.argv[2] || "http://localhost:8765/";
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } : undefined;
const browser = await chromium.launch({ proxy });
const page = await browser.newPage();
await page.goto(url);
await page.addScriptTag({ url: new URL("tinyvoca.js", url).href });
const files = await page.evaluate(async () => {
  const space = await (await fetch("voice-space.json")).json();
  const base = "https://austegard.com/experiments/tiny-voca-voices/web/";
  return Object.entries(PRESETS).map(([name, p]) => {
    const v = Object.fromEntries(CONTROLS.map((c) => [c.id, p[c.id] ?? c.value]));
    const settings = { mix: v.mix, axes: [v.a1, v.a2, v.a3, v.a4], pitch: v.pitch, range: v.range, speed: v.speed,
                       formant: v.formant, tilt: v.tilt, breath: v.breath };
    return [name, TinyVoca.voiceFromSettings(space, settings, name, base)];
  });
});
mkdirSync("web/voices", { recursive: true });
for (const [name, voice] of files) writeFileSync(`web/voices/${name}.voice.json`, JSON.stringify(voice));
console.log(files.map(([n]) => n).join(" "));
await browser.close();
