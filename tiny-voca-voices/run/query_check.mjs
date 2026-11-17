// ?text= on the tuner and example.html, &voice= on example.html, and the shipped preset voice files.
//   NODE_PATH=$(npm root -g) node run/query_check.mjs [BASE_URL]
import { createRequire } from "node:module";
import { join } from "node:path";
const { chromium } = createRequire(join(process.env.NODE_PATH || ".", "x.js"))("playwright");
const base = process.argv[2] || "http://localhost:8765/";
const TEXT = "Turn left at the second traffic light, then keep going until you see the bakery.";
const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: "localhost,127.0.0.1" } : undefined;
const browser = await chromium.launch({ proxy });
const errors = [];
const open = async (url) => {
  const page = await browser.newPage();
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto(url);
  return page;
};
const report = {};

// Tuner: text from the query, voice (M2-baritone) from the hash; Speak; then typing updates the address.
const tuner = await open(`${base}?text=${encodeURIComponent(TEXT)}#v=1,0,0,0,0,-5,1.1,0.95,0.92,0,0`);
await tuner.waitForFunction(() => ready === true, null, { timeout: 600000 });
report.tuner_text = (await tuner.inputValue("#text")) === TEXT;
report.tuner_status = await tuner.textContent("#status");
report.tuner_preset = await tuner.getAttribute("#p-M2-baritone", "aria-pressed");
await tuner.click("#speak");
await tuner.waitForFunction(() => lastClip && lastClip.id === request, null, { timeout: 60000 });
report.tuner_samples = await tuner.evaluate(() => lastClip.audio.length);
await tuner.fill("#text", "A new line to share.");
await tuner.waitForTimeout(700);
const shared = new URL(tuner.url());
report.tuner_url_text = shared.searchParams.get("text");
report.tuner_url_keeps_voice = shared.hash.startsWith("#v=1,");

// example.html: text and voice from the query.
const example = await open(`${new URL("example.html", base).href}?text=${encodeURIComponent(TEXT)}&voice=voices/M2-baritone.voice.json`);
await example.waitForFunction(() => /^(Ready\.|Could not)/.test(document.getElementById("status").textContent), null, { timeout: 600000 });
report.example_status = await example.textContent("#status");
report.example_voice = await example.textContent("#name");
report.example_text = (await example.inputValue("#text")) === TEXT;
await example.click("#speak");
await example.waitForFunction(() => / s$/.test(document.getElementById("status").textContent), null, { timeout: 60000 });
report.example_spoke = await example.textContent("#status");
report.example_samples = await example.evaluate(async ([text]) => {
  const tts = await TinyVoca.load();
  const voice = await (await fetch("voices/M2-baritone.voice.json")).json();
  return (await tts.speak(text, voice)).length;
}, [TEXT]);
report.same_samples = report.example_samples === report.tuner_samples;
report.errors = errors;
console.log(JSON.stringify(report, null, 1));
await browser.close();
