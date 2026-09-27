// Generate Strudel FM bars headlessly and dump each bar's parts as JSON.
// usage: node gen.mjs <strudel-fm.html> <out.json> [phrasesPerRun=24]
import fs from 'fs';
import { launch, wirePage, ORIGIN } from '/mnt/skills/user/listening-to-music/scripts/common.mjs';

const [src, out, nph = '24'] = process.argv.slice(2);
let html = fs.readFileSync(src, 'utf8');
const hook = `window.__fmReset = s => { seed = s >>> 0; bars.clear(); phrases = []; lastPhrase = null; prevVoicing = null; station.key = null; station.pending = []; if (typeof theme !== "undefined") theme = null; };
window.__fmGen = n => { let start = 0; for (let i = 0; i < n; i++){ const ph = planPhrase(start); writePhrase(ph); start += 4; } };
window.__fm = {`;
if (!html.includes('window.__fm = {')) throw new Error('hook point missing');
html = html.replace('window.__fm = {', hook);

const browser = await launch();
const page = await wirePage(browser, { html, quiet: true });
await page.goto(ORIGIN + 'strudel-fm.html');
await page.waitForFunction(() => window.__fmGen);
const data = await page.evaluate((nph) => {
  const F = window.__fm, rows = [];
  const styles = Object.keys(F.STYLES);
  for (const style of styles) for (const energy of [-1, 0, 1]) for (const seed of [1, 2, 3, 4]){
    F.station.style = style; F.station.bpm = F.STYLES[style].bpm; F.station.mood = F.STYLES[style].mood; F.station.energy = energy;
    window.__fmReset(seed * 7919 + energy * 31 + styles.indexOf(style));
    window.__fmGen(nph);
    for (const [b, bar] of [...F.bars.entries()].sort((x, y) => x[0] - y[0]))
      rows.push({ style, energy, seed, b, section: bar.info.section, e: bar.info.e, key: { tonic: bar.ph.key.tonic, mode: bar.ph.key.mode },
        chord: { root: bar.chord.root, iv: bar.chord.iv, sym: bar.chord.sym }, bip: b - bar.ph.start,
        lead: bar.parts.lead || null, chords: bar.parts.chords || null, bass: bar.parts.bass || null, texture: bar.parts.texture || null, compiled: Object.keys(bar.pats) });
  }
  return rows;
}, +nph);
fs.writeFileSync(out, JSON.stringify(data));
console.log(`${data.length} bars -> ${out}`);
await browser.close();
