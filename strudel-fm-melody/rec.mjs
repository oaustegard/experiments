// Record Strudel FM through the page's own audio (the listening-to-music tap), with the proxy's TLS tolerated
// so sample banks load. usage: node rec.mjs page.html out.wav query seed warmup seconds
import fs from 'fs';
import { execFileSync } from 'child_process';
import crypto from 'crypto';
// sample files come through curl with a disk cache: Chromium's fetches through the proxy intermittently fail
const CACHE = process.env.HOME + '/.cache/strudel-fm-samples';
fs.mkdirSync(CACHE, { recursive: true });
function cached(u){ const f = CACHE + '/' + crypto.createHash('sha1').update(u).digest('hex');
  if (!fs.existsSync(f)) execFileSync('curl', ['-sSfL', '--retry', '4', '-o', f, u]); return fs.readFileSync(f); }
import { launch, strudelBundle, writeWav, ORIGIN, HERE } from '/mnt/skills/user/listening-to-music/scripts/common.mjs';
const [src, out, query, seed, warm = '14', secs = '45'] = process.argv.slice(2);
const html = fs.readFileSync(src, 'utf8'), bundle = strudelBundle();
const browser = await launch();
try {
  const ctx = await browser.newContext({ ignoreHTTPSErrors: true });
  const page = await ctx.newPage();
  await page.addInitScript({ path: HERE + '/tap.js' });
  let fails = 0; page.on('requestfailed', r => { fails++; console.error('failed', r.url().slice(0, 100), r.failure()?.errorText); });
  await page.route('**/*', r => { const u = r.request().url();
    if (u.startsWith(ORIGIN)) return r.fulfill({ status: 200, contentType: 'text/html; charset=utf-8', body: html });
    if (/@strudel\/web(@[\d.]+)?\/dist\/index\.js/.test(u)) return r.fulfill({ status: 200, contentType: 'text/javascript', body: bundle });
    if (/fonts\.(googleapis|gstatic)\.com/.test(u)) return r.abort();
    if (u.startsWith('https://raw.githubusercontent.com/')){ try { return r.fulfill({ status: 200, body: cached(u), headers: { 'access-control-allow-origin': '*', 'content-type': u.endsWith('.json') ? 'application/json' : 'application/octet-stream' } }); } catch (e){ return r.abort(); } }
    return r.continue(); });
  await page.goto(ORIGIN + 'strudel-fm.html?' + query, { waitUntil: 'load' });
  await page.evaluate(s => window.__fmSetSeed(s), +seed);
  await page.click('#power');
  const res = await page.evaluate(async ([secs, warm]) => {
    await new Promise(r => setTimeout(r, warm * 1000)); window.__listen.start();
    await new Promise(r => setTimeout(r, secs * 1000)); return window.__listen.stop(); }, [+secs, +warm]);
  const info = writeWav(out, res);
  console.log(`${out}: ${info.seconds.toFixed(1)} s, peak ${info.peak.toFixed(2)}, failed requests ${fails}`);
} finally { await browser.close(); }
