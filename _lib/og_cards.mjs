// Renders the 1200x630 link-preview image of every pages.json entry into og/<slug>.jpg
// (a screenshot of the page in a card with its title and blurb), plus og/index.jpg for the listing.
//   NODE_PATH=$(npm root -g) node _lib/og_cards.mjs [--only slug,slug] [--force]
// Without --force an entry that already has an image is skipped. og/overrides.json sets, per slug,
//   {"mode":"text"} to skip the screenshot, {"wait":ms} for a slow page, {"viewport":[w,h]} for the screenshot size.
import {createRequire} from 'node:module';
import {createServer} from 'node:http';
import {readFileSync, writeFileSync, existsSync, mkdirSync, statSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {dirname, join, extname} from 'node:path';
const require = createRequire(import.meta.url);
const {chromium} = require('playwright');
const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const only = args.includes('--only') ? args[args.indexOf('--only') + 1].split(',') : null;
const force = args.includes('--force');
const manifest = JSON.parse(readFileSync(join(ROOT, 'pages.json'), 'utf8')).pages;
const over = existsSync(join(ROOT, 'og/overrides.json')) ? JSON.parse(readFileSync(join(ROOT, 'og/overrides.json'), 'utf8')) : {};
mkdirSync(join(ROOT, 'og'), {recursive: true});

const MIME = {'.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.css': 'text/css',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.wasm': 'application/wasm', '.bin': 'application/octet-stream', '.mp3': 'audio/mpeg', '.mp4': 'video/mp4', '.webp': 'image/webp'};
const server = createServer((q, r) => {
  let p = decodeURIComponent(q.url.split('?')[0]); if (p.endsWith('/')) p += 'index.html';
  const f = join(ROOT, p);
  if (!f.startsWith(ROOT) || !existsSync(f) || statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  const t = MIME[extname(f)] || 'application/octet-stream'; r.writeHead(200, {'content-type': t.startsWith('text/') || t.includes('javascript') || t.includes('json') ? t + '; charset=utf-8' : t}); r.end(readFileSync(f));
}).listen(0);
const base = `http://localhost:${server.address().port}/`;
const slugOf = e => e.slug || e.experiment;
const target = e => {
  if (e.url) return e.url;
  const p = Array.isArray(e.path) ? e.path[0] : e.path;
  return base + p + (existsSync(join(ROOT, p)) && statSync(join(ROOT, p)).isDirectory() && !p.endsWith('/') ? '/' : '');
};
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));

const FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,650&family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@500&display=swap">';
const CSS = `*{box-sizing:border-box}body{margin:0;width:1200px;height:630px;overflow:hidden;background:#f6f7f4;color:#1d2321;font-family:"IBM Plex Sans",system-ui,sans-serif;position:relative}
.eyebrow{position:absolute;left:56px;top:48px;font:500 18px "IBM Plex Mono",monospace;color:#5d6762}
h1{font:650 56px/1.04 Fraunces,Georgia,serif;margin:0;letter-spacing:-.01em}
.txt{position:absolute;left:56px;top:96px;bottom:84px;display:flex;flex-direction:column;justify-content:center;gap:22px}
p{margin:0;font-size:25px;line-height:1.4;color:#3d4743;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:5;overflow:hidden}
.foot{position:absolute;left:56px;bottom:40px;font:500 18px "IBM Plex Mono",monospace;color:#2f6b5a}
.shot{position:absolute;right:48px;top:48px;width:600px;height:534px;border-radius:14px;border:1px solid #d5dad5;box-shadow:0 12px 36px rgba(20,30,25,.18);background:#fff center top/cover no-repeat}
.bar{position:absolute;left:0;right:0;bottom:0;height:10px;background:#2f6b5a}`;
const card = (e, shotUrl) => `<!doctype html><meta charset="utf-8">${FONTS}<style>${CSS}
.txt{width:${shotUrl ? 440 : 1000}px}h1{font-size:${shotUrl ? 52 : 76}px}p{font-size:${shotUrl ? 24 : 32}px;-webkit-line-clamp:${shotUrl ? 6 : 5}}</style>
<div class="eyebrow">oaustegard/experiments</div><div class="txt"><h1>${esc(e.title)}</h1><p>${esc(e.blurb || '')}</p></div>
${shotUrl ? `<div class="shot" style="background-image:url(${shotUrl})"></div>` : ''}<div class="foot">${esc(e.experiment)}</div><div class="bar"></div>`;

const browser = await chromium.launch({args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--enable-unsafe-webgpu', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream']});
const report = [];
for (const e of manifest) {
  const slug = slugOf(e), out = join(ROOT, e.image || `og/${slug}.jpg`);
  if (only && !only.includes(slug)) continue;
  if (e.image) continue;               // a page that ships its own preview image
  if (!force && !only && existsSync(out)) continue;
  const o = over[slug] || {};
  let shotUrl = null, note = 'text';
  if (o.mode !== 'text') {
    const [w, h] = o.viewport || [800, 712];
    const ctx = await browser.newContext({viewport: {width: w, height: h}, deviceScaleFactor: 1, permissions: ['camera', 'microphone']});
    const pg = await ctx.newPage();
    try {
      await pg.goto(target(e), {waitUntil: 'load', timeout: 25000});
      await pg.waitForTimeout(o.wait || 3000);
      shotUrl = 'data:image/jpeg;base64,' + (await pg.screenshot({type: 'jpeg', quality: 85})).toString('base64'); note = 'shot';
    } catch (err) { note = 'text (' + String(err.message).split('\n')[0].slice(0, 60) + ')'; }
    await ctx.close();
  }
  const ctx = await browser.newContext({viewport: {width: 1200, height: 630}, deviceScaleFactor: 1});
  const pg = await ctx.newPage();
  await pg.setContent(card(e, shotUrl), {waitUntil: 'networkidle'}); await pg.waitForTimeout(400);
  writeFileSync(out, await pg.screenshot({type: 'jpeg', quality: 86}));
  await ctx.close(); report.push(`${slug}: ${note}`); console.log(`${slug}: ${note}`);
}
// listing card: a mosaic of the other cards
if (!only || only.includes('index')) {
  const imgs = manifest.filter(e => existsSync(join(ROOT, e.image || `og/${slugOf(e)}.jpg`))).slice(0, 12)
    .map(e => 'data:image/' + ((e.image || '').endsWith('.png') ? 'png' : 'jpeg') + ';base64,' + readFileSync(join(ROOT, e.image || `og/${slugOf(e)}.jpg`)).toString('base64'));
  const ctx = await browser.newContext({viewport: {width: 1200, height: 630}});
  const pg = await ctx.newPage();
  await pg.setContent(`<!doctype html><meta charset="utf-8">${FONTS}<style>${CSS}
.grid{position:absolute;right:-40px;top:-30px;width:760px;display:grid;grid-template-columns:repeat(3,1fr);gap:14px;transform:rotate(-6deg);transform-origin:center}
.grid img{width:100%;border-radius:8px;border:1px solid #d5dad5;box-shadow:0 6px 18px rgba(20,30,25,.15)}
.txt{width:430px}h1{font-size:64px}</style>
<div class="eyebrow">oaustegard/experiments</div><div class="txt"><h1>Experiments</h1><p>Every interactive page from the experiments repo: ${manifest.length} of them, generated on each push.</p></div>
<div class="grid">${imgs.map(s => `<img src="${s}">`).join('')}</div><div class="foot">austegard.com/experiments</div><div class="bar"></div>`, {waitUntil: 'networkidle'});
  await pg.waitForTimeout(400);
  writeFileSync(join(ROOT, 'og/index.jpg'), await pg.screenshot({type: 'jpeg', quality: 86}));
  await ctx.close(); console.log('index: mosaic');
}
await browser.close(); server.close();
