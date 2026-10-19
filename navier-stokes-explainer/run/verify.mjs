// Load the page in headless Chromium (SwiftShader WebGPU), drive each chapter,
// check the numbers it exposes on window.__ns, and screenshot every chapter.
//
// Usage: node run/verify.mjs [--wait 1.0]   (wait multiplies every settle time)
import { createRequire } from 'module';
import { spawn } from 'child_process';
import { mkdirSync, writeFileSync } from 'fs';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

const require = createRequire('/opt/node-tools/node_modules/');
const { chromium } = require('playwright');
const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SHOTS = resolve(ROOT, 'shots');
mkdirSync(SHOTS, { recursive: true });
const WAIT = +(process.argv[process.argv.indexOf('--wait') + 1] || 1) || 1;
const PORT = 8000 + Math.floor(Math.random() * 1000);
const CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
// --use-vulkan=swiftshader is what lets a WebGPU canvas present; without it the
// device is lost ("A valid external Instance reference no longer exists") on the first frame.
const GPU_ARGS = ['--no-sandbox', '--enable-unsafe-webgpu', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'];

const server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], { cwd: resolve(ROOT, 'web'), stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms * WAIT));
const results = { checks: [], errors: [] };
const check = (name, ok, detail) => { results.checks.push({ name, ok: !!ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'} ${name} ${JSON.stringify(detail)}`); };

async function drag(page, sel, pts) {
  const box = await page.locator(sel).boundingBox();
  const P = ([x, y]) => [box.x + x * box.width, box.y + y * box.height];
  await page.mouse.move(...P(pts[0])); await page.mouse.down();
  for (const p of pts.slice(1)) { await page.mouse.move(...P(p), { steps: 12 }); await page.waitForTimeout(30); }
  await page.mouse.up();
}
const ns = (page, fn) => page.evaluate(fn);
// Step a simulation in-page without waiting for frames: SwiftShader manages a few
// steps a second, too slow to watch a wake develop in real time.
const ff = (page, sim, steps, n = 1) => page.evaluate(async ([sim, steps, n]) => {
  const f = window.__ns.sims[sim];
  for (let i = 0; i < steps; i++) { f.step(n); if (i % 25 === 24) await f.device.queue.onSubmittedWorkDone(); }
  await f.device.queue.onSubmittedWorkDone();
}, [sim, steps, n]);
// Ask for a stats readback and wait until it lands.
const measured = (page, sim) => page.evaluate(async (sim) => {
  const f = window.__ns.sims[sim];
  const t0 = f.stats.time ?? -1;
  for (let k = 0; k < 100; k++) {
    f.measure();
    await new Promise((r) => setTimeout(r, 100));
    if ((f.stats.time ?? -1) !== t0 && f.stats.time !== undefined) break;
  }
  return { ...f.stats };
}, sim);

try {
  await sleep(800);
  const browser = await chromium.launch({ executablePath: CHROME, args: GPU_ARGS });
  const page = await browser.newPage({ viewport: { width: 1180, height: 900 }, deviceScaleFactor: 1 });
  page.on('console', (m) => { if (m.type() === 'error') results.errors.push(m.text()); });
  page.on('pageerror', (e) => results.errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${PORT}/index.html`);
  await page.waitForFunction(() => window.__ns?.ready, null, { timeout: 30000 });
  check('webgpu device', await ns(page, () => window.__ns.gpu), {});
  await page.screenshot({ path: `${SHOTS}/00-top.png` });

  // ---- chapter 1
  await page.locator('#ch1').scrollIntoViewIfNeeded();
  await sleep(4000);
  await drag(page, '#c1', [[0.15, 0.5], [0.4, 0.45], [0.65, 0.55], [0.85, 0.5]]);
  await sleep(1500);
  const s1 = { ...(await measured(page, 'c1')), frames: await ns(page, () => window.__ns.frames.ch1) };
  check('ch1 steps and carries momentum', s1.frames > 10 && s1.energy > 0 && s1.maxVort > 0, s1);
  await page.locator('#ch1').screenshot({ path: `${SHOTS}/01-terms.png` });
  // Pressure off: the divergence view should light up.
  await page.click('.switch[data-term=pressure]');
  await page.click('#c1-view button[data-v="3"]');
  await drag(page, '#c1', [[0.2, 0.3], [0.5, 0.35], [0.8, 0.3]]);
  await sleep(1500);
  const off = await ns(page, () => ({ pressure: window.__ns.sims.c1.terms.pressure, note: !document.querySelector('.term-note[data-for=pressure]').hidden, eqOff: document.querySelector('.eq .term[data-term=pressure]').classList.contains('off') }));
  check('ch1 pressure toggle', !off.pressure && off.note && off.eqOff, off);
  await page.locator('#ch1').screenshot({ path: `${SHOTS}/01b-terms-no-pressure.png` });
  await page.click('.switch[data-term=pressure]');
  await page.click('#c1-view button[data-v="0"]');

  // ---- chapter 2
  await page.locator('#ch2').scrollIntoViewIfNeeded();
  await page.locator('#c2-re').fill('2.2');
  await page.locator('#c2-re').dispatchEvent('input');
  await ff(page, 'c2', 1300);
  const s2 = { ...(await measured(page, 'c2')), ...(await ns(page, () => ({ re: window.__ns.re, frames: window.__ns.frames.ch2, regime: document.querySelector('#c2-regime').textContent }))) };
  check('ch2 channel flow runs', s2.frames > 10 && s2.energy > 0.05 && s2.maxVort > 0, s2);
  const wake = (page) => page.evaluate(() => window.__ns.c2.wakeRms());
  const w1 = await wake(page);
  check('ch2 cylinder wake sheds', w1 > 0.05, { vrms: w1 });
  await page.evaluate(() => window.__ns.c2.restart());
  await ff(page, 'c2', 1000);
  const w2 = await wake(page);
  check('ch2 wake sheds again after restart', w2 > 0.05, { vrms: w2 });
  await page.click('#c2-view button[data-v="1"]');
  await sleep(600);
  await page.locator('#ch2').screenshot({ path: `${SHOTS}/02-reynolds.png` });
  await page.evaluate(() => window.__ns.c2.setShape('airfoil'));
  await ff(page, 'c2', 1300);
  const w3 = await wake(page);
  const a3 = await ns(page, () => ({ re: window.__ns.re, aoa: +document.querySelector('#c2-aoa').value }));
  check('ch2 airfoil stalls at high angle', w3 > 0.05 && a3.re > 900, { vrms: w3, ...a3 });
  await page.locator('#ch2 figure').screenshot({ path: `${SHOTS}/02b-airfoil.png` });

  // ---- chapter 3
  await page.locator('#ch3 .two').scrollIntoViewIfNeeded();
  await sleep(6000);
  await ff(page, 'c3b', 50, 2);
  await sleep(8000);
  const s3 = await ns(page, () => ({ h2: window.__ns.hist2(), h3: window.__ns.hist3(), frames: window.__ns.frames.ch3 }));
  const max2 = Math.max(...s3.h2.map((p) => p[1])), last2 = s3.h2.at(-1)?.[1];
  const max3 = Math.max(...s3.h3.map((p) => p[1]));
  check('ch3 2D max vorticity never exceeds its start', s3.h2.length > 3 && max2 <= 1.05, { samples: s3.h2.length, max: max2, last: last2 });
  check('ch3 3D stretching raises vorticity', s3.h3.length > 3 && max3 > 1.3 && max3 < 6, { samples: s3.h3.length, maxStretch: max3, tLast: s3.h3.at(-1)?.[0] });
  await page.locator('#ch3').screenshot({ path: `${SHOTS}/03-stretching.png` });
  await page.click('#c3b-preset button[data-v="oblique"]');
  await drag(page, '#c3b', [[0.5, 0.5], [0.62, 0.45]]);
  await ff(page, 'c3b', 50, 2);
  await sleep(8000);
  const s3b = await ns(page, () => window.__ns.hist3());
  check('ch3 tilted preset runs', s3b.length > 2, { samples: s3b.length, maxStretch: Math.max(...s3b.map((p) => p[1])) });
  await page.locator('#ch3 .two').screenshot({ path: `${SHOTS}/03b-tilted.png` });

  // ---- chapter 4
  await page.locator('#p4a').scrollIntoViewIfNeeded();
  await page.locator('#p4a-alpha').fill('1.2'); await page.locator('#p4a-alpha').dispatchEvent('input');
  const v4a = await ns(page, () => document.querySelector('#p4a-verdict').textContent);
  check('ch4 toy verdict follows alpha', /diverges/.test(v4a), v4a.slice(0, 60));
  await page.locator('#c4').scrollIntoViewIfNeeded();
  await ff(page, 'c4', 300);
  await sleep(15000);
  const s4 = await ns(page, () => ({ spec: window.__ns.spec(), hist: window.__ns.hist4().length, frames: window.__ns.frames.ch4 }));
  check('ch4 spectrum and BKM trace', s4.spec && s4.spec.length === 128 && s4.hist > 3, { specLen: s4.spec?.length, hist: s4.hist, E1: s4.spec?.[0]?.[1], E64: s4.spec?.[63]?.[1], E128: s4.spec?.[127]?.[1] });
  await page.locator('#ch4').screenshot({ path: `${SHOTS}/04-bkm.png` });

  // ---- chapter 5
  await page.locator('#c5').scrollIntoViewIfNeeded();
  await sleep(5000);
  const before = await ns(page, () => document.querySelector('#c5-verdict').textContent);
  await page.locator('#c5w').scrollIntoViewIfNeeded();
  await drag(page, '#c5w', [[0.5, 0.5], [0.55, 0.25]]);
  const after = await ns(page, () => ({ v: document.querySelector('#c5-verdict').textContent, mu: +document.querySelector('#c5-mu').value, frames: window.__ns.frames.ch5 }));
  check('ch5 scaling window drag updates verdict', after.v !== before && after.frames > 10, { before: before.slice(0, 80), after: after.v.slice(0, 80), mu: after.mu });
  await page.locator('#ch5 figure').first().screenshot({ path: `${SHOTS}/05-cascade.png` });
  await page.locator('#c5w').screenshot({ path: `${SHOTS}/05b-window.png` });
  await page.locator('#ch5').screenshot({ path: `${SHOTS}/05c-chapter.png` });

  const nsErr = await ns(page, () => window.__ns.errors);
  results.errors.push(...nsErr);
  check('no page or WebGPU errors', results.errors.length === 0, results.errors.slice(0, 5));
  await browser.close();

  // ---- fallback: no WebGPU
  const b2 = await chromium.launch({ executablePath: CHROME, args: ['--no-sandbox', '--disable-features=WebGPU', '--disable-gpu'] });
  const p2 = await b2.newPage({ viewport: { width: 900, height: 800 } });
  await p2.goto(`http://127.0.0.1:${PORT}/index.html`);
  await p2.waitForFunction(() => window.__ns?.ready, null, { timeout: 30000 });
  const fb = await p2.evaluate(() => ({ gpu: window.__ns.gpu, nogpu: document.body.classList.contains('nogpu'), errors: window.__ns.errors }));
  check('fallback message without WebGPU', !fb.gpu && fb.nogpu, fb);
  await p2.locator('#ch1 figure').screenshot({ path: `${SHOTS}/06-fallback.png` });
  await b2.close();
} catch (e) {
  results.errors.push(String(e.stack || e));
  console.error(e);
} finally {
  server.kill();
  writeFileSync(resolve(SHOTS, 'verify.json'), JSON.stringify(results, null, 1));
  const failed = results.checks.filter((c) => !c.ok).length;
  console.log(`${results.checks.length - failed}/${results.checks.length} checks passed`);
  process.exitCode = failed || results.checks.length === 0 ? 1 : 0;
}
