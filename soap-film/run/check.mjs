// Headless checks of the Soap Film page (oaustegard.github.io/fun-and-games/soap-film.html) on SwiftShader.
//   node run/check.mjs            SITE_DIR=<site checkout> overrides ../../oaustegard.github.io
// The page's ?test=1 mode fixes the grid, steps on request and draws to a texture, so nothing here waits on a clock.
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { serve, open } from './lib.mjs';

const results = [];
const check = (name, pass, detail) => { results.push({ name, pass: !!pass, detail }); console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}  ${detail}`); };
const { srv, url } = await serve();

// A page per film: SwiftShader loses the first page's device when a second one closes, so use one browser each.
async function film(query, fn) {
  const { browser, page, errors } = await open(url, 'test=1&' + query);
  try { return await fn(page, errors); } finally { await browser.close(); }
}
// Runs in the page: summary numbers for the film's state.
const SUMMARY = async () => {
  const S = window.__soap, { NX, NY } = S.cfg, st = await S.state(), vel = await S.vel();
  let n = 0, sum = 0, sy = 0, sx = 0, vmax = 0, om = 0, black = 0, zero = 0, topBlack = 0, botBlack = 0, hmin = 1e9, hmax = 0;
  for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    if (!S.insideFilm(x + 0.5, y + 0.5)) continue;
    const i = y * NX + x, h = st[2 * i + 1];
    n++; sum += h; sx += h * x; sy += h * y; om += Math.abs(st[2 * i]);
    vmax = Math.max(vmax, Math.hypot(vel[2 * i], vel[2 * i + 1])); hmin = Math.min(hmin, h); hmax = Math.max(hmax, h);
    if (h < 0.5) zero++; else if (h < 20) { black++; if (y < NY / 2) topBlack++; else botBlack++; }
  }
  return { n, mean: sum / n, cx: sx / sum / NX, cy: sy / sum / NY, vmax, om: om / n, black: black / n, zero: zero / n, topBlack, botBlack, hmin, hmax,
    nan: st.some(v => v !== v) };
};
const sum = page => page.evaluate(SUMMARY);
const step = (page, n) => page.evaluate(k => window.__soap.step(k), n);

// 1. Colour table against the independent reference (run/ref_lut.py).
await film('n=64', async (page, errors) => {
  const ref = JSON.parse(await readFile(path.resolve(import.meta.dirname, '../data/ref_lut.json'), 'utf8'));
  const cfg = await page.evaluate(() => window.__soap.cfg);
  check('page loads with a WebGPU device', errors.length === 0 && cfg.NX === 64, `grid ${cfg.NX}x${cfg.NY}, adapter "${cfg.adapter}", ${errors.length} console errors ${errors.slice(0, 2).join(' | ')}`);
  const mine = await page.evaluate(({ t, c }) => c.map(cos => t.map(d => window.__soap.filmRGB(d, cos))), { t: ref.thickness_nm, c: ref.cos_incidence });
  const gamut = c => { const l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2], lo = Math.min(...c); if (lo < 0 && l > 0) { const t = l / (l - lo); return c.map(v => l + t * (v - l)); } return c.map(v => Math.max(v, 0)); };
  const lab = c => {                                    // linear sRGB, scaled so the brightest film is near white
    const v = c.map(x => Math.min(x * 12, 1)), X = 0.4124 * v[0] + 0.3576 * v[1] + 0.1805 * v[2], Y = 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2], Z = 0.0193 * v[0] + 0.1192 * v[1] + 0.9505 * v[2];
    const f = t => t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116, fx = f(X / 0.9505), fy = f(Y), fz = f(Z / 1.089);
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
  };
  for (const key of ['page_dispersion', 'constant_index']) {
    let worst = 0, total = 0, count = 0, at = 0;
    ref.cos_incidence.forEach((cos, j) => ref.thickness_nm.forEach((d, i) => {
      const a = lab(mine[j][i]), b = lab(gamut(ref[key][j][i])), de = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      total += de; count++; if (de > worst) { worst = de; at = d; }
    }));
    const limit = key === 'page_dispersion' ? [0.3, 1.5] : [4, 14];
    check(`colour table matches the reference (${key})`, total / count < limit[0] && worst < limit[1], `mean ΔE76 ${(total / count).toFixed(2)}, worst ${worst.toFixed(2)} at ${at} nm, ${count} colours`);
  }
  const k = await page.evaluate(() => [0, 6, 100, 3000].map(d => window.__soap.filmRGB(d, 1)));
  const lum = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  check('no film is black, black film nearly so, 100 nm is the bright first-order white', lum(k[0]) === 0 && lum(k[1]) < 0.002 && lum(k[2]) > 0.07 && Math.max(...k[2]) / Math.min(...k[2]) < 1.15,
    `luminance 0 nm ${lum(k[0])}, 6 nm ${lum(k[1]).toFixed(5)}, 100 nm ${lum(k[2]).toFixed(4)} rgb ${k[2].map(v => v.toFixed(3))}`);
});

// 2. A film stratified thin-over-thick is in balance: it must not move. Diffusion is off for this one, because
// thickness diffusing against the curved corners of the frame bends the layers there and drives a slow real flow
// (0.2 cells/s with it on).
await film('n=96&ar=1.5&kind=strat', async page => {
  await page.evaluate(() => { window.__soap.P.drainMul = 0; window.__soap.P.diff = 0; });
  const a = await sum(page); await step(page, 300); const b = await sum(page);
  check('stratified film at rest stays at rest', b.vmax < 1e-4 && Math.abs(b.mean - a.mean) < 0.5 && !b.nan, `after 300 steps: peak speed ${b.vmax.toExponential(2)} cells/s, mean thickness ${a.mean.toFixed(2)} -> ${b.mean.toFixed(2)} nm`);
  // Turn gravity sideways: the thick film now has to pour toward the new "down".
  await page.evaluate(() => { window.__soap.P.diff = 0.012; window.__soap.setGravity(1, 0); });
  await step(page, 420); const c = await sum(page);
  check('thick film pours toward gravity when it turns', c.cx - b.cx > 0.02 && c.vmax > 1, `thickness-weighted x ${b.cx.toFixed(3)} -> ${c.cx.toFixed(3)} of the width, peak speed ${c.vmax.toFixed(1)} cells/s`);
});

// 3. Thick over thin is unstable: it overturns, and thickness is conserved while it does.
await film('n=96&ar=1.5&kind=inverted', async page => {
  await page.evaluate(() => { window.__soap.P.drainMul = 0; });
  const a = await sum(page); await step(page, 1500); const b = await sum(page);
  check('thick film on top overturns', b.cy - a.cy > 0.05, `thickness-weighted height ${a.cy.toFixed(3)} -> ${b.cy.toFixed(3)} (1 = bottom) in 25 s`);
  check('thickness is conserved without drainage', Math.abs(b.mean / a.mean - 1) < 0.01 && !b.nan, `mean ${a.mean.toFixed(2)} -> ${b.mean.toFixed(2)} nm (${((b.mean / a.mean - 1) * 100).toFixed(3)} %) over 1500 steps`);
});

// 4. No gravity: nothing sorts by thickness, and the swirl dies away under drag.
await film('n=96&ar=1.5&kind=fresh&seed=3', async page => {
  await page.evaluate(() => { window.__soap.P.drainMul = 0; window.__soap.setGravity(0, 0); });
  await step(page, 30); const a = await sum(page); await step(page, 120); const b = await sum(page);
  const ratio = b.om / a.om, drag = Math.exp(-0.45 * 2);
  check('with no gravity the flow only decays', ratio < drag * 1.05 && ratio > 0.1 && Math.abs(b.cy - a.cy) < 0.01, `mean |vorticity| x${ratio.toFixed(3)} in 2 s (drag alone gives x${drag.toFixed(3)}), thickness-weighted height moved ${(b.cy - a.cy).toFixed(4)}`);
});

// 5. Blowing: a breath moving right leaves a vortex pair that carries film to the right.
await film('n=96&ar=1.5&kind=flat', async page => {
  await page.evaluate(() => { window.__soap.P.drainMul = 0; window.__soap.setGravity(0, 0); });
  const r = await page.evaluate(async () => {
    const S = window.__soap, { NX, NY } = S.cfg, cx = NX / 2, cy = NY / 2;
    S.blow(cx, cy, 200, 0); await S.step(12); S.blow();
    const st = await S.state(), vel = await S.vel(), at = (x, y) => (Math.round(y) * NX + Math.round(x));
    return { above: st[2 * at(cx, cy - 5)], below: st[2 * at(cx, cy + 5)], vx: vel[2 * at(cx, cy)], vy: vel[2 * at(cx, cy) + 1] };
  });
  check('a breath makes a vortex pair moving its way', r.above * r.below < 0 && r.vx > 2 && Math.abs(r.vy) < 0.2 * r.vx, `vorticity above ${r.above.toFixed(2)} and below ${r.below.toFixed(2)} 1/s, flow at the breath (${r.vx.toFixed(1)}, ${r.vy.toFixed(1)}) cells/s`);
});

// 5b. A hard breath across layered film: thickness must stay inside the range it started with (advection is
// clamped and diffusion only averages), and the flow must stay finite. An earlier version of the step computed
// buoyancy and diffusion from the thickness before it moved; the same stroke on a 192-cell loop with drainage on
// took thickness from 59..939 nm to 6..3,180 nm within 70 steps and pinned vorticity at its clamp.
await film('n=128&shape=loop&kind=fresh&seed=4', async page => {
  await page.evaluate(() => { window.__soap.P.drainMul = 0; });
  await step(page, 240); const a = await sum(page);
  await page.evaluate(async () => { const S = window.__soap, { NX, NY } = S.cfg;
    for (let k = 0; k < 24; k++) { S.blow(NX * (0.25 + 0.5 * k / 24), NY * 0.55, NX * 1.3, 0); await S.step(1); } S.blow(); await S.step(240); });
  const b = await sum(page);
  check('a hard breath keeps thickness inside its starting range', b.hmax <= a.hmax + 0.01 && b.hmin >= a.hmin - 0.01 && b.vmax < 400 && !b.nan,
    `thickness ${a.hmin.toFixed(1)}..${a.hmax.toFixed(1)} nm before, ${b.hmin.toFixed(1)}..${b.hmax.toFixed(1)} nm 4 s after; peak speed ${b.vmax.toFixed(1)} cells/s`);
});

// 6. Drainage: the film thins, black film appears at the top, and the picture shows it.
await film('n=96&ar=1.5&kind=thin', async page => {
  const a = await sum(page); await step(page, 900); const b = await sum(page);
  check('the film drains and black film forms at the top', b.mean < a.mean - 10 && b.topBlack > 50 && b.topBlack > 20 * (b.botBlack + 1) && !b.nan, `mean ${a.mean.toFixed(0)} -> ${b.mean.toFixed(0)} nm in 15 s; black cells: top half ${b.topBlack}, bottom half ${b.botBlack}; thinnest ${b.hmin.toFixed(1)} nm`);
  const px = await page.evaluate(async () => {
    const S = window.__soap, W = 192, H = 288, raw = atob(await S.shot(W, H)), at = (x, y) => { const i = (y * W + x) * 4; return [raw.charCodeAt(i), raw.charCodeAt(i + 1), raw.charCodeAt(i + 2)]; };
    return { top: at(96, 22), mid: at(96, 200), outside: at(2, 2), W, H };
  });
  const l = c => c[0] + c[1] + c[2];
  // 7. Popping empties the frame; dipping brings a film back.
  await page.evaluate(() => { const { NX, NY } = window.__soap.cfg; window.__soap.pop(NX / 2, NY / 3); });
  await step(page, 6); const mid = await sum(page); await step(page, 40); const c = await sum(page);
  const bare = await page.evaluate(async () => { const S = window.__soap, W = 192, H = 288, raw = atob(await S.shot(W, H)), at = (x, y) => { const i = (y * W + x) * 4; return [raw.charCodeAt(i), raw.charCodeAt(i + 1), raw.charCodeAt(i + 2)]; }; return { top: at(96, 22), mid: at(96, 200) }; });
  check('black film draws nearly as dark as no film, thicker film draws bright', l(px.top) - l(bare.top) < 30 && l(px.top) >= l(bare.top) && l(px.mid) - l(bare.mid) > 150,
    `rgb at the top with black film ${px.top} and with no film ${bare.top}; lower down with film ${px.mid} and without ${bare.mid}`);
  check('a pop opens a hole that empties the frame', mid.zero > 0.01 && mid.zero < 0.9 && c.zero === 1, `film gone: ${(mid.zero * 100).toFixed(0)} % after 0.1 s, ${(c.zero * 100).toFixed(0)} % after 0.77 s`);
  await page.evaluate(() => window.__soap.dip('fresh', 5)); await step(page, 5); const d = await sum(page);
  check('dipping again makes a new film', d.zero === 0 && d.mean > 300, `mean thickness ${d.mean.toFixed(0)} nm, ${d.n} cells of film`);
});

// 8. The round wand.
await film('n=96&shape=loop&kind=fresh&seed=2', async (page, errors) => {
  const a = await sum(page); await step(page, 240); const b = await sum(page);
  const { NX, margin } = await page.evaluate(() => window.__soap.cfg), disc = Math.PI * (NX / 2 - margin) ** 2;
  check('the loop holds a round film that sorts by thickness', Math.abs(a.n / disc - 1) < 0.03 && b.cy > a.cy && !b.nan && errors.length === 0, `${a.n} cells against ${disc.toFixed(0)} for the disc; thickness-weighted height ${a.cy.toFixed(3)} -> ${b.cy.toFixed(3)}`);
});

// 9. The page as a visitor gets it: canvas on screen, controls wired, no errors.
{
  const { browser, page, errors } = await open(url, 'n=96');
  await page.waitForTimeout(2500);
  const ui = await page.evaluate(() => ({ nogpu: document.getElementById('nogpu').hidden, canvas: [document.getElementById('film').width, document.getElementById('film').height],
    buttons: [...document.querySelectorAll('button')].filter(b => !b.hidden).map(b => b.textContent) }));
  await page.click('#info'); const open1 = await page.evaluate(() => document.getElementById('about').open);
  await page.click('#shape'); await page.waitForTimeout(600);
  await page.click('#close'); await page.click('#dip'); await page.waitForTimeout(400);
  await page.click('#full'); await page.waitForTimeout(500);
  const full = await page.evaluate(() => ({ hud: getComputedStyle(document.getElementById('hud')).display, exit: !document.getElementById('exit').hidden }));
  await page.screenshot({ path: path.resolve(import.meta.dirname, '../out/full.png') }).catch(() => {});
  await page.click('#exit'); await page.waitForTimeout(400);
  const back = await page.evaluate(() => getComputedStyle(document.getElementById('hud')).display);
  // A popped film comes back by itself.
  await page.mouse.click(210, 400); await page.waitForTimeout(900);
  const gone = await page.evaluate(() => window.__soapEmpty());
  await page.waitForFunction(() => !window.__soapEmpty(), null, { timeout: 30000 }).catch(() => {});
  const again = await page.evaluate(() => !window.__soapEmpty());
  const label = await page.textContent('#shape');
  check('the live page runs and its controls work', ui.nogpu && errors.length === 0 && open1 && label === 'Round wand' && full.hud === 'none' && full.exit && back !== 'none' && again, `canvas ${ui.canvas.join('x')}, buttons ${ui.buttons.join('/')}, about opened ${open1}, shape button now "${label}", full screen hides the controls ${full.hud === 'none'} and ✕ restores them ${back !== 'none'}, a tap pops the film (empty ${gone}) and a new one follows ${again}, errors: ${errors.slice(0, 3).join(' | ') || 'none'}`);
  await page.screenshot({ path: path.resolve(import.meta.dirname, '../out/page.png') }).catch(() => {});
  await browser.close();
}

srv.close();
const failed = results.filter(r => !r.pass).length;
await writeFile(path.resolve(import.meta.dirname, '../data/check.json'), JSON.stringify({ ran: new Date().toISOString().slice(0, 10), gpu: 'SwiftShader (headless Chromium)', results }, null, 1));
console.log(`\n${results.length - failed}/${results.length} checks pass`);
process.exit(failed ? 1 : 0);
