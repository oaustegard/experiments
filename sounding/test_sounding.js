// Playwright check of Sounding's rules, input and both renderers.
// Serve oaustegard.github.io/fun-and-games/sounding on :8766 first:
//   (cd fun-and-games/sounding && python3 -m http.server 8766) &   # in oaustegard.github.io
//   node sounding/test_sounding.js
const { chromium } = require('playwright');
const assert = (c, m) => { if (!c) { console.error('FAIL', m); process.exitCode = 1; } else console.log('ok  ', m); };
const URL = 'http://localhost:8766/';
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 390, height: 760 }, isMobile: true, hasTouch: true });
  const p = await ctx.newPage();
  const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  await p.goto(URL + '?debug&nogpu&seed=11');
  await p.waitForFunction(() => window.__sounding);
  const S = (f, ...a) => p.evaluate(([f, a]) => window.__sounding[f](...a), [f, a]);
  assert((await S('state')).renderer === '2d', '?nogpu draws the plain 2D view');

  // First reveal is never a mine and opens a region (its 3x3 is kept clear, so it is a zero).
  for (const [x, y] of [[0, 0], [4, 4], [8, 8], [0, 8]]) {
    await S('newGame', 'small');
    await S('reveal', x, y);
    const st = await S('state'), c = await S('cell', x, y);
    assert(st.status === 'play' && !c.mine && c.count === 0 && st.opened > 1, `first reveal at ${x},${y} is safe and floods (${st.opened} opened)`);
  }

  // Fixed layout: mines at (0,0) and (2,0) on a 9x9.
  await S('newGame', 'small');
  await S('setMines', [[0, 0], [2, 0]]);
  assert((await S('cell', 1, 0)).count === 2 && (await S('cell', 1, 1)).count === 2 && (await S('cell', 3, 1)).count === 1, 'neighbour counts');
  await S('reveal', 1, 1);
  assert((await S('state')).opened === 1, 'a numbered cell opens alone');
  await S('chord', 1, 1);
  assert((await S('state')).opened === 1, 'chord with too few marks does nothing');
  await S('flag', 0, 0);
  assert((await S('state')).flagged === 1 && (await p.textContent('#mineCount')) === '1', 'marking updates the mine counter');
  await S('reveal', 0, 0);
  assert((await S('cell', 0, 0)).state === 2, 'a marked cell ignores reveal');
  await S('flag', 2, 0);
  await S('chord', 1, 1);
  let st = await S('state');
  assert(st.status === 'won' && st.opened === 79, `chord on a satisfied 2 opens the rest and wins (${st.status}, ${st.opened})`);
  await p.waitForTimeout(1600);
  assert(!(await p.isHidden('#overlay')) && (await p.textContent('#ovTitle')) === 'FIELD CLEARED', 'win card shows');
  assert((await p.textContent('#best')) !== '—', 'best time recorded');
  await p.click('#againBtn');
  assert(await p.isHidden('#overlay') && (await S('state')).status === 'ready', 'Again starts a fresh field');

  // Wrong chord: mark the wrong neighbour, chord, and it detonates.
  await S('setMines', [[0, 0], [2, 0]]);
  await S('reveal', 1, 1); await S('flag', 0, 0); await S('flag', 1, 0);
  await S('chord', 1, 1);
  st = await S('state');
  assert(st.status === 'lost', 'chord with a wrong mark loses');
  assert((await S('cell', 0, 0)).state === 2 && (await S('cell', 2, 0)).state === 1, 'correct marks stay, unmarked mines open');

  // Touch: tap reveals, hold marks, MARK mode turns taps into marks.
  await S('newGame', 'small');
  await S('setMines', [[8, 8]]);
  const L = await S('layout');
  const at = (x, y) => [L.orgCss[0] + (x + 0.5) * L.cellCss, L.orgCss[1] + (y + 0.5) * L.cellCss];
  const hold = async (x, y, ms) => {
    const [cx, cy] = at(x, y);
    await p.dispatchEvent('#c2', 'pointerdown', { pointerId: 3, pointerType: 'touch', button: 0, clientX: cx, clientY: cy, bubbles: true });
    await p.waitForTimeout(ms);
    await p.dispatchEvent('#c2', 'pointerup', { pointerId: 3, pointerType: 'touch', button: 0, clientX: cx, clientY: cy, bubbles: true });
  };
  await hold(8, 8, 500);
  assert((await S('cell', 8, 8)).state === 2, 'long press marks');
  await hold(8, 8, 500);
  assert((await S('cell', 8, 8)).state === 0, 'long press again unmarks');
  await p.click('#flagBtn');
  await hold(7, 8, 60);
  assert((await S('cell', 7, 8)).state === 2, 'MARK mode: a tap marks');
  await p.click('#flagBtn');
  await hold(7, 8, 60);
  assert((await S('cell', 7, 8)).state === 2, 'a tap on a mark does not reveal it');
  await hold(0, 0, 60);
  st = await S('state');
  assert(st.status === 'play' && st.opened === 79, `a tap reveals (${st.opened} opened)`);

  // The wide board stands up on a portrait screen and lies down on a landscape one.
  await S('newGame', 'large');
  st = await S('state');
  assert(st.C === 16 && st.R === 30 && st.M === 99, `large on portrait is 16x30 (${st.C}x${st.R})`);
  await p.setViewportSize({ width: 1000, height: 700 });
  await S('newGame', 'large');
  st = await S('state');
  assert(st.C === 30 && st.R === 16, `large on landscape is 30x16 (${st.C}x${st.R})`);
  await p.reload();
  await p.waitForFunction(() => window.__sounding);
  assert((await S('state')).level === 'large', 'level persists across reloads');

  // WebGPU path: offscreen render on SwiftShader, no errors, and real pixels in the canvas.
  const g = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const gp = await g.newPage({ viewport: { width: 390, height: 760 } });
  const gerr = [];
  gp.on('pageerror', e => gerr.push(e.message));
  gp.on('console', m => { if (m.type() === 'error' || m.type() === 'warning') gerr.push(m.text()); });
  await gp.goto(URL + '?debug&offscreen&seed=5');
  await gp.waitForFunction(() => window.__sounding, null, { timeout: 30000 });
  await gp.evaluate(() => { __sounding.newGame('small'); __sounding.reveal(4, 4); });
  await gp.waitForTimeout(3000);
  await gp.evaluate(() => { const S = __sounding; for (let y = 0; y < 9; y++) for (let x = 0; x < 9; x++) if (S.cell(x, y).mine) return S.reveal(x, y); });
  await gp.waitForTimeout(3000);
  const px = await gp.evaluate(() => {
    const c = document.getElementById('c2'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let lit = 0; for (let i = 0; i < d.length; i += 4 * 37) if (d[i] + d[i + 1] + d[i + 2] > 60) lit++;
    return { lit, renderer: __sounding.state().renderer };
  });
  assert(px.renderer === 'webgpu', 'SwiftShader adapter runs the WebGPU renderer');
  assert(px.lit > 200, `WebGPU frames reach the canvas (${px.lit} lit samples)`);
  assert(!gerr.length, `no WebGPU errors (${gerr.slice(0, 3).join(' | ')})`);
  await g.close();

  assert(!errs.length, `no page errors (${errs.join(' | ')})`);
  await b.close();
})();
