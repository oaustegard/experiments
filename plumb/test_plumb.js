// Playwright check of Plumb's rules and controls. Serve web/ on :8765 first:
//   (cd plumb/web && python3 -m http.server 8765) & node plumb/test_plumb.js
const { chromium } = require('playwright');
const assert = (c, m) => { if (!c) { console.error('FAIL', m); process.exitCode = 1; } else console.log('ok  ', m); };
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true,
    permissions: ['accelerometer', 'gyroscope'] });
  const p = await ctx.newPage();
  const errs = [];
  p.on('pageerror', e => errs.push(e.message));
  await p.goto('http://localhost:8765/?debug');
  await p.click('#startBtn');
  await p.waitForTimeout(200);
  assert(await p.evaluate(() => document.querySelector('#overlay').hidden), 'overlay closes on Start');

  // Single line clear: fill the floor except four cells, drop a flat I into the gap.
  const r1 = await p.evaluate(() => {
    const P = window.__plumb, g = P.grid();
    for (let x = 4; x < 10; x++) g[21][x] = 'Z';
    P.place('I', 0); P.hardDrop();
    return new Promise(res => setTimeout(() => res({ ...P.stats(), floor: g[21].filter(Boolean).length }), 400));
  });
  assert(r1.lines === 1 && r1.floor === 0, `single clear empties the floor (lines ${r1.lines}, floor ${r1.floor})`);
  assert(r1.score >= 100, `single scores 100 + drop bonus (${r1.score})`);

  // Four-line clear: a vertical I into a four-deep well.
  const r4 = await p.evaluate(() => {
    const P = window.__plumb, g = P.grid();
    for (let y = 18; y < 22; y++) for (let x = 0; x < 9; x++) g[y][x] = 'S';
    P.place('I', 7, 1); P.hardDrop();
    return new Promise(res => setTimeout(() => res(P.stats()), 400));
  });
  assert(r4.lines === 5, `four-line clear counts four (lines ${r4.lines})`);

  // SRS kick: a T rotating against the left wall is kicked back inside.
  const kick = await p.evaluate(() => {
    const P = window.__plumb; P.place('T', 0, 3); P.tryMove(-1, 0); P.rotate(1);
    const pc = P.piece(); return { x: pc.x, r: pc.r };
  });
  assert(kick.r === 0 && kick.x >= 0, `T rotates off the left wall via kick (x ${kick.x}, r ${kick.r})`);

  // Tilt steering: roll right 30° for 0.6 s moves the piece right.
  const x0 = await p.evaluate(() => { const P = window.__plumb; P.place('O', 4);
    dispatchEvent(Object.assign(new Event('deviceorientation'), { beta: 45, gamma: 0 })); return P.piece().x; });
  await p.evaluate(() => { window.__t = setInterval(() => dispatchEvent(Object.assign(new Event('deviceorientation'), { beta: 45, gamma: 30 })), 20); });
  await p.waitForTimeout(600);
  const x1 = await p.evaluate(() => { clearInterval(window.__t); dispatchEvent(Object.assign(new Event('deviceorientation'), { beta: 45, gamma: 0 })); return window.__plumb.piece().x; });
  assert(x1 > x0, `right tilt steers right (${x0} -> ${x1})`);

  // Hold: swaps the active piece into the hold pod.
  const h = await p.evaluate(() => { const P = window.__plumb; P.place('J', 3); P.hold(); return document.querySelectorAll('#holdSvg .mini').length; });
  assert(h === 4, 'hold draws the held piece');

  // Touch: swipe down hard-drops.
  const before = await p.evaluate(() => window.__plumb.piece().t + window.__plumb.stats().score);
  const box = await p.locator('#well').boundingBox();
  await p.evaluate(({ x, y }) => {
    const w = document.querySelector('#well'), o = { pointerId: 7, bubbles: true, isPrimary: true, pointerType: 'touch' };
    w.dispatchEvent(new PointerEvent('pointerdown', { ...o, clientX: x, clientY: y }));
    w.dispatchEvent(new PointerEvent('pointermove', { ...o, clientX: x, clientY: y + 60 }));
    w.dispatchEvent(new PointerEvent('pointerup', { ...o, clientX: x, clientY: y + 140 }));
  }, { x: box.x + box.width / 2, y: box.y + 100 });
  const after = await p.evaluate(() => window.__plumb.piece().t + window.__plumb.stats().score);
  assert(before !== after, 'swipe down hard-drops');

  assert(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join('; ') : ''));
  if (process.env.SHOT) await p.screenshot({ path: process.env.SHOT });
  await b.close();
})();
