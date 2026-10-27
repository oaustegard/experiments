// Headless checks for Lantern Eel. Run from this directory with the site repo cloned beside experiments:
//   node test_lantern_eel.js [path-to-index.html]
const { chromium } = require('playwright');
const path = require('path');
const assert = require('assert');
const PAGE = 'file://' + path.resolve(process.argv[2] || '../../oaustegard.github.io/fun-and-games/lantern-eel/index.html');
const SENSORS = ['accelerometer', 'gyroscope', 'magnetometer'];

(async () => {
  const b = await chromium.launch();
  const errs = [];
  for (const vp of [{ width: 390, height: 800 }, { width: 844, height: 390 }, { width: 1280, height: 820 }]) {
    const ctx = await b.newContext({ viewport: vp, permissions: SENSORS });
    const p = await ctx.newPage();
    p.on('pageerror', e => errs.push(`${vp.width}: ${e.message}`));
    p.on('console', m => { if (m.type() === 'error') errs.push(`${vp.width} console: ${m.text()}`); });
    await p.goto(PAGE);
    assert.equal(await p.evaluate(() => __eel.state), 'title');
    // overlays honour [hidden] outside the artifact skeleton
    assert.equal(await p.evaluate(() => getComputedStyle(document.querySelector('#overView')).display), 'none');
    await p.click('#startBtn');
    assert.equal(await p.evaluate(() => __eel.state), 'play');
    assert.equal(await p.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'no horizontal scroll');

    // keys queue turns and refuse reversals
    const q = await p.evaluate(() => { const S = __eel.S; S.queue = []; S.dir = 'up'; __eel.turn('down'); __eel.turn('left'); return S.queue.slice(); });
    assert.deepEqual(q, ['left']);

    // tilt (a fresh dive each check: the live loop keeps stepping between evaluate calls)
    // first reading is level, then a tip past 9 degrees steers
    await p.click('#tiltBtn');
    await p.waitForFunction(() => __eel.tilt.on, null, { timeout: 2000 });
    const t = await p.evaluate(() => {
      const ev = (beta, gamma) => dispatchEvent(Object.assign(new Event('deviceorientation'), { beta, gamma }));
      const S = __eel.S, out = {};
      S.queue = []; S.dir = 'up'; __eel.tilt.neutral = null; ev(40, 0); ev(40, 5); out.dead = S.queue.slice();
      ev(40, 25); out.right = S.queue.slice();
      S.queue = []; S.dir = 'right'; ev(15, 0); out.up = S.queue.slice();
      S.queue = []; S.dir = 'right'; ev(70, 0); out.down = S.queue.slice();
      return out;
    });
    assert.deepEqual(t, { dead: [], right: ['right'], up: ['up'], down: ['down'] });

    // eating plankton scores, grows and chains
    const e = await p.evaluate(() => {
      __eel.start(); const S = __eel.S; S.jellies = []; S.fish = null; S.pearl = null;
      const h = S.snake[0], len = S.snake.length;
      S.food = { x: h.x, y: h.y - 1, rx: h.x, ry: h.y - 1, s: 1 };
      __eel.step(); __eel.step();
      return { score: S.score, grow: S.snake.length + S.grow - len, combo: S.combo };
    });
    assert.ok(e.score >= 10 && e.grow === 1 && e.combo === 1, JSON.stringify(e));

    // a jellyfish in the next cell ends the dive; a pearl's phase lets you through
    const j = await p.evaluate(() => {
      __eel.start(); const S = __eel.S, h = S.snake[0]; S.food = null;
      S.jellies = [{ x: h.x, y: h.y - 1, rx: h.x, ry: h.y - 1, s: 1, ph: 0, off: 1 }];
      S.phase = 3; __eel.step(); const alive = __eel.state;
      const h2 = S.snake[0]; S.phase = 0; S.jellies = [{ x: h2.x, y: h2.y - 1, rx: h2.x, ry: h2.y - 1, s: 1, ph: 0, off: 1 }];
      __eel.step(); return [alive, __eel.state];
    });
    assert.deepEqual(j, ['play', 'dying']);
    await p.waitForTimeout(1400);
    assert.equal(await p.evaluate(() => __eel.state), 'over');
    assert.equal(await p.evaluate(() => getComputedStyle(document.querySelector('#overView')).display) !== 'none', true);
    await ctx.close();
  }
  await b.close();
  assert.deepEqual(errs, []);
  console.log('lantern-eel: all checks passed');
})().catch(e => { console.error(e); process.exit(1); });
