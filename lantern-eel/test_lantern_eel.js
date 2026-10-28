// Headless checks for Lantern Eel. Run from this directory with the site repo cloned beside experiments:
//   node test_lantern_eel.js [path-to-index.html]
const { chromium } = require('playwright');
const path = require('path');
const assert = require('assert');
const PAGE = 'file://' + path.resolve(process.argv[2] || '../../oaustegard.github.io/fun-and-games/lantern-eel/index.html');
const SENSORS = ['accelerometer', 'gyroscope', 'magnetometer'];
const ev = () => { window.__ev = (beta, gamma) => dispatchEvent(Object.assign(new Event('deviceorientation'), { beta, gamma })); };

async function common(p, tag) {
  // overlays honour [hidden] outside the artifact skeleton, and nothing scrolls sideways
  assert.equal(await p.evaluate(() => getComputedStyle(document.querySelector('#overView')).display), 'none', tag);
  assert.ok(await p.evaluate(() => document.documentElement.scrollWidth <= innerWidth), tag + ' horizontal scroll');
  // turns queue and refuse reversals
  const q = await p.evaluate(() => { __eel.start(); const S = __eel.S; __eel.turn('down'); __eel.turn('left'); return S.queue.slice(); });
  assert.deepEqual(q, ['left'], tag);
  // plankton scores, grows and chains
  const e = await p.evaluate(() => {
    __eel.start(); const S = __eel.S; S.jellies = []; S.fish = null; S.pearl = null;
    const h = S.snake[0], len = S.snake.length;
    S.food = { x: h.x, y: h.y - 1, rx: h.x, ry: h.y - 1, s: 1 };
    __eel.step(); __eel.step();
    return { score: S.score, grow: S.snake.length + S.grow - len, combo: S.combo };
  });
  assert.ok(e.score >= 10 && e.grow === 1 && e.combo === 1, tag + JSON.stringify(e));
  // a pearl's phase passes a jellyfish; without it the jellyfish ends the dive
  const j = await p.evaluate(() => {
    __eel.start(); const S = __eel.S, h = S.snake[0]; S.food = null;
    S.jellies = [{ x: h.x, y: h.y - 1, rx: h.x, ry: h.y - 1, s: 1, ph: 0, off: 1 }];
    S.phase = 3; __eel.step(); const alive = __eel.state;
    const h2 = S.snake[0]; S.phase = 0; S.jellies = [{ x: h2.x, y: h2.y - 1, rx: h2.x, ry: h2.y - 1, s: 1, ph: 0, off: 1 }];
    __eel.step(); return [alive, __eel.state];
  });
  assert.deepEqual(j, ['play', 'dying'], tag);
  await p.waitForFunction(() => __eel.state === 'over', null, { timeout: 3000 });
  assert.notEqual(await p.evaluate(() => getComputedStyle(document.querySelector('#overView')).display), 'none', tag);
}

(async () => {
  const b = await chromium.launch();
  const errs = [];
  const open = async (opts, tag) => {
    const ctx = await b.newContext({ permissions: SENSORS, ...opts });
    const p = await ctx.newPage();
    p.on('pageerror', e => errs.push(`${tag}: ${e.message}`));
    p.on('console', m => { if (m.type() === 'error') errs.push(`${tag} console: ${m.text()}`); });
    await p.goto(PAGE); await p.evaluate(ev);
    return [ctx, p];
  };

  // phone: tilt is the default, the deck is hidden, tilt steers with a 6 degree dead zone
  {
    const [ctx, p] = await open({ viewport: { width: 390, height: 800 }, isMobile: true, hasTouch: true }, 'phone');
    assert.equal(await p.evaluate(() => __eel.ctl), 'tilt');
    assert.equal(await p.evaluate(() => getComputedStyle(document.querySelector('.dpad')).display), 'none');
    assert.equal(await p.evaluate(() => document.querySelector('#levelBtn').hidden), false);
    await p.tap('#startBtn');
    await p.waitForFunction(() => __eel.tilt.on, null, { timeout: 2000 });
    assert.notEqual(await p.evaluate(() => __eel.audio), null, 'audio context created on Dive');
    const t = await p.evaluate(() => {
      const S = __eel.S, out = {};
      S.queue = []; S.dir = 'up'; __eel.tilt.neutral = null; __ev(40, 0); __ev(40, 5); out.dead = S.queue.slice();
      __ev(40, 7); out.right = S.queue.slice();
      S.queue = []; S.dir = 'right'; __ev(15, 0); out.up = S.queue.slice();
      S.queue = []; S.dir = 'right'; __ev(70, 0); out.down = S.queue.slice();
      // diagonal hold: the axis already steering keeps priority until the other leads by 3 degrees
      S.queue = []; S.dir = 'up'; __eel.tilt.axis = null; __ev(40, 0); __ev(40, 10); __ev(48, 10); out.held = S.queue.slice();
      return out;
    });
    assert.deepEqual(t, { dead: [], right: ['right'], up: ['up'], down: ['down'], held: ['right'] });
    // tilt yields to a swipe while the finger is down, and ignores the tilt held at release until it changes
    await p.evaluate(() => { const S = __eel.S; S.acc = -1e6; S.queue = []; S.dir = 'right'; __eel.tilt.neutral = null; __ev(40, 0); });
    const box = await p.locator('#c').boundingBox(), cx = box.x + box.width / 2, cy = box.y + box.height / 2;
    await p.mouse.move(cx, cy); await p.mouse.down(); await p.mouse.move(cx, cy - 40, { steps: 4 });
    const during = await p.evaluate(() => { __ev(40, 25); return __eel.S.queue.slice(); });
    await p.mouse.up();
    const after = await p.evaluate(() => { const q = __eel.S.queue; __ev(40, 25); const stale = q.slice(); __ev(40, 0); __ev(40, -25); return [stale, q.slice()]; });
    assert.deepEqual([during, ...after], [['up'], ['up'], ['up', 'left']]);
    await p.evaluate(() => { __eel.S.acc = 0; });
    // switching to arrows in pause shows the deck and stops tilt
    await p.evaluate(() => document.querySelector('#pauseBtn').click());
    await p.locator('#pauseView [data-ctl="arrows"]').tap();
    assert.equal(await p.evaluate(() => [__eel.ctl, __eel.tilt.on, getComputedStyle(document.querySelector('.dpad')).display]).then(r => r.join()), 'arrows,false,grid');
    await p.locator('#pauseView [data-ctl="tilt"]').tap();
    await p.locator('#resumeBtn').tap();
    // rotating mid-dive re-fits the grid to the new shape and carries the eel over in one piece
    const fill = () => p.evaluate(() => { const g = __grid(), c = document.querySelector('#c').getBoundingClientRect(); return [g.w / c.width, g.h / c.height]; });
    await p.evaluate(() => { __eel.S.acc = -1e6; });
    await p.setViewportSize({ width: 844, height: 390 }); await p.waitForTimeout(300);
    const rot = await p.evaluate(() => {
      const S = __eel.S, C = __eel.dims[0], Rr = __eel.dims[1];
      const adj = S.snake.every((q, i) => { if (!i) return true; const a = S.snake[i - 1], dx = Math.abs(a.x - q.x), dy = Math.abs(a.y - q.y);
        return (Math.min(dx, C - dx) + Math.min(dy, Rr - dy)) === 1; });
      return { state: __eel.state, adj, wide: C > Rr };
    });
    assert.deepEqual(rot, { state: 'play', adj: true, wide: true });
    const [fw, fh] = await fill();
    assert.ok(fw > 0.9 && fh > 0.75, `landscape grid fills the screen: ${fw.toFixed(2)} x ${fh.toFixed(2)}`);
    await p.setViewportSize({ width: 390, height: 800 }); await p.waitForTimeout(300);
    await p.evaluate(() => { __eel.S.acc = 0; });
    await common(p, 'phone');
    await ctx.close();
  }

  // desktop and short landscape: arrows by default, left or right, hold the water to boost
  for (const vp of [{ width: 1280, height: 820 }, { width: 844, height: 390 }]) {
    const tag = `${vp.width}`;
    const [ctx, p] = await open({ viewport: vp }, tag);
    assert.equal(await p.evaluate(() => __eel.ctl), 'arrows', tag);
    const sideOf = () => p.evaluate(() => { const d = document.querySelector('.dpad').getBoundingClientRect(), s = document.querySelector('#stage').getBoundingClientRect(); return d.x + d.width / 2 < s.x + s.width / 2 ? 'left' : 'right'; });
    assert.equal(await sideOf(), 'left', tag);
    await p.click('#titleView [data-side="right"]');
    assert.equal(await sideOf(), 'right', tag);
    await p.click('#startBtn');
    const clear = await p.evaluate(() => { const S = __eel.S, d = document.querySelector('.dpad').getBoundingClientRect(), w = document.querySelector('.actions').getBoundingClientRect();
      const g = window.__grid(); return [d, w].every(r => r.right <= g.x || r.left >= g.x + g.w || r.bottom <= g.y || r.top >= g.y + g.h); });
    assert.ok(clear, tag + ' grid overlaps the floating controls');
    const box = await p.locator('#c').boundingBox();
    await p.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await p.mouse.down();
    await p.waitForTimeout(350);
    assert.equal(await p.evaluate(() => __eel.S.boosting), true, tag + ' hold to boost');
    await p.mouse.up(); await p.waitForTimeout(60);
    assert.equal(await p.evaluate(() => __eel.S.boosting), false, tag + ' release ends boost');
    await common(p, tag);
    await ctx.close();
  }
  await b.close();
  assert.deepEqual(errs, []);
  console.log('lantern-eel: all checks passed');
})().catch(e => { console.error(e); process.exit(1); });
