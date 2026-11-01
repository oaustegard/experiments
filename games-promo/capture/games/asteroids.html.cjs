// Asteroids (2D, Preact): keyboard play. The fake clock goes in before load (reload) so
// rAF timestamps share one time base with the frozen clock; otherwise the dt goes negative.
// pauseAt with a margin; under heavy load the round trip can outlast the margin, so retry
async function pauseClock(page) {
  for (let i = 0; ; i++) {
    try { return await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1500)); }
    catch (e) { if (i > 5 || !/past/.test(e.message)) throw e; }
  }
}
module.exports = async (b, h) => {
  const page = await h.open(b, 'asteroids.html');
  // seeded Math.random so the rock field (and whether a rock hits the ship) repeats run to run
  await page.context().addInitScript((seed) => { let s = seed >>> 0;
    Math.random = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }, +(process.env.AST_SEED || 3));
  // The main-loop effect re-runs on every score change and re-sets canvas.width, which clears
  // the canvas until the next rAF; a screenshot in between came out blank. Skip same-size sets.
  await page.context().addInitScript(() => {
    for (const k of ['width', 'height']) {
      const d = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, k);
      Object.defineProperty(HTMLCanvasElement.prototype, k, { configurable: true, get: d.get,
        set(v) { if (d.get.call(this) !== Math.floor(v)) d.set.call(this, v); } });
    }
  });
  await page.clock.install();
  await page.reload({ waitUntil: 'load' });
  await page.getByRole('button', { name: 'Start', exact: true }).waitFor({ timeout: 60000 });
  await pauseClock(page);
  await page.getByRole('button', { name: 'Start', exact: true }).click();
  await page.clock.runFor(100);
  const k = page.keyboard;
  await k.down('Space');
  // plan, in frames of the whole run (warm-up + sequence): turn, thrust, turn the other way
  const held = new Set();
  const want = async (keys) => {
    for (const x of [...held]) if (!keys.includes(x)) { await k.up(x); held.delete(x); }
    for (const x of keys) if (!held.has(x)) { await k.down(x); held.add(x); }
  };
  const plan = (t) => { // t in page seconds since start
    const s = t % 4;
    if (s < 1.5) return ['ArrowLeft'];
    if (s < 1.7) return ['ArrowUp'];
    if (s < 3.4) return ['ArrowRight'];
    return ['ArrowLeft'];
  };
  // warm-up 2.5 s (past the start shield's blinking) of shooting so rocks are already splitting
  for (let i = 0; i < 75; i++) { await want(plan(i / 30)); await page.clock.runFor(33); }
  await page.screenshot({ path: require('path').join(require('os').tmpdir(), 'asteroids-warm.jpg') }); // first screenshot comes out blank
  await h.sequence(page, 'asteroids.html', 2.5, 75, async (i) => want(plan(2.5 + i / 30)));
  await page.context().close();
};
