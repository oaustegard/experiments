// Asteroids 3D: Launch, then an autopilot in this script steers the arrow keys toward the
// nearest rock ahead while Space is held (rapid fire, aim assist) and R thrusts in bursts.
// Extra rocks are placed ahead of the ship through the page's test hook (window.__a3d) so the
// view is busy. Clock installed before load (reload) so rAF timestamps share one time base.
// pauseAt with a margin; under heavy load the round trip can outlast the margin, so retry
async function pauseClock(page) {
  for (let i = 0; ; i++) {
    try { return await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1500)); }
    catch (e) { if (i > 5 || !/past/.test(e.message)) throw e; }
  }
}
module.exports = async (b, h) => {
  const page = await h.open(b, 'asteroids-3d/');
  await page.context().addInitScript((seed) => { let s = seed >>> 0;
    Math.random = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }, +(process.env.A3D_SEED || 5));
  await page.clock.install();
  await page.reload({ waitUntil: 'load' });
  await page.waitForFunction(() => window.__a3d, null, { timeout: 120000 });
  await page.waitForTimeout(3000); // WebGPU init and the nebula bake
  await pauseClock(page);
  await page.locator('#start').click();
  await page.clock.runFor(100);
  // more rocks, spread in a cone ahead (forward is -z at launch), all three sizes
  await page.evaluate(() => {
    const A = window.__a3d, p0 = A.ship.p, R = Math.random;
    for (let i = 0; i < 14; i++) {
      const d = 45 + R() * 120, x = (R() - 0.5) * d * 0.9, y = (R() - 0.5) * d * 0.6;
      A.rocks.push(A.makeRock(i % 3 === 2 ? 1 : 0, [p0[0] + x, p0[1] + y, p0[2] - d]));
    }
  });
  const k = page.keyboard, held = new Set();
  const want = async (keys) => {
    for (const x of [...held]) if (!keys.includes(x)) { await k.up(x); held.delete(x); }
    for (const x of keys) if (!held.has(x)) { await k.down(x); held.add(x); }
  };
  // target in ship-local coordinates: x right, y up, z ahead
  const aim = () => page.evaluate(() => {
    const A = window.__a3d, s = A.ship, q = s.q;
    const cross = (a, c) => [a[1] * c[2] - a[2] * c[1], a[2] * c[0] - a[0] * c[2], a[0] * c[1] - a[1] * c[0]];
    const rot = (v) => { const u = [q[0], q[1], q[2]], t = cross(u, v).map(e => e * 2), ut = cross(u, t); return [0, 1, 2].map(i => v[i] + t[i] * q[3] + ut[i]); };
    const r = rot([1, 0, 0]), up = rot([0, 1, 0]), f = rot([0, 0, -1]);
    const wr = (a) => ((a + 180) % 360 + 360) % 360 - 180;
    let best = null;
    for (const k of A.rocks) {
      const d = [0, 1, 2].map(i => wr(k.p[i] - s.p[i]));
      const z = d[0] * f[0] + d[1] * f[1] + d[2] * f[2], x = d[0] * r[0] + d[1] * r[1] + d[2] * r[2], y = d[0] * up[0] + d[1] * up[1] + d[2] * up[2];
      const dist = Math.hypot(...d), score = dist * (z > 0 ? 1 : 3);
      if (!best || score < best.score) best = { x, y, z, dist, score };
    }
    return { best, phase: A.game.phase, score: A.game.score, lives: A.game.lives, rocks: A.rocks.length };
  });
  const drive = async (t) => {
    const { best } = await aim();
    const keys = ['Space'];
    if (best) {
      const ax = Math.atan2(best.x, best.z), ay = Math.atan2(best.y, best.z);
      if (ax > 0.05) keys.push('ArrowRight'); else if (ax < -0.05) keys.push('ArrowLeft');
      if (ay > 0.05) keys.push('ArrowDown'); else if (ay < -0.05) keys.push('ArrowUp');
      if ((t % 2) < 0.6 && best.dist > 50) keys.push('KeyR');
    }
    await want(keys);
  };
  for (let i = 0; i < 75; i++) { await drive(i / 30); await page.clock.runFor(33); } // past the WAVE 1 banner
  console.log('start', JSON.stringify(await aim()));
  await h.sequence(page, 'asteroids-3d', 3, 90, (i) => drive(2.5 + i / 30));
  console.log('end', JSON.stringify(await aim()));
  await page.context().close();
};
