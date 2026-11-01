// Tilt Invaders: wave 7 (shaped formation, divers, split drones, aimed shots)
// with spread, wingmen, missiles and a shield. The ship is flown by synthetic
// tilt in a sweep back and forth under the formation; the guns autofire.
const { sequenceGPU, gammaFor, drain } = require('./_gpu.cjs');
const WAVE = Number(process.env.TI_WAVE || 7);
const BETA0 = 50, RANGE = 22, W = 360;
const DEV = '() => window.__ti.renderer && window.__ti.renderer.device';

module.exports = async (b, h) => {
  const page = await h.open(b, 'tilt-invaders/');
  page.setDefaultTimeout(180000);
  // Freeze, then reload so the game boots on the fake clock (freezing a running
  // page makes rAF time jump back and stalls the step accumulator). fastForward
  // fires timers and one rAF per call, so booting renders only a few frames.
  await h.freezeClock(page);
  await page.reload({ waitUntil: 'load' });
  for (let k = 0; k < 30; k++) {
    await page.clock.fastForward(300); await page.waitForTimeout(300);
    if (await page.evaluate(() => !!(window.__ti && window.__ti.renderer))) break;
  }
  await drain(page, DEV);
  await page.locator('#startBtn').click();
  await page.clock.runFor(100);
  await h.tilt(page, BETA0, 0);
  await page.clock.runFor(300);
  await h.tilt(page, BETA0, 0);
  await page.clock.runFor(300);
  if (await page.locator('#plumbBtn').isVisible()) await page.locator('#plumbBtn').click();
  await page.clock.runFor(200);
  await page.evaluate((wave) => {
    const T = window.__ti, g = T.game;
    g.wave = wave; T.startWave();
    for (const id of ['spread', 'spread', 'wing', 'missile', 'shield']) T.applyPower(id);
  }, WAVE);

  const steer = async () => {
    // sweep under the formation so the tilt steering shows; the shield is kept topped up
    const tx = await page.evaluate(() => 180 + 115 * Math.sin(window.__ti.game.t * 2.4));
    const s = Math.max(-1, Math.min(1, (tx - W / 2) / (W / 2 - 16)));
    await h.tilt(page, BETA0, gammaFor(s * RANGE, BETA0));
  };
  // warm up: formation drops in and the guns start; banner removed by hand
  for (let i = 0; i < 40; i++) { await steer(); await page.clock.runFor(33); await drain(page, DEV); }
  await page.evaluate(() => { const m = document.getElementById('msg'); if (m) m.hidden = true; });

  await sequenceGPU(page, h, 'tilt-invaders', 3, 90, async (i) => {
    await steer();
    const st = await page.evaluate(() => { const g = window.__ti.game; return { ph: g.phase, lives: g.lives, inv: g.inv.filter(v => v.alive).length, sh: g.shield, sc: g.score, x: Math.round(window.__ti.ship.x), aim: Math.round(window.__ti.demoTarget()), ok: window.__ti.motion.ok, tilt: window.__ti.settings.tilt }; });
    if (!st.sh) await page.evaluate(() => window.__ti.applyPower('shield'));
    if (i % 15 === 0) console.log(i, JSON.stringify(st));
  }, DEV);
  await page.context().close();
};
