// Tilt Rally: the ?demo autopilot races. The fake clock is installed before the page
// loads (reload), so rAF timestamps share one time base. It is paused at once and the
// countdown plus the first seconds of the race are stepped 100 ms per rAF (the game's dt
// clamp), which is far faster than waiting in real time on SwiftShader.
// pauseAt with a margin; under heavy load the round trip can outlast the margin, so retry
async function pauseClock(page) {
  for (let i = 0; ; i++) {
    try { return await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1500)); }
    catch (e) { if (i > 5 || !/past/.test(e.message)) throw e; }
  }
}
module.exports = async (b, h) => {
  const page = await h.open(b, 'tilt-rally/index.html?demo&seed=7', { landscape: true });
  await page.clock.install();
  await page.reload({ waitUntil: 'load' });
  await page.waitForFunction(() => window.__game?.()?.cars, null, { timeout: 120000, polling: 250 });
  await pauseClock(page);
  const T0 = +(process.env.RALLY_T || 7);
  for (let i = 0; i < 400; i++) {
    await page.clock.fastForward(100);
    if (i % 10 === 9) { const t = await page.evaluate(() => window.__game().raceT); if (t > T0) break; }
  }
  console.log('raceT', await page.evaluate(() => window.__game().raceT));
  await h.sequence(page, 'tilt-rally', 5, 150);
  console.log('raceT end', await page.evaluate(() => window.__game().raceT));
  await page.context().close();
};
