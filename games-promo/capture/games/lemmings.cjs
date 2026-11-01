// Lemmings: level 8 "Exodus" (10,000 lemmings). Clock installed before load (reload) so rAF
// timestamps share the frozen clock's time base. Fast-forward ("F", 4x) until thousands are
// out, then capture at normal speed while the camera pans across the eight hatches.
// pauseAt with a margin; under heavy load the round trip can outlast the margin, so retry
async function pauseClock(page) {
  for (let i = 0; ; i++) {
    try { return await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1500)); }
    catch (e) { if (i > 5 || !/past/.test(e.message)) throw e; }
  }
}
module.exports = async (b, h) => {
  const page = await h.open(b, 'lemmings/', { landscape: true });
  await page.clock.install();
  await page.reload({ waitUntil: 'load' });
  await page.locator('#bStart').waitFor({ state: 'attached', timeout: 60000 });
  await page.waitForFunction(() => window.gpuLemmings?.game, null, { timeout: 240000, polling: 500 });
  await pauseClock(page);
  await page.getByRole('button', { name: /Exodus/ }).click();
  await page.locator('#bStart').click();
  await page.clock.runFor(100);
  await page.keyboard.press('f'); // fast
  // SwiftShader runs the compute far slower than rAF fires: wait for the GPU after each step,
  // or the queue backs up and the stats readback and screenshots stall
  const gpu = () => page.evaluate(() => window.gpuLemmings.game.device.queue.onSubmittedWorkDone());
  const W = () => page.evaluate(() => { const w = window.gpuLemmings.game.world; return w && { out: w.out, saved: w.saved, time: w.time }; });
  const OUT_TARGET = +(process.env.LEM_OUT || 4000);
  for (let i = 0; i < 600; i++) {
    await page.clock.fastForward(200); // dt is capped at 200 ms; x4 fast, up to 16 ticks per frame
    await gpu();
    if (i % 5 === 4) { const w = await W(); if (i % 20 === 19) console.log('warm', JSON.stringify(w)); if (w && w.out >= OUT_TARGET) break; }
  }
  console.log('start', JSON.stringify(await W()));
  await page.keyboard.press('f'); // back to normal speed
  await page.clock.runFor(100);
  // pan right across the hatches: hold the right arrow for part of every frame
  let held = false;
  await h.sequence(page, 'lemmings', 5, 150, async (i) => {
    await gpu();
    const want = i >= 15 && i < 100; // ~700 level px: hatches, then the stream pouring over the drop
    if (want !== held) { if (want) await page.keyboard.down('ArrowRight'); else await page.keyboard.up('ArrowRight'); held = want; }
  });
  console.log('end', JSON.stringify(await W()));
  await page.context().close();
};
