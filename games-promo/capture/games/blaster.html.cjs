// Blaster: keyboard play. Clock installed before load (reload) so rAF timestamps share the
// frozen clock's time base. Warm-up holds fire so saucers fill the sky; during the sequence
// the ship tracks the lowest saucer (found by its orange pixels on the canvas) and fires.
// pauseAt with a margin; under heavy load the round trip can outlast the margin, so retry
async function pauseClock(page) {
  for (let i = 0; ; i++) {
    try { return await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1500)); }
    catch (e) { if (i > 5 || !/past/.test(e.message)) throw e; }
  }
}
module.exports = async (b, h) => {
  const page = await h.open(b, 'blaster.html');
  await page.clock.install();
  await page.reload({ waitUntil: 'load' });
  await page.getByRole('button', { name: 'Start Game' }).waitFor({ timeout: 60000 });
  await pauseClock(page);
  await page.getByRole('button', { name: 'Start Game' }).click();
  const k = page.keyboard;
  let dir = null;
  const press = async (d) => { if (d !== dir) { if (dir) await k.up(dir); if (d) await k.down(d); dir = d; } };
  // lowest saucer x and player x as fractions of the canvas width
  const look = () => page.evaluate(() => {
    const c = document.querySelector('canvas'), x = c.getContext('2d');
    const W = c.width, H = c.height, d = x.getImageData(0, 0, W, H).data;
    const at = (px, py) => { const i = (py * W + px) * 4; return [d[i], d[i + 1], d[i + 2]]; };
    let enemy = null, ply = null;
    for (let py = Math.floor(H * 0.8); py > H * 0.05 && !enemy; py -= 3)
      for (let px = 0; px < W; px += 3) { const [r, g, bb] = at(px, py); if (r > 220 && g > 140 && g < 200 && bb < 60) { enemy = px / W; break; } }
    for (let py = Math.floor(H * 0.97); py > H * 0.8 && ply == null; py -= 2)
      for (let px = 0; px < W; px += 2) { const [r, g, bb] = at(px, py); if (g > 200 && r > 90 && r < 160 && bb < 60) { ply = px / W; break; } }
    return { enemy, ply };
  });
  const track = async () => {
    const { enemy, ply } = await look();
    if (enemy == null || ply == null) return press(null);
    const dx = enemy - ply - 0.02; // ply is the left edge of the ship's sprite rows
    await press(Math.abs(dx) < 0.02 ? null : dx < 0 ? 'ArrowLeft' : 'ArrowRight');
  };
  for (let i = 0; i < 165; i++) await page.clock.runFor(33); // ~5.5 s: saucers drift down unopposed
  await k.down('Space');
  await h.sequence(page, 'blaster.html', 2.5, 75, track);
  console.log('last look', JSON.stringify(await look()));
  await page.context().close();
};
