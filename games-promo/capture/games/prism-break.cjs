// Prism Break: level 8 "Fuse" (rows of bombs), multiball, lasers and a wide paddle.
// The paddle is steered by synthetic tilt aimed under the lowest falling ball.
// Clock frozen before Start so rAF time never runs backwards inside the game.
const LEVEL = Number(process.env.PB_LEVEL || 8);
const BETA0 = 50, RANGE = 22; // plumbed pose; default full-sweep tilt in degrees

const fs = require('fs'), path = require('path');

// SwiftShader WebGPU falls behind a frozen clock: frames queue up and a
// screenshot shows one from a second or more earlier. Wait for the GPU queue
// to drain after each step. Same stepping as lib.sequence otherwise.
const drain = (page) => page.evaluate(() => { const r = window.__pb.renderer; return r && r.device ? r.device.queue.onSubmittedWorkDone() : null; });
async function sequenceGPU(page, h, name, pageSeconds, frames, onFrame) {
  const dir = path.join(h.OUT, name);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const per = (pageSeconds * 1000) / frames, TICK = Math.min(34, per), step = per - TICK;
  for (let i = 0; i < frames; i++) {
    if (onFrame) await onFrame(i);
    if (step > 0) await page.clock.fastForward(step);
    await page.clock.runFor(TICK);
    await drain(page);
    await page.waitForTimeout(40);
    await page.screenshot({ path: path.join(dir, `f${String(i).padStart(4, '0')}.jpg`), quality: 90 });
  }
  console.log(name, frames, 'frames ->', dir);
}

module.exports = async (b, h) => {
  const page = await h.open(b, 'prism-break/');
  page.setDefaultTimeout(180000); // SwiftShader WebGPU on a shared, loaded box can take a while per frame
  await page.waitForTimeout(3000);
  await h.freezeClock(page);
  // The fake clock starts behind the real rAF timestamps, so the game's first
  // dt is a large negative number and its step accumulator stalls until page
  // time catches up. Run the menu's attract mode until the sim moves again.
  for (let k = 0; k < 40; k++) {
    const t0 = await page.evaluate(() => __pb.game.t);
    await page.clock.runFor(500); await drain(page);
    if (await page.evaluate((t0) => __pb.game.t > t0 + 0.2, t0)) break;
  }
  await page.locator('#startBtn').click();
  await page.clock.runFor(100);
  await h.tilt(page, BETA0, 0);
  await page.clock.runFor(300);
  await h.tilt(page, BETA0, 0);
  await page.clock.runFor(300);
  if (await page.locator('#plumbBtn').isVisible()) await page.locator('#plumbBtn').click();
  await page.clock.runFor(200);
  await page.evaluate((lvl) => {
    const P = window.__pb, g = P.game;
    g.level = lvl; P.buildLevel(lvl);
    P.applyPower('wide'); P.applyPower('laser'); P.applyPower('laser');
    P.action(); // launch
  }, LEVEL);

  // gamma that puts the paddle at x (game units, W = 360)
  const steer = async () => {
    const tx = await page.evaluate(() => {
      const g = __pb.game, W = 360, PY = 552;
      const free = g.balls.filter(b => !b.stuck);
      if (!free.length) return W / 2;
      // the falling ball that reaches the paddle first, else the lowest
      const down = free.filter(b => b.vy > 0).map(b => ({ b, t: (PY - b.y) / b.vy })).sort((a, c) => a.t - c.t)[0];
      if (!down) return free.sort((a, c) => c.y - a.y)[0].x;
      let x = down.b.x + down.b.vx * down.t; // reflect off the side walls
      for (let k = 0; k < 4; k++) { if (x < 0) x = -x; if (x > W) x = 2 * W - x; }
      return x;
    });
    const half = await page.evaluate(() => __pb.paddle.half);
    const span = 180 - half - 3, s = Math.max(-1, Math.min(1, (tx - 180) / span));
    const sd = s * RANGE * Math.PI / 180; // wanted asin(gx) relative to level (gx0 = 0)
    const gamma = Math.asin(Math.max(-1, Math.min(1, Math.sin(sd) / Math.cos(BETA0 * Math.PI / 180)))) * 180 / Math.PI;
    await h.tilt(page, BETA0, gamma);
  };
  // short warm-up: the ball leaves the paddle and splits; the bricks are still whole
  for (let i = 0; i < 18; i++) {
    await steer();
    if (i === 8) await page.evaluate(() => __pb.applyPower('multi'));
    if (i % 3 === 0) await page.evaluate(() => __pb.action());
    await page.clock.runFor(33);
    await drain(page);
  }
  await page.evaluate(() => { document.getElementById('msg').hidden = true; }); // drop the LEVEL 8 banner early
  await sequenceGPU(page, h, 'prism-break', 3, 90, async (i) => {
    await steer();
    if (i % 3 === 0) await page.evaluate(() => __pb.action()); // fire lasers
    const st = await page.evaluate(() => ({ n: __pb.game.balls.length, left: __pb.game.bricks.filter(b => b.hp > 0 && b.type !== 3).length, ph: __pb.game.phase, caps: __pb.game.caps.length }));
    if (st.n < 2 && st.ph === 'play') await page.evaluate(() => __pb.applyPower('multi'));
    if (i % 15 === 0) console.log(i, JSON.stringify(st));
  });
  await page.context().close();
};
