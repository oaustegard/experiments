// Tilt Chomp: an in-page autopilot steers the chomper with synthetic tilt
// (deviceorientation), chasing pellets, running from ghosts, taking a power
// pellet when they close in and then hunting the blue ghosts.
// The game has no debug hook, so the served page gets one spliced into its
// closure (window.__chomp); the site files are not touched.
const { sequenceGPU, injectAndReload } = require('./_gpu.cjs');
const DEV = '() => window.__gpuDevice';
const WARM = Number(process.env.TC_WARM || 7000); // ms of play before the sequence

const HOOK = `window.__chomp = { game, pac, ghosts, pellets, MAP, solid, DX, DY, W, H };`;

module.exports = async (b, h) => {
  const page = await h.open(b, 'tilt-chomp/');
  page.setDefaultTimeout(180000);
  // freeze first: the reload then boots the game on the fake clock
  await h.freezeClock(page);
  await injectAndReload(page, '/tilt-chomp/', { tail: HOOK });
  for (let k = 0; k < 20; k++) {
    await page.clock.fastForward(300); await page.waitForTimeout(300);
    if (await page.evaluate(() => !!window.__gpuDevice)) break;
  }
  await page.clock.runFor(200);
  await page.locator('#start').click();
  await h.tilt(page, 50, 0); // first reading becomes the base pose
  await page.evaluate(() => {
    const C = window.__chomp, { game, pac, ghosts, pellets, solid, DX, DY, W, H } = C;
    const tile = (e) => [((Math.floor(e.x) % W) + W) % W, Math.floor(e.y)];
    const key = (c, r) => r * W + c;
    let lastDir = 1, tx = 0, ty = 0;
    const plan = () => {
      const [pc, pr] = tile(pac);
      const threats = ghosts.filter(g => g.state === 'active' && !g.fright).map(tile);
      const prey = ghosts.filter(g => g.state === 'active' && g.fright).map(tile);
      const near = threats.some(([c, r]) => Math.abs(c - pc) + Math.abs(r - pr) < 8);
      const danger = new Set();
      for (const [c, r] of threats) for (let dc = -2; dc <= 2; dc++) for (let dr = -2; dr <= 2; dr++)
        if (Math.abs(dc) + Math.abs(dr) <= 2) danger.add(key(((c + dc) % W + W) % W, r + dr));
      const goal = (c, r) => {
        if (prey.length) return prey.some(([gc, gr]) => gc === c && gr === r);
        const v = pellets[key(c, r)];
        return near ? v === 2 || (v === 1 && false) : v > 0;
      };
      // BFS; first step remembered per node
      const seen = new Map([[key(pc, pr), -1]]); let q = [[pc, pr]];
      let fallback = null;
      while (q.length) {
        const nq = [];
        for (const [c, r] of q) {
          const first = seen.get(key(c, r));
          if (first >= 0 && goal(c, r)) return first;
          for (let d = 0; d < 4; d++) {
            const nc = ((c + DX[d]) % W + W) % W, nr = r + DY[d];
            if (nr < 0 || nr >= H || solid(nc, nr) || seen.has(key(nc, nr))) continue;
            if (danger.has(key(nc, nr)) && !prey.length) continue;
            const f = first < 0 ? d : first;
            seen.set(key(nc, nr), f); nq.push([nc, nr]);
            if (fallback === null && first >= 0 && pellets[key(nc, nr)]) fallback = f;
          }
        }
        q = nq;
      }
      return fallback;
    };
    const loop = () => {
      requestAnimationFrame(loop);
      if (game.phase !== 'play') return;
      let d = plan();
      if (d == null) { // nothing safe found while near: any pellet, ignoring danger
        d = lastDir;
      }
      lastDir = d;
      // ease the tilt toward the wanted direction so the dial marble moves smoothly
      const wx = d === 3 ? 16 : d === 1 ? -16 : 0, wy = d === 2 ? 16 : d === 0 ? -16 : 0;
      tx += (wx - tx) * 0.5; ty += (wy - ty) * 0.5;
      window.dispatchEvent(new DeviceOrientationEvent('deviceorientation', { alpha: 0, beta: 50 + ty, gamma: tx, absolute: false }));
    };
    requestAnimationFrame(loop);
  });
  for (let t = 0; t < WARM; t += 250) await page.clock.runFor(250);
  console.log(await page.evaluate(() => { const { game } = window.__chomp; return JSON.stringify({ ph: game.phase, sc: game.score, left: game.dotsLeft, lives: game.lives }); }));
  await sequenceGPU(page, h, 'tilt-chomp', 3, 90, async (i) => {
    if (i % 15 === 0) console.log(i, await page.evaluate(() => { const { game, ghosts } = window.__chomp; return JSON.stringify({ ph: game.phase, sc: game.score, left: game.dotsLeft, lives: game.lives, fr: game.frightT.toFixed(1), g: ghosts.map(g => g.state[0] + (g.fright ? 'F' : '')).join('') }); }));
  }, DEV);
  await page.context().close();
};
