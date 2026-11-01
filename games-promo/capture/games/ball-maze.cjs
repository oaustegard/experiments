// Ball Maze: an in-page autopilot rolls the ball along the solution path.
// Fixes from the first run: the clock is frozen BEFORE Start (installing the
// fake clock mid-play made rAF time jump back ~3.5 s, so dt went negative, the
// timer sat at 0:00 and the physics accumulator stalled), and the ball is
// steered by tilt toward path waypoints instead of weak half-tilt arrow keys.
// Math.random is seeded before the level is built so the maze is repeatable.
const LEVEL = Number(process.env.BM_LEVEL || 2); // 0-based; 2 = level 3, a few holes
const SEED = Number(process.env.BM_SEED || 1);

async function setup(b, h, level, seed) {
  const page = await h.open(b, 'ball-maze/?debug');
  await page.evaluate(() => localStorage.clear());
  await page.waitForTimeout(1500);
  await h.freezeClock(page);
  await page.clock.runFor(500); // let the rAF time jump happen while idle
  await page.getByRole('button', { name: 'Start', exact: true }).click();
  await page.clock.runFor(100);
  await page.evaluate(([lvl, seed]) => {
    let s = seed >>> 0;
    Math.random = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const B = window.__ballMaze, g = B.game;
    B.startLevel(lvl);
    window.__falls = 0;
    // autopilot: PD steer toward the next path waypoint, nudged off any hole
    let wp = 0, prev = 'playing';
    const tick = () => {
      requestAnimationFrame(tick);
      if (g.state === 'falling' && prev !== 'falling') { window.__falls++; wp = 0; }
      prev = g.state;
      if (g.state !== 'playing') return;
      const m = g.maze, bl = g.ball, cs = m.cs, P = m.path;
      const cx = (c) => m.ox + (c + 0.5) * cs, cy = (r) => m.oy + (r + 0.5) * cs;
      while (wp < P.length - 1 && Math.hypot(bl.x - cx(P[wp][0]), bl.y - cy(P[wp][1])) < cs * 0.35) wp++;
      let tx = cx(P[wp][0]), ty = cy(P[wp][1]);
      let near = false;
      for (const ho of m.holes) {
        const dx = tx - ho.x, dy = ty - ho.y, d = Math.hypot(dx, dy);
        if (d < cs * 0.6) { near = true; tx += (dx / (d || 1)) * cs * 0.12; ty += (dy / (d || 1)) * cs * 0.12; }
        if (Math.hypot(bl.x - ho.x, bl.y - ho.y) < cs * 0.9) near = true;
      }
      const gain = near ? 0.25 : 0.5;
      const k = 5 / cs, kd = 0.55 / cs;
      const cl = (v) => Math.max(-1, Math.min(1, v));
      g.raw.x = cl(k * (tx - bl.x) * gain - kd * bl.vx * 0.4);
      g.raw.y = cl(k * (ty - bl.y) * gain - kd * bl.vy * 0.4);
      window.__wp = wp;
    };
    requestAnimationFrame(tick);
  }, [level, seed]);
  return page;
}

module.exports = async (b, h) => {
  const page = await setup(b, h, LEVEL, SEED);
  await page.clock.runFor(1600); // ball under way, timer already counting
  await h.sequence(page, 'ball-maze', 3, 90, async (i) => {
    if (i % 15 === 0) console.log(i, await page.evaluate(() => { const g = __ballMaze.game; return `${g.state} t=${g.elapsed.toFixed(1)} wp=${__wp}/${g.maze.path.length} falls=${__falls} score=${g.score}`; }));
  });
  await page.context().close();
};
module.exports.setup = setup;
