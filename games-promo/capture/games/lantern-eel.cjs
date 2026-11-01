// Lantern Eel: a deep dive (950 m, the light down to the lure, jellyfish about)
// with a long eel. An in-page autopilot steers by synthetic tilt: a BFS to the
// nearest plankton or lanternfish that keeps clear of the body and jellyfish.
// Canvas 2D, so no GPU drain; the clock is frozen before Dive.
const EATEN = Number(process.env.EEL_EATEN || 38); // 25 m per bite
const WARM = Number(process.env.EEL_WARM || 4000);
const FRAMES = Number(process.env.EEL_FRAMES || 90);

const SEED = () => Number(process.env.EEL_SEED || 6);

module.exports = async (b, h) => {
  const page = await h.open(b, 'lantern-eel/');
  page.setDefaultTimeout(120000);
  // Freeze, then reload so the game boots on the fake clock: freezing a running
  // page makes rAF time jump back, and a negative dt here throws in drawPlankton.
  await h.freezeClock(page);
  await page.reload({ waitUntil: 'load' });
  for (let i = 0; i < 8; i++) { await page.clock.runFor(250); await page.waitForTimeout(100); }
  // seeded Math.random: the same prey, jellyfish and fish every run
  await page.evaluate((seed) => {
    let s = seed >>> 0;
    Math.random = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }, SEED());
  // a phone-sized page starts in tilt mode
  await page.getByRole('button', { name: 'Dive', exact: true }).click();
  await h.tilt(page, 50, 0); // neutral pose
  await page.clock.runFor(100);
  await page.evaluate((eaten) => {
    const E = window.__eel, S = E.S;
    S.eaten = eaten; S.grow = 18; S.glow = 100;
    const OPP = { up: 'down', down: 'up', left: 'right', right: 'left' };
    const D = { up: [0, -1], down: [0, 1], left: [-1, 0], right: [1, 0] };
    let tx = 0, ty = 0;
    const choose = () => {
      const S = E.S, [C, R] = E.dims, md = (v, n) => ((v % n) + n) % n;
      const k = (x, y) => y * C + x;
      const cur = S.queue.length ? S.queue[S.queue.length - 1] : S.dir;
      const blocked = new Set();
      S.snake.slice(0, -1).forEach(p => blocked.add(k(p.x, p.y)));
      for (const j of S.jellies) for (const [dx, dy] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) blocked.add(k(md(j.x + dx, C), md(j.y + dy, R)));
      const jcell = new Set(S.jellies.map(j => k(j.x, j.y)));
      const goals = [S.fish, S.food, S.pearl].filter(Boolean);
      const isGoal = (x, y) => goals.some(g => g.x === x && g.y === y);
      const h0 = S.snake[0];
      // flood size from a cell, to break ties and escape when no goal is reachable
      const room = (sx, sy) => { const seen = new Set([k(sx, sy)]); const q = [[sx, sy]]; while (q.length && seen.size < 200) { const [x, y] = q.pop(); for (const [dx, dy] of Object.values(D)) { const nx = md(x + dx, C), ny = md(y + dy, R); if (!blocked.has(k(nx, ny)) && !seen.has(k(nx, ny))) { seen.add(k(nx, ny)); q.push([nx, ny]); } } } return seen.size; };
      let best = null;
      for (const d of Object.keys(D)) {
        if (d === OPP[cur]) continue;
        const sx = md(h0.x + D[d][0], C), sy = md(h0.y + D[d][1], R);
        if (blocked.has(k(sx, sy)) && !(isGoal(sx, sy) && !jcell.has(k(sx, sy)))) continue;
        // BFS distance to a goal starting with move d
        const seen = new Set([k(h0.x, h0.y), k(sx, sy)]); let q = [[sx, sy]], dist = 0, found = isGoal(sx, sy) ? 0 : -1;
        while (found < 0 && q.length) { dist++; const nq = []; for (const [x, y] of q) for (const [dx, dy] of Object.values(D)) { const nx = md(x + dx, C), ny = md(y + dy, R), kk = k(nx, ny); if (seen.has(kk)) continue; if (isGoal(nx, ny) && !jcell.has(kk)) { found = dist; break; } if (blocked.has(kk)) continue; seen.add(kk); nq.push([nx, ny]); } q = nq; }
        const r = room(sx, sy);
        const score = (r < S.snake.length ? -1000 + r : 0) + (found >= 0 ? 500 - found : 0) + r * 0.01 + (d === cur ? 0.5 : 0);
        if (!best || score > best.score) best = { d, score };
      }
      return best ? best.d : cur;
    };
    let lastTick = -1, want = S.dir;
    const loop = () => {
      requestAnimationFrame(loop);
      const S = E.S; if (E.state !== 'play') return;
      // plan once per tick, after the step has used up the queued turn
      if (S.tick !== lastTick && !S.queue.length) { lastTick = S.tick; want = choose(); }
      // the tilt that steers this way; the game's own onOrient turns the eel.
      // turn() too, as a key press would, so a turn is never a tick late
      const wx = want === 'right' ? 14 : want === 'left' ? -14 : 0, wy = want === 'down' ? 14 : want === 'up' ? -14 : 0;
      tx = wx; ty = wy;
      if (want !== S.dir && !S.queue.length) E.turn(want);
      window.dispatchEvent(new DeviceOrientationEvent('deviceorientation', { alpha: 0, beta: 50 + ty, gamma: tx, absolute: false }));
    };
    requestAnimationFrame(loop);
  }, EATEN);
  for (let t = 0; t < WARM; t += 250) await page.clock.runFor(250);
  const st = () => page.evaluate(() => { const E = window.__eel, S = E.S; return JSON.stringify({ hd: S.snake[0], dir: S.dir, food: S.food && [S.food.x, S.food.y], fish: S.fish && [S.fish.x, S.fish.y], dims: E.dims, st: E.state, m: S.eaten * 25, len: S.snake.length, sc: S.score, j: S.jellies.length, combo: S.combo }); });
  console.log('warm', await st());
  if (process.env.EEL_DRY) { for (let t = 0; t < 3000; t += 250) await page.clock.runFor(250); console.log('dry seed', SEED(), await st()); await page.context().close(); return; }
  if (await page.evaluate(() => window.__eel.state !== 'play')) throw new Error('the eel died in warm-up: ' + await page.evaluate(() => window.__eel.S.cause));
  await h.sequence(page, 'lantern-eel', 3 * FRAMES / 90, FRAMES, async (i) => { if (i % (FRAMES < 90 ? 1 : 15) === 0) console.log(i, await st()); });
  await page.context().close();
};
