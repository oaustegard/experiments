// Shared driver for Sounding and Scoop (same minefield engine, window.__sounding / window.__scoop).
// Timeline over 4 s / 120 frames: open a cell (flood ripple), open a second region, mark three
// mines (Sounding beacons / Scoop bags), then hit a mine near the middle so the chain plays out.
module.exports = (game, hook) => async (b, h) => {
  const seed = process.env.SEED || 7;
  const page = await h.open(b, `${game}/?offscreen&debug&seed=${seed}`);
  await page.waitForFunction((k) => window[k], hook, { timeout: 30000 });
  await page.waitForTimeout(2500);
  // The end card would slide over the board after the chain; keep it out of the shot.
  await page.addStyleTag({ content: '#overlay{display:none!important}' });
  await page.evaluate((k) => window[k].newGame('medium'), hook);
  await page.waitForTimeout(800);
  await h.freezeClock(page);
  const S = (fn, ...a) => page.evaluate(([k, src, args]) => (new Function('S', 'args', src))(window[k], args), [hook, fn, a]);
  const st = await S('return S.state()');
  const { C, R } = st, cx = C >> 1, cy = R >> 1;
  console.log(game, JSON.stringify(st));
  await h.sequence(page, game, 4, 120, async (i) => {
    if (i === 4) await S('S.reveal(args[0], args[1])', cx, cy);
    if (i === 30) await S(`const st = S.state();  // open the largest still-hidden zero region far from the first
      for (const [x, y] of [[1,1],[st.C-2,1],[1,st.R-2],[st.C-2,st.R-2],[1,(st.R/2)|0],[st.C-2,(st.R/2)|0]]) {
        const c = S.cell(x, y); if (c.state === 0 && !c.mine && c.count === 0) { S.reveal(x, y); return; } }`);
    if (i === 42 || i === 50 || i === 58) await S(`const st = S.state(); // mark a mine that borders an open cell
      for (let y = 0; y < st.R; y++) for (let x = 0; x < st.C; x++) { const c = S.cell(x, y); if (!c.mine || c.state !== 0) continue;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const n = x+dx >= 0 && y+dy >= 0 && x+dx < st.C && y+dy < st.R && S.cell(x+dx, y+dy);
          if (n && n.state === 1) { S.flag(x, y); return; } } }`);
    if (i === 68) await S(`const st = S.state(), cx = st.C / 2, cy = st.R / 2; let best = null, bd = 1e9;
      for (let y = 0; y < st.R; y++) for (let x = 0; x < st.C; x++) { const c = S.cell(x, y);
        if (c.mine && c.state === 0) { const d = Math.hypot(x + .5 - cx, y + .5 - cy); if (d < bd) { bd = d; best = [x, y]; } } }
      if (best) S.reveal(best[0], best[1]);`);
  });
  console.log(game, JSON.stringify(await S('return S.state()')));
  await page.context().close();
};
