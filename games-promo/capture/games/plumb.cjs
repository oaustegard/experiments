// Plumb: a near-full well; the I piece is steered by synthetic tilt over the one
// open column, tipped to pour it down, and clears four rows ("Plumb!") into
// WebGPU shards. The next pieces are steered and tipped down after it.
// The clock is frozen before Start: installing it mid-game makes rAF time jump
// backwards and the game's dt goes negative.
const BETA0 = 50; // the pose pressed as "plumb": phone held up at 50 degrees

module.exports = async (b, h) => {
  const page = await h.open(b, 'plumb/?debug');
  await page.waitForTimeout(2500);
  await h.freezeClock(page);
  await page.clock.runFor(300);
  // music on needs a gesture; sound effects stay as they are
  await page.locator('#startBtn').click();
  await page.clock.runFor(200);
  await h.tilt(page, BETA0, 0);
  await page.clock.runFor(300);
  await h.tilt(page, BETA0, 0);
  await page.clock.runFor(300);
  if (await page.locator('#plumbBtn').isVisible()) await page.locator('#plumbBtn').click();
  await page.clock.runFor(200);

  // Stack: bottom four rows full but for column 7, a ragged pair above them.
  await page.evaluate(() => {
    const P = window.__plumb, g = P.grid();
    const rows = {
      14: '..........',
      15: '..........',
      16: 'TT......OO',
      17: 'TJJLL...OO',
      18: 'ZZSSIII.LL',
      19: 'JZZSSTT.LL',
      20: 'JJJOOTL.SS',
      21: 'IIIIOOL.SS',
    };
    for (const [y, s] of Object.entries(rows)) [...s].forEach((c, x) => { g[y][x] = c === '.' ? 0 : c; });
    // an O drops onto the left of the stack; its lock redraws the well and spawns the next piece
    P.place('O', 3, 0);
    P.hardDrop();
    P.place('I', 3, 1); // vertical I (its cells sit in column x+2), left of the gap
  });
  await page.clock.runFor(500);
  console.log(await page.evaluate(() => JSON.stringify({ p: __plumb.piece(), s: __plumb.stats() })));

  // Steering plan: target column for each successive piece (by count of spawns).
  const targets = [{ col: 7, r: 1 }, { col: 1, r: 0 }, { col: 6, r: 1 }, { col: 0, r: 0 }];
  let k = -1, lastKey = '', lastY = -1, still = 0;
  await h.sequence(page, 'plumb', 3, 90, async (i) => {
    const st = await page.evaluate(() => { const p = __plumb.piece(); return p && { ...p, n: __plumb.stats().lines }; });
    if (!st) { await h.tilt(page, BETA0, 0); return; }
    const key = st.t + ':' + st.y;
    if (k < 0 || (lastY >= 0 && st.y < lastY - 1)) k++; // a new piece spawned at the top
    lastKey = key;
    // grounded and lined up: jab it home instead of sitting out the lock delay
    still = st.y === lastY ? still + 1 : 0; lastY = st.y;
    if (still >= 2 && st.y > 6) { await page.evaluate(() => __plumb.hardDrop()); still = 0; return; }
    const tg = targets[Math.max(0, Math.min(k, targets.length - 1))];
    // rotate toward the target state
    if (st.r !== tg.r) await page.evaluate((d) => __plumb.rotate(d), 1);
    // leftmost occupied column of this piece in this state
    const left = await page.evaluate(() => {
      const p = __plumb.piece(); const g = __plumb.grid();
      // probe: count how far it can shift left; leftmost cell = x - 0 offset unknown, so use the I special case
      return p.x + (p.t === 'I' && p.r === 1 ? 2 : (p.t === 'I' && p.r === 3 ? 1 : 0));
    });
    const dx = tg.col - left;
    const gamma = dx === 0 ? 0 : Math.sign(dx) * Math.min(22, 11 + 4 * Math.abs(dx));
    const beta = dx === 0 && st.r === tg.r && i > 2 ? BETA0 + 34 : BETA0; // tip to pour once lined up
    await h.tilt(page, beta, gamma);
    if (i % 10 === 0) console.log(i, JSON.stringify(st), 'k', k, 'dx', dx);
  });
  await page.context().close();
};
