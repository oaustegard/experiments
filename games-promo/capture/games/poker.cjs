// Poker: No-limit Hold'em against the computer at Brisk pace, captured mid-hand.
// Canvas 2D renderer: under SwiftShader on this loaded box the WebGPU tables (Mahjong measured ~57 s a
// frame, with tiles missing) are too slow to step; the game's own 2D renderer draws the same table.
// The clock is frozen in the lobby (installing it mid-animation sends one frame's dt hugely negative),
// then play warms up on the fake clock. Seat 0 waits a few frames with its buttons up, then plays the
// game's own hint (a computer's TAG choice), checking or calling rather than ever folding,
// so its cards stay in the shot. Warm-up runs until a flop is on the board.
module.exports = async (b, h) => {
  const page = await h.open(b, 'poker/?renderer=2d&seed=' + (process.env.SEED || 3));
  await page.waitForFunction(() => window.__poker && window.__poker.renderer, null, { timeout: 30000 });
  await page.waitForTimeout(1500);
  await h.freezeClock(page);
  await page.clock.runFor(300);
  await page.locator('#set-pace button[data-v="brisk"]').click();
  await page.locator('#btn-solo').click();
  let seen = null; // frame index when the current prompt first appeared
  const drive = async (i) => page.evaluate(([i, seen]) => {
    const P = window.__poker, G = P.G, pr = G.prompt;
    if (!pr) return null;
    if (pr.kind === 'rebuy') { P.act(true); return null; }
    if (pr.kind !== 'turn') return seen;
    if (seen === null) return i;
    if (i - seen < 10) return seen;
    const o = pr.options;
    let a = G.hint && G.hint.action;
    if (o.kind === 'draw') a = { type: 'draw', discard: (G.hint && G.hint.discard) || [] };
    if (!a || a.type === 'fold') a = o.canCheck ? { type: 'check' } : { type: 'call' };
    P.act(a);
    return null;
  }, [i, seen]);
  for (let i = 0; i < 1500; i++) {
    seen = await drive(i); await page.clock.runFor(33);
    if (i > 60 && i % 5 === 0 && await page.evaluate(() => { const v = window.__poker.G.view; return v && v.phase !== 'over' && v.board.length >= 3 && !v.players[v.seat].folded; })) break;
  }
  // Settle: the hole cards only fly in once real time has passed (their faces load asynchronously),
  // so give it real seconds, then a few rendered frames.
  await page.waitForTimeout(4000);
  for (let i = 1500; i < 1525; i++) { seen = await drive(i); await page.clock.runFor(33); await page.waitForTimeout(40); }
  await h.sequence(page, 'poker', 4, 120, async (i) => { seen = await drive(i + 2000); }); // indices stay above the warm-up's
  console.log('poker', JSON.stringify(await page.evaluate(() => { const v = window.__poker.G.view; return { phase: v.phase, hand: v.handNo, pot: v.potTotal, board: v.board.length, toAct: v.toAct }; })));
  await page.context().close();
};
