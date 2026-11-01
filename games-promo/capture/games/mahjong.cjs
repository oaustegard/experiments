// Mahjong (Canvas 2D renderer: under SwiftShader on a loaded box the WebGPU table took ~57 s a frame
// and dropped tiles; the game's own 2D renderer draws the same table in ~0.5 s): Hong Kong rules against three computer players at Brisk pace, captured mid-hand.
// The clock is frozen in the lobby (installing it mid-animation sends one frame's dt hugely negative);
// play is then warmed up on the fake clock. Seat 0 plays the computer's own choice (ai.js botTurn /
// botClaim): it lifts the tile for a few frames, then discards it.
module.exports = async (b, h) => {
  const page = await h.open(b, 'mahjong/?renderer=2d&seed=' + (process.env.SEED || 5));
  await page.waitForFunction(() => window.__mahjong && window.__mahjong.renderer, null, { timeout: 30000 });
  await page.waitForTimeout(1500);
  await h.freezeClock(page);
  const run = async (ms) => { for (let t = 0; t < ms; t += 100) { await page.clock.runFor(100); } };
  await run(300);
  await page.locator('#set-pace button[data-v="brisk"]').click();
  await page.locator('#btn-solo').click();
  let liftAt = null; // frame index of the current lift, null when none
  const drive = async (i) => page.evaluate(async ([i, liftAt]) => {
    const M = window.__mahjong, G = M.G, ai = await import('./ai.js');
    if (!G.prompt || !G.view) return null;
    if (G.prompt.kind === 'claim') { M.act(ai.botClaim(G.view, G.prompt.options)); return null; }
    const a = ai.botTurn(G.view, G.prompt.options);
    if (a.type !== 'discard') { M.act(a); return null; }
    if (liftAt === null) {   // lift the tile: select it, then nudge the selection out and back so the table redraws
      const hand = G.view.players[G.view.seat].hand, k = hand.indexOf(a.id);
      G.ui.selected = a.id;
      const [x, y] = k === 0 ? ['ArrowRight', 'ArrowLeft'] : ['ArrowLeft', 'ArrowRight'];
      dispatchEvent(new KeyboardEvent('keydown', { key: x })); dispatchEvent(new KeyboardEvent('keydown', { key: y }));
      return i;
    }
    if (i - liftAt >= 8) { M.act({ type: 'discard', id: G.ui.selected }); return null; }
    return liftAt;
  }, [i, liftAt]);
  // Warm up past the deal and the first round of discards, driving seat 0 at the same rhythm.
  for (let i = 0; i < 240; i++) { liftAt = await drive(i); await page.clock.runFor(33); }
  await page.waitForTimeout(300);
  await h.sequence(page, 'mahjong', 4, 120, async (i) => { liftAt = await drive(i + 1000); });
  console.log('mahjong', JSON.stringify(await page.evaluate(() => { const v = window.__mahjong.G.view; return { phase: v.phase, wall: v.wallLeft, turn: v.turn, disc: v.players.map(p => p.discards.length) }; })));
  await page.context().close();
};
