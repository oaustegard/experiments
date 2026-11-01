// Careen: tilt steering off (arrow keys turn gravity), Normal board, then 3 s of play.
// Gravity turns every 0.6 s make the whole board slide; between turns the board's own hint
// (a turn or a swap that clears) is played so matches pop into sparks.
module.exports = async (b, h) => {
  const page = await h.open(b, 'careen/?seed=' + (process.env.SEED || 11));
  await page.evaluate(() => { localStorage.setItem('careen-tilt', 'false'); localStorage.setItem('careen-level', '"normal"'); localStorage.setItem('careen-sound', 'false'); });
  await page.reload({ waitUntil: 'load' });
  await page.waitForFunction(() => window.__careen && window.__careen.state.renderer, null, { timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.getByRole('button', { name: 'Start', exact: true }).click();
  await page.waitForTimeout(3500); // opening drop settles, the tip fades
  // Installing the fake clock makes rAF time jump backwards, and Careen's dt = min(0.05, now - last)
  // would go hugely negative (tiles fling off and balloon). Pause the game across the switch; resuming
  // resets `last` to the fake clock's time.
  await page.keyboard.press('p');
  await h.freezeClock(page);
  await page.clock.runFor(100);
  await page.keyboard.press('p');
  await page.clock.runFor(100);
  const turns = ['left', 'up', 'right', 'down'];
  let k = 0;
  await h.sequence(page, 'careen', 3, 90, async (i) => {
    if (i % 18 === 1) await page.keyboard.press({ left: 'ArrowLeft', up: 'ArrowUp', right: 'ArrowRight', down: 'ArrowDown' }[turns[k++ % 4]]);
    if (i % 18 === 10) await page.evaluate(() => {
      const C = window.__careen; if (C.state.phase !== 'idle') return;
      const hnt = C.board.findHint(C.state.g);
      if (hnt && hnt.kind === 'tilt') C.setGravity(hnt.dir); else if (hnt) C.trySwap(hnt.a, hnt.b);
    });
  });
  console.log('careen', JSON.stringify(await page.evaluate(() => window.__careen.state)));
  await page.context().close();
};
