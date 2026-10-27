// Checks that a real touch tap (not a mouse click) unlocks Web Audio, with the browser's
// user-gesture autoplay policy on. Touch pointerdown is not a user activation; touchend is.
//   node scoop/test_audio.js [port]   (8767 Scoop, 8766 Sounding)
const { chromium } = require('playwright');
const port = process.argv[2] || 8767;
(async () => {
  const b = await chromium.launch({ args: ['--autoplay-policy=user-gesture-required'] });
  const ctx = await b.newContext({ viewport: { width: 390, height: 760 }, isMobile: true, hasTouch: true });
  const p = await ctx.newPage();
  await p.addInitScript(() => {
    const AC = window.AudioContext;
    window.__nodes = 0;
    window.AudioContext = class extends AC {
      constructor(...a) { super(...a); window.__ac = this; }
      createOscillator() { window.__nodes++; return super.createOscillator(); }
      createBufferSource() { window.__nodes++; return super.createBufferSource(); }
    };
  });
  await p.goto(`http://localhost:${port}/?debug&nogpu&seed=4`);
  await p.waitForFunction(() => window.__scoop || window.__sounding);
  const L = await p.evaluate(() => (window.__scoop || window.__sounding).layout());
  await p.touchscreen.tap(L.orgCss[0] + 4.5 * L.cellCss, L.orgCss[1] + 4.5 * L.cellCss);
  await p.waitForTimeout(400);
  const { st, nodes } = await p.evaluate(() => ({ st: window.__ac ? window.__ac.state : 'no context', nodes: window.__nodes }));
  console.log(st === 'running' ? 'ok  ' : 'FAIL', `audio after a touch tap: ${st}`);
  console.log(nodes > 1 ? 'ok  ' : 'FAIL', `the tap scheduled sound nodes (${nodes})`);
  if (st !== 'running' || nodes <= 1) process.exitCode = 1;
  await b.close();
})();
