// Renders Scoop offscreen in headless Chromium (SwiftShader WebGPU) and saves frames as PNGs.
//   (cd fun-and-games/scoop && python3 -m http.server 8767) &   # in oaustegard.github.io
//   node scoop/shots.js [outdir]
const { chromium } = require('playwright');
const out = process.argv[2] || '.';
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-features=Vulkan'] });
  const p = await b.newPage({ viewport: { width: 390, height: 760 }, deviceScaleFactor: 1 });
  const logs = [];
  p.on('console', m => logs.push(`${m.type()}: ${m.text()}`));
  p.on('pageerror', e => logs.push('pageerror: ' + e.message));
  await p.goto('http://localhost:8767/?offscreen&debug&seed=7');
  await p.waitForFunction(() => window.__scoop, null, { timeout: 20000 });
  await p.waitForTimeout(1500);
  console.log(JSON.stringify(await p.evaluate(() => __scoop.state())));
  await p.screenshot({ path: `${out}/1-start.png` });
  await p.evaluate(() => __scoop.reveal(4, 4));
  await p.waitForTimeout(250);
  await p.screenshot({ path: `${out}/2-flood.png` });
  await p.waitForTimeout(1500);
  await p.screenshot({ path: `${out}/3-open.png` });
  // flag a few hidden cells that are mines
  await p.evaluate(() => { const S = __scoop, st = S.state(); let k = 0;
    for (let y = 0; y < st.R && k < 3; y++) for (let x = 0; x < st.C && k < 3; x++) { const c = S.cell(x, y); if (c.mine) { S.flag(x, y); k++; } } });
  await p.waitForTimeout(800);
  await p.screenshot({ path: `${out}/4-flags.png` });
  // lose: reveal an unflagged mine
  await p.evaluate(() => { const S = __scoop, st = S.state();
    for (let y = st.R - 1; y >= 0; y--) for (let x = st.C - 1; x >= 0; x--) { const c = S.cell(x, y); if (c.mine && c.state === 0) { S.reveal(x, y); return; } } });
  for (const ms of [120, 300, 700, 2600]) { await p.waitForTimeout(ms); await p.screenshot({ path: `${out}/5-boom-${ms}.png` }); }
  console.log(logs.slice(0, 30).join('\n'));
  await b.close();
})();
