// Self-update check: copy fun-and-games/plumb to a scratch dir, serve it on :8766, then
//   node plumb/test_update.js <scratch dir>
// The test edits sw.js and index.html in that copy to simulate shipping a new version.
const { chromium } = require('playwright');
const fs = require('fs');
const dir = process.argv[2];
const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) process.exitCode = 1; };
(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 390, height: 844 } });
  const p = await ctx.newPage();
  const errs = []; p.on('pageerror', e => errs.push(e.message));
  await p.goto('http://localhost:8766/');
  await p.waitForFunction(() => navigator.serviceWorker.controller, null, { timeout: 10000 });
  await p.reload(); await p.waitForTimeout(500);
  ok(await p.evaluate(() => !!navigator.serviceWorker.controller), 'page is controlled by the worker');
  const v0 = await p.textContent('#titleCard .status');
  // Start a game, then ship a new version
  await p.click('#startBtn'); await p.waitForTimeout(300);
  if (await p.isVisible('#plumbCard')) await p.click('#plumbBtn');
  await p.evaluate(() => { window.__mark = 1; });
  for (const f of ['sw.js', 'index.html']) {
    let t = fs.readFileSync(`${dir}/${f}`, 'utf8');
    t = f === 'sw.js' ? t.replace("'plumb-v8'", "'plumb-v9-test'") : t.replace('8 (self-updating)', '9 (test build)');
    fs.writeFileSync(`${dir}/${f}`, t);
  }
  await p.evaluate(() => navigator.serviceWorker.getRegistration().then(r => r.update()));
  await p.waitForTimeout(2500);
  ok(await p.evaluate(() => window.__mark === 1), 'mid-game: new version waits, game not reloaded');
  // Game over -> reload into the new build
  await p.keyboard.press('p'); await p.click('#menuBtn');
  await p.waitForTimeout(2000);
  const marked = await p.evaluate(() => window.__mark).catch(() => undefined);
  ok(marked === undefined, 'on the title screen the page reloads');
  const v1 = await p.textContent('#titleCard .status');
  ok(/9 \(test build\)/.test(v1) && /8 \(self-updating\)/.test(v0), 'reloaded page is the new version');
  ok(errs.length === 0, 'no page errors ' + errs.join('; '));
  await b.close();
})();
