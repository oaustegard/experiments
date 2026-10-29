// Capture aeyu.io demo-mode screens at iPhone 14 size (390x844 @3x) into public/shots.
// Stills are full-page PNGs; motion shots are CDP screencast frames plus a
// timestamp manifest, replayed in Remotion by wall-clock time.
const { chromium, devices } = require('playwright');
const fs = require('fs');
const path = require('path');
const OUT = path.resolve(__dirname, '../public/shots');
const BASE = process.env.AEYU_URL || 'https://aeyu.io';

// clock.install() alone does NOT freeze time: the fake clock keeps ticking at
// real speed, so a slow screenshot lets the app run ahead. pauseAt() freezes it;
// after that only fastForward/runFor move the page's time.
async function freezeClock(page) {
  await page.clock.install();
  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 50));
}

// Motion: freeze the page clock, then step it one output frame at a time and
// screenshot. rAF-driven animations land on exact times however slow the
// software renderer is, which a real-time screencast cannot promise.
async function sequence(page, name, pageSeconds, frames) {
  const dir = path.join(OUT, name);
  fs.mkdirSync(dir, { recursive: true });
  const per = (pageSeconds * 1000) / frames; // page ms per output frame
  const TICK = Math.min(34, per);             // runFor below also advances page time
  const step = per - TICK;
  for (let i = 0; i < frames; i++) {
    await page.clock.fastForward(step);
    await page.clock.runFor(TICK); // at least one rAF tick so the canvas repaints
    await page.waitForTimeout(40); // real time for the GPU/compositor to present
    await page.screenshot({ path: path.join(dir, `f${String(i).padStart(4, '0')}.jpg`), quality: 92 });
  }
  console.log(name, frames, 'frames');
}

const goto = (page, r) => page.evaluate(r => { history.pushState(null, '', '/' + r); dispatchEvent(new PopStateEvent('popstate')); }, r);

// Each motion shot gets a fresh page: a page whose clock is installed stays
// frozen, and later waitForTimeout calls no longer advance the app's timers.
async function demoPage(ctx, route) {
  const page = await ctx.newPage();
  page.on('pageerror', e => console.log('pageerror', e.message));
  await page.goto(BASE + '/demo', { waitUntil: 'networkidle' });
  await page.waitForTimeout(4000);
  if (route) { await goto(page, route); await page.waitForTimeout(8000); }
  return page;
}

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium',
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader'] });
  // The 'iPhone 14' preset's viewport is 390x664 (Safari's chrome subtracted);
  // the reel shows a full-height screen, so use the 844 px screen height.
  const ctx = await browser.newContext({ ...devices['iPhone 14'], viewport: { width: 390, height: 844 }, deviceScaleFactor: 3 });
  const only = process.argv[2]; // optional: capture one shot by name

  if (!only || only === 'stills') {
    const page = await ctx.newPage();
    await page.goto(BASE + '/', { waitUntil: 'networkidle' });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${OUT}/landing.png`, fullPage: true });
    await page.goto(BASE + '/demo', { waitUntil: 'networkidle' });
    await page.waitForTimeout(4000);
    await page.screenshot({ path: `${OUT}/dashboard.png`, fullPage: true });
    await page.getByRole('button', { name: /River Loop/ }).first().click();
    await page.waitForTimeout(6000);
    await page.screenshot({ path: `${OUT}/activity.png`, fullPage: true });
    await goto(page, 'power');
    await page.waitForTimeout(6000);
    await page.screenshot({ path: `${OUT}/power.png`, fullPage: true });
    await page.close();
  }
  if (!only || only === 'replay') {
    const page = await demoPage(ctx);
    await page.getByRole('button', { name: /River Loop/ }).first().click();
    await page.waitForTimeout(8000);
    await freezeClock(page);
    // exact name: /play/i also matches "Make a 30 second replay video"
    await page.getByRole('button', { name: 'Play replay', exact: true }).click();
    await sequence(page, 'replay', 30, 150); // the 30s pace setting
    await page.close();
  }
  if (!only || only === 'heatmap') {
    const page = await demoPage(ctx, 'heatmap');
    await page.screenshot({ path: `${OUT}/heatmap.png` });
    await freezeClock(page);
    await page.getByRole('button', { name: 'Play', exact: true }).click();
    await sequence(page, 'heatmap-play', 22, 135); // 20 s sweep plus the glow tail
    await page.close();
  }
  await browser.close();
})();
