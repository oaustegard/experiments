// Shared capture harness for the games promo. Each game script exports
// async (page, h) => { ...set up..., await h.sequence(name, pageSeconds, frames, onFrame) }.
// Run: node capture/run.cjs <game> [<game>...]   (site served at http://localhost:8080)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const OUT = path.resolve(__dirname, '../public/shots');
const BASE = process.env.SITE || 'http://localhost:8080/games/';
const ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader',
  '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist', '--autoplay-policy=no-user-gesture-required'];
const PORTRAIT = { width: 390, height: 844 }, LANDSCAPE = { width: 844, height: 390 };

async function launch() { return chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ARGS }); }

async function open(browser, url, { landscape = false, dpr = 2 } = {}) {
  const ctx = await browser.newContext({ viewport: landscape ? LANDSCAPE : PORTRAIT, deviceScaleFactor: dpr,
    isMobile: true, hasTouch: true, ignoreHTTPSErrors: true,
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1' });
  const page = await ctx.newPage();
  page.on('pageerror', e => console.log('pageerror', e.message.slice(0, 200)));
  await page.goto(url.startsWith('http') ? url : BASE + url, { waitUntil: 'load', timeout: 60000 });
  return page;
}

// install() alone keeps ticking at real speed; pauseAt() freezes it.
async function freezeClock(page) {
  await page.clock.install();
  await page.clock.pauseAt(await page.evaluate(() => Date.now() + 50));
}

// Fire a deviceorientation event (degrees). beta = front/back tip, gamma = left/right roll.
async function tilt(page, beta, gamma, alpha = 0) {
  await page.evaluate(([a, b, g]) => {
    const e = new DeviceOrientationEvent('deviceorientation', { alpha: a, beta: b, gamma: g, absolute: false });
    window.dispatchEvent(e);
  }, [alpha, beta, gamma]);
}

// Step page time pageSeconds over `frames` captures. onFrame(i) runs before each
// capture and may press keys, tilt, click or call game hooks.
async function sequence(page, name, pageSeconds, frames, onFrame) {
  const dir = path.join(OUT, name);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const per = (pageSeconds * 1000) / frames;
  const TICK = Math.min(34, per), step = per - TICK;
  for (let i = 0; i < frames; i++) {
    if (onFrame) await onFrame(i);
    if (step > 0) await page.clock.fastForward(step);
    await page.clock.runFor(TICK);
    await page.waitForTimeout(40);
    await page.screenshot({ path: path.join(dir, `f${String(i).padStart(4, '0')}.jpg`), quality: 90 });
  }
  console.log(name, frames, 'frames ->', dir);
}

module.exports = { launch, open, freezeClock, tilt, sequence, OUT, BASE };
