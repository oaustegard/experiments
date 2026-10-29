// Record page-space positions (CSS px) the reel animates to: scroll targets,
// tap points and the sticky header's height. Written to public/shots/layout.json.
const { chromium, devices } = require('playwright');
const fs = require('fs');
const path = require('path');
const BASE = process.env.AEYU_URL || 'https://aeyu.io';
const box = (page, loc) => loc.first().evaluate(el => {
  const r = el.getBoundingClientRect();
  return { x: r.x + scrollX, y: r.y + scrollY, w: r.width, h: r.height };
});
(async () => {
  const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
  const ctx = await browser.newContext({ ...devices['iPhone 14'], viewport: { width: 390, height: 844 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  await page.goto(BASE + '/demo', { waitUntil: 'networkidle' });
  await page.waitForTimeout(4000);
  const L = { viewport: { w: 390, h: 844 } };
  L.header = await box(page, page.locator('header'));
  L.dashRiverLoop = await box(page, page.getByRole('button', { name: /River Loop/ }));
  L.dashRecent = await box(page, page.getByText('Recent Activities'));
  await page.getByRole('button', { name: /River Loop/ }).first().click();
  await page.waitForTimeout(6000);
  L.actHeader = await box(page, page.locator('header'));
  L.actAwards = await box(page, page.getByText('Awards earned', { exact: false }));
  L.actZones = await box(page, page.getByText('Time in zones', { exact: false }));
  fs.writeFileSync(path.resolve(__dirname, '../public/shots/layout.json'), JSON.stringify(L, null, 1));
  console.log(JSON.stringify(L));
  await browser.close();
})();
