const { chromium } = require('playwright');
const GAMES = process.argv.slice(2);
(async () => {
  const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', args: ['--enable-unsafe-webgpu','--enable-features=Vulkan','--use-vulkan=swiftshader','--use-angle=swiftshader','--use-webgpu-adapter=swiftshader','--ignore-gpu-blocklist'] });
  for (const g of GAMES) {
    const ctx = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, hasTouch: true, isMobile: true, ignoreHTTPSErrors: true });
    const p = await ctx.newPage(); const logs = [];
    p.on('pageerror', e => logs.push('ERR ' + e.message)); p.on('console', m => m.type()==='error' && logs.push('cerr ' + m.text().slice(0,120)));
    const url = 'http://localhost:8080/games/' + g + (g.endsWith('.html') ? '' : '/');
    try { await p.goto(url, { timeout: 30000 }); } catch (e) { logs.push('goto ' + e.message.slice(0,80)); }
    await p.waitForTimeout(4000);
    await p.screenshot({ path: `capture/probe/${g.replace(/\W/g,'_')}.png` });
    const btns = await p.$$eval('button', bs => bs.filter(b => b.offsetParent).map(b => (b.innerText||b.getAttribute('aria-label')||'').trim().slice(0,30)));
    console.log(g, JSON.stringify(btns), logs.slice(0,3).join(' | '));
    await ctx.close();
  }
  await b.close();
})();
