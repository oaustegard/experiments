// Loads test/embed.html: three <raking-light> elements (inline SVG with transparent wall, src= with a fixed light,
// opaque wall with an orbiting light), moves the pointer, saves a screenshot and checks toBlob().
// Usage: (cd raking-light && python3 -m http.server 8765 &) ; node raking-light/test/embed.js
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader',
    '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader'] });
  const p = await b.newPage({ viewport: { width: 1100, height: 460 } });
  let errors = 0;
  p.on('console', m => { if (m.type() === 'error') { errors++; console.log('console error:', m.text()); } });
  p.on('pageerror', e => { errors++; console.log('pageerror:', e.message); });
  await p.goto('http://localhost:8765/test/embed.html');
  await p.waitForTimeout(15000);
  await p.mouse.move(40, 40); await p.waitForTimeout(4000);
  await p.screenshot({ path: path.join(__dirname, 'embed.png') });
  const size = await p.evaluate(async () => (await document.getElementById('b').toBlob()).size);
  const lightingK = await p.evaluate(() => document.getElementById('b').cfg.lights[0].k);
  console.log('lighting="golden hour" key light K', lightingK);
  if (lightingK !== 2100) errors++;
  const shown = await p.evaluate(() => [...document.querySelectorAll('raking-light')].map(e => !e.shadowRoot.querySelector('canvas').hidden));
  console.log('toBlob bytes', size, 'canvases shown', shown.join(','));
  await b.close();
  if (errors || !(size > 10000) || shown.includes(false)) { console.log('FAIL'); process.exit(1); }
  console.log('PASS');
})();
