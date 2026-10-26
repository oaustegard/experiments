// Renders each sample under its preset through the real WebGPU canvas path on SwiftShader and saves screenshots.
// Usage: (cd raking-light && python3 -m http.server 8765 &) ; node raking-light/test/render.js
// --use-vulkan=swiftshader is required for canvas presentation (METHODS.md, Environment gotchas).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-webgpu-adapter=swiftshader',
    '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-angle=swiftshader'] });
  const p = await b.newPage({ viewport: { width: 1280, height: 800 } });
  let errors = 0;
  p.on('console', m => { if (m.type() === 'error') { errors++; console.log('console error:', m.text()); } });
  p.on('pageerror', e => { errors++; console.log('pageerror:', e.message); });
  await p.goto('http://localhost:8765/web/index.html');
  await p.waitForTimeout(15000);
  const mode = await p.textContent('#hud-mode');
  console.log('mode:', mode);
  for (const [sample, preset] of [['Card', 'Gallery'], ['Poster', 'Raking'], ['Type', 'Letterpress'], ['Icons', 'Studio']]) {
    await p.click(`#samples .chip:text-is("${sample}")`); await p.waitForTimeout(3000);
    await p.click(`#presets .chip:text-is("${preset}")`); await p.waitForTimeout(12000);
    await p.screenshot({ path: path.join(__dirname, `${sample.toLowerCase()}-${preset.toLowerCase()}.png`) });
  }
  await b.close();
  if (!/^WebGPU ·/.test(mode) || errors) { console.log('FAIL'); process.exit(1); }
  console.log('PASS');
})();
