// Drives web/index.html end to end in headless Chromium: WebGPU on SwiftShader, and a fake camera that plays a
// .y4m file (make one from a still: ffmpeg -loop 1 -i in.jpg -vf "scale=960:540:force_original_aspect_ratio=decrease,
// pad=960:540:(ow-iw)/2:(oh-ih)/2,format=yuv420p" -t 2 -r 15 cam.y4m). Models load from their CDNs through
// $HTTPS_PROXY. Prints the start-up steps and the subject readout, and writes a screenshot.
// usage: node run-page.js cam.y4m out.png [auto|human|cat|dog]   (VW / VH env set the viewport, default 960x540)
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const [y4m, out, mode] = process.argv.slice(2);
(async () => {
  const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: 'localhost' } : undefined;
  const b = await chromium.launch({ proxy, args: ['--ignore-certificate-errors', '--enable-unsafe-webgpu', '--enable-features=Vulkan',
    '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--use-vulkan=swiftshader',
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + path.resolve(y4m)] });
  const ctx = await b.newContext({ ignoreHTTPSErrors: true, permissions: ['camera'], viewport: { width: +(process.env.VW || 960), height: +(process.env.VH || 540) } });
  const p = await ctx.newPage();
  p.on('pageerror', e => console.log('pageerror:', e.message));
  // localhost is a secure context, which WebGPU and the camera need
  await p.route('http://localhost/', r => r.fulfill({ contentType: 'text/html', body: fs.readFileSync(path.join(__dirname, '..', 'web', 'index.html'), 'utf8') }));
  await p.route('http://localhost/rtm_animal_fp16.bin', r => r.fulfill({ path: path.join(__dirname, '..', 'web', 'rtm_animal_fp16.bin') }));
  await p.goto('http://localhost/');
  const t0 = Date.now();
  await p.click('#start');
  await p.waitForFunction(() => document.getElementById('boot').hidden || [...document.querySelectorAll('#steps li')].some(l => l.className === 'err' && !/Face/.test(l.textContent)), null, { timeout: 180000 });
  const steps = await p.evaluate(() => [...document.querySelectorAll('#steps li')].map(l => `${l.className}: ${l.textContent}`));
  console.log(`started in ${((Date.now() - t0) / 1000).toFixed(1)} s\n  ` + steps.join('\n  '));
  if (mode) await p.click(`[data-mode="${mode}"]`);
  await p.waitForTimeout(12000);  // SwiftShader runs ~1 frame/s; give the filters time to settle
  console.log('readout:', await p.evaluate(() => document.getElementById('subject').innerText.replace(/\n/g, ' ')), '| marker', await p.evaluate(() => document.getElementById('marker').className));
  await p.screenshot({ path: out });
  await b.close();
})().catch(e => { console.error(e.message); process.exit(1); });
