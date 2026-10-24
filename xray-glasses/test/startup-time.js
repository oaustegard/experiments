// How long from pressing Start until the page is tracking? Boots web/index.html (or an older version of it,
// given as a git revision) in headless Chromium with SwiftShader and a fake camera playing test/pose.jpg,
// and records when each start-up step reports done. Writes ../results/startup.json, one entry per page.
// usage: node startup-time.js [git-rev ...]      (no rev: the working copy; needs ffmpeg and $HTTPS_PROXY)
const { chromium } = require('playwright');
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const D = __dirname, y4m = path.join(D, 'facing-cache', 'pose.y4m');
if (!fs.existsSync(y4m)) {
  fs.mkdirSync(path.dirname(y4m), { recursive: true });
  execFileSync('ffmpeg', ['-loglevel', 'error', '-y', '-loop', '1', '-i', path.join(D, 'pose.jpg'), '-vf',
    'scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2,format=yuv420p', '-t', '2', '-r', '15', y4m]);
}
const revs = process.argv.slice(2);
const pages = revs.length ? revs.map(r => ({ name: r, html: execFileSync('git', ['show', `${r}:xray-glasses/web/index.html`], { cwd: D }).toString() }))
  : [{ name: 'working copy', html: fs.readFileSync(path.join(D, '..', 'web', 'index.html'), 'utf8') }];
(async () => {
  const out = {};
  for (const pg of pages) {
    const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: 'localhost' } : undefined;
    const b = await chromium.launch({ proxy, args: ['--ignore-certificate-errors', '--enable-unsafe-webgpu', '--enable-features=Vulkan',
      '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--use-vulkan=swiftshader',
      '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-video-capture=' + y4m] });
    const p = await (await b.newContext({ ignoreHTTPSErrors: true, permissions: ['camera'] })).newPage();
    await p.route('http://localhost/', r => r.fulfill({ contentType: 'text/html', body: pg.html }));
    await p.goto('http://localhost/');
    const t0 = Date.now(); await p.click('#start');
    const done = {};
    for (;;) {
      const st = await p.evaluate(() => ({ steps: [...document.querySelectorAll('#steps li')].map(l => [l.className, l.textContent]), boot: document.getElementById('boot').hidden }));
      for (const [c, t] of st.steps) if (c && !(t in done)) done[t] = +((Date.now() - t0) / 1000).toFixed(1);
      if (st.boot || st.steps.some(([c, t]) => c === 'err' && !/Face/.test(t)) || Date.now() - t0 > 240000) { done.tracking_starts_s = +((Date.now() - t0) / 1000).toFixed(1); break; }
      await p.waitForTimeout(100);
    }
    await b.close();
    out[pg.name] = done; console.log(pg.name, JSON.stringify(done));
  }
  const f = path.join(D, '..', 'results', 'startup.json');
  const prev = fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, 'utf8')) : {};
  fs.writeFileSync(f, JSON.stringify({ ...prev, ...out, _note: 'seconds from pressing Start; headless Chromium, SwiftShader, fake camera' }, null, 1));
})().catch(e => { console.error(e.message); process.exit(1); });
