// Helpers for the WebGPU games captured by this agent (not a game; run.cjs never loads it by name).
const fs = require('fs'), path = require('path');

// Wait for the page's WebGPU queue to finish. getDevice is evaluated in the page
// and returns a GPUDevice or null.
async function drain(page, getDevice) {
  await page.evaluate(`(async () => { const d = (${getDevice})(); if (d) await d.queue.onSubmittedWorkDone(); })()`);
}

// lib.sequence plus a GPU drain before each screenshot. SwiftShader falls
// behind a frozen clock: frames queue up and a screenshot shows one from a
// second or more earlier.
async function sequenceGPU(page, h, name, pageSeconds, frames, onFrame, getDevice) {
  const dir = path.join(h.OUT, name);
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  const per = (pageSeconds * 1000) / frames, TICK = Math.min(34, per), step = per - TICK;
  for (let i = 0; i < frames; i++) {
    if (onFrame) await onFrame(i);
    if (step > 0) await page.clock.fastForward(step);
    await page.clock.runFor(TICK);
    await drain(page, getDevice);
    await page.waitForTimeout(40);
    await page.screenshot({ path: path.join(dir, `f${String(i).padStart(4, '0')}.jpg`), quality: 90 });
  }
  console.log(name, frames, 'frames ->', dir);
}

// After h.freezeClock the fake clock sits behind the real rAF timestamps, so a
// game that does dt = min(.05, now - last) gets one large negative dt and its
// fixed-step accumulator stalls until page time catches up. Step page time
// until simTime() (evaluated in the page) moves again.
async function waitSim(page, simTime, getDevice) {
  for (let k = 0; k < 40; k++) {
    const t0 = await page.evaluate(simTime);
    await page.clock.runFor(500);
    if (getDevice) await drain(page, getDevice);
    if (await page.evaluate(`(${simTime})() > ${t0} + 0.2`)) return;
  }
  console.log('waitSim: simulation never resumed');
}

// gamma (degrees) that makes asin(cos(beta)*sin(gamma)) equal `steer` degrees:
// the roll these games read from gravity along the screen's right edge.
function gammaFor(steer, beta) {
  const r = Math.PI / 180;
  return Math.asin(Math.max(-1, Math.min(1, Math.sin(steer * r) / Math.cos(beta * r)))) / r;
}

module.exports = { drain, sequenceGPU, waitSim, gammaFor };

// Serve a game's index.html with extra code spliced in (the site checkout is
// untouched): `head` runs before the game's own scripts, `tail` replaces the
// last "})();\n</script>" so it runs inside the game's closure. Then reload.
// The head always records the WebGPU device as window.__gpuDevice.
async function injectAndReload(page, urlPart, { head = '', tail = '' } = {}) {
  const HEAD = `<script>(() => { if (!window.GPUAdapter) return; const rd = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (...a) { const d = await rd.apply(this, a); window.__gpuDevice = d; return d; }; })();${head}</script>`;
  await page.route(u => u.pathname.endsWith(urlPart), async (route) => {
    const res = await route.fetch();
    let body = await res.text();
    body = body.replace(/<head>/i, '<head>' + HEAD);
    if (tail) {
      const i = body.lastIndexOf('})();');
      if (i < 0) throw new Error('injectAndReload: no closing IIFE');
      body = body.slice(0, i) + tail + '\n' + body.slice(i);
    }
    await route.fulfill({ response: res, body, headers: { ...res.headers(), 'content-type': 'text/html' } });
  });
  await page.reload({ waitUntil: 'load' });
}
module.exports.injectAndReload = injectAndReload;
