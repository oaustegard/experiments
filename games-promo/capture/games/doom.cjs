// gPu(doom): E1M1 played from the keyboard (the ?bot TRM planner was too slow on SwiftShader:
// it lagged its own plans and hugged walls). ?embed=E1M1 starts the map straight away.
// iddqd + idkfa, chaingun, then run up the long corridor firing, stop short of its end
// strafe side to side firing into the zombies coming down it. The fake clock goes in before load (reload) and is paused as soon
// as the shader compile starts, so the page's slow-GPU watchdog (which measures page time)
// never sees SwiftShader's slow tics; after each step the script waits for the GPU queue.
async function pauseClock(page) {
  for (let i = 0; ; i++) {
    try { return await page.clock.pauseAt(await page.evaluate(() => Date.now() + 1500)); }
    catch (e) { if (i > 5 || !/past/.test(e.message)) throw e; }
  }
}
module.exports = async (b, h) => {
  const page = await h.open(b, 'doom/?embed=E1M1', { landscape: true });
  await page.context().addInitScript(() => {
    const rd = GPUAdapter.prototype.requestDevice; // keep the device reachable to wait on its queue
    GPUAdapter.prototype.requestDevice = async function (...a) { const d = await rd.apply(this, a); window.__dev = d; return d; };
  });
  await page.clock.install();
  await page.reload({ waitUntil: 'load' });
  await page.waitForFunction(() => /Compiling/.test(document.getElementById('status')?.textContent || ''), null, { timeout: 900000, polling: 300 });
  await pauseClock(page);
  await page.waitForFunction(() => document.getElementById('status').textContent === '' && document.getElementById('banner').hidden,
    null, { timeout: 900000, polling: 1000 });
  await page.addStyleTag({ content: '#fps, #pos { display: none !important; }' }); // the frame-rate meter
  const gpu = () => page.evaluate(() => window.__dev.queue.onSubmittedWorkDone());
  const tics = async (n) => { for (let i = 0; i < n; i += 2) { await page.clock.runFor(57); await gpu(); } };
  await tics(10);
  for (const c of ['iddqd', 'idkfa']) { await page.keyboard.type(c); await tics(6); }
  await page.keyboard.press('Digit4'); // chaingun
  await tics(320); // the cheat message fades
  const k = page.keyboard, held = new Set();
  const want = async (keys) => {
    for (const x of [...held]) if (!keys.includes(x)) { await k.up(x); held.delete(x); }
    for (const x of keys) if (!held.has(x)) { await k.down(x); held.add(x); }
  };
  const FWD = +(process.env.DOOM_FWD || 50);
  await h.sequence(page, 'doom', 5, 150, async (i) => {
    if (i < FWD) await want(['ShiftLeft', 'KeyW', 'ControlLeft']);
    else await want(['ControlLeft', ((i - FWD) % 30) < 15 ? 'KeyA' : 'KeyD']);
    await gpu();
  });
  await page.context().close();
};
