// GPU Bridge job: times the room-drum membrane on real hardware. engine.js arrives as an input string (so an
// unpushed engine can be timed) and is imported from a Blob URL. Runs the float64 accuracy check, then times
// sequential blocks, submit to readback as the page's audio path, for each kernel, grid, workgroup size and block.
const { src, block = 512, sr = 48000, blocks = 40, tile = [], multi = [] } = ctx.inputs.meta;
const E = await import(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
const dev = await ctx.gpu();
const lim = E.tileLimits(dev);
ctx.log(`tile limits: ${JSON.stringify(lim)}`);
const check = [];
for (const kernel of ["tile", "multi"]) check.push(await E.selfTest(dev, { kernel, nx: 40, ny: 30, block: 256 }));
ctx.log(`accuracy: ${check.map((c) => `${c.kernel} ${c.rmsDb.toFixed(1)} dB`).join(", ")}`);
const rows = [], jobs = [...tile.map(([nx, ny, wg, b]) => ["tile", nx, ny, wg, b || block]), ...multi.map(([nx, ny, b]) => ["multi", nx, ny, 0, b || block])];
for (const [i, [kernel, nx, ny, wg, B]] of jobs.entries()) {
  const blockMs = 1000 * B / sr;
  ctx.progress(i / jobs.length, `${kernel} ${nx}x${ny} wg ${wg}`);
  if (kernel === "tile" && (nx * ny > lim.maxCells || wg > lim.wg)) { rows.push({ kernel, nx, ny, wg, skipped: "over limits" }); continue; }
  try {
    const e = await E.DrumEngine.create(dev, { nx, ny, block: B, sr, kernel, wg: wg || undefined, ...E.coeffs({ sr }) });
    e.setPickups(3 + 3 * nx, nx * ny - 4 - 3 * nx);
    await e.render([{ x: nx * 0.4, y: ny * 0.4, r: 2, amp: 0.2, js: 0, T: 20 }]);
    const ms = [];
    for (let k = 0; k < blocks; k++) { const t = performance.now(); await e.render(k % 8 ? [] : [{ x: nx * 0.6, y: ny * 0.3, r: 2, amp: 0.2, js: 100, T: 30 }]); ms.push(performance.now() - t); }
    e.destroy(); ms.sort((a, b) => a - b);
    const r = { kernel, nx, ny, cells: nx * ny, wg, block: B, blockMs: +blockMs.toFixed(3), medianMs: +ms[blocks >> 1].toFixed(3), p90Ms: +ms[Math.floor(blocks * 0.9)].toFixed(3), load: +(ms[blocks >> 1] / blockMs).toFixed(3) };
    rows.push(r); ctx.log(JSON.stringify(r));
  } catch (err) { rows.push({ kernel, nx, ny, wg, error: String(err.message || err) }); ctx.log(`${kernel} ${nx}x${ny}: ${err.message || err}`); }
}
return { check, sr, limits: lim, rows, adapter: await ctx.describe() };
