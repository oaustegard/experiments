// Runs the page's two WGSL passes, unmodified, on a real WebGPU implementation: headless Chromium
// (Playwright) with SwiftShader as the WebGPU adapter. Same three panels as render-cpu.js. Writes
// test-frame-gpu.png and reports how far each GPU panel is from the CPU port.
// Pipelines, blend states, formats, uniforms and bind groups follow the page's render(). The page copies each
// camera frame into an rgba8unorm texture (frameTexture); here the test JPEG goes into one at full size.
// Needs: npm i (this dir) and Playwright with Chromium (preinstalled in Claude Code cloud containers).
// Dawn on Mesa lavapipe (npm `webgpu`) compiles and validates (wgsl-check.js) but crashes on the first
// draw in this container, so execution goes through Chromium.
const fs = require('fs');
const path = require('path');
const { PNG } = require('pngjs');
const { chromium } = require('playwright');
const src = require('./extract.js');
const { panels, assemble } = require('./render-cpu.js');

const T = 1.234;  // the CPU port's grain time
const harness = (bonesCode, compCode) => `<!doctype html><script type="module">
const add = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
const max = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'max' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'max' } };
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
const errors = [];
device.onuncapturederror = (e) => errors.push(e.error.message);
const mods = { bones: device.createShaderModule({ code: ${JSON.stringify(bonesCode)} }), composite: device.createShaderModule({ code: ${JSON.stringify(compCode)} }) };
for (const [k, m] of Object.entries(mods)) for (const msg of (await m.getCompilationInfo()).messages) errors.push(k + ' ' + msg.type + ': ' + msg.message);
const pipes = {
  bones: device.createRenderPipeline({ layout: 'auto', vertex: { module: mods.bones, entryPoint: 'vs' },
    fragment: { module: mods.bones, entryPoint: 'fs', targets: [{ format: 'r16float', blend: add }, { format: 'r16float', blend: max }] }, primitive: { topology: 'triangle-strip' } }),
  composite: device.createRenderPipeline({ layout: 'auto', vertex: { module: mods.composite, entryPoint: 'vsFull' },
    fragment: { module: mods.composite, entryPoint: 'fsComp', targets: [{ format: 'bgra8unorm' }] }, primitive: { topology: 'triangle-list' } }),
};
const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
const bin = async (u, T) => new T(await (await fetch(u)).arrayBuffer());
window.renderPanel = async (name, W, H, nPrims, maskW, maskH, tissue) => {
  const img = await bin('/data/' + name + '/img', Uint8Array), f = await bin('/data/' + name + '/prims', Float32Array);
  const uniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // screen == video: scale 1, no offset, no mirror (the CPU port makes the same assumption)
  device.queue.writeBuffer(uniform, 0, new Float32Array([W, H, W, H, 1, 0, 0, 0, ${T}, tissue, 0, 1]));
  const prims = device.createBuffer({ size: f.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(prims, 0, f, 0, nPrims * 12);
  const mask = device.createTexture({ size: [maskW, maskH], format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: mask }, await bin('/data/' + name + '/mask', Uint8Array), { bytesPerRow: maskW }, [maskW, maskH]);
  const mk = () => device.createTexture({ size: [W, H], format: 'r16float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  const dens = mk(), glow = mk();
  const out = device.createTexture({ size: [W, H], format: 'bgra8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const bpr = 256 * Math.ceil(W * 4 / 256);
  const read = device.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const vid = device.createTexture({ size: [W, H], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: vid }, img, { bytesPerRow: W * 4 }, [W, H]);
  const t0 = performance.now();
  const enc = device.createCommandEncoder();
  const bp = enc.beginRenderPass({ colorAttachments: [
    { view: dens.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] },
    { view: glow.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
  bp.setPipeline(pipes.bones);
  bp.setBindGroup(0, device.createBindGroup({ layout: pipes.bones.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: { buffer: prims } }] }));
  bp.draw(4, nPrims); bp.end();
  const cp = enc.beginRenderPass({ colorAttachments: [{ view: out.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
  cp.setPipeline(pipes.composite);
  cp.setBindGroup(0, device.createBindGroup({ layout: pipes.composite.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniform } }, { binding: 1, resource: sampler },
    { binding: 2, resource: vid.createView() },
    { binding: 3, resource: mask.createView() }, { binding: 4, resource: dens.createView() }, { binding: 5, resource: glow.createView() }] }));
  cp.draw(3); cp.end();
  enc.copyTextureToBuffer({ texture: out }, { buffer: read, bytesPerRow: bpr }, [W, H]);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const ms = performance.now() - t0;
  const raw = new Uint8Array(read.getMappedRange()), px = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const s = y * bpr + x * 4, d = (y * W + x) * 4;
    px[d] = raw[s + 2]; px[d + 1] = raw[s + 1]; px[d + 2] = raw[s]; px[d + 3] = 255;  // BGRA -> RGBA
  }
  read.unmap();
  [uniform, prims, read].forEach(b => b.destroy()); [vid, mask, dens, glow, out].forEach(t => t.destroy());
  let s = ''; for (let i = 0; i < px.length; i += 0x8000) s += String.fromCharCode(...px.subarray(i, i + 0x8000));
  return { ms, px: btoa(s), errors };
};
window.ready = JSON.stringify(adapter.info);
</script>`;

(async () => {
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--use-vulkan=swiftshader'] });
  const page = await browser.newPage();
  page.on('pageerror', e => console.log('pageerror:', e.message));
  const data = {};
  for (const [k, p] of Object.entries(panels)) {
    data[k] = { img: Buffer.from(p.img.buffer, p.img.byteOffset, p.img.byteLength), prims: Buffer.from(p.pr.f.buffer, p.pr.f.byteOffset, p.pr.f.byteLength),
      mask: p.mask ? Buffer.from(p.mask) : Buffer.from([0]) };
  }
  await page.route('http://localhost/**', (route) => {
    const u = new URL(route.request().url());
    if (u.pathname === '/') return route.fulfill({ contentType: 'text/html', body: harness(src.bones, src.composite) });
    const [, , name, kind] = u.pathname.split('/');
    return route.fulfill({ contentType: 'application/octet-stream', body: data[name][kind] });
  });
  await page.goto('http://localhost/');
  const info = await (await page.waitForFunction(() => window.ready, null, { timeout: 60000 })).jsonValue();
  console.log('adapter:', info);
  const g = {}, stats = {}; let errors = [];
  for (const [k, p] of Object.entries(panels)) {
    const { W, H } = p;
    const r = await page.evaluate(([k, W, H, n, mw, mh, t]) => window.renderPanel(k, W, H, n, mw, mh, t),
      [k, W, H, p.pr.n, p.mask ? p.maskW : 1, p.mask ? p.maskH : 1, p.mask ? 1 : 0]);
    errors = r.errors;
    const px = Buffer.from(r.px, 'base64');
    const hist = new Float64Array(256); let sum = 0;
    for (let i = 0; i < W * H; i++) for (let c = 0; c < 3; c++) { const dd = Math.abs(px[i * 4 + c] - p.px[i * 4 + c]); hist[dd]++; sum += dd; }
    const n = W * H * 3, pct = (q) => { let c = 0; for (let v = 0; v < 256; v++) { c += hist[v]; if (c >= q * n) return v; } return 255; };
    console.log(`${p.label}: ${W}x${H}, ${p.pr.n} prims, ${r.ms.toFixed(0)} ms; vs CPU port per 8-bit channel: mean ${(sum / n).toFixed(2)}, p50 ${pct(0.5)}, p99 ${pct(0.99)}, max ${pct(1)}`);
    g[k] = { ...p, px };
    stats[k] = { mean: +(sum / n).toFixed(2), p50: pct(0.5), p99: pct(0.99), max: pct(1), prims: p.pr.n };
  }
  await browser.close();
  if (errors.length) { console.log('GPU errors:', errors); process.exit(1); }
  const png = assemble(g.human, g.dog, g.cat);
  const outPath = path.join(__dirname, '..', 'test-frame-gpu.png');
  fs.writeFileSync(outPath, PNG.sync.write(png));
  fs.mkdirSync(path.join(__dirname, '..', 'results'), { recursive: true });
  fs.writeFileSync(path.join(__dirname, '..', 'results', 'gpu_vs_cpu.json'), JSON.stringify({ adapter: 'SwiftShader (headless Chromium)', per_8bit_channel: stats }, null, 1));
  console.log('wrote', outPath, png.width + 'x' + png.height, 'and results/gpu_vs_cpu.json');
})().catch(e => { console.error(e); process.exit(1); });
