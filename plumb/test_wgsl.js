// Compiles Plumb's two shard WGSL modules, builds both pipelines and runs 30 steps offscreen.
// Headless Chromium loses any device that presents to a canvas, so this renders to a texture.
// Serve oaustegard.github.io/fun-and-games/plumb on :8765 first, as for test_plumb.js.
const { chromium } = require('playwright');
const src = require('fs').readFileSync(process.env.PLUMB_HTML || require('path').join(__dirname, '../../oaustegard.github.io/fun-and-games/plumb/index.html'), 'utf8');
const grab = n => src.match(new RegExp(`const ${n} = /\\* wgsl \\*/\`([\\s\\S]*?)\`;`))[1];
(async () => {
  const b = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'] });
  const p = await b.newPage();
  await p.goto('http://localhost:8765/manifest.json');
  console.log(JSON.stringify(await p.evaluate(async ({ C, R }) => {
    const d = await (await navigator.gpu.requestAdapter()).requestDevice();
    const msgs = [];
    for (const [n, code] of [['compute', C], ['render', R]]) {
      const info = await d.createShaderModule({ code }).getCompilationInfo();
      for (const m of info.messages) msgs.push(`${n} ${m.type} L${m.lineNum}: ${m.message}`);
    }
    d.pushErrorScope('validation');
    const N = 4096, S = 12, init = new Float32Array(N * S);
    for (let i = 0; i < 64; i++) { init.set([0, 0, 0.5, -0.5, 1, 0.5, 0.2, 1.0, 0.02, 3, 0, 0], i * S); }  // 64 live, rest dead
    const sbuf = d.createBuffer({ size: init.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    d.queue.writeBuffer(sbuf, 0, init);
    const ubuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const comp = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: C }), entryPoint: 'sim' } });
    const rm = d.createShaderModule({ code: R });
    const rend = d.createRenderPipeline({ layout: 'auto', vertex: { module: rm, entryPoint: 'vs' },
      fragment: { module: rm, entryPoint: 'fs', targets: [{ format: 'rgba8unorm', blend: { color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } } }] } });
    const entries = [{ binding: 0, resource: { buffer: sbuf } }, { binding: 1, resource: { buffer: ubuf } }];
    const cbg = d.createBindGroup({ layout: comp.getBindGroupLayout(0), entries });
    const rbg = d.createBindGroup({ layout: rend.getBindGroupLayout(0), entries });
    const tex = d.createTexture({ size: [64, 64], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    d.queue.writeBuffer(ubuf, 0, new Float32Array([0, 2.4, 1, 1 / 60]));
    for (let s = 0; s < 30; s++) {
      const e = d.createCommandEncoder(), cp = e.beginComputePass();
      cp.setPipeline(comp); cp.setBindGroup(0, cbg); cp.dispatchWorkgroups(N / 64); cp.end();
      const rp = e.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: 'clear', storeOp: 'store' }] });
      rp.setPipeline(rend); rp.setBindGroup(0, rbg); rp.draw(6, N); rp.end();
      d.queue.submit([e.finish()]);
    }
    const err = await d.popErrorScope();
    const rb = d.createBuffer({ size: init.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = d.createCommandEncoder(); e.copyBufferToBuffer(sbuf, 0, rb, 0, init.byteLength); d.queue.submit([e.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Float32Array(rb.getMappedRange());
    return { msgs, validation: err && err.message,
      live_after_half_s: { x: +out[0].toFixed(3), y: +out[1].toFixed(3), vy: +out[3].toFixed(3), life: +out[7].toFixed(3) },
      dead_untouched: out[64 * S + 7] === 0 && out[64 * S + 1] === 0 };
  }, { C: grab('SHARD_COMPUTE'), R: grab('SHARD_RENDER') })));
  await b.close();
})();
