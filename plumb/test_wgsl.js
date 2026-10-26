// Compiles Plumb's two WGSL modules, builds both pipelines and runs 240 sim steps offscreen.
// Headless Chromium loses any device that presents to a canvas, so this renders to a texture.
// Serve web/ on :8765 first, as for test_plumb.js.
const { chromium } = require('playwright');
const src = require('fs').readFileSync('/home/user/experiments/plumb/web/index.html', 'utf8');
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
    const N = 1024, init = new Float32Array(N * 4);
    for (let i = 0; i < N; i++) { init[i * 4] = Math.random() * 2 - 1; init[i * 4 + 1] = Math.random() * 2 - 1; }
    const pbuf = d.createBuffer({ size: init.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    d.queue.writeBuffer(pbuf, 0, init);
    const ubuf = d.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const comp = d.createComputePipeline({ layout: 'auto', compute: { module: d.createShaderModule({ code: C }), entryPoint: 'sim' } });
    const rm = d.createShaderModule({ code: R });
    const rend = d.createRenderPipeline({ layout: 'auto', vertex: { module: rm, entryPoint: 'vs' },
      fragment: { module: rm, entryPoint: 'fs', targets: [{ format: 'rgba8unorm', blend: { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } } }] } });
    const entries = [{ binding: 0, resource: { buffer: pbuf } }, { binding: 1, resource: { buffer: ubuf } }];
    const cbg = d.createBindGroup({ layout: comp.getBindGroupLayout(0), entries });
    const rbg = d.createBindGroup({ layout: rend.getBindGroupLayout(0), entries });
    const tex = d.createTexture({ size: [64, 64], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const ub = new ArrayBuffer(96), uf = new Float32Array(ub), uu = new Uint32Array(ub);
    uf[0] = 0; uf[1] = 1; uf[2] = 1; uf[3] = 1 / 60; uu[5] = N; uu[6] = 0;
    for (let s = 0; s < 240; s++) {
      uf[4] = s / 60; d.queue.writeBuffer(ubuf, 0, ub);
      const e = d.createCommandEncoder(), cp = e.beginComputePass();
      cp.setPipeline(comp); cp.setBindGroup(0, cbg); cp.dispatchWorkgroups(N / 64); cp.end();
      const rp = e.beginRenderPass({ colorAttachments: [{ view: tex.createView(), loadOp: 'clear', storeOp: 'store' }] });
      rp.setPipeline(rend); rp.setBindGroup(0, rbg); rp.draw(6, N); rp.end();
      d.queue.submit([e.finish()]);
    }
    const err = await d.popErrorScope();
    const rb = d.createBuffer({ size: init.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = d.createCommandEncoder(); e.copyBufferToBuffer(pbuf, 0, rb, 0, init.byteLength); d.queue.submit([e.finish()]);
    await rb.mapAsync(GPUMapMode.READ); const out = new Float32Array(rb.getMappedRange());
    let y0 = 0, y1 = 0; for (let i = 0; i < N; i++) { y0 += init[i * 4 + 1]; y1 += out[i * 4 + 1]; }
    return { msgs, validation: err && err.message, meanY_before: +(y0 / N).toFixed(3), meanY_after_4s_gravity_down: +(y1 / N).toFixed(3) };
  }, { C: grab('COMPUTE_WGSL'), R: grab('RENDER_WGSL') })));
  await b.close();
})();
