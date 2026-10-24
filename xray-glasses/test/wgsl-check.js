// Compile both WGSL modules with Dawn and build the two render pipelines exactly as the page does.
// Execution is not attempted: queue work never completes under this sandbox's lavapipe.
const { create, globals } = require('webgpu');
Object.assign(globalThis, globals);
const src = require('./extract.js');
(async () => {
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  const info = adapter.info || {};
  console.log('adapter:', info.vendor, info.architecture, info.device, info.description);
  const device = await adapter.requestDevice();
  const errors = [];
  device.addEventListener?.('uncapturederror', (e) => errors.push(e.error.message));
  device.onuncapturederror = (e) => errors.push(e.error.message);
  let bad = 0;
  const mods = {};
  for (const name of ['bones', 'composite']) {
    const m = device.createShaderModule({ code: src[name] });
    const ci = await m.getCompilationInfo();
    for (const msg of ci.messages) { console.log(`${name} ${msg.type} ${msg.lineNum}:${msg.linePos} ${msg.message}`); if (msg.type === 'error') bad++; }
    mods[name] = m;
  }
  device.pushErrorScope('validation');
  const add = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } };
  const max = { color: { srcFactor: 'one', dstFactor: 'one', operation: 'max' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'max' } };
  const bones = device.createRenderPipeline({ layout: 'auto', vertex: { module: mods.bones, entryPoint: 'vs' },
    fragment: { module: mods.bones, entryPoint: 'fs', targets: [{ format: 'r16float', blend: add }, { format: 'r16float', blend: max }] }, primitive: { topology: 'triangle-strip' } });
  const comp = device.createRenderPipeline({ layout: 'auto', vertex: { module: mods.composite, entryPoint: 'vsFull' },
    fragment: { module: mods.composite, entryPoint: 'fsComp', targets: [{ format: 'bgra8unorm' }] }, primitive: { topology: 'triangle-list' } });
  bones.getBindGroupLayout(0); comp.getBindGroupLayout(0);
  const e = await device.popErrorScope();
  if (e) { console.log('pipeline validation error:', e.message); bad++; }
  await new Promise(r => setTimeout(r, 200));
  if (errors.length) { console.log('uncaptured:', errors); bad++; }
  console.log(bad ? 'WGSL_FAIL' : 'WGSL_OK (2 modules compiled, 2 pipelines created)');
  process.exit(bad ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
