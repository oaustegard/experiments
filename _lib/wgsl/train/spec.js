// The training kernels' spec format and its shared WGSL helpers (kernel library, _lib/wgsl/README.md).
//
// A kernel spec is { u, b, wg, pre, body }: u the uniform names in order (suffix ":f" for f32, else
// u32), b the storage bindings [name, "r" | "rw", element type] from binding 1 up, wg the workgroup
// size, pre any helper functions or workgroup arrays, body the code after `let gi = ...` (the
// workgroup's linear index over a 2-D grid). wgsl(name, spec) assembles the shader; uniforms are one
// struct at binding 0, read with a dynamic offset so one buffer holds every dispatch's values.
// Moved from ettin-wgsl-train/web/train.js (step 3); the shaders are byte-for-byte the same.

export const SMAX = 1024;   // longest sequence: the attention kernels' workgroup arrays

// Abramowitz & Stegun 7.1.26 erf (|error| < 1.5e-7): exact-erf GELU as in PyTorch, and its derivative.
export const GELU = /* wgsl */ `
fn erf_(x: f32) -> f32 {
  let a = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * a);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
  return sign(x) * y;
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.7071067811865476)); }
fn gelu_d(x: f32) -> f32 { return 0.5 * (1.0 + erf_(x * 0.7071067811865476)) + x * exp(-0.5 * x * x) * 0.3989422804014327; }
`;

export const REDUCE = (n) => /* wgsl */ `
var<workgroup> red: array<f32, ${n}>;
fn wsum(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) {
    if (t < s) { red[t] = red[t] + red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
fn wmax(v: f32, t: u32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = ${n / 2}u; s > 0u; s = s >> 1u) {
    if (t < s) { red[t] = max(red[t], red[t + s]); }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
`;

export const ENTRY = (wg) => `@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
  let gi = wid.x + wid.y * nwg.x;`;

export function wgsl(name, k) {
  const fields = k.u.map((s) => { const [n, t] = s.split(":"); return `  ${n}: ${t === "f" ? "f32" : "u32"},`; });
  const st = `struct Uni {\n${fields.join("\n")}\n}\n@group(0) @binding(0) var<uniform> p: Uni;\n`;
  const binds = k.b.map(([n, m, t], i) => `@group(0) @binding(${i + 1}) var<storage, ${m === "rw" ? "read_write" : "read"}> ${n}: array<${t}>;`).join("\n");
  return `// ${name}\n${st}${binds}\n${k.pre || ""}\n${ENTRY(k.wg)}\n${k.body}\n}\n`;
}

export function grid(n) {
  if (n <= 65535) return [n, 1, 1];
  return [65535, Math.ceil(n / 65535), 1];
}

// SpecRunner: compiles a set of specs and records dispatches against them. Uniforms go into one buffer
// with a 256-byte slot per dispatch (dynamic offsets), so a flush is one writeBuffer, one command
// buffer and cached bind groups; clear(buf) ends the pass, clears, and resumes. Both trainers extend it.
const SLOT = 256;   // uniform slot bytes (minUniformBufferOffsetAlignment)

export class SpecRunner {
  constructor(device, { maxSlots = 2048 } = {}) {
    this.device = device;
    this.maxSlots = maxSlots;
    this.ops = [];
    this.bgCache = new Map();
    this.nid = 0;
  }

  // Compile { name: spec } into this.K (pipeline, bind-group layout, uniform names); a WGSL error names
  // the kernel, line and column.
  async compile(K) {
    const dev = this.device;
    this.K = {};
    await Promise.all(Object.entries(K).map(async ([name, k]) => {
      const mod = dev.createShaderModule({ code: wgsl(name, k), label: name });
      const entries = [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
        ...k.b.map(([, mode], i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: mode === "rw" ? "storage" : "read-only-storage" } }))];
      const bgl = dev.createBindGroupLayout({ entries, label: name });
      let pipe;
      try {
        pipe = await dev.createComputePipelineAsync({ layout: dev.createPipelineLayout({ bindGroupLayouts: [bgl] }), compute: { module: mod, entryPoint: "main" }, label: name });
      } catch (e) {
        const info = await mod.getCompilationInfo();
        throw new Error(`${name}: ${e.message}\n${info.messages.map((x) => `${x.lineNum}:${x.linePos} ${x.message}`).join("\n")}`);
      }
      this.K[name] = { pipe, bgl, u: k.u.map((s) => s.split(":")), nb: k.b.length };
    }));
    this.ubo = dev.createBuffer({ size: SLOT * this.maxSlots, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "uniforms" });
    this.udata = new ArrayBuffer(SLOT * this.maxSlots);
    this.uview = new DataView(this.udata);
  }

  // A storage buffer of n 4-byte values (f32 or u32), optionally initialized from a typed array.
  buf(label, n, init = null) {
    const size = Math.max(16, Math.ceil((n * 4) / 16) * 16);
    const b = this.device.createBuffer({ size, label, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, mappedAtCreation: !!init });
    if (init) {
      new Uint8Array(b.getMappedRange()).set(new Uint8Array(init.buffer, init.byteOffset, init.byteLength));
      b.unmap();
    }
    b._id = this.nid++;
    b._n = n;
    return b;
  }

  // Record one dispatch: bufs bind from 1 up in the spec's order, vals by uniform name (missing = 0),
  // groups a workgroup count (split over y past 65,535) or an [x, y, z] grid.
  op(name, bufs, vals, groups) {
    const k = this.K[name];
    if (!k) throw new Error(`no kernel ${name}`);
    if (bufs.length !== k.nb || bufs.some((b) => !b)) throw new Error(`${name}: expected ${k.nb} buffers`);
    this.ops.push({ k, name, bufs, vals, groups: Array.isArray(groups) ? groups : grid(groups) });
  }

  clear(buf) { this.ops.push({ clear: buf }); }

  // Submit the recorded dispatches; wait = await the GPU (queue order already keeps later uploads behind it).
  async flush(wait = true) {
    if (!this.ops.length) { if (wait) await this.device.queue.onSubmittedWorkDone(); return; }
    const dev = this.device;
    let slots = 0;
    for (const o of this.ops) {
      if (o.clear) continue;
      if (slots >= this.maxSlots) throw new Error("too many dispatches in one flush");
      o.slot = slots++;
      o.k.u.forEach(([n, t], i) => {
        const v = o.vals[n] ?? 0;
        if (t === "f") this.uview.setFloat32(o.slot * SLOT + 4 * i, v, true);
        else this.uview.setUint32(o.slot * SLOT + 4 * i, v, true);
      });
    }
    dev.queue.writeBuffer(this.ubo, 0, this.udata, 0, slots * SLOT);
    const enc = dev.createCommandEncoder();
    let pass = null;
    for (const o of this.ops) {
      if (o.clear) {
        if (pass) { pass.end(); pass = null; }
        enc.clearBuffer(o.clear);
        continue;
      }
      if (!pass) pass = enc.beginComputePass();
      const key = o.name + "|" + o.bufs.map((b) => b._id).join(",");
      let bg = this.bgCache.get(key);
      if (!bg) {
        bg = dev.createBindGroup({ layout: o.k.bgl, entries: [{ binding: 0, resource: { buffer: this.ubo, offset: 0, size: SLOT } },
          ...o.bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))] });
        this.bgCache.set(key, bg);
      }
      pass.setPipeline(o.k.pipe);
      pass.setBindGroup(0, bg, [o.slot * SLOT]);
      pass.dispatchWorkgroups(...o.groups);
    }
    if (pass) pass.end();
    dev.queue.submit([enc.finish()]);
    this.ops = [];
    if (wait) await dev.queue.onSubmittedWorkDone();
  }

  // Flush, then copy n values of buf back as a typed array.
  async read(buf, n = buf._n, type = Float32Array) {
    await this.flush();
    const bytes = n * 4;
    const st = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, st, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new type(st.getMappedRange(0, bytes).slice(0));
    st.unmap();
    st.destroy();
    return out;
  }
}
