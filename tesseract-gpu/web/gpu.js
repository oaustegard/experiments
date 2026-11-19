// Minimal WebGPU helper shared by the tesseract-gpu kernels and engine.
//
//   const gpu = await Gpu.create();
//   const a = gpu.upload(new Float32Array(...));          // storage buffer
//   const u = gpu.uniform(new Uint32Array([n, k, ...]));   // uniform buffer (16-byte padded)
//   const out = gpu.buffer(bytes);
//   gpu.begin();
//   gpu.dispatch(WGSL_SOURCE, [u, a, out], [gx, gy, gz]);  // binding i = buffers[i], group 0
//   gpu.submit();
//   const host = await gpu.read(out);                      // Float32Array (or pass a type)
//
// Pipelines are cached per WGSL source string and entry point. All dispatches between
// begin() and submit() share one command encoder and one compute pass, so they run in
// order on the device.

export class Gpu {
  static async create(opts = {}) {
    if (!navigator.gpu) throw new Error("WebGPU unavailable");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const L = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
        maxBufferSize: L.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(10, L.maxStorageBuffersPerShaderStage),
        maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
      },
      requiredFeatures: (opts.timestamps && adapter.features.has("timestamp-query")) ? ["timestamp-query"] : [],
    });
    return new Gpu(adapter, device);
  }

  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
    this.pipelines = new Map();
    this.encoder = null;
    this.pass = null;
    this.dispatches = 0;
  }

  get info() {
    const i = this.adapter.info || {};
    return { vendor: i.vendor, architecture: i.architecture, device: i.device, description: i.description };
  }

  buffer(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 16) * 16), usage });
  }

  upload(arr, usage) {
    const buf = this.buffer(arr.byteLength, usage);
    this.device.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
    return buf;
  }

  // Uniform buffer from a Uint32Array / Float32Array / ArrayBuffer (mix types with a DataView).
  uniform(arr) {
    return this.upload(arr instanceof ArrayBuffer ? new Uint8Array(arr) : arr,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  }

  pipeline(code, entry = "main") {
    const key = entry + "\u0000" + code;
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code });
      p = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: entry } });
      this.pipelines.set(key, p);
    }
    return p;
  }

  begin() {
    this.encoder = this.device.createCommandEncoder();
    this.pass = this.encoder.beginComputePass();
  }

  dispatch(code, buffers, grid, entry = "main") {
    const p = this.pipeline(code, entry);
    const bind = this.device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    if (!this.pass) this.begin();
    this.pass.setPipeline(p);
    this.pass.setBindGroup(0, bind);
    const [x, y = 1, z = 1] = grid;
    this.pass.dispatchWorkgroups(x, y, z);
    this.dispatches++;
  }

  submit() {
    if (!this.pass) return;
    this.pass.end();
    this.device.queue.submit([this.encoder.finish()]);
    this.pass = null;
    this.encoder = null;
  }

  async read(buf, Type = Float32Array, bytes = buf.size) {
    this.submit();
    const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Type(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  // Splits a flat workgroup count into a 2-D grid within the 65535 per-dimension limit.
  static grid(n) {
    if (n <= 65535) return [n, 1, 1];
    const y = Math.ceil(n / 65535);
    return [Math.ceil(n / y), y, 1];
  }
}

// Test helpers shared by web/test/*.js modules.
export async function loadCase(base) {
  const man = await (await fetch(base + "manifest.json")).json();
  const f32 = async (key) => {
    const f = man.files[key];
    if (!f) return null;
    return new Float32Array(await (await fetch(base + f.file)).arrayBuffer());
  };
  return { man, f32 };
}

// max |got - want| and max |want|, plus the count of entries off by more than tol.
export function compare(got, want, tol = 1e-4) {
  let d = 0, m = 0, bad = 0, at = -1;
  for (let i = 0; i < want.length; i++) {
    const e = Math.abs(got[i] - want[i]);
    if (!(e <= d)) { d = e; at = i; }
    if (!(e <= tol)) bad++;
    m = Math.max(m, Math.abs(want[i]));
  }
  return { maxDiff: d, maxRef: m, bad, n: want.length, worstIndex: at };
}
