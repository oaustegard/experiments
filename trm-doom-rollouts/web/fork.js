// Save and restore gPu(doom)'s game state on the GPU, and run tics without drawing them.
// The whole game lives in four storage buffers (sectors, things, line state, World); copying them aside
// and back is a fork, and every copy of the future runs on the same GPU device as the game.

export class Forker {
  constructor(eng, device) { this.eng = eng; this.d = device; this.bufs = null; this.rb = []; }

  live() { const e = this.eng; return [e.secBuf, e.mobjBuf, e.lsBuf, e.worldBuf]; }

  // loadMap() replaces the buffers, so the saved copies follow the live ones
  ensure() {
    const live = this.live();
    if (this.bufs && this.bufs.every((b, i) => b.src === live[i])) return;
    this.bufs?.forEach((b) => b.copy.destroy());
    this.bufs = live.map((src) => ({ src, copy: this.d.createBuffer({ size: src.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }) }));
  }

  save() {
    this.ensure();
    const enc = this.d.createCommandEncoder();
    for (const b of this.bufs) enc.copyBufferToBuffer(b.src, 0, b.copy, 0, b.src.size);
    this.d.queue.submit([enc.finish()]);
  }

  restore() {
    const enc = this.d.createCommandEncoder();
    for (const b of this.bufs) enc.copyBufferToBuffer(b.copy, 0, b.src, 0, b.src.size);
    this.d.queue.submit([enc.finish()]);
  }

  // World word `k` (u32), written after a restore: the reseed of a scenario
  setWord(k, v) { this.d.queue.writeBuffer(this.eng.worldBuf, k * 4, new Uint32Array([v >>> 0])); }

  // One tic of game logic. encodeFrame submits each tic itself and records the drawing passes into the
  // encoder it is given; that encoder is never submitted, so nothing is drawn.
  tic(input) { this.eng.encodeFrame(this.d.createCommandEncoder(), 1, { ...input, frac: 1 }); }

  // Copy byte ranges of the live buffers back: [[buffer, offset, size], ...] -> ArrayBuffers
  async read(parts) {
    const total = parts.reduce((a, [, , n]) => a + n, 0);
    let rb = this.rb.find((r) => !r.busy && r.size >= total);
    if (!rb) { rb = { buf: this.d.createBuffer({ size: Math.max(total, 4096), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), size: Math.max(total, 4096) }; this.rb.push(rb); }
    rb.busy = true;
    const enc = this.d.createCommandEncoder();
    let o = 0;
    for (const [b, off, n] of parts) { enc.copyBufferToBuffer(b, off, rb.buf, o, n); o += n; }
    this.d.queue.submit([enc.finish()]);
    await rb.buf.mapAsync(GPUMapMode.READ, 0, total);
    const all = rb.buf.getMappedRange(0, total).slice(0);
    rb.buf.unmap(); rb.busy = false;
    const out = []; o = 0;
    for (const [, , n] of parts) { out.push(all.slice(o, o + n)); o += n; }
    return out;
  }
}
