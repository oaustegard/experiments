// Plays the stereo blocks the GPU renders, in order. Reports each finished block so the page can keep the queue
// at its target depth, and each dropout (queue ran dry while playing). A peak limiter with instant attack and a
// ~0.5 s release keeps a pile-up of strikes from clipping.
class ResonatorOut extends AudioWorkletProcessor {
  constructor() {
    super();
    this.q = []; this.cur = null; this.pos = 0; this.done = 0; this.under = 0; this.playing = false; this.env = 0;
    this.rel = Math.exp(-1 / (0.5 * sampleRate));
    this.port.onmessage = (e) => {
      if (e.data.type === "block") this.q.push(e.data.data);
      else if (e.data.type === "reset") { this.q = []; this.cur = null; this.playing = false; }
    };
  }
  process(_in, outs) {
    const L = outs[0][0], R = outs[0][1] || L;
    for (let i = 0; i < L.length; i++) {
      if (!this.cur || this.pos >= this.cur.length) {
        if (this.cur) { this.done++; this.port.postMessage({ type: "done", n: this.done }); }
        this.cur = this.q.shift() || null; this.pos = 0;
        if (!this.cur) {
          if (this.playing) { this.under++; this.port.postMessage({ type: "under", n: this.under }); }
          this.playing = false;
          for (; i < L.length; i++) { L[i] = 0; R[i] = 0; }
          return true;
        }
        this.playing = true;
      }
      const l = this.cur[this.pos], r = this.cur[this.pos + 1]; this.pos += 2;
      const pk = Math.max(Math.abs(l), Math.abs(r));
      this.env = pk > this.env ? pk : this.env * this.rel;
      const g = this.env > 0.89 ? 0.89 / this.env : 1;
      L[i] = l * g; R[i] = r * g;
    }
    return true;
  }
}
registerProcessor("resonator-out", ResonatorOut);
