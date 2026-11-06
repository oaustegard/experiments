// A deterministic stand-in for a training backend, for tests only (run/check_js.mjs, and the page with
// ?backend=fake in run/check.mjs resume). A bag-of-token-ids softmax model with Adam, no GPU, implementing
// the same interface as backends.js: step, score, exportSpike, loadSpike, saveState, loadState.
import { softmax } from "./creator.js";

export class FakeBackend {
  constructor(D = 97, delay = 0) { this.D = D; this.delay = delay; this.w = new Float64Array(D); this.m = new Float64Array(D); this.v = new Float64Array(D); this.t = 0; this.name = "fake (test backend, no GPU)"; }
  static async create({ delay = 0 } = {}) { return new FakeBackend(97, delay); }   // delay: ms per step, so a test can press Stop mid-run
  idx(id, j) { return (id * 7 + 13 * j) % this.D; }
  z(e) { return Float64Array.from(e.pos, (p, j) => { let s = 0; for (const id of e.ids) s += this.w[this.idx(id, j)]; return s / e.ids.length; }); }
  async step(seqs, targets, { lr }) {
    const g = new Float64Array(this.D);
    let loss = 0;
    seqs.forEach((e, i) => {
      const q = softmax(this.z(e));
      e.pos.forEach((_, j) => {
        if (targets[i][j] > 0) loss -= targets[i][j] * Math.log(q[j] + 1e-12);
        const d = (q[j] - targets[i][j]) / seqs.length;
        for (const id of e.ids) g[this.idx(id, j)] += d / e.ids.length;
      });
    });
    this.t++;
    if (this.delay) await new Promise((r) => setTimeout(r, this.delay));
    for (let k = 0; k < this.D; k++) {
      this.m[k] = 0.9 * this.m[k] + 0.1 * g[k]; this.v[k] = 0.999 * this.v[k] + 0.001 * g[k] * g[k];
      this.w[k] -= lr * (this.m[k] / (1 - 0.9 ** this.t)) / (Math.sqrt(this.v[k] / (1 - 0.999 ** this.t)) + 1e-8);
    }
    return { loss: loss / seqs.length };
  }
  async score(seqs) { return seqs.map((e) => Float32Array.from(this.z(e))); }
  exportSpike() { return { sm: { fake: true, w: Array.from(this.w) }, bytes: new Uint8Array(4) }; }
  loadSpike(sm) { if (sm.fake) this.w = Float64Array.from(sm.w); }
  saveState() { return { kind: "fake", w: this.w.slice(), m: this.m.slice(), v: this.v.slice(), t: this.t }; }
  loadState(st) { this.w = st.w.slice(); this.m = st.m.slice(); this.v = st.v.slice(); this.t = st.t; }
}
