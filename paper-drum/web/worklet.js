// One AudioWorklet node does both ends: it plays the stereo blocks the GPU renders (as room-drum's worklet does) and
// listens to the microphone on its input for taps (taps.js). Keeping them in one node gives the tap detector the
// exact samples the speaker is playing, which it uses to tell the drum's own sound from a new tap.
//
// To the page: { type: "tap", n, frame, level, db } per onset (n counts mic samples, frame is currentFrame when it
// was reported), "done" per finished block, "under" per dropout, and "meter" about ten times a second.
// { type: "train" } plays the start-up noise burst the echo canceller learns the speaker from (taps.js).
import { TapDetector, trainingNoise } from "./taps.js";

class PaperDrum extends AudioWorkletProcessor {
  constructor() {
    super();
    this.q = []; this.cur = null; this.pos = 0; this.done = 0; this.under = 0; this.playing = false; this.env = 0;
    this.rel = Math.exp(-1 / (0.5 * sampleRate));
    this.det = new TapDetector(sampleRate); this.played = new Float32Array(128); this.meterAt = 0; this.mx = 0; this.extra = null; this.ei = 0;
    this.port.onmessage = ({ data: d }) => {
      if (d.type === "block") this.q.push(d.data);
      else if (d.type === "reset") { this.q = []; this.cur = null; this.playing = false; }
      else if (d.type === "learn") this.det.learnFrom(d.n);
      else if (d.type === "set") this.det.set(d);
      else if (d.type === "train") { this.extra = trainingNoise(sampleRate); this.ei = 0; }
    };
  }
  process(ins, outs) {
    const L = outs[0][0], R = outs[0][1] || L, P = this.played;
    for (let i = 0; i < L.length; i++) {
      if (!this.cur || this.pos >= this.cur.length) {
        if (this.cur) { this.done++; this.port.postMessage({ type: "done", n: this.done }); }
        this.cur = this.q.shift() || null; this.pos = 0;
        if (!this.cur) {
          if (this.playing) { this.under++; this.port.postMessage({ type: "under", n: this.under }); }
          this.playing = false;
          L[i] = R[i] = P[i] = this.mix(); continue;
        }
        this.playing = true;
      }
      const l = this.cur[this.pos], r = this.cur[this.pos + 1]; this.pos += 2;
      const pk = Math.max(Math.abs(l), Math.abs(r));
      this.env = pk > this.env ? pk : this.env * this.rel;
      const g = this.env > 0.89 ? 0.89 / this.env : 1;
      const x = this.mix(); L[i] = l * g + x; R[i] = r * g + x; P[i] = 0.5 * (L[i] + R[i]);
    }
    const mic = ins[0] && ins[0][0];
    if (mic) {
      for (const t of this.det.process(mic, P.length === mic.length ? P : null)) this.port.postMessage({ type: "tap", ...t, frame: currentFrame });
      for (let i = 0; i < mic.length; i++) this.mx = Math.max(this.mx, Math.abs(mic[i]));
      if (currentFrame >= this.meterAt) {
        this.port.postMessage({ type: "meter", n: this.det.n, peakDb: 20 * Math.log10(this.mx + 1e-9), baseDb: 10 * Math.log10(this.det.base), cpl: this.det.cpl, erleDb: this.det.erleDb(), training: !!this.extra });
        this.mx = 0; this.meterAt = currentFrame + Math.round(0.1 * sampleRate);
      }
    }
    return true;
  }
  mix() {   // the next sample of the training noise, if it is playing
    if (!this.extra) return 0;
    const v = this.extra[this.ei]; this.ei += 2;
    if (this.ei >= this.extra.length) this.extra = null;
    return v;
  }
}
registerProcessor("paper-drum", PaperDrum);
