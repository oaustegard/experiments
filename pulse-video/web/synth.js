// A synthetic camera: a skin-coloured face on a textured background, with a pulse of known rate, sensor noise,
// 8-bit quantisation, a little head sway, and an auto-exposure model that scales the whole frame. Deterministic.
//   bpm      pulse rate (0 = no pulse); the instantaneous rate wanders +-3% over 10 s
//   amp      pulsatile amplitude relative to skin DC, along de Haan's normalised blood vector (0.33, 0.77, 0.53)
//   drift    auto-exposure: target gain jumps by up to +-4% every 3.7 s, approached with a 0.4 s time constant,
//            on top of a slow 1.5% swell
//   flicker  [Hz, depth]: an in-band oscillation of the whole frame's gain (a light source or exposure hunting)
//   noise    per-pixel sensor noise, standard deviation in 8-bit levels

export const PULSE_VECTOR = [0.33, 0.77, 0.53];

function rng(seed) { let s = seed >>> 0 || 1; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

export class Synth {
  constructor({ w = 640, h = 480, bpm = 72, amp = 0.004, drift = 1, flicker = null, noise = 2, seed = 1, sway = 1.5 } = {}) {
    Object.assign(this, { w, h, bpm, amp, drift, flicker, noise, sway });
    this.rand = rng(seed); this.img = new ImageData(w, h);
    const r = rng(seed + 7), bgc = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {   // a wall with soft blotches and a gradient
      const v = 0.55 + 0.12 * Math.sin(x / 37) * Math.cos(y / 53) + 0.08 * (y / h) + 0.03 * (r() - 0.5), i = (y * w + x) * 3;
      bgc[i] = 120 * v; bgc[i + 1] = 135 * v; bgc[i + 2] = 150 * v;
    }
    this.bgc = bgc; this.skin = [190, 140, 115];
    this.phase = 0; this.lastT = 0; this.gain = 1; this.target = 1; this.nextJump = 3.7;
    // where the face sits, normalised, and the box a face detector would report for it
    this.cx = 0.5; this.cy = 0.47; this.rx = 0.2; this.ry = 0.32;
    this.box = [this.cx - 0.75 * this.rx, this.cy - 0.8 * this.ry, this.cx + 0.75 * this.rx, this.cy + 0.75 * this.ry];
  }

  pulse(phase) { return Math.sin(phase) + 0.35 * Math.sin(2 * phase - 0.8); }

  gainAt(t, dt) {
    if (!this.drift) return 1;
    while (t >= this.nextJump) { this.target = 1 + 0.08 * (this.rand() - 0.5); this.nextJump += 3.7; }
    this.gain += (this.target - this.gain) * (1 - Math.exp(-dt / 0.4));
    return this.gain * (1 + 0.015 * Math.sin(2 * Math.PI * 0.05 * t));
  }

  // Draws the frame at time t (seconds, increasing) into a 2D context.
  draw(ctx, t) {
    const dt = Math.max(0, t - this.lastT); this.lastT = t;
    const f = this.bpm / 60 * (1 + 0.03 * Math.sin(2 * Math.PI * 0.1 * t));
    this.phase += 2 * Math.PI * f * dt;
    let g = this.gainAt(t, dt);
    if (this.flicker) g *= 1 + this.flicker[1] * Math.sin(2 * Math.PI * this.flicker[0] * t);
    const p = this.bpm ? this.amp * this.pulse(this.phase) : 0, { w, h, bgc, img, noise } = this, d = img.data, rnd = this.rand;
    const skin = this.skin.map((s, c) => s * (1 + p * PULSE_VECTOR[c]));
    const cx = (this.cx * w) + this.sway * Math.sin(2 * Math.PI * 0.23 * t), cy = this.cy * h, irx = 1 / (this.rx * w), iry = 1 / (this.ry * h);
    const ns = noise * 2.449;   // sum of two uniforms minus one has sd 1/sqrt(6)
    for (let y = 0; y < h; y++) {
      const dy = (y + 0.5 - cy) * iry, dy2 = dy * dy;
      for (let x = 0; x < w; x++) {
        const dx = (x + 0.5 - cx) * irx, i = y * w + x, o = i * 4;
        if (dx * dx + dy2 < 1) {
          const shade = 1 - 0.25 * (dx * dx + dy2);
          for (let c = 0; c < 3; c++) d[o + c] = skin[c] * shade * g + ns * (rnd() + rnd() - 1);
        } else for (let c = 0; c < 3; c++) d[o + c] = bgc[i * 3 + c] * g + ns * (rnd() + rnd() - 1);
        d[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }
}
