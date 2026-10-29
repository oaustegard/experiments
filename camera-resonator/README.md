# camera-resonator

A page that turns movement in front of the phone camera into strikes on a virtual resonator. The resonator is
either a steel plate shaped like the camera frame or a row of eight bells. Both are synthesised on the GPU as
thousands of damped sinusoids. Live at https://austegard.com/experiments/camera-resonator/web/.

The question: can a phone's GPU run a physically modelled instrument with thousands of modes in real time, played
by the camera, inside a browser?

## How it works

- **Motion.** Each camera frame is drawn into a 160×120 canvas and copied into a texture, the route that keeps
  iOS frames upright (METHODS.md). A compute pass sums the change in luma per 10×10-pixel cell over 16×12 cells.
  A cell fires a strike when its motion jumps above its own running baseline, it is the largest of its
  neighbours, and it has rested for 140 ms. Faster motion strikes harder and with a harder mallet, which means a
  shorter contact time and a brighter sound.
- **Where the strike lands.** On the plate, the cell's position is the strike point, so where you move decides
  which modes ring: mode (m, n) is excited by `sin(mπx)·sin(nπy)`. With the bells, the column picks the bell
  and the height picks the angle around its rim. Tapping the screen strikes too.
- **Synthesis.** `web/engine.js` keeps every mode's complex state on the GPU and renders one audio block per
  submit. A block takes three passes: `synth` (one thread per sample, with the modes split into slices),
  `advance` (one thread per mode) and `reduce`. As in `webgpu-bells`, `e^{st}` comes from float64-built tables
  (`hi[t >> 6]·lo[t & 63]`), so the GPU runs no sin or exp. Strike excitations are computed in float64 JS.
- **Playback.** An AudioWorklet (`web/worklet.js`) plays the blocks in order and reports dropouts. A peak
  limiter catches strikes that pile up. The page keeps 2–6 blocks queued and shows the GPU time per block
  against the block's own duration.

## Checks

`node test/verify.mjs selftest` runs the GPU block stream on SwiftShader against `engine.reference()`, a
float64 implementation. `node test/verify.mjs live` opens the page with Chromium's fake camera and checks that
strikes fire, blocks stream and no WebGPU error is raised. `bench/job.js` is a GPU Bridge job that times the
engine on real hardware. Results are in `RESULTS.md`.
