# camera-resonator

A page that turns movement in front of the phone camera into strikes on a virtual resonator. The resonator is
either a steel plate shaped like the camera frame or a row of eight bells. Both are synthesised on the GPU as
thousands of damped sinusoids. Live at https://austegard.com/experiments/camera-resonator/web/.

The question: can a phone's GPU run a physically modelled instrument with thousands of modes in real time, played
by the camera, inside a browser?

## How it works

- **What counts as a strike.** A switch on the page picks one of three detectors (`web/detect.js`):
  - *Fingertips* (the default): MediaPipe hand tracking. A fingertip that moves fast and then stops strikes where
    it stopped. Only the fastest fingertip of each hand strikes, and the hand then rests for 180 ms, so a
    whole-hand tap is one note.
  - *Motion impacts*: block matching on the GPU gives each cell of a 16×12 grid a velocity. Each candidate shift
    is scored with the cell's mean brightness removed, so an exposure change doesn't read as motion; the frame is
    low-passed first, and larger shifts cost a little extra. A strike is something that moved fast for at least
    two frames and then stopped, with nothing nearby or further along its path still moving. When most of the
    frame moves at once (shake, a pan), strikes are muted. This works with any object: a hand, a stick, a pen.
  - *Any motion*: the first version's rule, kept for comparison. A cell fires when its brightness change jumps
    above its own baseline.
  Frames are copied through a 2D canvas into a texture, the route that keeps iOS frames upright (METHODS.md).
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
strikes fire, blocks stream and no WebGPU error is raised. `synthetic` plays a scripted scene (a disc that drops and stops, drifts, pans with the frame, rises and stops) and checks that impacts land only where it stopped. `switch` checks that the audio queue stays bounded after the resonator changes, and `hands` that the hand model loads and runs. `bench/job.js` is a GPU Bridge job that times the
engine on real hardware. Results are in `RESULTS.md`.
