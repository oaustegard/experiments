# camera-resonator: results

The Camera Resonator page turns movement in front of a camera into strikes on a virtual steel plate or a row of
eight bells. A vibrating plate or bell is the sum of its *modes*. Each mode is one natural frequency of the
object, ringing as a sine wave that dies away (a damped sinusoid). A *strike* is a tap at a point; where it lands
decides how loudly each mode rings. The page sums thousands of modes on the GPU, one *block* of 512 samples at a
time. At 48 kHz, that block is 10.7 ms of sound, so each block has to be computed in under 10.7 ms or the sound
drops out. See `README.md` for how the page works.

## Answer

The question was whether a GPU can sum thousands of modes fast enough to play live from the camera, inside a
browser. On a Mac (Apple GPU, Chrome) it can, with room to spare, up to about 32,000 modes. At that size a block
takes 2 ms median <!-- @evidence[[results/resonator-bench-1-mac.json#plate.32000.medianMs]] --> and 6.3 ms at the
90th percentile <!-- @evidence[[results/resonator-bench-1-mac.json#plate.32000.p90Ms]] -->, inside the 10.7 ms
budget. At 64,000 modes the 90th percentile is 12.9 ms <!-- @evidence[[results/resonator-bench-1-mac.json#plate.64000.p90Ms]] -->,
which would drop out. The GPU result matches the same sum done in double precision (float64) in JavaScript, with
an RMS error of −141.5 dB relative to peak <!-- @evidence[[results/resonator-bench-1-mac.json#check.rmsDb]] -->.
That is below what 24-bit audio can represent, so the GPU adds no audible error. Only the engine was timed, and
only on one Mac. The full loop from camera to speaker is unmeasured, and the iPhone is untested. A phone run
would change this answer.

## Findings

1. On the Mac the usable ceiling is about 32,000 modes. The 90th-percentile block time crosses the 10.7 ms budget
   between 32,000 (6.3 ms) and 64,000 modes (12.9 ms <!-- @evidence[[results/resonator-bench-1-mac.json#plate.64000.p90Ms]] -->),
   while the median stays at 2 ms <!-- @evidence[[results/resonator-bench-1-mac.json#plate.64000.medianMs]] -->.
   The likely cause, which is inferred and not separately timed, is that one block in eight carries a strike, and
   a strike block cannot skip modes that have gone silent. (`resonator-bench-1`)
2. Up to 8,000 modes the median block takes 0.8 ms <!-- @evidence[[results/resonator-bench-1-mac.json#plate.8000.medianMs]] -->
   regardless of size, which looks like the fixed cost of a GPU round trip (submit plus readback). The 90th
   percentile does grow with size, from 1.2 ms at 1,000 modes to 2.5 ms at 8,000. (`resonator-bench-1`)
3. Keeping mode state on the GPU from block to block costs no accuracy. With float64-built lookup tables in place
   of sin/exp on the GPU (the trick from `webgpu-bells`, see METHODS.md), the streamed output matched float64 at
   −141.5 dB RMS on the Mac <!-- @evidence[[results/resonator-bench-1-mac.json#check.rmsDb]] -->. On SwiftShader,
   Chrome's software GPU, it matched at −140.9 to −141.1 dB for a plate and −139.7 dB for the bells. (`resonator-bench-1`, `test/verify.mjs selftest`)
4. Camera-triggered strikes work end to end in a headless browser. With Chromium's fake camera the page fired
   strikes, streamed blocks and raised no WebGPU errors. After the gain fix the output peaked at 0.45 of full
   scale. SwiftShader is far too slow to keep up in real time, so it says nothing about dropouts on real devices. (`test/verify.mjs live`)

## Method

- Engine: `web/engine.js` at commit bc7f76f, with 512-sample blocks at 48 kHz. The plate presets use one strike
  point and two pickups. The bell preset is 8 bells of 360 modes each (2,880 in all): 30 partials as split pairs,
  plus 300 short modes for the clapper's click.
- Accuracy: `selfTest` renders a plate (2,000 modes on the Mac, 1,200 in the headless check) over 16 blocks, with
  random strikes at random sample offsets, and compares it sample by sample with `reference()`, a float64
  implementation.
- Timing: `bench/job.js` on the GPU Bridge, which runs jobs on a browser tab on Oskar's machine. It imports the
  engine at the commit, strikes twice so every mode rings, then times 60 blocks one after another, from submit to
  readback. That is the same path the page waits on. Every 8th block carries a strike. "Load" is block time
  divided by the block's 10.7 ms. Chrome rounds the times to about 0.1 ms.
- Not measured: the camera-to-sound latency on a device, the iPhone GPU, and iOS audio with the silent switch on.

## Log

### 2026-10-29: build, headless checks, Mac bench

Built the page, engine, worklet and checks. On SwiftShader, `selftest` on a 1,200-mode plate gave −140.9 dB with
512-sample blocks and −141.1 dB with 1,024-sample blocks; the bells gave −139.7 dB.

A first `live` run with the fake camera fired 50 strikes in 12 s, but the output peaked at 2.5, far over full
scale. The preset gains were cut (plate 0.35 to 0.12, bells 0.35 to 0.20). A second 8 s run fired 30 strikes and
peaked at 0.45.

Bridge job `resonator-bench-1` ran on the Mac tab (apple metal-3, Chrome 156) in 3.6 s:

| modes | median ms | p90 ms | median load |
|---|---|---|---|
| 1,000 | 0.8 | 1.2 | 7.5% |
| 2,000 | 0.8 | 1.3 | 7.5% |
| 4,000 | 0.8 | 1.4 | 7.5% |
| 8,000 | 0.8 | 2.5 | 7.5% |
| 16,000 | 1.3 | 3.5 | 12.2% |
| 32,000 | 2.0 | 6.3 | 18.8% |
| 64,000 | 2.0 | 12.9 | 18.8% |
| bells (2,880) | 0.8 | 1.6 | 7.5% |

The median is flat from 32,000 to 64,000 modes, presumably because the highest modes decay below the
skip threshold (1e-10) within a few blocks. That was not checked directly.
