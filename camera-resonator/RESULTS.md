# camera-resonator: results

The Camera Resonator page turns movement in front of a camera into strikes on a virtual steel plate or a row of
bells. Every mode is a damped sinusoid summed on the GPU, one audio block at a time. See `README.md` for how it
works.

## Answer

The question was whether a GPU can run a physically modelled instrument with thousands of modes in real time,
played by the camera, inside a browser. On a Mac (Apple GPU, Chrome) the engine has ample headroom. A 32,000-mode
plate renders a 10.7 ms block in 2 ms median <!-- @evidence[[results/resonator-bench-1-mac.json#plate.32000.medianMs]] -->,
and in 6.3 ms at the 90th percentile <!-- @evidence[[results/resonator-bench-1-mac.json#plate.32000.p90Ms]] -->,
the percentile that the blocks carrying a strike fall into. The output matches a float64 reference with an RMS
error of −141.5 dB relative to peak <!-- @evidence[[results/resonator-bench-1-mac.json#check.rmsDb]] -->. The bench times
the engine alone. The full loop of camera, motion pass, strike, audio queue and speaker is still unmeasured on
real hardware, and so is the iPhone. A phone run of the page, or of the bench, would change this answer.

## Findings

1. Blocks that carry a strike set the real-time limit, not the quiet ones. Strike blocks cannot skip silent
   modes, so at 64,000 modes the median block is 2 ms <!-- @evidence[[results/resonator-bench-1-mac.json#plate.64000.medianMs]] -->
   but the 90th percentile is 12.9 ms <!-- @evidence[[results/resonator-bench-1-mac.json#plate.64000.p90Ms]] -->,
   over the 10.7 ms block. On the Mac the usable ceiling is therefore about 32,000 modes, not 64,000.
2. Below 8,000 modes, the GPU round trip, not the modes, sets the block time. 1,000 and 8,000 modes both take
   0.8 ms median <!-- @evidence[[results/resonator-bench-1-mac.json#plate.8000.medianMs]] -->: submit plus
   readback.
3. The streaming engine keeps the bells' precision. Keeping mode state on the GPU across blocks, with float64
   tables and float64 strike excitations, matched float64 at −141.5 dB RMS on the Mac
   <!-- @evidence[[results/resonator-bench-1-mac.json#check.rmsDb]] --> and −141 dB on SwiftShader
   (`test/verify.mjs selftest`).
4. Motion-triggered strikes work end to end headless. With Chromium's fake camera, the page fired 30 strikes in
   8 s and streamed blocks with no WebGPU errors. Peak output was 0.45 once the preset gains were lowered.
   SwiftShader is far too slow to play in real time, so its dropouts say nothing about real devices.

## Method

- Engine: `web/engine.js` at commit bc7f76f, 512-sample blocks at 48 kHz. Plates use a strike point and two
  pickups. The bells are 8 bells with 2,880 modes.
- Accuracy: `selfTest` renders a 2,000-mode plate over 16 blocks with random strikes at random sample offsets,
  compared with `reference()` (float64) sample by sample.
- Timing: `bench/job.js` on the GPU Bridge. The job imports the engine from jsDelivr at the commit. Two strikes
  first, so every mode rings. Then 60 sequential blocks, each timed from submit to readback, the page's latency
  path. Every 8th block carries a strike. Chrome's coarsened timer rounds the times to about 0.1 ms.
- Not measured: the camera-to-sound latency on a device, the iPhone GPU, and iOS audio with the silent switch on.

## Log

### 2026-10-29: build, headless checks, Mac bench

Built the page, engine, worklet and checks. On SwiftShader, `selftest` gave −140.9 dB (512-sample blocks) and
−141.1 dB (1,024) for a 1,200-mode plate, and −139.7 dB for the bells. A `live` run with the fake camera fired
50 strikes in 12 s, but the peak hit 2.5. The plate and bell gains were cut from 0.35 to 0.12 and 0.20, and the
peak then read 0.45. Bridge job `resonator-bench-1` ran on the Mac tab (apple metal-3, Chrome 156) in 3.6 s:

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

The prediction was that the cost would scale with the mode count. It does only on strike blocks. The median is
flat from 32,000 to 64,000 modes because the high modes decay below the skip threshold within a few blocks.
