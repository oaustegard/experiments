# room-drum: results

The page turns the camera image into a drum. Edges become the walls of a simulated drum skin, and the skin is
stepped once per audio sample on the GPU. The question was whether a browser GPU can do that in real time on a
grid fine enough to trace the image, with walls that change every frame.

## Answer

On the Mac, yes. Its Apple GPU, running Chrome 156, steps a 256×192 skin (49,152 cells) once
per sample, 512 steps per 10.7 ms block, in 2.2 ms. That is 21% of the block's playing time.
<!-- @evidence[[results/summary.json#mac.multi_256x192_wg0_b512.load]] -->
A 640×480 skin, one cell per pixel of a VGA frame, takes 5.1 ms (48%).
<!-- @evidence[[results/summary.json#mac.multi_640x480_wg0_b512.medianMs]] -->
Up to about 50k cells most of that is a fixed 1.8 ms for 512 dispatches.

The CPU limits the grid before the GPU does. It finds the walls once per 33 ms camera frame, on the main thread, which also feeds the audio: 9 ms at 256×192
<!-- @evidence[[results/summary.json#macWalls.256x192.wallsMedianMs]] -->
and 47.5 ms at 640×480, longer than a frame,
<!-- @evidence[[results/summary.json#macWalls.640x480.wallsMedianMs]] -->
so the page's default grid, the largest under 30% GPU load on the visitor's device, stops at 256×192.

The numerics hold. The GPU output matches a float64 run of the same scheme to −114 dB on the Mac,
<!-- @evidence[[results/summary.json#accuracy.macDb]] -->
and a rectangular drum rings within 0.09 cents of the scheme's predicted modes.
<!-- @evidence[[results/summary.json#modes.rectMaxCentsVsDiscrete]] -->

No phone has been measured. A slow phone, on the GPU or in the wall pass, would lower its default grid; the page times itself and has a Copy timings button.

## Findings

1. **Real time on the Mac with one simulation step per sample.** At 256×192 (Round 1, `results/summary.json`),
   the `multi` kernel takes 2.2 ms per 10.7 ms block. <!-- @evidence[[results/summary.json#mac.multi_256x192_wg0_b512.medianMs]] -->
   At 640×480 it takes 5.1 ms, and the p90 is 5.4 ms (Round 1). <!-- @evidence[[results/summary.json#mac.multi_640x480_wg0_b512.p90Ms]] -->
2. **Up to about 50k cells, the number of dispatches sets the cost.** From 6,912 to 49,152 cells (Round 1),
   `multi` takes 1.8 to 2.2 ms per block, roughly 3.5 µs per dispatch. <!-- @evidence[[results/summary.json#mac.multi_96x72_wg0_b512.medianMs]] -->
3. **The single-workgroup kernel wins only on small grids.** `tile` runs all 512 steps in one dispatch, so it pays
   no per-dispatch price. It takes 1.1 ms at 64×48, against 1.9 ms for `multi`.
   <!-- @evidence[[results/summary.json#mac.tile_64x48_wg1024_b512.medianMs]] -->
   But one workgroup uses only one GPU core, so at 96×72 it is slower, 2.7 ms against 1.8 ms (Round 1).
   <!-- @evidence[[results/summary.json#mac.tile_96x72_wg1024_b512.medianMs]] -->
   Its workgroup size matters: 2.5 ms at 256 threads against 1.1 ms at 1,024 threads for 64×48.
   <!-- @evidence[[results/summary.json#mac.tile_64x48_wg256_b512.medianMs]] -->
4. **The CPU wall pass sets the grid limit.** Sobel, hysteresis and region labelling take 9 ms per frame at
   256×192 and 18.4 ms at 384×288 on the Mac, in the bridge's Worker with a synthetic frame (`bench/walls-job.js`, Round 1).
   <!-- @evidence[[results/summary.json#macWalls.384x288.wallsMedianMs]] -->
   A main thread that stalls for longer than the queued audio, about 32 ms, causes dropouts.
5. **GPU and float64 agree.** Over both kernels, two grid shapes and a wall change mid-stream, the worst run on
   SwiftShader is −107.7 dB, RMS error relative to the reference's peak, (`test/verify.mjs selftest`). <!-- @evidence[[results/summary.json#accuracy.swiftshaderWorstDb]] -->
6. **The modes match theory.** On a 33×23 rectangle, the ten lowest peaks sit within 0.09 cents of the discrete dispersion relation
   (`test/verify.mjs modes`). <!-- @evidence[[results/summary.json#modes.rectMaxCentsVsDiscrete]] -->
   They sit within 5.33 cents of the continuous membrane, a gap that is the scheme's own dispersion.
   <!-- @evidence[[results/summary.json#modes.rectMaxCentsVsContinuous]] -->
7. **A staircase disc keeps the Bessel ratios.** On a disc of radius 15.5 cells, the first six partial ratios are within
   0.26% of the Bessel-zero ratios (`test/verify.mjs modes`). <!-- @evidence[[results/summary.json#modes.discFirst6RatioErrPct]] -->
   Every partial reads 26 to 30 cents flat, because the effective radius is about a quarter cell larger than the
   nominal one. The (2,2) mode lies 2.39% off the ratio, and (4,1) did not show at that strike and pickup.
   <!-- @evidence[[results/summary.json#modes.discAllRatioErrPct]] -->

## Method

**Scheme.** u[n+1] = (2u[n] − (1 − a0)u[n−1] + λ²Lu[n] + b(Lu[n] − Lu[n−1])) / (1 + a0), then a force term,
then multiplication by the wall mask. L is the 5-point Laplacian, and outside the grid counts as wall. λ² runs
from 0.10 to 0.45 (Tension). a0 is set from a decay time (T60, the Ring slider) of 0.3 to 6.3 s, and b ≤ 0.02 adds loss that grows with
frequency. The scheme is stable for λ² + 2b ≤ ½. Each cell keeps its previous Laplacian, so a step needs only
its neighbours' current values. The output is the skin's velocity at two pickup cells.

**Kernels.** `tile`: one workgroup of up to 1,024 threads holds u in workgroup memory, at most 8,192 cells on the
Mac. Each thread keeps its cells' history in registers, and one dispatch runs the whole block. `multi`: one 8×8
workgroup dispatch per step, with ping-pong buffers and one small bind group per step for the step index.

**Timing.** The GPU Bridge (a page on Oskar's laptop that runs jobs pushed from a session) runs jobs in a Worker in a Mac tab ("apple metal-3", Chrome 156, macOS). Each
measurement is 40 sequential blocks of 512 samples at 48 kHz, timed from submit to readback, which is the
page's audio path. The tables report the median and p90 (the 90th-percentile block); load is GPU time per block divided by the block's playing time. The engine source travels as a job input and is
imported from a Blob URL, so the timed code is the committed `web/engine.js`. The Worker's `performance.now()`
has 0.1 ms resolution.

**Correctness.** SwiftShader (Chromium's software GPU) in headless Chromium checks numerics only; its timings are 100 to 1,000 times off.
`selftest` compares GPU blocks against `reference()` in float64 with random walls, a wall change and random
strikes. `modes` uses a lossless skin (a0 = 10⁻⁶, b = 0, λ² = 0.45) and 16,384 samples (2.9 Hz bins). It finds
peaks with a Hann window and parabolic interpolation and matches each predicted mode to its nearest peak. A cent is 1/100 of a semitone.

**Not measured.** Any phone. The CPU wall pass on a live camera frame (the bridge has no camera, so a synthetic
640×480 frame stands in). Audio dropouts on real hardware. Perceived latency.

## Log

### Round 1 — 2026-10-29 build and Mac timing

Asked: build "the room as a drum". Camera edges become walls in a 2D wave equation stepped once per audio
sample on the GPU, read out at pickup points into an AudioWorklet, live on Oskar's phone or laptop. Time the
step on the bridge before choosing the grid.

Prior art, in three web searches: GPU finite-difference drum membranes in real time exist (Sosnick and Hsu,
NIME 2011; Renney et al., OpenCL drum membrane, 2019; a NIME 2017 fragment-shader solver). A drum whose shape the
player can change during play also exists (Tycowicz and Loviscach, "A malleable drum", SIGGRAPH 2008 poster).
Boundary Synthesizer II (Matsui and Ogusu, 2015) turns a video outline into a waveform directly, with no
simulation. I did not find the composition, camera edges as live boundaries of a membrane in the browser, in
those searches. `xr` across the account found `fun-and-games/python-gpu/ripples` (a visual wave sim) and nothing
audio. `repo-index/ask.py` refused to run because its index is stale ("Lloyd-Max codebook does not match").

Two kernels (`tile`, one workgroup looping over the block; `multi`, one dispatch per step) were written and checked on SwiftShader first. `selftest` gave −114.3 dB on the 40×30 grid and −107.7 dB on 27×19, identical for both kernels.
`modes` passed: the rectangle within 0.09 cents, the disc as in Finding 7.

Bridge jobs, all on the Mac tab:

| job | what | result |
|---|---|---|
| `room-drum-1` | tile at 4 workgroup sizes, multi up to 256×192, block 512 | multi flat at 1.8–2.2 ms; tile 1.0–3.3 ms, slower above ~5k cells |
| `room-drum-2` | multi to 640×480; blocks 256 and 1024 | 640×480 5.1 ms (48%); at 256×192 a 256-sample block takes 1.3 ms of 5.3 (24%), a 1024-sample block 4.2 ms of 21.3 (20%) |
| `room-drum-3-walls` | `walls.js` per grid, synthetic frame | 2.1 ms at 48×36 up to 47.5 ms at 640×480 |

Mac, block 512 (10.67 ms), median and p90 in ms:

| grid | cells | tile (wg 1024) | multi |
|---|---|---|---|
| 48×36 | 1,728 | 1.0 / 1.2 | — |
| 64×48 | 3,072 | 1.1 / 1.4 | 1.9 / 2.2 |
| 80×60 | 4,800 | 1.9 / 2.2 | — |
| 96×72 | 6,912 | 2.7 / 3.1 | 1.8 / 2.0 |
| 128×96 | 12,288 | over limit | 1.9 / 2.0 |
| 256×192 | 49,152 | over limit | 2.2 / 2.5 |
| 384×288 | 110,592 | over limit | 3.0 / 3.4 |
| 640×480 | 307,200 | over limit | 5.1 / 5.4 |

Predicted before the first job: the single-workgroup kernel would win everywhere it fits, because 512 dispatches
per block sounded expensive. It won only up to about 5k cells. The dispatch price on the Mac (about 3.5 µs) is
lower than I assumed, and one workgroup leaves all but one GPU core idle.

Two page bugs were found by the live check and fixed before the push. Two blocks in flight shared one readback
buffer ("Buffer already has an outstanding map pending"), and the error path re-pumped in a tight loop that hung
the tab. The first output level peaked at 8.7 and was cut to about 0.3.
