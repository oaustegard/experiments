# room-drum

A page that turns the camera image into a drum. Edges in the picture become walls in a simulated drum skin, and
tapping or moving inside a shape strikes the skin there, so you hear how a drum with exactly that outline rings.
Hold up a hand and the walls move with it. Live at https://austegard.com/experiments/room-drum/web/ (timed on a
Mac with a synthetic frame and on Chromium's fake camera so far; no phone and no real camera yet).

The question: can a browser GPU advance a 2D wave equation by one time step per audio sample, 48,000 times a second, on a grid
large enough to trace the camera image, while the walls change with every frame? Answer and numbers are in
[`RESULTS.md`](RESULTS.md).

## Trying it

Tap Start, allow the camera (the front camera opens; Flip switches) and wait a second while the page times your
GPU. Walls show as pale cells over the dimmed video, and the skin's motion as amber and blue ripples. Tap inside
an outlined shape: you should hear a drum-like thump whose pitch drops as the shape gets bigger, and see two
small squares, the left and right pickups. Moving in front of the camera strikes too (Motion strikes), Auto beat
repeats the last strike, and Freeze walls holds the current shape. The Walls from menu switches between edges and dark or
bright areas, which suits a drawing on paper. Tension sets the pitch, Ring the decay. The status line shows the
GPU's share of each block's time, how late the sound is (the three queued blocks, about 32 ms, plus the
browser's own output latency), dropouts, and the struck shape's
estimated pitch. Headphones keep the speaker out of the motion detector.

## Simulation, walls and playback

- **The skin.** `web/engine.js` steps Bilbao's lossy membrane scheme (a 5-point Laplacian with a loss term that
  grows with frequency) once per sample. A wall cell has mask 0 and is held at zero. Each 512-sample block (10.7 ms at 48 kHz) is
  one GPU submit, and two kernels run the same arithmetic:
  - `tile` keeps the whole grid in one workgroup's memory and runs all 512 steps in a single dispatch, with two
    barriers per step. Workgroup memory caps it at 8,192 cells on the Mac.
  - `multi` dispatches once per step over 8×8 workgroups in storage buffers, 512 dispatches per block, at any
    grid size.
- **The walls.** `web/walls.js` draws each camera frame into a 2D canvas at twice the grid's resolution. The
  canvas route keeps iOS frames upright (METHODS.md, xray-glasses). From that image it builds one wall bit per
  cell: Sobel edges, or dark or bright areas, against a threshold with hysteresis so a flickering edge does not
  crackle. It then labels the connected regions of skin.
- **Strikes and pickups.** A tap, or motion against a running baseline as in camera-resonator, strikes the
  nearest skin cell with a half-sine force pulse. A harder strike makes a shorter pulse and a brighter sound. Two
  pickups go about four cells either side of the strike, inside the same region, and the skin's velocity there is
  the left and right channel. The status line shows the struck region's area and the pitch of a disc with the
  same area.
- **Playback.** An AudioWorklet (`web/worklet.js`, copied from camera-resonator) plays the blocks in order with a
  peak limiter. The page keeps three blocks queued, about 32 ms.
- **Grid choice.** Before the sound starts, the page times a few blocks of every grid on both kernels on the
  visitor's own GPU. It then picks the largest grid up to 256×192 that uses under 30% of the block's playing time.
  Larger grids stay in the menu.

## Checks

`node test/verify.mjs selftest` runs both kernels on SwiftShader against `reference()`, a float64 version of the
same scheme, with walls changing mid-stream. `node test/verify.mjs modes` strikes a rectangular drum and a disc
drum on the GPU and compares the spectral peaks with the scheme's dispersion relation, the continuous membrane
and the Bessel zeros. `node test/verify.mjs live grid=48x36` opens the page with Chromium's fake camera and checks
that walls appear, blocks stream and no error is raised. `bench/job.js` and `bench/walls-job.js` are GPU Bridge
jobs that time the simulation and the CPU wall pass on real hardware. `python3 bench/summarize.py` collects the
headline numbers into `results/summary.json`.
