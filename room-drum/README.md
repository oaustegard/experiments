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
GPU. Walls show as pale cells over the dimmed video, and they settle only where the picture holds still for about a
third of a second: a waving hand does not repaint them, and a hand held still becomes a wall. Every enclosed shape
is a drum. The nine largest get numbered labels with their estimated note; press 1 to 9 to hit them (1 is the
largest and lowest, Shift for a softer stroke), Space to repeat the last hit, or click or tap inside any shape.
The skin's motion shows as amber and blue ripples, and two small squares mark the left and right pickups.

Motion strikes (off by default) turn a sudden movement into one strike, at most four a second; Sensitivity sets
how sudden. Auto beat repeats the last strike, and Freeze walls holds the current shape. The Walls from menu
switches between edges and dark or bright areas, which suits a drawing on paper. Tension sets the pitch, Ring the
decay. The status line shows the GPU's share of each block's time, how late the sound is (two queued 5.3 ms
blocks plus the browser's own output latency), dropouts, and the wall pass's time per frame. Headphones keep the
speaker out of the motion detector.

## Simulation, walls and playback

- **The skin.** `web/engine.js` steps Bilbao's lossy membrane scheme (a 5-point Laplacian with a loss term that
  grows with frequency) once per sample. A wall cell has mask 0 and is held at zero. Each block (256 samples, 5.3 ms at 48 kHz, on the page; 512 in
  the timings) is one GPU submit, and two kernels run the same arithmetic:
  - `tile` keeps the whole grid in one workgroup's memory and runs all 512 steps in a single dispatch, with two
    barriers per step. Workgroup memory caps it at 8,192 cells on the Mac.
  - `multi` dispatches once per step over 8×8 workgroups in storage buffers, 512 dispatches per block, at any
    grid size.
- **The walls.** `web/walls.js` runs in a Worker (`web/walls-worker.js`), so it never holds up the audio. The page
  draws each camera frame into a 2D canvas at twice the grid's resolution and posts it there. The
  canvas route keeps iOS frames upright (METHODS.md, xray-glasses). From that image it builds one wall bit per
  cell: Sobel edges, or dark or bright areas, against a threshold with hysteresis, averaged over about a third of a
  second and held where the image moves. It drops isolated specks, fills skin islands too small to ring, labels
  the connected regions, and marks each large region's deepest cell as its pad.
- **Strikes and pickups.** A key, a tap, or a motion onset against a running baseline strikes the
  nearest skin cell with a half-sine force pulse. A harder strike makes a shorter pulse and a brighter sound. Two
  pickups go about four cells either side of the strike, inside the same region, and the skin's velocity there is
  the left and right channel. Each pad's label shows the pitch of a disc with the same area as its region.
- **Playback.** An AudioWorklet (`web/worklet.js`, copied from camera-resonator) plays the blocks in order with a
  peak limiter. The page keeps two 256-sample blocks queued, about 11 ms.
- **Grid choice.** Before the sound starts, the page times a few blocks of every grid on both kernels on the
  visitor's own GPU. It then picks the largest grid up to 128×96 that uses under 30% of the block's playing time.
  Larger grids stay in the menu; a coarser grid gives fewer, larger drums that are easier to aim at.

## Checks

`node test/verify.mjs selftest` runs both kernels on SwiftShader against `reference()`, a float64 version of the
same scheme, with walls changing mid-stream. `node test/verify.mjs modes` strikes a rectangular drum and a disc
drum on the GPU and compares the spectral peaks with the scheme's dispersion relation, the continuous membrane
and the Bessel zeros. `node test/verify.mjs live grid=48x36` opens the page with Chromium's fake camera and checks
that walls appear, blocks stream and no error is raised. `bench/job.js` and `bench/walls-job.js` are GPU Bridge
jobs that time the simulation and the CPU wall pass on real hardware. `python3 bench/summarize.py` collects the
headline numbers into `results/summary.json`.
