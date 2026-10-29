# pulse-video

A page that makes the colour change of your pulse visible in live camera video, and estimates your heart rate
from it. Live at https://austegard.com/experiments/pulse-video/web/. A curiosity, not a medical device.

Each heartbeat pushes a little more blood into the skin of the face, which changes its colour by a fraction of a
percent: far too little to see, but enough for a camera to record. Eulerian video magnification (Wu et al.
2012) amplifies it. Blur each frame down to a coarse grid, keep only the colour changes that repeat at heart-rate
frequencies, multiply them, and add them back. The page runs this on the GPU for every camera frame. It also
reads the heart rate off the face's average colour and measures how much of that colour signal is really the
camera's auto-exposure.

## How it works

- **Frames.** Each camera frame is drawn into a canvas of at most 640 px on its long side and copied into a
  WebGPU texture. The canvas route keeps iOS frames upright (METHODS.md, xray-glasses).
- **Pyramid.** `web/evm.js` blurs and halves the frame four times with a 5-tap binomial kernel, so 640×480
  becomes a 40×30 grid. Each coarse cell averages 256 pixels, which pulls the sensor noise below the pulse.
- **Band-pass.** Each coarse cell's colour goes through a temporal band-pass: a two-pole smoother at 3 Hz minus
  one at 0.7 Hz (42 to 180 bpm). Each pole is an exponential smoother whose gain comes from that frame's own
  time step, so a dropped or late camera frame does not shift the band.
- **Magnify.** A render pass upsamples the filtered grid bilinearly, multiplies it by the gain (60 by default)
  and adds it to the frame. "Changes only" shows the amplified signal on grey.
- **Face.** MediaPipe's short-range face detector runs on the CPU every 200 ms. The skin rectangle is the face box
  without its outer sides and chin. If the detector cannot load, the page uses a fixed box for you to sit in.
- **Heart rate.** A compute pass averages three regions on the GPU: the face's colour, the face's filtered
  colour, and a background column with no skin in it. Those 48 bytes come back each frame. `web/rate.js`
  resamples the last 10 s to 30 Hz and finds the spectral peak between 42 and 180 bpm for three signals:
  - POS (Wang et al. 2017), which projects out any change that scales red, green and blue alike. Auto-exposure
    is such a change. This is the number shown.
  - The face's magnified green, the signal the video shows.
  - The face's raw green.
- **Exposure.** Web cameras give a page no exposure control, so the camera keeps re-adjusting brightness. The
  background column has no pulse, so its brightness measures that drift directly. The page shows how far the
  background brightness moved in the window, and how strongly the face's in-band green moves with it. When the
  background shows the same rhythm as the POS estimate at comparable strength, it marks the reading as light
  or exposure, not pulse. It shows no number below 0 dB SNR, while the face's brightness moves by more than 8%
  (you are moving), or when the peak sits on the band's lower edge. The number is the median of the last 5 s.
- **Comparing with a watch.** "Compare with a watch" opens a field for the rate your watch shows; Save it a few
  times, then Copy JSON copies the page's per-second readings together with yours.

## Checks

- `node test/verify.mjs filter` compares the GPU pyramid, band-pass, region means and composite against
  `Reference` in `evm.js`, a float64 implementation, on SwiftShader.
- `node test/verify.mjs rate` runs the whole page on synthetic video from `web/synth.js`: a face with a pulse of
  known rate, sensor noise, head sway and an auto-exposure model. The cases cover several rates, a faint pulse,
  heavy noise, no pulse at all, and a light that flickers at a heart-like rate.
- `node test/verify.mjs live` runs the page with Chromium's fake camera.
- `bench/job.js` is a GPU Bridge job that times whole frames on real hardware.

Results are in `RESULTS.md`.
