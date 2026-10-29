# pulse-video: results

A browser page amplifies the colour change of your pulse in live camera video on the GPU and reads your heart
rate off the face. The amplification is Eulerian video magnification (Wu et al. 2012): blur each frame to a
coarse grid, keep the colour changes at heart-rate frequencies (the band from 42 to 180 beats per minute, bpm),
multiply them and add them back. The questions are whether the GPU pipeline computes what it should, whether it
is fast enough to run on every camera frame, and how close the heart rate comes. That last question has two
parts. One is synthetic video with a known pulse. The other is a real face against a watch, and that comparison
has not been run yet.

The page computes the heart rate three ways, from the face's average colour over the last 10 s:

- **POS** (plane orthogonal to skin, Wang et al. 2017) combines red, green and blue so that any change scaling all
  three alike cancels. A change in brightness or camera gain scales all three alike; blood absorbs mostly green.
  The page displays this estimate.
- **Magnified green** is the face's green after the same band-pass the video shows.
- **Raw green** is the face's unfiltered green.

Each is the peak of the signal's spectrum. SNR, in decibels, is how far that peak stands above the rest of the
spectrum. Error is the mean absolute error (MAE) in bpm across the windows of a run. SwiftShader is Chromium's
software GPU, used for the headless checks.

## Answer

The GPU pipeline matches a float64 implementation to within 1.82 × 10⁻⁷. <!-- @evidence[[results/summary.json#filter.maxFiltErrE7]] -->
On a Mac's GPU it takes a median of 1.9 ms per 640×480 frame, <!-- @evidence[[results/summary.json#mac.640x480.medianMs]] -->
about a seventeenth of the 33 ms between camera frames.

On synthetic video with a known pulse, sensor noise and simulated auto-exposure (the camera re-adjusting
its brightness every few seconds), POS has an error of at most 1.45 bpm in every pulse case. <!-- @evidence[[results/summary.json#drift.posMAEmax]] -->
Magnified green misses by up to 18.42 bpm, <!-- @evidence[[results/summary.json#drift.evmMAEmax]] -->
because the exposure steps leak through the band-pass and pull its peak to the band's lower edge, 42 bpm.
With steady exposure, all three estimates are within 0.82 bpm. <!-- @evidence[[results/summary.json#steady.posMAE]] -->
A background region with no skin measures the drift: under drift, the face's band-passed green correlates with
the background's brightness at 0.85 to 0.95, <!-- @evidence[[results/summary.json#drift.expCorrMax]] --> and with steady exposure at 0.11.
The page shows no number when POS's SNR is below 0 dB; with no pulse, no window reached −2 dB.

Real-face accuracy is unknown until someone checks the page against a watch. Real pulses are weaker and less
regular than these synthetic ones, so expect a larger error.

## Findings

1. **The GPU matches float64.** The pyramid, band-pass and region means agree with `Reference` in `evm.js`
   to within 1.82 × 10⁻⁷ on SwiftShader, <!-- @evidence[[results/summary.json#filter.maxFiltErrE7]] -->
   at three sizes and pyramid depths with uneven frame spacing. The magnified frame is off by at most 1 level. <!-- @evidence[[results/summary.json#filter.maxCompLevels]] -->
   On the Mac's Metal GPU the filtered level is within 1.84 × 10⁻⁷. <!-- @evidence[[results/summary.json#mac.check.maxErrE7]] -->
   (`test/verify.mjs filter`, `bench/job.js`)
2. **A whole frame takes about 2 ms on a Mac.** That covers the canvas upload, four pyramid levels, the band-pass,
   the region means, the composite draw and a 48-byte readback. At 640×480 the median is 1.9 ms
   and the 90th percentile 2.8 ms; <!-- @evidence[[results/summary.json#mac.640x480.p90Ms]] -->
   at 1280×720 the median is 2.1 ms. <!-- @evidence[[results/summary.json#mac.1280x720.medianMs]] -->
   Frame size barely matters, so the time is the fixed cost of one GPU submission and readback, not the work. Generating the synthetic frame on the
   CPU takes 36 ms. (GPU Bridge job `pulse-1`, Mac, Chrome 156)
3. **Auto-exposure breaks the green-channel estimate, and POS removes it.** With drift, raw green missed by up
   to 12.72 bpm <!-- @evidence[[results/summary.json#drift.greenMAEmax]] -->
   and magnified green by up to 18.42 bpm, <!-- @evidence[[results/summary.json#drift.evmMAEmax]] -->
   while POS stayed within 1.45. In the last window of the 96 and 128 bpm cases the magnified green's peak sat at
   41.9 bpm, <!-- @evidence[[results/summary.json#finalEvmBpm.128]] --> the lower edge of the band, where the exposure steps leak through the band-pass's gentle low side.
   Raw green, which is not filtered, still found 96 and 128, at −5 dB. Without drift, magnified green was within 0.8 bpm. <!-- @evidence[[results/summary.json#steady.evmMAE]] -->
   A gain change scales red, green and blue alike, and POS projects that direction out. The magnified video
   therefore shows exposure ripple as well as pulse, and the page reports POS. (`test/verify.mjs rate`)
4. **The background region measures the drift.** The correlation between the face's in-band green and the
   background's brightness was 0.85 to 0.95 under drift <!-- @evidence[[results/summary.json#drift.expCorrMin]] -->
   and 0.11 with steady exposure. <!-- @evidence[[results/summary.json#steady.meanExpCorr]] -->
   The page shows this number live, and on a real camera it is the direct measure of how much auto-exposure is in
   the signal.
5. **SNR separates a pulse from no pulse on synthetic video.** Windows with a pulse averaged at least +8.33 dB. <!-- @evidence[[results/summary.json#snr.pulseMeanMin]] -->
   Across 42 windows of three no-pulse cases, none reached −2.06 dB, <!-- @evidence[[results/summary.json#snr.noPulseMax]] -->
   and the no-pulse peak wandered from 48 to 180 bpm. The page shows a number only at 0 dB or above. Chromium's
   fake camera, a moving test pattern with no face, passed the gate at +5 to +7 dB, so something other than a
   pulse can pass it; I did not establish which part of the pattern did.
6. **A flickering light fools green and not POS.** A whole-frame flicker at 1.3 Hz (78 bpm) and 0.3% depth, with
   no pulse present, put the magnified-green peak at 78 bpm. <!-- @evidence[[results/summary.json#flicker.noPulseEvmBpmMedian]] -->
   POS stayed below 0 dB SNR. With a 72 bpm pulse added, POS read 72 and magnified green read 80.1. <!-- @evidence[[results/summary.json#finalEvmBpm.72-flicker]] -->
   A real light can also shift colour, which this flicker does not, and POS cancels only the colourless part.
   The page's exposure flag fires when the background shows the same peak as POS; POS removed the flicker here,
   so the flag never fired in these cases.

## Method

**Pipeline** (`web/evm.js`). Each frame goes into a canvas and then a texture. A 5-tap binomial pyramid reduces
640×480 to 40×30 over four levels. Each coarse pixel's RGB goes through a band-pass: a two-pole exponential
smoother at 3 Hz minus one at 0.7 Hz, with gains `1 − exp(−2π f dt)` taken from each frame's own `dt`. A render
pass draws the frame plus 60× the bilinearly upsampled band. Region means come from one 256-thread workgroup
and are read back.

**Rate** (`web/rate.js`). The last 10 s of region means, resampled to 30 Hz. Three signals: POS with 1.6 s
windows; the face's mean band-passed green from the GPU; the face's raw green normalised by its mean. Each is
detrended and Hann-windowed, the DFT runs at 0.005 Hz steps, the peak is searched in 0.7 to 3 Hz, and SNR follows
de Haan & Jeanne 2013 (±0.1 Hz around the peak and ±0.2 Hz around its harmonic, against the rest of 0.5 to 4 Hz).
The displayed number is the median of the last five windows with SNR ≥ 0 dB.

**Exposure measures.** The background region's brightness range over the window. The background's peak. The
correlation between the face's band-passed green and the background's band-passed brightness, both normalised by
their means.

**Synthetic video** (`web/synth.js`). A 640×480 skin ellipse on a textured wall. The pulse runs along de Haan's
blood vector (0.33, 0.77, 0.53) at 0.4% amplitude (0.2% in the faint case), and the rate wanders ±3% over 10 s.
Sensor noise has a standard deviation of 2 levels (6 in the noisy case), the head sways 1.5 px at 0.23 Hz, and
values are rounded to 8 bits. The auto-exposure model jumps its target gain by up to ±4% every 3.7 s, approaches
it with a 0.4 s time constant, and adds a 1.5% swell. Frames are 30 fps with ±2 ms timestamp jitter. The face box
is given, not detected. Each case runs 20 s through the whole page on SwiftShader. Error is the mean absolute
error over the windows ending at 10 s or later.

**Cases** (`results/rate.json`). 52, 72, 96 and 128 bpm with drift; 72 faint; 72 noisy; 72 steady (no drift);
72 with flicker; and no pulse, no pulse with steady exposure, and no pulse with flicker.

**Real-device timing** (`bench/job.js` through the GPU Bridge). 150 frames per size, with the first 10 dropped.
The time runs from `frame()` to the readback resolving, the same latency path the page uses.

## Log

### Round 1 — 2026-10-29: build, SwiftShader checks, Mac timing

Asked: build the page, check the filter against a JS reference and the rate estimator on synthetic video,
and measure auto-exposure drift rather than assume it away.

The filter check passed on the first run: max error 1.37e-7 to 1.82e-7 on the coarse level, and the composite
within 1 level on 0.011 to 0.013% of bytes.

| case | POS MAE | magnified green MAE | raw green MAE | mean POS SNR (dB) | face/background correlation |
|---|---|---|---|---|---|
| 52 | 0.63 | 2.05 | 2.90 | 10.2 | 0.85 |
| 72 | 0.79 | 4.05 | 4.06 | 9.9 | 0.85 |
| 96 | 1.06 | 6.39 | 1.02 | 9.5 | 0.87 |
| 128 | 1.45 | 18.42 | 1.35 | 8.3 | 0.90 |
| 72 faint (0.2%) | 0.78 | 9.90 | 12.72 | 9.7 | 0.95 |
| 72 noisy (σ 6) | 0.75 | 4.04 | 6.93 | 9.4 | 0.85 |
| 72 steady | 0.82 | 0.80 | 0.78 | 9.9 | 0.11 |
| 72 + flicker | 0.79 | 5.02 | 5.02 | 9.9 | 0.90 |
| no pulse | — | — | — | −4.3 | 1.00 |
| no pulse, steady | — | — | — | −5.9 | −0.16 |
| no pulse + flicker | — | — | — | −4.1 | 1.00 |

Prediction against measurement: I expected green to suffer under drift and POS not to, and both held. I did
not expect POS's error to grow with rate (0.63 at 52, 1.45 at 128); it is about 1% of the rate in every case.
The synthetic rate wanders ±3%, which spreads the spectral peak in proportion to the rate. That is a likely
cause, not a tested one.

The first no-pulse run showed 72 bpm, which looked like a POS window artifact: the 1.6 s window's second harmonic
sits at 75 bpm. A noise-only model showed no such artifact. The per-window log showed the peak wandering from 48
to 180 bpm, and 72 was where the median of five happened to land. The page originally displayed the median of all
windows with an "unsure" label below −2 dB. It now shows no number below 0 dB, and low-SNR windows no longer enter
the median.

Mac timing (Chrome 156, Metal): 1.6 ms at 320×240, 1.9 ms at 640×480, 1.7 ms at 480×640, 2.1 ms at 1280×720.
The job took 32.5 s.

With Chromium's fake camera, the page ran at 8 fps on SwiftShader with no errors, and MediaPipe's face detector
loaded from jsDelivr. The fake camera has no face, so the fixed box was used.

Real-face accuracy: pending Oskar's comparison against a watch, using the page's Record and Copy JSON buttons.
