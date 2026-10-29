# paper-drum: results

## Answer

The question is whether a webcam and a microphone together can tell when and where a finger taps a drawing on paper,
well enough to play it as a drum. Not yet shown on real hardware. In an automated browser with one still photo and
synthetic taps, the parts work together: the microphone finds the moment of the tap and the camera's hand tracker
finds the place.

With a generated photo of a child's finger inside a drawn circle and seven synthetic taps,
5 of 5 taps heard while the hand was in view struck the circle, at the fingertip's cell. <!-- @evidence[[results/e2e.json#final.hits]] -->
That run accepted a hand seen up to 1.5 s from a tap, because the emulated test browser ran the hand tracker at about
3 frames a second; at the page's own 250 ms window it matched no taps.

The drum's own sound, coming back through the mic, was the main risk. With the drum's echo up to about −2 dBFS,
louder than any tap, the detector found 48 of 51 taps with its echo canceller and 22 of 53 without it. <!-- @evidence[[results/taps.json#runs.4.found]] -->

What would change this: real taps on a real table through a laptop mic (the synthetic taps are noise bursts), a real
speaker's distortion, and the hand tracker's speed on a real device. The container ran it at about 3 frames a second.

## Findings

1. **Mic for when, camera for where works end to end in a headless browser with a still image.** With the hand in view,
   5 of 5 heard taps struck the circle under the fingertip. <!-- @evidence[[results/e2e.json#final.hits]] --> MediaPipe put the index fingertip inside the drawn circle,
   and every strike landed on cell (46, 45) of the 128×96 grid, the fingertip's position. The finger never moves in
   the photo, and the hand window was widened to 1.5 s for the slow emulated browser. (e2e)
2. **The page must cancel its own speaker.** Closed loop, with the drum's echo up to −2 dBFS,
   the detector found 48 of 51 taps with the adaptive filter. <!-- @evidence[[results/taps.json#runs.4.found]] -->
   Without it, it found 22 of 53. <!-- @evidence[[results/taps.json#runs.6.found]] --> A plain energy detector
   cannot hear a −30 dBFS tap under a −15 dBFS ring; subtracting the known playback can. (taps, round 2)
3. **A refractory window and an attack gate keep the drum from triggering itself.** Each detected tap plays a strike
   that returns through the mic, and the typical scenario still had only 1 false onset in 30 s. <!-- @evidence[[results/taps.json#runs.0.falseOnsets]] --> (taps, round 2)
4. **Taps near the room's noise are missed.** With noise at −45 dBFS and taps down to −36 dBFS,
   the detector found 27 of 51. <!-- @evidence[[results/taps.json#runs.2.found]] --> Those taps sit within about 10 dB of the noise. The page's Tap sensitivity slider lowers the threshold; its effect here is untested. (taps)
5. **The start-up noise burst matters.** Without it the canceller learns only from the drum itself, and the loud
   scenario found 48 of 57. <!-- @evidence[[results/taps.json#runs.7.found]] --> (taps)
6. **The echo canceller costs about a seventh of a server core while the drum plays.** On this container's Xeon, the
   120 ms filter took 142.8 ms per second of audio, against 6.4 ms with it off. <!-- @evidence[[results/taps-cpu.json#aec120.msPerAudioSecond]] --> It skips the filter when nothing
   has played for 120 ms. Phone cost unmeasured. (taps-cpu)
7. **MediaPipe tasks-vision 1.1.0 fails in a module Worker with "ModuleFactory not set."** Evaluating its wasm loader
   script globally first fixes it (`web/hands-worker.js`). (e2e)

## Method

**Tap detector (`test/taps.mjs`).** Each scenario is 30 s at 48 kHz: Gaussian room noise, and taps (12 ms noise
bursts with a 120-200 Hz thump, 2.5 ms decay) at random gaps and peak levels from −36 to −14 dBFS. Closed loop:
each detected onset starts a six-mode drum strike 35 ms later, which the detector sees as the page's output. The
output returns into the mic through a tanh soft clipper (drive 1.5, standing in for a small speaker's distortion), a
direct path and two reflections at 1.3 and 3.6 ms, after 28 ms. A tap counts as found if an onset falls within 15 ms
after it starts; any other onset is false. The page's start-up noise burst plays first. The gate is recall ≥ 0.9 and
at most 2 false onsets on four scenarios: typical, fast taps, a misjudged echo delay, and a loud speaker. Noisy room,
long ring and the two ablations are reported, not gated.

**Detector.** Mic and playback are averaged down to 16 kHz. An NLMS filter of 120 ms (1,920 taps, step 0.4) cancels
the playback; it adapts only while something has played and holds for 30 ms after each onset. The residual is
high-passed at 100 Hz. An onset is the 2 ms energy envelope crossing 8 times (9 dB) a 250 ms baseline and a −62 dB
floor, with 150 ms refractory. For 40 ms after each attack in the delayed playback envelope it must also beat twice the
learned speaker coupling.

**Page (`test/e2e.mjs`).** Headless Chromium with SwiftShader WebGPU (WebGPU emulated on the CPU). The fake camera is `fixtures/hand.jpg` (made
with Gemini through Muninn's image tool) at 640×480, 30 fps. The first 1.5 s are a copy with the hand painted out and
the circle's hidden arc redrawn. The fake mic is 10 s with seven taps at −20 dBFS over −60 dBFS noise. Both loop.
Grid 128×96, hand tracker on the CPU delegate. SwiftShader ran the hand tracker at about 3 frames a second, so the
page was told to accept a hand seen up to 1.5 s from a tap (`handwin=1500`); in use the window is 250 ms. Dropouts on
SwiftShader say nothing about real GPUs.

## Log

### Round 1, 2026-10-29: build and tap detector without echo cancelling

Asked (Oskar, by voice): a kids' game where you draw an instrument on paper and play it, the camera and the real
world being the point; start with the drum. Built: room-drum's engine and walls (copied at 57b5a070), an 'ink' wall
source, a hand hold box, gap closing, MediaPipe hands in a Worker, and a combined play-and-listen worklet.

Prior art (three searches). Google's AR Music Kit (Yuri Suzuki) reads printed paper markers that a hand covers, not
taps. Samsung patents US 8866846 / US 2012/0007884 recognise a drawn instrument and play it. US 9971457 fuses a mic
with a depth camera for touch on a projected surface. A paper-piano paper (Araku University) reports 77% key accuracy
from vision. Did not find, in three searches, a plain webcam and mic fused to play a physically simulated drum whose
shape is the drawing.

First detector: energy onset, refractory window and an output-envelope gate, no cancelling. At an unrealistically loud
echo (2× the playback) it found 15 of 54 taps with 9 false onsets. With realistic levels and a gate limited to 40 ms
after attacks, false onsets fell to 1, but recall stayed at 35 of 54. The misses were taps under the drum's ring.

### Round 2, 2026-10-29: NLMS echo cancelling, start-up noise, headless page

Added the NLMS canceller, a speaker model with distortion and reflections in the test, and the start-up noise burst.

| scenario | found | false |
|---|---|---|
| typical (echo to −16 dBFS) | 52/53 | 1 |
| fast taps, 0.16-0.3 s | 118/125 | 1 |
| noisy room, −45 dBFS | 27/51 | 4 |
| echo delay misjudged (60 vs 28 ms) | 52/53 | 2 |
| loud speaker (echo to −2 dBFS) | 48/51 | 1 |
| loud speaker, 2.5 s ring | 42/52 | 3 |
| loud speaker, no cancelling | 22/53 | 0 |
| loud speaker, no start-up noise | 48/57 | 0 |

Page: the first run failed with "ModuleFactory not set." from MediaPipe in the module Worker, fixed by a global eval
of the loader. The second run matched no taps: hand frames arrived 420 ms apart, outside the 250 ms window. With
`handwin=1500` the third run passed. It heard 5 taps and struck all 5 at cell (46, 45), and the walls found two drums
of 1,401 and 609 cells.

Pending: a run on a Mac with a real camera, mic and paper, reported with the page's Copy timings button.
