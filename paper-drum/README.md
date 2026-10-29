# paper-drum

A kids' game: draw a drum on paper, point the camera at it, and tap it with your finger. The camera sees where the
finger is, the microphone hears the tap, and the GPU simulates a drum skin whose edge follows the outline you drew,
traced onto a grid of about 128×96 cells. A bigger drum should sound lower.

So far it has only been tested in an automated browser with a fake camera (one still photo) and synthetic taps, not
on a real camera, microphone and speaker. The page is at https://austegard.com/experiments/paper-drum/web/.

The question: can a plain webcam and a laptop or phone microphone together tell when and where a finger taps a
drawing on paper, well enough to play it like a drum? A single camera cannot see touch: a finger 1 cm above the paper
looks the same as one on it. The microphone hears the tap within a few milliseconds but cannot say where it was. The
page uses each for what it can do. Answer and numbers are in [`RESULTS.md`](RESULTS.md).

## Trying it

It needs a browser with WebGPU (recent Chrome or Edge on a laptop; Safari 26 or later), an internet connection the
first time (the hand tracker, about 8 MB, loads from Google's servers), and the device's own speaker, not headphones:
the tap detector learns the speaker's echo, and with headphones there is none to learn. Nothing is uploaded; the
camera and microphone are processed on the device.

Draw closed shapes with a thick dark marker on white paper. Open the page, tap Start and allow the camera and the
microphone. You will hear a short whoosh: the page is learning how its own speaker sounds through the mic, so keep the volume
where you will play and stay quiet for that half second. Point the
camera at the paper: each closed shape gets a colour once it is recognised as a drum. Draw each drum
smaller than about a third of what the camera sees; a larger enclosed area is taken to be the paper itself. Tap inside
one with a finger. A white dot marks each fingertip the page sees. A tap it hears but cannot place shows a line at the
bottom of the screen: "I heard a tap, but I can't see a finger" or "Tap inside a drum you drew". The bar at the top
right is the microphone level.

On a laptop, lay the paper in front of the keyboard where the webcam sees it; on a phone, prop it above the paper with
the back camera down. Tapping on a table carries well; tapping on a soft surface may be too quiet to hear. If taps go
unheard, raise Tap sensitivity; if the drawing is not picked up, move Ink threshold. Freeze drawing stops the page
from re-reading the paper. Tapping the screen also strikes the drum there, for trying it without paper.

## How it works

- **Where: the hand.** MediaPipe's Hand Landmarker runs in a Worker (`web/hands-worker.js`) on each camera frame and
  gives the index fingertip of up to two hands. The page keeps the last second of fingertips with their capture times.
- **When: the tap.** One AudioWorklet node (`web/worklet.js`) both plays the drum and listens to the mic, so the tap
  detector (`web/taps.js`) knows exactly what the speaker is playing. It first cancels that from the mic with an
  adaptive filter, then looks for a sudden rise in the remaining energy against a slow baseline. The browser's own
  echo cancellation, noise suppression and gain control are switched off: they are tuned for speech.
- **Matching them.** A tap is matched to the fingertips seen closest to the moment it happened. A fingertip has to be
  inside a drum; with two candidates, the one that moved more in the 150 ms before wins. A tap with no fingertip on a
  drum makes no sound and the page says why.
- **The drawing.** `web/walls.js` (in `web/walls-worker.js`) marks a cell as wall where its darkest pixel is well
  below the local paper brightness, closes one-cell gaps in the outline, and splits the rest into regions. A region is
  a drum if it does not touch the frame's edge and covers under 30% of the frame. The cells under the hand keep the
  walls they had before the hand arrived.
- **The drum.** `web/engine.js` is the membrane from the sibling experiment `room-drum`, unchanged: Bilbao's lossy 2D wave scheme stepped once per
  audio sample on WebGPU, with the drawing's outline as the clamped edge. A louder tap is a harder, brighter strike.

## Checks

- `node test/taps.mjs` runs the tap detector on 30 s synthetic scenarios, closed loop (each detected tap plays a drum
  that comes back into the mic through a distorting speaker and a room), and gates on recall and false onsets.
- `node test/taps-cpu.mjs` times the detector per second of audio.
- `python3 test/make-fixtures.py && node test/e2e.mjs` runs the whole page in headless Chromium with a fake camera
  (a generated photo of a drawn circle and square, first without and then with a child's finger in the circle) and a
  fake microphone (seven taps). MediaPipe and its model come from their CDNs.
