# Lantern Eel

Snake in the deep sea, steered by tilting your phone. Installable PWA; Canvas 2D and Web Audio, no libraries.

Live: <https://austegard.com/fun-and-games/lantern-eel/>. The page's source lives in
[`oaustegard.github.io/fun-and-games/lantern-eel/`](https://github.com/oaustegard/oaustegard.github.io/tree/main/fun-and-games/lantern-eel);
this directory keeps the writeup and the tests. The page also runs as a claude.ai Artifact, where
tilt is unavailable because the artifact frame blocks motion sensors.

## Rules beyond classic snake

- **Depth.** Every bite takes you 25 m deeper. Speed rises with depth, and past about 150 m the water
  darkens until only a radius around the eel's lure is lit; prey and jellyfish still show as glows.
- **Lanternfish** (+40, grow 3) flee the head every other tick and leave after 11 s.
- **Jellyfish** drift in from 200 m (Twilight Zone) and multiply with depth. Touching one is fatal.
- **Pearls** appear every 20 to 30 s and give 6 s of phasing through your own tail and jellyfish.
- **Chain bites** within 3.2 s raise a multiplier up to ×8.
- **Boost** doubles speed and points while it burns the glow meter, which refills slowly and with each bite.
- Edges wrap, which suits tilt steering.

## Controls and tilt

Phones (a coarse pointer plus `DeviceOrientationEvent`) start in tilt mode, which hides the D-pad,
meters and Boost button and gives the stage the whole screen; glow and phase are drawn on the water
instead. Arrows mode can put the D-pad on either side. Swiping steers and press-and-hold on the water
boosts in both modes. The choices persist in `localStorage`.

The angle the phone is held at when a dive starts, play resumes or the screen rotates counts as
level, and a vial in the HUD resets it. A tip of 6° past level along one axis turns the eel (first
version: 9°). The axis already steering keeps priority until the other leads it by 3°, so a diagonal
hold does not alternate turns. Beta and gamma are mapped through `screen.orientation.angle`, the same
mapping Tilt Chomp uses. If permission is refused or no data arrives within 1.6 s, the game falls
back to arrows for that visit without saving it.

## Sound

The first version made sound only through a bare `AudioContext`, which iOS silences on the
ring/silent switch, so on an iPhone it played nothing. It now does what Plumb and Scoop do: sets
`navigator.audioSession.type = 'playback'` and plays a one-sample buffer on each pointerup,
touchend, click or keydown until the context runs. On top of that sit an ambient bed (looped noise
through a slowly swelling low-pass, and a 55/82 Hz drone, both lowering with depth) and event sounds
for bites, turns, boost, phase, zones, sonar below 1000 m and nearby jellyfish.

## Tests

`node test_lantern_eel.js` with the site repo cloned beside this one. It runs phone, short-landscape
and desktop viewports in headless Chromium and checks: a mobile context defaults to tilt
and hides the deck, Dive creates the audio context and enables tilt, synthetic `deviceorientation`
events respect the 6° dead zone, steer all four ways and keep axis priority on a diagonal, switching
to arrows in the pause menu shows the deck and stops tilt, desktop defaults to arrows and moves the
D-pad to the right on request, holding the water boosts and releasing stops it, overlays honour
`[hidden]` outside the Artifact skeleton, no horizontal scroll, reversals are refused, plankton
scores and grows, a pearl's phase passes through a jellyfish, a jellyfish without it ends the
dive, and no console errors.

## Notes

- Headless Chromium returns `denied` from `DeviceOrientationEvent.requestPermission()` unless the
  context grants `accelerometer`, `gyroscope` and `magnetometer`.
- The first build was authored for the Artifact skeleton, whose reset supplies
  `[hidden]{display:none!important}`. Served standalone, overlays with `display: grid` ignored
  `hidden` and covered the start button; the page now carries the rule itself.
