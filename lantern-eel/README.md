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

## Tilt

The angle the phone is held at when tilt is switched on, a dive starts, play resumes or the screen
rotates counts as level. A tip of 9° past it along one axis turns the eel, provided that axis leads
the other by 15%. Beta and gamma are mapped through `screen.orientation.angle`, the same mapping
Tilt Chomp uses, so landscape works.

## Tests

`node test_lantern_eel.js` with the site repo cloned beside this one. It runs phone, short-landscape
and desktop viewports in headless Chromium and checks: overlays honour `[hidden]` outside the
Artifact skeleton, no horizontal scroll, turn queueing refuses reversals, synthetic
`deviceorientation` events respect the dead zone and steer all four ways, plankton scores and
grows, a pearl's phase passes through a jellyfish, a jellyfish without it ends the dive, and no
console errors.

## Notes

- Headless Chromium returns `denied` from `DeviceOrientationEvent.requestPermission()` unless the
  context grants `accelerometer`, `gyroscope` and `magnetometer`.
- The first build was authored for the Artifact skeleton, whose reset supplies
  `[hidden]{display:none!important}`. Served standalone, overlays with `display: grid` ignored
  `hidden` and covered the start button; the page now carries the rule itself.
