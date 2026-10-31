# Tilt Rally

A WebGPU racing game for a phone held sideways like a steering wheel. Turning the phone steers. Tilting its top edge away from you is the throttle, and tilting it toward you is the brake. You race three laps of a generated valley circuit against three computer cars.

**Play:** <https://austegard.com/experiments/tilt-racer/web/> (landscape phone; arrow keys or WASD on a computer). `?seed=N` picks another valley.

Origin: Oskar, 2026-10-31: "Can we do a GPU based racing game where the user holds the phone sideways and uses the phone as steering wheel? And use tilt as well"

## Status

Built. The checks below pass in headless Chromium on SwiftShader, a software stand-in for a GPU, with the phone's tilt sensor replaced by synthetic events. **Not yet played on a real phone**, so the feel of the controls is untuned.

## Playing it

Hold the phone at whatever angle is comfortable. The pose you hold when the countdown lights go out becomes straight ahead, so there is no calibration step. Double-tap the screen or press **Center wheel** to reset it mid-race. On iPhone, Safari asks for motion access when you tap Start engine; if you refuse, only a keyboard can drive.

- **Steering** follows the rotation of the phone in the plane of its screen, like turning a wheel. Rotations under 2° are ignored, full lock is at about 57°, and the response curve is gentle near the center.
- **Gas and brake** follow how far the screen leans back or forward compared with the starting pose. Leaning the top away 3° to 20° opens the throttle. Leaning it toward you 4° to 18° brakes, and holding the brake at a standstill reverses.
- **Keep horizon level** (a checkbox, on by default) tilts the 3D picture opposite to the phone, so the road's horizon stays level with the real one while you steer.

## How the phone becomes a wheel

The browser reports the phone's orientation as three angles: alpha (the compass heading), beta and gamma. The page converts beta and gamma into the direction of gravity in the phone's own coordinates, `(-cos β sin γ, sin β, cos β cos γ)`, and ignores alpha, so compass drift cannot move the wheel. Steering is the angle that gravity direction has turned within the screen's plane since the start. That angle means the same thing whichever way round the phone is held. The pedal is the change in `asin(up.z)`, the component of gravity that points out of the screen.

Held sideways, the phone sits exactly where the raw beta value jumps by 180° as the screen tilts past vertical. The gravity vector computed from the angles stays smooth through that jump, so the controls do not glitch there (`test/verify.mjs math`).

## What runs on the GPU

- **Terrain carving (compute).** Each of 160,000 terrain vertices finds its nearest track sample (a coarse pass over every 8th of 2,048 samples, then a fine pass). It sinks into a ditch beside the road and blends out to noise-generated hills and a mountain ring. Heights are read back once, for car physics and tree placement.
- **Tyre smoke and dust (compute).** A pool of 4,096 particles is spawned from up to 8 wheel emitters per frame and drawn as instanced billboards straight from the storage buffer.
- **Rendering.** 4× MSAA. The sky has a sun and procedural clouds; the road shader draws the kerbs, lines and start checker; trees, chevron boards and cars are instanced; every shader shares the fog.

Car physics runs on the CPU at 120 Hz. It uses a simple bicycle model: turning rate is capped by tyre grip, and sideways slide dies away faster on tarmac than on gravel or grass. The AI aims at a point ahead in its lane and follows a speed plan: a grip limit from curvature, then a backward braking pass.

## Checks

```bash
node test/verify.mjs math   # orientation -> wheel/pedal on known poses (9 checks)
node test/verify.mjs demo   # autopilot race on SwiftShader: no WebGPU errors, all cars moving and on the road
node test/verify.mjs tilt   # synthetic deviceorientation events: tilt away accelerates, clockwise steers right, tilt back brakes
```

These use synthetic orientation events, not a phone. Last run (seed 7: 2.87 km lap, tightest corner 30 m, 2,400 trees): all pass. Tilt run: 83 km/h after 3 s of 15° tilt; 30° clockwise moved the car 11 m right in 1.2 s; 14° back braked from 89 to 4 km/h in 1.5 s. Screenshots and the race state are in `results/`.

## Issues found while building

- In headless Chromium, `DeviceOrientationEvent.requestPermission` exists and never settles. Awaiting it before attaching the listener meant no events ever arrived. The page now attaches the listener first and asks for permission in parallel.
- `requestFullscreen`, `screen.orientation.lock` and `navigator.wakeLock.request` can also hang. Awaiting them before attaching the listener had the same effect, so the page now fires them without waiting.
- A class that sets `display: flex` overrides the `hidden` attribute. A `[hidden] { display: none !important }` rule fixed the overlays.

## Not done

- Tuning on a real phone (iOS Safari 26+ and Android Chrome).
- A touch-only fallback for devices with no motion sensor and no keyboard.
- Lap records saved between visits.
