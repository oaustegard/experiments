# Lantern Eel

Snake as a deep-sea anglerfish-eel, in one self-contained HTML file (Canvas 2D, Web Audio, no libraries).
Live: https://austegard.com/experiments/lantern-eel/index.html

## Rules beyond classic snake

- **Depth.** Every bite takes you 25 m deeper. Speed rises with depth, and past ~150 m the water darkens
  until only a radius around the eel's lure is lit; prey and jellyfish still show as glows.
- **Lanternfish** (+40, grow 3) flee the head every other tick and vanish after 11 s.
- **Jellyfish** drift in from 200 m (Twilight Zone) and multiply with depth. Touching one is fatal.
- **Pearls** appear every 20–30 s and give 6 s of phasing through your own tail and jellyfish.
- **Chain bites** within 3.2 s raise a multiplier up to ×8.
- **Boost** doubles speed and points while it burns the glow meter, which refills slowly and with each bite.
- Edges wrap.

## Controls

Arrows/WASD, Space or Shift to boost, P to pause; swipe on the board; on-screen D-pad and Boost;
Tilt (DeviceOrientation). Tilt treats the angle the phone is held at when it is enabled, when a dive
starts, or on resume as level, and steers once the device tips 9° past it along one axis. It maps
beta/gamma through `screen.orientation.angle`, so it works in landscape too. iOS asks for motion
permission on the button press. Tilt does not work inside a claude.ai Artifact frame, which blocks motion sensors.
