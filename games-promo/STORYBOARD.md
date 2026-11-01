# Games promo: narrative and storyboard

A 1080×1920 vertical promo, about 55 s at 30 fps, for the 17 browser games at
<https://austegard.com/games/>. Built with Remotion from gameplay captured from the
real games in headless Chromium (SwiftShader WebGPU, frozen clock).

## Narrative

The READMEs share three ideas, and the film moves through them in order:

1. **The phone is the controller.** Nine of the games are steered by tilting the
   phone: a pose you choose becomes level, and gravity does the rest.
2. **The GPU does the work.** Doom's engine and a Lemmings game run as WebGPU
   compute shaders; ten thousand lemmings at once.
3. **Nothing to install, no one in the middle.** Mahjong and poker tables connect
   phones directly by text or QR, with no server and no accounts. Every sound is
   synthesized; every game installs from the page and plays offline.

It closes on the count and the address.

## Scenes

| # | scene | shots | read |
|---|---|---|---|
| 1 | Hook | phone tilting, Ball Maze | Your phone is the controller |
| 2 | Tilt montage | Plumb, Prism Break, Tilt Invaders, Tilt Chomp, Lantern Eel, Careen, Asteroids 3D | one verb per game |
| 3 | Wheel | Tilt Rally, phone held sideways | turn the phone like a wheel |
| 4 | GPU | gPu(doom), Lemmings Exodus | the engine runs on the GPU |
| 5 | Minesweeper twice | Sounding and Scoop side by side | same rules, two worlds |
| 6 | The table | Mahjong, Poker | friends join by QR, no server |
| 7 | Where it started | Asteroids, Blaster | the first two |
| 8 | Call to action | grid of all 17 | austegard.com/games |

## Captures

`node capture/run.cjs <game>` writes `public/shots/<game>/f0000.jpg…`. Portrait
captures are 390×844 CSS px at DPR 2; landscape ones 844×390.

| game | orientation | frames | page seconds | must be visible |
|---|---|---|---|---|
| ball-maze | P | 90 | 3 | ball rolling through corridors |
| plumb | P | 90 | 3 | pieces falling, a row clearing into shards |
| prism-break | P | 90 | 3 | ball breaking bricks, particles |
| tilt-invaders | P | 90 | 3 | formation, ship firing, explosions |
| tilt-chomp | P | 90 | 3 | chomper eating pellets, ghosts chasing |
| lantern-eel | P | 90 | 3 | eel swimming and biting prey |
| careen | P | 90 | 3 | tiles sliding after a gravity turn, a clear |
| asteroids-3d | P | 90 | 3 | rocks around, shots, a rock exploding |
| tilt-rally | L | 150 | 5 | `?demo` autopilot race, other cars |
| doom | L | 150 | 5 | `?bot` gameplay, moving through a level |
| lemmings | L | 150 | 5 | Exodus, thousands of lemmings pouring out |
| sounding | P | 120 | 4 | tap, flood ripple, then a mine chain reaction |
| scoop | P | 120 | 4 | sweep, bag, step in one, flies |
| mahjong | P | 120 | 4 | a hand in progress against the computer |
| poker | P | 120 | 4 | a hand against the computer: cards, bets, chips |
| asteroids.html | P | 75 | 2.5 | the original 2D game being played |
| blaster.html | P | 75 | 2.5 | saucers descending, shots |
