# navier-stokes-explainer

An interactive explainer of the incompressible Navier–Stokes equations, built on WebGPU compute
shaders. Five chapters, one static page, no backend. Tracking issue:
[#101](https://github.com/oaustegard/experiments/issues/101).

Live: <https://austegard.com/experiments/navier-stokes-explainer/web/> (via `pages.json`).
Locally: `python3 -m http.server -d navier-stokes-explainer/web` and open `http://127.0.0.1:8000`.

## Chapters

| # | chapter | what runs |
|---|---|---|
| 1 | The four terms | 320 × 180 stable-fluids solver; advection, pressure, viscosity and forcing each switch off, from buttons or by clicking the term in the equation; dye, vorticity, speed and divergence views |
| 2 | Reynolds number | 320 × 108 channel past a cylinder 22 cells across; Re slider from 1 to 1000 sets ν = UD/Re |
| 3 | Vortex stretching in 2D and 3D | 256² periodic 2D vortex field with a live max\|ω\| trace, beside a 3D vortex-filament model (2 rings × 384 points, all-pairs regularized Biot–Savart, RK2) with 8,000 smoke tracers and a live max-stretch trace; head-on, tilted and leapfrog presets |
| 4 | The Beale–Kato–Majda criterion | toy plot of ∫(T* − t)^−α dt with an α slider; 128²/256²/512² forced 2D turbulence with max\|ω\|, the running BKM average, and a live energy spectrum E(k) from a JS FFT of the read-back velocity |
| 5 | Layered cascades and the 2026 results | Canvas2D cartoon of a geometric cascade (size ratio λ, time ratio μ) with a T* timeline; draggable (λ, μ) plane showing the 3D window λ^5/2 ≤ μ ≤ λ² against the 2D line; Clay alternatives table, claims table, timeline, sources |

## Files

| file | what |
|---|---|
| `web/index.html` | page and prose |
| `web/fluid2d.js` | 2D solver: semi-Lagrangian RK2 backtrace with MacCormack correction, implicit viscosity by Jacobi, Jacobi pressure projection, box / periodic / channel boundaries, obstacle, Kolmogorov forcing, atomic max\|ω\| + energy reduction |
| `web/filament3d.js` | 3D filaments: tiled all-pairs Biot–Savart in workgroup memory, RK2, stretch reduction, smoke tracers, instanced screen-space ribbons |
| `web/plots.js` | Canvas2D line plots, radix-2 FFT, shell-summed energy spectrum |
| `web/cascade.js` | chapter 5 cascade animation and scaling window |
| `web/main.js` | wiring; steps only the chapters on screen; exposes `window.__ns` for the test driver |
| `run/verify.mjs` | headless Chromium (SwiftShader WebGPU) driver: interacts with every chapter, checks the numbers on `window.__ns`, screenshots each chapter and the no-WebGPU fallback |
| `shots/` | screenshots and `verify.json` from the last run |

## Verification

`node run/verify.mjs` (needs the container's Playwright at `/opt/node-tools`, Chromium at
`/opt/pw-browsers/chromium-1194`). The last run (2026-10-19) passed 12 of 12 checks:
a WebGPU device; chapter 1 momentum from a drag (max|ω| 4.3) and the pressure toggle; a developed
cylinder wake after 1,734 steps at Re 158; 2D max|ω| never above its start (0.40 of it after 290
steps); 3D head-on stretch 3.75 at t = 7.2 and tilted 4.72; the BKM verdict following α; a
128-bin spectrum; the chapter 5 verdict following a drag; no page or WebGPU errors; the
no-WebGPU fallback message. `shots/verify.json` has the numbers, `shots/*.png` the pictures.

## Chapter 5: forced, not unforced

Muninn's memory `c43844eb` (from Quanta, 2026-09-08) described the OpenAI result as blowup
"without forcing functions". That was wrong. OpenAI's manuscript states its main theorem as a
solution of Clay alternative (C), finite-time breakdown on ℝ³ with a smooth, decaying force, and
derives (D), the periodic case, as a corollary. The unforced alternatives (A) and (B) are untouched.
Quanta's own text agrees: it calls a smooth forcing function the remaining hurdle after Córdoba and
Martínez-Zoroa's 2023 result. arXiv:2609.17642 calls it "the OpenAI 2026 forced blow-up
construction". The openai.com announcement returned 403 to every fetch, so the theorem and
corollary numbers come from the lilting.ch explainer, not the manuscript itself. The memory was
superseded by `edba5729`, and the conflict note `7ba0692a` was retired.

## Limits

- The 2D solver's interpolation diffuses momentum on its own. Chapter 2's top Reynolds numbers and
  chapter 4's smallest ν are dominated by that numerical diffusion; the page says so.
- The filament model is inviscid and cannot reconnect. It shows stretching raising vorticity, and
  it cannot blow up.
- The chapter 5 cascade is a scaling cartoon, not the Córdoba–Martínez-Zoroa or OpenAI construction.
- SwiftShader timings say nothing about real-GPU speed: about 3 ms per dispatch there, so a 50-pass
  fluid step takes ~165 ms headless.
