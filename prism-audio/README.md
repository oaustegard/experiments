# Two Prisms

Five-band spectral audio processor running in one WebGPU compute dispatch. Each block of samples from an AudioWorklet goes to the GPU, where a single 256-thread workgroup walks the hops sequentially: Hann window, 1024-point radix-2 FFT in workgroup memory, band energies via logistic crossovers in log-frequency, per-band compressor/gate/mute envelope (state kept in a storage buffer across dispatches), optional spectral freeze (held magnitudes, re-rolled phases), Hermitian mirror, inverse FFT, synthesis window, overlap-add. The same dispatch writes one row per hop into a storage texture that a render pass draws as a waterfall.

Hop 256 at 75% overlap; added latency is N−HOP = 768 samples plus the block and queue sizes chosen in the UI.

## Files

- `web/index.html` — the page: UI, AudioWorklet, demo loop. Live at https://austegard.com/experiments/prism-audio/web/
- `web/engine.js` — `PrismEngine` (WGSL compute + render shaders, host code). Runs in browsers and in Node via the `webgpu` package.
- `cpu-mirror.js` — line-by-line JS mirror of the kernel, used for the checks below.

## What was verified (2026-10-24)

- Dawn (lavapipe adapter) compiles both WGSL modules with no messages and validates the compute pipeline, render pipeline and bind groups.
- The CPU mirror: flat settings reconstruct the input to 1.2e-7 max error at 768 samples delay; muting the 120–500 Hz band drops a 300 Hz tone from 0.30 to 0.005 while 3 kHz stays at 0.20; threshold −30 dB, ratio 4 on a −14 dB band gives 12.0 dB gain reduction; freeze over silence sustains the band at 0.208 RMS against 0.212 input (after a 1.85 gain that offsets random-phase overlap-add loss).
- GPU execution is untested: in this sandbox llvmpipe cannot complete a queue submission (futex error under gVisor), so `test.js` hangs. Run `node test.js` on a machine with a real GPU to check the WGSL path against the same numbers.
