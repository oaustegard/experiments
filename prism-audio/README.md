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
- The WGSL path, run on SwiftShader WebGPU in headless Chromium: the same four checks give the same numbers (300 Hz 0.0048 muted, 3 kHz 0.20; 11.96 dB gain reduction; freeze 0.208 RMS), no validation errors, and the waterfall render pass completes. Flat-settings reconstruction error is 4.0e-4 (about −68 dB), larger than the CPU mirror's because the GPU computes twiddles and the window in f32. SwiftShader took 29 ms per 1024-sample block, over the 21 ms budget; it is a CPU emulation of a GPU, so this says nothing about real hardware.
- Not tested: a hardware GPU, the AudioWorklet round trip, Safari and Firefox.
