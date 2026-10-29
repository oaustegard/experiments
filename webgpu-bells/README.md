# webgpu-bells

A six-bell ring (rounds twice, plain hunt down to back rounds) and a 110 Hz bourdon, synthesised by
summing 3,320 damped sinusoids on the GPU at 96 kHz and saved as 24-bit WAV.

Each bell has 30 partials at church-bell ratios: hum 0.5, prime 1, minor-third tierce 1.19, quint 1.5,
nominal 2 and upward. Every partial is a doublet split by 0.02–0.14 %, which gives a real bell its
slow warble. The bell's nodal pattern `cos(n(θ − θ0))` sets how hard each partial is excited, so
handstrokes and backstrokes, which strike opposite sides, interfere differently with what is still
ringing. A half-sine clapper contact of about 0.3 ms shapes the excitation spectrum, and 400
transient modes between 2.2 and 19 kHz decay in tens of milliseconds. Stereo comes from per-bell
panning and an inter-mic delay of up to 0.6 ms, applied as a complex gain on each mode.

`score.py` builds the mode table and the strike list (49 strikes, 22,639 mode excitations).
`job.js` is a GPU Bridge job. Its first pass runs in float64 JS and computes each mode's complex
state at every 4,096-sample block start, keeping only modes above −130 dB. Strikes that land inside
a block keep their own sample offset, so onsets are sample-exact. Its second pass is a WGSL kernel
with one thread per sample. Each thread sums `Im(z · e^{s j})` over the block's live modes, taking
`e^{s j}` as `hi[j >> 6] · lo[j & 63]` from tables built in float64. The GPU evaluates no sin or exp,
because WGSL only promises sin to 2^-11 absolute, about −66 dB per mode.

## Results

| run | device | GPU | JS prep | error vs float64 reference (RMS, re peak) |
|---|---|---|---|---|
| `bells-96k-1` | iPhone (iOS 27, Chrome) | 79 ms | 73 ms | −135.7 dB |
| local | SwiftShader, 4 vCPU container | 6.1–7.0 s | 64–121 ms | −134.6 dB |

The iPhone run did 0.73 G mode-samples, 237 times faster than real time, and agrees with the
SwiftShader render to −157 dB.

`master.py` high-passes at 18 Hz, normalises to −1 dBFS and writes the dry render. It also writes a
version convolved with a synthetic stone-square impulse response: frequency-dependent T60 from
2.6 s down to 0.5 s, plus 18 early reflections. The reverb is FFT convolution in the container, not
on the GPU.

```bash
python3 score.py OUT                     # modes.bin, strike_*.bin, meta.json
# job inputs: modes (f64 N x 6), strike_mode, strike_on (u32), strike_amp (f64), meta (json)
python3 master.py audio.npy OUT          # bells-dry-96k24.wav, bells-hall-96k24.wav
```
