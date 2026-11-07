# Errors

## 1. A GPU "mismatch" that was two different checkpoints

**What was wrong.** The first comparison of a Mac GPU sample with PyTorch gave 28 dB PSNR, with a mean
difference of 0.06 and a visible colour shift. The GPU was right. The torch reference had been run on
`ema.pt` copied from the mirror directory several minutes after `web/model/weights.bin` was exported
from it, and the keepalive loop had replaced `ema.pt` with the next checkpoint (15k steps instead of
10k) in between.

**How it was caught.** Two checks ruled out the GPU. The fixture cases (random weights) passed on the
same Mac at 3e-6, and the sampler was stable under perturbation: f32 against f16 weights agreed at
72 dB, and 1e-4 relative weight jitter at 81 dB. So the difference had to come from the inputs. One
conv weight compared between `wd_w10k.bin` and the copied `ema.pt` differed by 0.035.

**Fix.** References are now built from the exported `weights.bin` itself (`run/ref_from_bin.py`). Built
that way, the same Mac result matched to 8.6e-7 (142 dB).

**Direction.** It would have pushed the conclusion wrong ("the WGSL runtime is broken on Metal"). It
cost three bridge jobs.
