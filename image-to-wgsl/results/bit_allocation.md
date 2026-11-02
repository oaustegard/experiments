# Bit allocation of the 61-bit splat (no retraining)

Question: which fields of `codec.Layout` earn their bytes, and is there a layout with more PSNR per gzipped byte?

## What was measured

- Six prefix-trained runs `results/runs/kodim{01,05,07,15,20,23}_prefix1024.json`, rendered at full resolution (`codec.render_image`, 8-bit PSNR via `codec.psnr8`) at the first 256 and first 1024 splats.
- For each field group and k in {1, 2}: the stored integer q is requantized onto the coarser grid the reduced `Layout` decodes (`round(q * M'/M)` for x, y, sigma, colour; `round(q/2^k) mod 2^(b-k)` for angle, which wraps; `round((q+1)/2^k - 1)` for alpha; shape exponent clamped to the smaller range). Groups: x+y (both axes), sigma (both axes), angle, colour (all three channels), alpha, shape. The reduced `Layout` goes to `codec.emit_wgsl` and `codec.rate`, so bytes are gzip-9 of the minified shader (data words and decoder).
- All numbers are means over the 6 images. Baseline: 22.59 dB / 3483 B at n=256, 25.13 dB / 11456 B at n=1024.
- Splat slope (what the same bytes buy as more splats, per-image slope of the prefix curve in `summary.json`, averaged): **0.253 dB/KB** between 512 and 1024 splats, 0.449 dB/KB between 256 and 512. The n=1024 rows compare against the first, the n=256 rows against the second.
- A field is over-provisioned when its dB lost per byte saved is below the slope. `net` = bytes saved x slope - dB lost, i.e. the dB gained if the freed bytes buy splats on the prefix curve.
- Every number is a lower bound on the gain: the splats were trained at the current layout and not refit, and a refit at fewer bits recovers part of the loss. (Counterpoint: the slope is also from this training recipe.)

## Per-field tables

dB/KB columns are loss per byte saved; compare with the slope (0.449 at n=256, 0.253 at n=1024). Bits is the splat width after the change (baseline 61).


### x and y

| change | bits | dB lost @256 | bytes saved @256 | dB/KB @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | dB/KB @1024 | net dB @1024 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| -1 bit | 59 | 0.069 | 89 | 0.772 | -0.029 | 0.328 | 349 | 0.941 | -0.240 | keep |
| -2 bits | 57 | 0.318 | 180 | 1.771 | -0.237 | 0.967 | 702 | 1.378 | -0.789 | keep |

### log2 sigma (both axes)

| change | bits | dB lost @256 | bytes saved @256 | dB/KB @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | dB/KB @1024 | net dB @1024 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| -1 bit | 59 | 0.066 | 87 | 0.754 | -0.027 | 0.080 | 346 | 0.230 | +0.008 | over-provisioned at 1024 only |
| -2 bits | 57 | 0.309 | 180 | 1.713 | -0.228 | 0.334 | 702 | 0.476 | -0.156 | keep |

### angle

| change | bits | dB lost @256 | bytes saved @256 | dB/KB @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | dB/KB @1024 | net dB @1024 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| -1 bit | 60 | 0.126 | 53 | 2.377 | -0.103 | 0.250 | 227 | 1.101 | -0.193 | keep |
| -2 bits | 59 | 0.439 | 92 | 4.751 | -0.398 | 0.623 | 367 | 1.697 | -0.530 | keep |

### colour (3 channels)

| change | bits | dB lost @256 | bytes saved @256 | dB/KB @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | dB/KB @1024 | net dB @1024 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| -1 bit | 58 | 0.020 | 129 | 0.154 | +0.038 | 0.039 | 527 | 0.074 | +0.094 | over-provisioned |
| -2 bits | 55 | 0.079 | 260 | 0.305 | +0.037 | 0.144 | 1021 | 0.141 | +0.114 | over-provisioned |

### alpha

| change | bits | dB lost @256 | bytes saved @256 | dB/KB @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | dB/KB @1024 | net dB @1024 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| -1 bit | 60 | 0.044 | 50 | 0.888 | -0.022 | 0.047 | 206 | 0.228 | +0.005 | over-provisioned at 1024 only |
| -2 bits | 59 | 0.149 | 88 | 1.690 | -0.109 | 0.195 | 369 | 0.528 | -0.102 | keep |

### shape

| change | bits | dB lost @256 | bytes saved @256 | dB/KB @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | dB/KB @1024 | net dB @1024 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| -1 bit | 60 | 0.000 | 44 | 0.000 | +0.020 | 0.000 | 163 | 0.000 | +0.041 | over-provisioned |
| -2 bits | 59 | 0.000 | 78 | 0.000 | +0.035 | 0.000 | 324 | 0.000 | +0.082 | over-provisioned |

Shape: the trained exponent is 0 for all 6 x 1024 splats (every splat is a plain gaussian), so the 2 bits carry nothing and the loss is exactly 0.00 dB. Removing the field's bits still leaves the `for` loop in the shader; the `shape-drop` row below also deletes that line.


## Combined and extra layouts

| layout | bits | dB lost @256 | bytes saved @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | net dB @1024 | share of bytes @1024 |
|---|---|---|---|---|---|---|---|---|
| shape removed (0 bits, loop deleted) | 59 | 0.000 | 100 | +0.045 | 0.000 | 348 | +0.088 | 3.0% |
| RECOMMENDED: shape removed, colour 4 bits/channel, sigma 5 bits/axis | 51 | 0.147 | 454 | +0.056 | 0.222 | 1722 | +0.214 | 15.0% |
| colour -3 bits (3 per channel) | 52 | 0.334 | 385 | -0.161 | 0.615 | 1536 | -0.226 | 13.4% |
| rec_A with colour 3 bits/channel | 48 | 0.394 | 590 | -0.129 | 0.689 | 2283 | -0.111 | 19.9% |

Per-image net at 1024 for the recommended layout: kodim01 +0.32, kodim05 +0.37, kodim07 +0.23, kodim15 +0.10, kodim20 +0.20, kodim23 +0.06 dB; positive on all six. Colour at 3 bits goes negative on 4 of 6 images (kodim07, 15, 20, 23), so 4 bits is the floor for colour.

## Recommended layout

`Layout(bx=10, by=9, bs=5, ba=6, bc=4, bal=4, bm=0)`: **51 bits per splat (from 61)**. The shape field is deleted, colour goes from 6 to 4 bits per channel, sigma from 6 to 5 bits per axis; x, y, angle and alpha are untouched.

Predicted, as a lower bound, with no retraining: it costs 0.22 dB at 1024 splats and saves 1722 B (15% of the shader). Spending those bytes on more splats at the prefix slope returns 0.44 dB, for a net of **+0.21 dB at equal bytes** (n=256: +0.06 dB). The gain is small because the slope is shallow (about 0.25 dB per extra KB at the top of the curve). At 256 splats the net is thin (+0.06 dB) because the slope there is steeper; the shape removal alone is the part that is free at every size.

The byte savings run above the raw bit count: 10 bits per splat is 1.25 B raw, but the sweep saves 1722 B / 1024 splats = 1.7 B per splat, because the packed words are emitted as decimal literals and the shape loop leaves the decoder.

"## What stays

- x/y: -1 bit costs 0.33 dB at 1024 for 349 B, 0.94 dB/KB against a 0.25 slope: 4x too expensive to cut. Position precision is the best-spent field.
- Angle: worst ratio of any field (1.10 dB/KB at -1 bit, 2.38 at n=256); keep 6 bits.
- Alpha: -1 bit is break-even at 1024 (net +0.005 dB) and negative at 256; leave it.
- Sigma -1 is a marginal win only at n=1024 (net +0.008 dB) and a small loss at n=256 (-0.027); it is in the recommendation for the full-size shader and should be dropped for small prefixes. Without it the layout is shape + colour only, 53 bits, net +0.21 dB at 1024 by addition (not remeasured).

## Extra bits (k = -1), argued, not measured

Adding a bit needs retraining, so it is bounded from the quantization error. Each bit removed multiplies the loss by about 3 (x/y: 0.97 dB at -2 vs 0.33 at -1, a factor of 2.9; colour 3.7), consistent with quantization noise variance falling 4x per bit. So the 11th x/y bit would recover roughly 0.11 dB at 1024 splats for about +349 B, 0.32 dB/KB against a slope of 0.25: roughly break-even to slightly positive at n=1024 and negative at n=256 (about 0.17 dB/KB vs 0.45). An extra colour bit would recover about 0.011 dB for +527 B, far below the slope. Neither is worth a retrain on this evidence; a full-resolution x/y bit is the only candidate, and only for the largest shader.

## Caveats

- Lower bound on retraining only in the sense above. Quantization-aware retraining at the new layout would recover some of each loss; it could also shift the shape field away from all-zero (the trained runs never used it, which may be an effect of the init rather than a property of the images).
- "Net" assumes freed bytes buy splats at the prefix-curve slope, which is itself an average over prefixes of one training recipe; actually adding splats means training them.
- Six images, one seed each. Per-image nets for the recommendation are positive on all six, with the smallest on kodim23.
- gzip savings come from the real minified shader, not from the nominal bit count.
- Raw sweeps: `results/bit_allocation_raw.json` (field x k), `results/bit_allocation_raw2.json` (combined layouts); `results/bit_allocation.json` has the merged, aggregated numbers.
