import json
from pathlib import Path
R = Path("/home/user/experiments/image-to-wgsl/results")
d = json.loads((R/"bit_allocation.json").read_text())
F, C, B, S = d["fields"], d["combined"], d["base"], d["slope"]
s256, s1024 = S["256-512"]["mean_db_per_byte"]*1000, S["512-1024"]["mean_db_per_byte"]*1000
o = []
o.append(f"""# Bit allocation of the 61-bit splat (no retraining)

Question: which fields of `codec.Layout` earn their bytes, and is there a layout with more PSNR per gzipped byte?

## What was measured

- Six prefix-trained runs `results/runs/kodim{{01,05,07,15,20,23}}_prefix1024.json`, rendered at full resolution (`codec.render_image`, 8-bit PSNR via `codec.psnr8`) at the first 256 and first 1024 splats.
- For each field group and k in {{1, 2}}: the stored integer q is requantized onto the coarser grid the reduced `Layout` decodes (`round(q * M'/M)` for x, y, sigma, colour; `round(q/2^k) mod 2^(b-k)` for angle, which wraps; `round((q+1)/2^k - 1)` for alpha; shape exponent clamped to the smaller range). Groups: x+y (both axes), sigma (both axes), angle, colour (all three channels), alpha, shape. The reduced `Layout` goes to `codec.emit_wgsl` and `codec.rate`, so bytes are gzip-9 of the minified shader (data words and decoder).
- All numbers are means over the 6 images. Baseline: {B['256']['psnr']:.2f} dB / {B['256']['gz']:.0f} B at n=256, {B['1024']['psnr']:.2f} dB / {B['1024']['gz']:.0f} B at n=1024.
- Splat slope (what the same bytes buy as more splats, per-image slope of the prefix curve in `summary.json`, averaged): **{s1024:.3f} dB/KB** between 512 and 1024 splats, {s256:.3f} dB/KB between 256 and 512. The n=1024 rows compare against the first, the n=256 rows against the second.
- A field is over-provisioned when its dB lost per byte saved is below the slope. `net` = bytes saved x slope - dB lost, i.e. the dB gained if the freed bytes buy splats on the prefix curve.
- Every number is a lower bound on the gain: the splats were trained at the current layout and not refit, and a refit at fewer bits recovers part of the loss. (Counterpoint: the slope is also from this training recipe.)

## Per-field tables

dB/KB columns are loss per byte saved; compare with the slope ({s256:.3f} at n=256, {s1024:.3f} at n=1024). Bits is the splat width after the change (baseline 61).
""")
names = {"xy": "x and y", "sigma": "log2 sigma (both axes)", "angle": "angle", "colour": "colour (3 channels)", "alpha": "alpha", "shape": "shape"}
for g, nm in names.items():
    o.append(f"\n### {nm}\n\n| change | bits | dB lost @256 | bytes saved @256 | dB/KB @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | dB/KB @1024 | net dB @1024 | verdict |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for k in (1, 2):
        r = F[f"{g}-{k}"]; a, b = r["256"], r["1024"]
        v = "over-provisioned" if b["over_provisioned"] and a["over_provisioned"] else ("over-provisioned at 1024 only" if b["over_provisioned"] else "keep")
        o.append(f"| -{k} bit{'s' if k>1 else ''} | {b['bits']} | {a['psnr_loss_db']:.3f} | {a['bytes_saved']:.0f} | {1000*a['loss_db_per_byte']:.3f} | {a['net_gain_db_if_bytes_respent']:+.3f} | {b['psnr_loss_db']:.3f} | {b['bytes_saved']:.0f} | {1000*b['loss_db_per_byte']:.3f} | {b['net_gain_db_if_bytes_respent']:+.3f} | {v} |")
o.append("""
Shape: the trained exponent is 0 for all 6 x 1024 splats (every splat is a plain gaussian), so the 2 bits carry nothing and the loss is exactly 0.00 dB. Removing the field's bits still leaves the `for` loop in the shader; the `shape-drop` row below also deletes that line.
""")
o.append("\n## Combined and extra layouts\n\n| layout | bits | dB lost @256 | bytes saved @256 | net dB @256 | dB lost @1024 | bytes saved @1024 | net dB @1024 | share of bytes @1024 |\n|---|---|---|---|---|---|---|---|---|")
lab = {"shape-drop": "shape removed (0 bits, loop deleted)", "colour-3": "colour -3 bits (3 per channel)", "rec_A": "RECOMMENDED: shape removed, colour 4 bits/channel, sigma 5 bits/axis", "rec_B": "rec_A with colour 3 bits/channel"}
for k in ("shape-drop", "rec_A", "colour-3", "rec_B"):
    a, b = C[k]["256"], C[k]["1024"]
    o.append(f"| {lab[k]} | {b['bits']} | {a['psnr_loss_db']:.3f} | {a['bytes_saved']:.0f} | {a['net_gain_db_if_bytes_respent']:+.3f} | {b['psnr_loss_db']:.3f} | {b['bytes_saved']:.0f} | {b['net_gain_db_if_bytes_respent']:+.3f} | {100*b['bytes_saved']/B['1024']['gz']:.1f}% |")
pi = C["rec_A"]["1024"]["per_image_net"]
o.append(f"""
Per-image net at 1024 for the recommended layout: {', '.join(f'{k} {v:+.2f}' for k, v in pi.items())} dB; positive on all six. Colour at 3 bits goes negative on 4 of 6 images (kodim07, 15, 20, 23), so 4 bits is the floor for colour.

## Recommended layout

`Layout(bx=10, by=9, bs=5, ba=6, bc=4, bal=4, bm=0)`: **51 bits per splat (from 61)**. The shape field is deleted, colour goes from 6 to 4 bits per channel, sigma from 6 to 5 bits per axis; x, y, angle and alpha are untouched.

Predicted, as a lower bound, with no retraining: it costs {C['rec_A']['1024']['psnr_loss_db']:.2f} dB at 1024 splats and saves {C['rec_A']['1024']['bytes_saved']:.0f} B ({100*C['rec_A']['1024']['bytes_saved']/B['1024']['gz']:.0f}% of the shader). Spending those bytes on more splats at the prefix slope returns {C['rec_A']['1024']['bytes_saved']*s1024/1000:.2f} dB, for a net of **{C['rec_A']['1024']['net_gain_db_if_bytes_respent']:+.2f} dB at equal bytes** (n=256: {C['rec_A']['256']['net_gain_db_if_bytes_respent']:+.2f} dB). The gain is small because the slope is shallow (about 0.25 dB per extra KB at the top of the curve). At 256 splats the net is thin ({C['rec_A']['256']['net_gain_db_if_bytes_respent']:+.2f} dB) because the slope there is steeper; the shape removal alone is the part that is free at every size.

The byte savings run above the raw bit count: 10 bits per splat is 1.25 B raw, but the sweep saves 1722 B / 1024 splats = 1.7 B per splat, because the packed words are emitted as decimal literals and the shape loop leaves the decoder.

"## What stays

- x/y: -1 bit costs {F['xy-1']['1024']['psnr_loss_db']:.2f} dB at 1024 for {F['xy-1']['1024']['bytes_saved']:.0f} B, {1000*F['xy-1']['1024']['loss_db_per_byte']:.2f} dB/KB against a {s1024:.2f} slope: 4x too expensive to cut. Position precision is the best-spent field.
- Angle: worst ratio of any field ({1000*F['angle-1']['1024']['loss_db_per_byte']:.2f} dB/KB at -1 bit, {1000*F['angle-1']['256']['loss_db_per_byte']:.2f} at n=256); keep 6 bits.
- Alpha: -1 bit is break-even at 1024 (net {F['alpha-1']['1024']['net_gain_db_if_bytes_respent']:+.3f} dB) and negative at 256; leave it.
- Sigma -1 is a marginal win only at n=1024 (net {F['sigma-1']['1024']['net_gain_db_if_bytes_respent']:+.3f} dB) and a small loss at n=256 ({F['sigma-1']['256']['net_gain_db_if_bytes_respent']:+.3f}); it is in the recommendation for the full-size shader and should be dropped for small prefixes. Without it the layout is shape + colour only, 53 bits, net {C['rec_A']['1024']['net_gain_db_if_bytes_respent'] - F['sigma-1']['1024']['net_gain_db_if_bytes_respent']:+.2f} dB at 1024 by addition (not remeasured).

## Extra bits (k = -1), argued, not measured

Adding a bit needs retraining, so it is bounded from the quantization error. Each bit removed multiplies the loss by about 3 (x/y: {F['xy-2']['1024']['psnr_loss_db']:.2f} dB at -2 vs {F['xy-1']['1024']['psnr_loss_db']:.2f} at -1, a factor of {F['xy-2']['1024']['psnr_loss_db']/F['xy-1']['1024']['psnr_loss_db']:.1f}; colour {F['colour-2']['1024']['psnr_loss_db']/F['colour-1']['1024']['psnr_loss_db']:.1f}), consistent with quantization noise variance falling 4x per bit. So the 11th x/y bit would recover roughly {F['xy-1']['1024']['psnr_loss_db']/(F['xy-2']['1024']['psnr_loss_db']/F['xy-1']['1024']['psnr_loss_db']):.2f} dB at 1024 splats for about +{F['xy-1']['1024']['bytes_saved']:.0f} B, {1000*F['xy-1']['1024']['psnr_loss_db']/(F['xy-2']['1024']['psnr_loss_db']/F['xy-1']['1024']['psnr_loss_db'])/F['xy-1']['1024']['bytes_saved']:.2f} dB/KB against a slope of {s1024:.2f}: roughly break-even to slightly positive at n=1024 and negative at n=256 (about {1000*F['xy-1']['256']['psnr_loss_db']/(F['xy-2']['256']['psnr_loss_db']/F['xy-1']['256']['psnr_loss_db'])/F['xy-1']['256']['bytes_saved']:.2f} dB/KB vs {s256:.2f}). An extra colour bit would recover about {F['colour-1']['1024']['psnr_loss_db']/(F['colour-2']['1024']['psnr_loss_db']/F['colour-1']['1024']['psnr_loss_db']):.3f} dB for +{F['colour-1']['1024']['bytes_saved']:.0f} B, far below the slope. Neither is worth a retrain on this evidence; a full-resolution x/y bit is the only candidate, and only for the largest shader.

## Caveats

- Lower bound on retraining only in the sense above. Quantization-aware retraining at the new layout would recover some of each loss; it could also shift the shape field away from all-zero (the trained runs never used it, which may be an effect of the init rather than a property of the images).
- "Net" assumes freed bytes buy splats at the prefix-curve slope, which is itself an average over prefixes of one training recipe; actually adding splats means training them.
- Six images, one seed each. Per-image nets for the recommendation are positive on all six, with the smallest on kodim23.
- gzip savings come from the real minified shader, not from the nominal bit count.
- Raw sweeps: `results/bit_allocation_raw.json` (field x k), `results/bit_allocation_raw2.json` (combined layouts); `results/bit_allocation.json` has the merged, aggregated numbers.
""")
(R/"bit_allocation.md").write_text("\n".join(o))
