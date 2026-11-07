# texture-collage: a picture cut into regions, each filled with a texture from a diffusion model

Tracking issue: [experiments#109](https://github.com/oaustegard/experiments/issues/109). Live page:
<https://austegard.com/experiments/texture-collage/web/>.

A vectoriser such as the image-to-svg skill cuts a picture into regions and paints each one a flat
colour. This experiment paints each region with a texture instead: one of the 47 classes of the
Describable Textures Dataset (marbled, woven, cracked, scaly…), generated on the GPU by the
[`wgsl-diffusion`](../wgsl-diffusion/RESULTS.md) model and recoloured to the region. The question is
which choices make such a collage still read as the picture: how a region picks its texture class, how
the texture is recoloured, how neighbouring textures meet, and whether the texture's grain should follow
the picture's.

The page takes an uploaded, dropped or pasted picture, shows a collage at once from one stored tile
per class, and lets you set the number of regions, how classes are picked and how textures are
recoloured, change any region's class from a menu, generate fresh tiles on your GPU, and export a PNG.

Likeness is measured three ways: SSIM (structural similarity, 1 is identical), GMSD (gradient
difference, 0 is identical) and ΔE (CIE colour difference; about 2 is just visible).

![Rows 1-3: Kodak input, regions in their mean colour, texture collage (pattern-matched classes, stored
tiles). Rows use 12, 12 and 6 regions. Row 4: the parrots with classes picked by colour, at random, and by pattern
with fresh tiles generated on a Mac](results/figure.jpg)

## Answer

Pick each region's class by matching its pattern, recolour the texture to the region's mean colour,
and turn it to follow the region's grain. Over six Kodak pictures at 6 and 12 regions, picking by
pattern kept SSIM at 0.360, <!-- @evidence[[results/eval_pooled.json#structure.ssim]] --> against
0.280 <!-- @evidence[[results/eval_pooled.json#colour.ssim]] --> picking by colour and 0.211 <!-- @evidence[[results/eval_pooled.json#random.ssim]] -->
 at random, and beat both on SSIM and GMSD
in 11 or 12 of the 12 picture × region-count cells. Recolouring holds each region's average colour (ΔE between region means) to
within 1.32 ΔE; <!-- @evidence[[results/eval_pooled.json#structure.dEreg]] --> the textures' own
colours miss by 38.59. <!-- @evidence[[results/eval_pooled.json#/tint:none structure/dEreg]] --> A
textured collage gives up some likeness: flat colour scores SSIM 0.492 <!-- @evidence[[results/eval_pooled.json#solid.ssim]] -->
 but is
worse on gradients (GMSD 0.315 against 0.244). Edge blending, texture scale and the contrast floor move
the numbers little and are left as page controls.

On a Mac, 8 regions of a 768×512 picture take 1.4 s to find and describe, eight fresh 128 px
textures take 2977 ms, <!-- @evidence[[results/mac/pipeline.json#outputs.S128.textures_ms]] --> and compositing 0.8 s. Stored tiles show a collage first.

What would change this: people judging which collages look good; every metric here measures likeness
to the photo.

## Findings

1. **Picking by pattern beats picking by colour, at random, or one class for all.** SSIM 0.360 against
   0.280, 0.211 and 0.245 (`marbled` everywhere); it beat colour on SSIM in 11/12 cells <!-- @evidence[[results/structure_wins.json#/colour.ssim]] -->
    and random in 12/12. <!-- @evidence[[results/structure_wins.json#/random.ssim]] -->
    Against `marbled` everywhere it won
   SSIM in 11/12 and GMSD in only 6/12 <!-- @evidence[[results/structure_wins.json#/fixed:marbled.gmsd]] -->:
   one mild texture everywhere is a strong default. (`results/eval_pooled.json`)
2. **The pattern choice carries over to a tile it never saw.** The class table was built from two
   seeds and the collages filled with a third. The descriptor distance between input region and
   collage was 1.53 <!-- @evidence[[results/eval_pooled.json#structure.dist]] --> picking by pattern,
   against 4.18 <!-- @evidence[[results/eval_pooled.json#random.dist]] --> at random. Among the 47
   classes the seven-number descriptor names a held-out tile's own class first 69% of the time over 94 held-out tiles (bank seeds 2 and 3; chance 2%). <!-- @evidence[[results/descriptor_separability.json#top1]] -->
   (`run/build_table.mjs`)
3. **Recolour, or the collage stops being the picture.** Region recolouring: 1.32 ΔE between region
   means; the texture's own colours with pattern choice, 38.59; picking by colour and keeping the
   texture's colours, 14.15. <!-- @evidence[[results/eval_pooled.json#/tint:none colour/dEreg]] -->
   Recolouring to the blurred photo instead of the region mean lowers the blurred-colour error from
   7.78 to 6.03 <!-- @evidence[[results/eval_pooled.json#/tint:local/dE]] --> and keeps the photo's
   gradients, at the cost of the cut-paper look. Capping the gain of the colour transfer (4 for L,
   1.5 for a and b) cut the region-colour error from 1.43 to 1.32. <!-- @evidence[[results/eval_pooled.json#/gain:uncapped/dEreg]] -->
   
4. **Turning textures to follow the grain halves the orientation error.** 0.280 rad with alignment,
   0.549 <!-- @evidence[[results/eval_pooled.json#/align:off/orientErr]] --> without, for oriented
   regions; SSIM is unchanged (0.360 vs 0.357).
5. **Textures cost likeness by SSIM and gain it by gradients.** Flat colour: SSIM 0.492, GMSD 0.315; <!-- @evidence[[results/eval_pooled.json#solid.gmsd]] -->
    pattern collage: SSIM 0.360, GMSD 0.244. <!-- @evidence[[results/eval_pooled.json#structure.gmsd]] -->
    Flat colour wins SSIM in every
   cell and loses GMSD in every cell. The photo has texture, and flat fills have none.
6. **Edges, scale and contrast are tastes, not fixes.** A 6 px blend instead of a hard cut lowered
   GMSD at region edges from 0.274 <!-- @evidence[[results/eval_pooled.json#/feather:0/gmsdEdge]] -->
   to 0.248; <!-- @evidence[[results/eval_pooled.json#/feather:6/gmsdEdge]] --> the default is 2 px.
   Letting each region pick 1× or 2× scored between the fixed scales. A contrast floor of 0 scored
   best (SSIM 0.377 <!-- @evidence[[results/eval_pooled.json#/strength:0/ssim]] -->) because flat
   regions then stay flat; the page defaults to 8 so every region shows its texture. Giving each region
   a distinct class cost 0.011 SSIM.
7. **Speed.** Mac (its Apple GPU, reached through the GPU bridge: a page on Oskar's laptop that runs jobs pushed from this container): a 128 px texture at 24 steps took 540 ms <!-- @evidence[[bank/meta.json#median_ms]] -->
    median in the bank job, including readback, and 360 ms at the page's
   20 steps; 256 px took 9.0 s for eight. Regions 0.42 s, descriptors 1.03 s, compositing 0.78 s, all
   on the CPU. SwiftShader (a software GPU, the only one in the cloud container) took 153867 ms for one 64 px texture at 12 steps, <!-- @evidence[[results/swiftshader_speed.json#S64_steps12_ms]] -->
    which is why the page ships a
   stored tile per class. (`results/mac/pipeline.json`)
8. **`freckled` tiles grow eyes.** The model draws skin and eyes for this class (freckles are
   photographed on faces; the dataset was not checked), and pattern matching picks it for bright,
   smooth regions. The page leaves it out of
   automatic choice; it stays in each region's menu.

## Method

**Pipeline** (`web/collage.js`, the same module in the page's worker, the bridge job and the Node harness):

1. **Regions.** The picture is scaled to at most 768 px on its long side and converted to CIELAB. k-means
   (k-means++ start, 20 Lloyd steps on 20,000 sampled pixels) clusters the colours of a blurred copy
   (blur radius = long side / 150) into K clusters. Every pixel takes its nearest cluster, a 5×5 majority
   filter runs twice, and components smaller than 0.2% of the picture, or with an area under 2.5 times
   their boundary length, are merged into the neighbour they share most border with. The second test
   removes the thin band of in-between colour that k-means otherwise gives every blurred edge. A region
   is a colour cluster, so it can be several separate blobs sharing one texture.
2. **Description.** Each region's L channel gives seven numbers: busyness (mean gradient magnitude over
   standard deviation plus 2) at full, half and quarter resolution; the coherence of the region-wide
   structure tensor (does one orientation dominate?) and that orientation; the energy-weighted local
   coherence (lines against blobs); and the skew and log kurtosis of L. Contrast is divided out, since
   the recolouring sets it. Each texture tile gets the same seven numbers at 1× and 2× scale.
3. **Class choice.** *Pattern* (`structure` in code): the class and scale whose mean descriptor is
   nearest the region's, each feature divided by its spread across classes. *Colour*: the class whose
   tiles' mean Lab is nearest the region's mean. *Random* and *fixed* (`marbled` everywhere) are
   controls. *Distinct* lets each class be used once, largest region first.
4. **Fill.** The tile repeats across the region at the chosen scale, turned so its dominant orientation
   matches the region's when both are oriented (coherence above 0.15), sampled bilinearly with
   wrap-around (the tiles are seamless). **Recolour** *to region*: per Lab channel, the texture's
   deviation from its own mean, scaled to the region's spread, is added to the region's mean colour.
   The spread of L has a floor (`strength`, default 8) so flat regions still show texture, and gains are
   capped. *To photo*: the same detail added to a blurred copy of the picture. *Off*: the texture's colours.
5. **Edges.** Each region's mask is blurred by `feather` px and the fills blended by those weights.

**Textures.** Bridge job `tc-bank-1` generated 4 tiles of each class on the Mac: 128×128, 24 steps,
guidance 3, int8 weights, circular padding, bank seeds 0–3, sampler seeds 1000–1003 (`bank/bank.png`, one row per class). The
page starts from the first tile of each class (`web/bank.png`, 1.8 MB) and generates fresh ones on
request. It loads the int8 weights because they are half the download (8.4 MB against 16.8 MB for f16);
`wgsl-diffusion` measured them at 34 dB PSNR against f32. Their effect on the collage was not measured.

**Fixture.** Each *arm* below is one setting of the pipeline; the table names arms by their code
values (`structure` = pick by pattern, `tint:local` = recolour to photo, `solid` = flat colour,
`strength` = contrast floor, `feather` = edge blend px).  Six Kodak pictures (768×512: kodim01, 05, 07, 15, 20, 23) at K = 6 and 12. The class
table comes from seeds 0–1 (`results/table_train.json`); every collage is filled with the seed-2
tiles. Random is averaged over 3 seeds. Every arm uses the defaults (pattern choice, region recolour,
feather 2, strength 8, alignment on, scales {1, 2}) except the setting it names.

**Metrics** (`run/metrics.mjs`), each against the input:
- *ΔE blur*: mean CIE76 colour difference after a blur of radius 8 px. Any texture raises it.
- *ΔE region*: area-weighted CIE76 difference between each region's mean colour in input and collage.
- *SSIM* on L with 7×7 windows; *GMSD* (gradient magnitude similarity deviation, lower is better) on
  L, over the whole picture and within 3 px of region boundaries.
- *Descriptor distance*: between each input region and the same pixels of the collage, area-weighted.
  The choice step minimises the same distance against the table, so this measures whether a choice made
  on two seeds holds for a third tile once it is rotated and recoloured.
- *Orientation error*: the angle between each oriented input region's dominant orientation and the
  collage's there, weighted by coherence and area.

**Checks.** `run/unit.mjs`: colour round trip, a synthetic segmentation, descriptor orientation,
alignment in `render()`. `run/check_page.mjs`: the page on SwiftShader (`results/page_check.json`).
Bridge job `tc-pipe-1`: the whole pipeline on the Mac.

**Prior art.** Hertzmann et al.'s *texture-by-numbers* (Image Analogies, 2001) fills a label map with
texture synthesised from an example pair, and Ritter et al.'s *Painting with Texture* (2006) made it
interactive; both copy texture from a given example. Three searches found no tool that segments a photo
and fills the regions with diffusion-generated textures in the browser. Inside this account, `xr`
found the image-to-svg skill and two emoji-collage pages, which fill regions with colour and emoji.

**Reproduce.** `sh ../image-to-wgsl/data/fetch.sh`, `python3 run/prep_images.py`,
`python3 run/bank_files.py --from-png`, `node run/build_table.mjs`, `node run/eval.mjs 6 12`,
`python3 run/summarize.py`, `node run/unit.mjs`, `node run/check_page.mjs` (Node with Playwright).

## Log

### Round 1, 2026-11-07

Asked (issue #109): turn the texture model into a collage maker that decomposes a picture as
image-to-wgsl and image-to-svg do and fills the regions with textures, as an interactive page.

SwiftShader generated one 64 px texture in 154 s, too slow to build the class bank, so the bank came from
the Mac (188 tiles in 100 s). The first segmentation of a two-colour test picture gave three regions:
k-means had made a cluster of the blend band along the blurred edge, which the boundary-ratio merge
fixed. The first collages had brightened, greener backgrounds where a tile with little colour spread
was stretched to the region's spread; capping the gain fixed it. The Mac collage of the parrots put
`freckled` tiles, with eyes, on the out-of-focus highlights, so the page leaves that class out of
automatic choice.

Full table (`results/eval_table.md`, both region counts pooled):

| arm | n | ΔE blur | ΔE region | SSIM | GMSD | GMSD edges | structure dist | orient err (rad) |
|---|---|---|---|---|---|---|---|---|
| solid | 12 | 5.00 | 0.00 | 0.492 | 0.315 | 0.331 | 9.39 | 1.026 |
| structure | 12 | 7.78 | 1.32 | 0.360 | 0.244 | 0.258 | 1.53 | 0.280 |
| colour | 12 | 7.28 | 1.01 | 0.280 | 0.270 | 0.275 | 3.53 | 0.392 |
| random | 36 | 6.67 | 0.97 | 0.211 | 0.284 | 0.280 | 4.18 | 0.336 |
| fixed:marbled | 12 | 7.39 | 0.99 | 0.245 | 0.252 | 0.255 | 3.03 | 0.615 |
| structure+distinct | 12 | 7.60 | 1.33 | 0.349 | 0.247 | 0.258 | 1.53 | 0.271 |
| tint:none structure | 12 | 37.89 | 38.59 | 0.262 | 0.258 | 0.268 | 2.22 | 0.311 |
| tint:none colour | 12 | 15.74 | 14.15 | 0.237 | 0.287 | 0.288 | 3.98 | 0.382 |
| tint:local | 12 | 6.03 | 3.18 | 0.355 | 0.241 | 0.254 | 1.52 | 0.338 |
| feather:0 | 12 | 7.79 | 1.26 | 0.347 | 0.253 | 0.274 | 1.93 | 0.343 |
| feather:6 | 12 | 7.78 | 1.87 | 0.365 | 0.238 | 0.248 | 1.46 | 0.296 |
| align:off | 12 | 7.93 | 1.43 | 0.357 | 0.245 | 0.259 | 1.52 | 0.549 |
| scale:1 | 12 | 7.32 | 0.95 | 0.319 | 0.250 | 0.260 | 1.79 | 0.327 |
| scale:2 | 12 | 8.02 | 1.48 | 0.367 | 0.244 | 0.258 | 1.67 | 0.221 |
| strength:0 | 12 | 7.51 | 1.16 | 0.377 | 0.243 | 0.258 | 0.96 | 0.314 |
| strength:16 | 12 | 9.46 | 1.99 | 0.282 | 0.263 | 0.271 | 1.89 | 0.307 |
| gain:uncapped | 12 | 8.12 | 1.43 | 0.360 | 0.244 | 0.258 | 1.53 | 0.280 |

Page check on SwiftShader (`results/page_check.json`): first collage 3.6 s after load, seven
re-renders on control changes, a region's menu overriding its class, one fresh 64 px texture in
69 s, a PNG export, no horizontal scroll at 390 px wide, no page errors.
