# RESULTS: live camera tagging in the browser against precomputed tag banks

Built 2026-10-23 for the "Embedded Embedder" project. Oskar asked whether an in-browser, mobile,
live image tagger is plausible on our WebGPU stack. A stock MobileCLIP-S0 demo ran at about 30 fps
on his iPhone 16 Pro with a handful of typed labels. This experiment replaces the typed labels with
a 4,585-tag vocabulary embedded offline, so the camera page shows the top 5 tags with no typing.
Live page: https://austegard.com/experiments/live-tagger/web/

## Answer

Can a phone page tag a live camera feed against thousands of tags without running a text model?
Yes, in the headless check: the page runs MobileCLIP-S0's vision tower on WebGPU and scores 4,585
tags embedded offline by the same model's text tower. Its vectors match onnxruntime CPU at cosine
0.999942 or better on six COCO images, <!-- @evidence[[results/check_swiftshader.json#/images/000000000724.jpg/cos]] -->
with identical top-5 tags on all six. The page's default bank adds Open Images V7's class names,
for 22,690 tags in 11.7 MB of int8. It shows the top 5 general (RAM++) tags and, under them, the
top 3 Open Images labels, because ranking all 22,690 together lets rare specific labels crowd out
the general ones. Scoring the whole bank is one dot product in plain JavaScript (24.7 ms in
headless Chromium; 4.1 ms for RAM++ alone, which `?vocab=small` selects).

On 500 COCO val images, the top 5 tags contain 25.5% of the image's COCO object categories <!-- @evidence[[results/eval_coco.json#/by_beta/0/object_recall@5]] -->
and 27.1% of the top-5 tags appear word for word in the image's captions. <!-- @evidence[[results/eval_coco.json#/by_beta/0/caption_precision@5]] -->
Both are crude proxies (COCO has no tag ground truth), kept as the baseline for later model
swaps.

What would change this: phone timings. Every timing here comes from SwiftShader,
which says nothing about a phone GPU.

## Findings

1. **A bigger vocabulary has to be ranked in tiers.** Ranked together, the 22,690-tag bank drops
   object recall@5 from 0.255 to 0.158 <!-- @evidence[[results/eval_vocab.json#by_bank.big_int8.object_recall@5_shared]] -->
   and caption precision@5 to 0.196. <!-- @evidence[[results/eval_vocab.json#by_bank.big_int8.caption_precision@5]] -->
   Some of that is the proxies penalising correct specificity ("Sommelier" for a wine pourer,
   "California-style pizza"). The rest is rare labels ("Pi day", "Kuih kochi") outranking plain
   ones. Showing the RAM++ top 5 unchanged with an Open Images top 3 beside it keeps the 0.255 and
   still surfaces the specific labels. (`run/eval_vocab.py`)
2. **Canvas `drawImage` does not antialias when it downscales.** A 480 px crop drawn to 256 px
   lands at cosine 0.975 to 0.9996 against PIL's resize, the same as naive bilinear sampling in
   NumPy, and it changes the top 5 on four of six images. A JavaScript copy of PIL's
   separable triangle filter, with integer crop offsets, brings every image to 0.99994 or better. (`run/check.py`, Log rounds 1 to 4)
3. **Hub correction does not help this tag bank.** Some tags ("dot", "twin", "cat furniture")
   show up near the top for many images. Subtracting a per-tag CSLS bias, computed on 995 Bluesky
   images, lowers object recall@5 from 0.255 to 0.244 at full strength. <!-- @evidence[[results/eval_coco.json#/by_beta/1.0/object_recall@5]] -->
   Subtracting each tag's mean cosine instead lowers it to 0.221. <!-- @evidence[[results/eval_coco_meanbias.json#/by_beta/1.0/object_recall@5]] -->
   No strength in the sweep improves both proxies, so the page ranks by raw cosine. (`run/eval_coco.py`)
4. **Only 69 of COCO's 80 categories are RAM++ tags verbatim.** <!-- @evidence[[results/eval_coco.json#coco_categories_in_tags]] -->
   Missing ones include "airplane", "tv" and "dining table", so object recall understates the tagger. (`run/eval_coco.py`)
5. **int8 rows cost nothing measurable.** The big bank at int8 with a float32 scale per row
   scores 0.158 object recall against 0.161 for float32 rows, <!-- @evidence[[results/eval_vocab.json#by_bank.big_f32.object_recall@5_shared]] -->
   at half the bytes of float16. (`run/eval_vocab.py`)
6. **The text tower never runs on the phone.** Tags are the mean of four prompt templates, embedded
   once with the fp32 `text_model.onnx` (3 minutes on 4 vCPU). The page downloads only the 46 MB
   fp32 vision model and the 4.7 MB bank. (`run/embed_tags.py`)

## Method

- **Model.** `Xenova/mobileclip_s0` (MobileCLIP v1 S0) through transformers.js 3.5.0, the
  version and model the working phone demo uses. The vision model runs in fp32 on WebGPU, because
  METHODS.md records transformers.js int8 on WebGPU collapsing embeddings. It falls back to q8 on
  wasm when there is no WebGPU adapter.
- **Tags.** The RAM++ tag list (4,585 tags, `xinyu1205/recognize-anything`), and for the big
  bank Open Images V7's 20,931 class display names after it, deduplicated case-insensitively
  (22,690 in all; `n_general` in `tags-big.json` marks where RAM++ ends). Each tag is the
  normalized mean of "a photo of a {}.", "a photo of the {}.", "a close-up photo of a {}." and
  "{}", padded to 77 tokens with the pad token as CLIPTokenizer does (`run/embed_tags.py`).
- **Preprocessing.** Center square crop at integer offsets, PIL-style bilinear resize to 256,
  divided by 255 with no mean or std normalization, per `preprocessor_config.json`. Camera
  frames over 1,024 px are halved on a canvas first.
- **Loading screen.** A full-screen card lists four stages (runtime, tag bank, vision model,
  shader compile) with byte progress, warns after 15 s without progress, shows errors, and ends
  with a Start camera button, so the iOS permission prompt follows a tap.
- **Live loop.** One frame in flight at a time. The image embedding is smoothed with an EMA (0.5
  on the newest frame), renormalized and scored. The page shows fps and per-stage milliseconds.
- **Checks.** `run/ref_tags.py` produces onnxruntime CPU embeddings and top tags for six COCO
  images. `run/check.py` loads the page in headless Chromium on SwiftShader WebGPU, calls
  `window.tagger.tagImage()` on the same files and requires cosine ≥ 0.999 and the same top-5 set.
- **Quality proxies** (`run/eval_coco.py`, 500 random COCO val2017 images, seed 1): object
  recall@5 over COCO categories that are tags verbatim, and caption precision@5, where a tag counts
  when all of its words appear in the image's five captions (lowercased, trailing "s" stripped).
- **Hub bias** (`run/calibrate.py`): 995 random thumbnails from the Bluesky image-search index.
  CSLS r_tag is each tag's mean cosine to its 10 nearest images. The `--bias mean` arm uses each
  tag's mean cosine over all 995.

## Log

### 2026-10-23: build, parity, calibration

Asked: build the big-tag-bank page after the feasibility note, once Oskar measured about 30 fps for
the stock demo (iPhone 16 Pro, Safari). Feasibility notes are in the project's Claude Doc "Live
mobile image tagger on WGSL".

Ran: tag bank from RAM++ with the MobileCLIP-S0 text tower. Spot checks on six COCO images looked
right (bear: bear, brown bear, bear cub; tennis: play tennis, tennis match), with generic hubs
such as "dot" and "dolly" on the stop-sign image.

Parity rounds in headless Chromium:

| round | preprocessing | min cos | top-5 identical |
|---|---|---|---|
| 1 | `drawImage` crop to 256 | 0.9749 | 2 of 6 |
| 2 | same, `imageSmoothingQuality = "high"` | 0.9749 (no change on SwiftShader) | 2 of 6 |
| 3 | JS triangle filter, fractional crop offset | 0.9976 | 5 of 6 |
| 4 | JS triangle filter, integer crop offset | 0.99994 | 6 of 6 |

Naive bilinear in NumPy reproduced round 1's per-image cosines to within 0.002, which pinned the gap
on resampling rather than on the WebGPU model. Round 2 says nothing about Safari's high-quality
smoothing, only that SwiftShader ignores it.

Hub correction sweep, 500 COCO images:

| bias | β | object recall@5 | caption precision@5 | distinct tags in top 5 |
|---|---|---|---|---|
| none | 0 | 0.255 | 0.271 | 1065 |
| CSLS | 0.25 | 0.249 | 0.274 | 1075 |
| CSLS | 0.5 | 0.245 | 0.267 | 1076 |
| CSLS | 1.0 | 0.244 | 0.252 | 1048 |
| mean | 0.25 | 0.251 | 0.273 | 1051 |
| mean | 1.0 | 0.221 | 0.246 | 1008 |

β 0.25 gains 0.003 caption precision and loses 0.006 object recall, which is within noise and not
worth a second file on the page. Shipped raw cosine.

Follow-up the same day: Oskar's first phone visit showed only a caption and nothing loading
after a minute on unknown Wi-Fi. Added the loading screen above; the stall cause is not known yet.

Open: phone fps and per-stage timings (the stats line shows them), MobileCLIP2-S0 (71.5% against
S0 v1's 67.8% zero-shot IN-1k) once an ONNX export that transformers.js loads exists, and the WGSL
port.

### 2026-10-23: bigger vocabulary

Asked: Oskar, after using the page: "that's pretty damn good, but is it possible to expand the tag
vocabulary?"

Ran: RAM++ plus all Open Images V7 display names (22,690 tags, 15 minutes of CPU for the text
tower; a first pass crashed saving its cache because `np.savez(**cache)` met a tag named "file").
Compared on the same 500 COCO images (`run/eval_vocab.py`):

| bank | tags | object recall@5 (69 shared categories) | caption precision@5 | distinct tags in top 5 |
|---|---|---|---|---|
| RAM++ float16 | 4,585 | 0.255 | 0.271 | 1065 |
| RAM++ + Open Images, float32 | 22,690 | 0.161 | 0.198 | 1443 |
| RAM++ + Open Images, int8 | 22,690 | 0.158 | 0.196 | 1441 |
| RAM++ + Open Images trainable (9,668), float32 | 11,740 | 0.177 | 0.214 | not recorded |

The trainable-subset row was an inline check, not a script. Examples, RAM++ against the big bank:
a dessert table went from "moon cake, pudding, steamed bread" to "South asian sweets, Water
chestnut cake, Kuih kochi", and a winery shot from "bartender, wine tasting, chef" to "Winemaker,
Sommelier, bartender". Shipped the tiered display: RAM++ top 5 plus Open Images top 3, both from
one int8 bank. Headless check passes with the big bank (same top-5 sets; int8 swaps near-tied
neighbours such as "bear" and "brown bear").
