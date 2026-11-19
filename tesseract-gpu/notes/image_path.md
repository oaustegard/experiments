# Tesseract 5: how a page image reaches the LSTM

This file explains how Tesseract 5 turns a page image into the pixels its LSTM
recognizer reads, and what a GPU pipeline could change on that path. The GPU
pipeline means the two tesseract-gpu engines (`server/torch_engine.py` and
`web/engine.js`) fed by Indra's line detector, the detector from Datalab's Indra
OCR. That pipeline does not use Tesseract's own page layout, so the sections
below describe the path it replaces.

Source: a shallow clone of `tesseract-ocr/tesseract` at commit `9c1f34d3`, read but not built or run.
Every reference is `src/...:line` relative to the clone root, and the lines were checked against that tree.
Leptonica (`pixOtsuAdaptiveThreshold`, `pixSauvolaBinarizeTiled`, `pixRotateOrth`) is external and is not described here.
`Input::PrepareLSTMInputs` and `NetworkIO::FromPix` are not described here.

## Order of operations

1. `TessBaseAPI::SetImage` copies the image into an `ImageThresholder`, keeps the original pixels as `pix_original_`, and sets the resolution (`src/api/baseapi.cpp:505-511`, `529-541`; `src/ccmain/thresholder.cpp:163-183`).
2. `TessBaseAPI::Threshold` makes the binary image, and for non-Otsu methods also the grey image and a per-pixel threshold image (`src/api/baseapi.cpp:2012-2085`).
3. `Tesseract::SegmentPage` runs connected components on the binary image, finds tabs and skew, and builds text rows (`src/ccmain/pagesegmain.cpp:101`, `309`, `329-334`; `src/textord/textord.cpp:177-231`).
4. For each word or line, `LSTMRecognizeWord` crops `BestPix()` and runs the recogniser (`src/ccmain/linerec.cpp:229-258`).

No stage resamples the page. `ImageThresholder::SetImage` sets `scale_ = 1` (`src/ccmain/thresholder.cpp:181`), and no other assignment to `scale_` exists in the tree.

## 1. Thresholding, grey and binary images, DPI

**Methods.** `ThresholdMethod` is `Otsu = 0`, `LeptonicaOtsu = 1`, `Sauvola = 2` (`src/ccmain/thresholder.h:30-35`). The variable is `thresholding_method`, default `Otsu` (`src/ccmain/tesseractclass.cpp:84-88`).

**Otsu (default).** The binary image is made by `ThresholdToPix` → `OtsuThresholdRectToPix` → `ThresholdRectToPix` (`src/api/baseapi.cpp:2047-2052`; `src/ccmain/thresholder.cpp:288-312`, `382-389`). The threshold comes from `OtsuThreshold` in `src/ccstruct/otsuthr.cpp:35`, one value per channel. A pixel is white when it is above the threshold, per channel, in the `hi_values` test (`src/ccmain/thresholder.cpp:407-420`). The grey image comes from `GetPixRectGrey` (`src/ccmain/thresholder.cpp:365-379`, which converts to 8-bit). The threshold image is a single constant from one global Otsu over the rectangle (`src/ccmain/thresholder.cpp:321-336`, using `pixSetAllArbitrary` at line 334). Both are stored at `src/api/baseapi.cpp:2055-2056`.

**LeptonicaOtsu.** `pixOtsuAdaptiveThreshold` is called with a tile size of `thresholding_tile_size` (default 0.33) times the DPI, with a floor of 16 px (`src/ccmain/thresholder.cpp:251-256`; default at `src/ccmain/tesseractclass.cpp:102`). The smoothing kernel is `thresholding_smooth_kernel_size`, default 0.0 (`src/ccmain/thresholder.cpp:258-264`; `src/ccmain/tesseractclass.cpp:107`). The score fraction is `thresholding_score_fraction`, default 0.1 (`src/ccmain/thresholder.cpp:266-267`; `src/ccmain/tesseractclass.cpp:113`).

**Sauvola.** This runs only when the rectangle is larger than 6×6 px. Otherwise the code falls through to the LeptonicaOtsu branch (`src/ccmain/thresholder.cpp:214`, `251`). The window is `thresholding_window_size` (default 0.33) times the DPI, at least 7 px, and capped at `min(w, h) - 3` (`src/ccmain/thresholder.cpp:220-223`; `src/ccmain/tesseractclass.cpp:92`). `thresholding_kfactor` is 0.34 (`src/ccmain/thresholder.cpp:240-242`; `src/ccmain/tesseractclass.cpp:97`). Tiles are about 250 px (`src/ccmain/thresholder.cpp:227-238`). The call is `pixSauvolaBinarizeTiled` (`src/ccmain/thresholder.cpp:248-250`).

**Where the images are made.** The source is kept in its own depth. `pix_channels_` is derived from the depth (`src/ccmain/thresholder.cpp:173-179`), and the rectangle is cropped with `GetPixRect` (`src/ccmain/thresholder.cpp:348-359`). `pix_original_` is set from that crop (`src/api/baseapi.cpp:509`, `539`, `864-866`).

**DPI.** The image's Y resolution is read from the Pix (`src/ccmain/thresholder.cpp:182`). The priority is:
- `user_defined_dpi` overrides everything (`src/api/baseapi.cpp:2024-2035`). The CLI sets it at `src/tesseract.cpp:756`, and the default is 0 (`src/ccmain/tesseractclass.cpp:355`).
- Otherwise, a resolution outside `[70, 2400]` is replaced by 70 (`src/api/baseapi.cpp:2036-2043`; `include/tesseract/publictypes.h:36`, `38`).

The layout resolution is then clipped to the same range and stored (`src/api/baseapi.cpp:2077-2085`).

**Estimate from text size.** This runs only when the source resolution is still 70 (`src/ccmain/pagesegmain.cpp:318-319`). The estimate is `to_block->line_size × kResolutionEstimationFactor`, where `kResolutionEstimationFactor` is 10 (`src/ccmain/pagesegmain.cpp:321`; `include/tesseract/publictypes.h:43`). It is accepted only if it is larger than 70 and below 2400 (`src/ccmain/pagesegmain.cpp:322-325`).

Note: the code uses `line_size`, the median height of text partitions (`src/textord/colpartitiongrid.cpp:727-731`; `src/textord/colpartition.cpp:1676`). It does not use the x-height. The per-row x-height is computed later in `src/textord/makerow.cpp:1379`.

## 2. The image the LSTM reads

`BestPix()` returns `pix_original_` when its width equals `ImageWidth()` (the width of the binary image). Otherwise it returns the grey image, and failing that the binary (`src/ccmain/tesseractclass.h:240-248`, `259-261`).

`pix_original_` is the cropped source in its own depth (`src/ccmain/thresholder.cpp:348-359`). Because `scale_` is 1 and the rectangle is shared, the widths match. In the normal path, then, the LSTM reads **the original colour or 8-bit pixels, not the binary and not the grey copy**. The binary is used only for layout.

**Word box.** `LSTMRecognizeWord` takes the word's bounding box (`src/ccmain/linerec.cpp:231`). In `PSM_SINGLE_WORD` and `PSM_RAW_LINE` the box is the whole image (`src/ccmain/linerec.cpp:233-236`). Otherwise it is clipped to the row: bottom is `baseline + descenders`, top is `baseline + x_height + ascenders`, with the baseline read at the box's mid-x (`src/ccmain/linerec.cpp:238-244`).

**Crop.** `GetRectImage` pads the box by `kImagePadding = 4` px on each side (`src/ccmain/linerec.cpp:164-167`; `src/ccstruct/imagedata.h:39`). If the block is rotated by 90, 180 or 270 degrees, it rotates the box back (`src/ccmain/linerec.cpp:172-183`). It clips to the image (`src/ccmain/linerec.cpp:185-194`), flips the y-axis for `pixClipRectangle` (`src/ccmain/linerec.cpp:195-197`), and rotates the crop orthogonally with `pixRotateOrth` (`src/ccmain/linerec.cpp:202-206`). Depths below 8 bit go to 8 bit (`src/ccmain/linerec.cpp:208-214`).

**No fine deskew of pixels.** The only rotation in the recognition path is the orthogonal one above. No other rotation call exists in `linerec.cpp`, `control.cpp` or `lstm/lstmrecognizer.cpp`, checked by grep. A skewed line is therefore cropped axis-aligned, and its slope is left to the network input step, which is not described here.

**Training.** `GetLineData` uses the same `GetRectImage` with the same padding to build training lines (`src/ccmain/linerec.cpp:132-156`, called from `116`).

## 3. Inversion and other per-line retries

**Settings.** `tessedit_do_invert` defaults to true, and is marked deprecated (`src/ccmain/tesseractclass.cpp:66-70`). `invert_threshold` defaults to 0.7 (`src/ccmain/tesseractclass.cpp:71-73`).

**Gate.** `LSTMRecognizeWord` passes `invert_threshold` when `tessedit_do_invert` is set, and 0 otherwise (`src/ccmain/linerec.cpp:251-253`). The wrapper `RecognizeLine` calls the core with `re_invert = false` and `upside_down = false` (`src/lstm/lstmrecognizer.cpp:279-286`).

**Core logic.** The core prepares the inputs (`src/lstm/lstmrecognizer.cpp:358`) and runs one forward pass (`src/lstm/lstmrecognizer.cpp:378`). If `invert_threshold > 0` and the mean of the best non-null output is below the threshold (`src/lstm/lstmrecognizer.cpp:380-383`), it inverts the prepared image with `pixInvert` (`src/lstm/lstmrecognizer.cpp:388`) and runs a second forward pass (`src/lstm/lstmrecognizer.cpp:389-390`). It keeps the inverted result only if its mean is higher (`src/lstm/lstmrecognizer.cpp:393-400`). The `re_invert` branch (`src/lstm/lstmrecognizer.cpp:401-406`) is not used on this path.

**Statistic.** `OutputStats` takes the best non-null label probability at each time step, scales it by `INT8_MAX`, and returns the mean (`src/lstm/lstmrecognizer.cpp:325-347`). If there are no non-null outputs, the mean is 0 and the inversion is always tried (`src/lstm/lstmrecognizer.cpp:338-341`).

The inversion is global to the prepared line image. It is not per-pixel, and it happens after the input preparation, not on the raw crop.

**Other retries.** `classify_word_pass1` skips the LSTM for `odd_size` words in combined mode (`src/ccmain/control.cpp:1511`). If the LSTM returns no words in combined mode, it falls through to the legacy classifier (`src/ccmain/control.cpp:1513-1516`). In `OEM_LSTM_ONLY` mode it gives a fake word and does not retry (`src/ccmain/control.cpp:1517-1519`). The dictionary certainty passed to the decoder is a constant (`src/ccmain/linerec.cpp:254`), not a retry.

Page orientation is a separate step. `OrientationIdToValue` maps OSD ids to 0, 90, 180 or 270 degrees (`src/ccmain/osdetect.cpp:560-572`).

## 4. Page layout, and failure modes that follow from the binary image

**Pipeline.**
- `SegmentPage` runs `AutoPageSeg` for the OSD and block-finding modes (`src/ccmain/pagesegmain.cpp:101`, `137-140`).
- `find_components` runs on `pix_binary_` (`src/ccmain/pagesegmain.cpp:309`). It calls `extract_edges` for each text block (`src/textord/tordmain.cpp:225-244`, `237`). `extract_edges` traces outlines on the thresholded image and builds blobs (`src/textord/edgblob.cpp:351-362`).
- The photo mask comes from `ImageFind::FindImages(pix_binary_)` (`src/ccmain/pagesegmain.cpp:293`).
- `ColumnFinder` builds tab vectors when column finding is on (`src/ccmain/pagesegmain.cpp:329-334`; `src/textord/colfind.cpp:343-352`). When it is off, the deskew is the identity (`src/textord/colfind.cpp:343-346`; `src/textord/tabfind.cpp:449-456`).
- `FindTabVectors` estimates skew from vertical tabs, then rotates the blobs and tab vectors (`src/textord/tabfind.cpp:422-434`, `1287-1318`; `ComputeDeskewVectors` at `1404-1411`).
- `TextordPage` runs `make_rows` (`src/textord/textord.cpp:224`; `src/textord/makerow.cpp:224`), then baseline detection (`src/textord/textord.cpp:231`).

**Grey is used for edge offsets, not for components.** Each blob's edge offset is computed from the grey image and the per-blob threshold (`src/textord/textord.cpp:192`; `src/ccstruct/blobbox.cpp:388-412`; `src/ccstruct/coutln.cpp:736-745`). Components themselves come from the binary.

**Failure modes.**
- **Touching lines and characters.** Components are extracted from the binary (`src/textord/tordmain.cpp:225-237`). Two lines or characters that merge into one outline cannot be separated by a later stage, because the LSTM only sees the row's box.
- **Coloured or shaded backgrounds.** With Otsu, one threshold per channel is used for the whole rectangle (`src/ccmain/thresholder.cpp:382-389`, `321-336`). A gradient or a coloured highlight then produces wrong black and white. Only LeptonicaOtsu and Sauvola adapt locally (`src/ccmain/thresholder.cpp:248`, `273`).
- **Low resolution.** A missing DPI is taken as 70 (`src/api/baseapi.cpp:2036-2043`). The text-size estimate only raises it, and only by `line_size × 10` (`src/ccmain/pagesegmain.cpp:318-326`). Layout then works on sizes that are too small.
- **Skew and rotation.** Deskew is refused above about 30 degrees, because `kCosMaxSkewAngle = 0.866025` (`src/textord/tabfind.cpp:60`, `1290-1292`). It runs only with column finding. Page orientation is limited to multiples of 90 degrees (`src/ccmain/osdetect.cpp:560-572`).
- **`--psm` modes.** The enum is at `include/tesseract/publictypes.h:158-174`. `PSM_OSD_ONLY` runs only `FindLines` (`src/api/baseapi.cpp:1224-1225`). `PSM_AUTO_ONLY` runs only `AnalyseLayout` (`src/api/baseapi.cpp:1219-1223`). The default is `PSM_SINGLE_BLOCK` (`src/ccmain/tesseractclass.cpp:77-79`; `include/tesseract/publictypes.h:166`). `PSM_SINGLE_LINE` (7) and `PSM_SINGLE_WORD` (8) still run layout analysis. The modes that run line finding are the range that `PSM_LINE_FIND_ENABLED` tests (`include/tesseract/publictypes.h:201-206`). `PSM_RAW_LINE` (13) bypasses layout and uses the whole image (`src/ccmain/linerec.cpp:233-236`).

## 5. Ranked improvement candidates for a GPU pipeline

Each candidate gives the current Tesseract behaviour, the expected effect and a measurement. The measurements use the line sets in `data/lines` (synthetic, see `data.md`) and `data/real` (scanned FUNSD and SROIE lines, see `data_real.md`), scored by character error rate (CER) on the text. Neither set has polygon boxes, so the line-detection measures need page-level line labels that do not exist yet. No measurement in this section has been run. Each should use the same weights, so that only the image path changes.

1. **Learned line detector giving grey or colour crops.** Now: lines come from binary components and `make_rows` (`src/ccmain/pagesegmain.cpp:309`; `src/textord/textord.cpp:224`), and crops are axis-aligned from the row box (`src/ccmain/linerec.cpp:238-246`). Expected effect: fixes touching and merged lines and shaded backgrounds, and is likely the largest gain. Measure (not yet run; needs page-level line boxes): line-level AP at IoU 0.5, CER on the matched lines, and split and merge counts, against Tesseract's line boxes on the same pages.
2. **GPU deskew of each line crop by its baseline slope.** Now: crops are axis-aligned, with only orthogonal rotation (`src/ccmain/linerec.cpp:172-206`), and page deskew stops at about 30 degrees (`src/textord/tabfind.cpp:1290`). Expected effect: fewer errors on slanted glyphs and on skewed scans. Measure (not yet run): CER on the line sets with injected rotations of 0, 2, 5, 10 and 15 degrees. `data/lines` already has a `rotate` condition at plus or minus 2 degrees.
3. **Per-line contrast normalisation on the grey crop before the network.** Now: the network reads the original pixels, and any normalisation sits in `PrepareLSTMInputs`, which is not described here (`src/lstm/lstmrecognizer.cpp:358`). Expected effect: helps low-contrast and shaded lines, where the global Otsu of `src/ccmain/thresholder.cpp:382-389` is wrong. Measure: CER on a low-contrast subset, with and without the step.
4. **Inversion decided per line before the forward pass.** Now: one forward pass, then a second one with the whole image inverted if the mean is below 0.7 (`src/lstm/lstmrecognizer.cpp:380-400`; `src/ccmain/tesseractclass.cpp:71`). Expected effect: roughly halves the cost of inverted lines, and may catch the lines the mean rule misses. Measure: fraction of lines inverted, GPU ms per line, CER on inverted lines, and the false-invert rate.
5. **Upscaling small text before height normalisation.** Now: the page path never resamples (`src/ccmain/thresholder.cpp:181`), and small-text handling sits in the input step, which is not described here. Expected effect: better CER for x-heights below about 12 px. Measure (not yet run): CER binned by x-height, with the line sets downsampled to the x-heights that 150, 200 and 300 dpi scans give, comparing super-resolution, bicubic and no upscaling.
6. **Resolution estimate from detected line heights.** Now: a missing DPI falls to 70, and the estimate is `line_size × 10`, only ever raising (`src/api/baseapi.cpp:2036-2043`; `src/ccmain/pagesegmain.cpp:318-326`). Expected effect: correct layout sizes on pages with no DPI metadata. Measure: estimate error on pages with known DPI, and layout F1 or CER on pages without it.
7. **Dewarping, or curved baselines in the crop.** Now: baselines are fitted with `BaselineDetect` (`src/textord/textord.cpp:231`) and spline rows exist (`src/textord/makerow.cpp:2006`), but crops are rectangles (`src/ccmain/linerec.cpp:164-225`). Expected effect: better reading on curved or photographed pages. Measure: CER on a set of pages with known curvature, before and after dewarping.
8. **Per-channel or colour-aware binarisation for coloured backgrounds.** Now: one per-channel Otsu over the rectangle (`src/ccmain/thresholder.cpp:382-411`). The LSTM reads colour, but edge offsets and layout depend on the global threshold (`src/api/baseapi.cpp:2055`). Expected effect: better layout on highlighted or coloured-background pages. Measure: layout precision and recall, and CER on a coloured subset.
9. **Batched line recognition on the GPU.** Now: `classify_word_pass1` recognises one word at a time (`src/ccmain/control.cpp:1499-1512`). Expected effect: throughput only, with no change to accuracy. Measure: lines per second at batch sizes 1, 8 and 32 on the same lines, with output checked to be identical.
10. **Photo mask computed on the grey image.** Now: `FindImages` runs on the binary (`src/ccmain/pagesegmain.cpp:293`). Expected effect: fewer shaded text regions wrongly masked as photos. Measure: region precision and recall on a labelled text and photo set.

## 6. Limits of this note

- Nothing was built or run. Every claim is from reading the source.
- Leptonica internals (`pixOtsuAdaptiveThreshold`, `pixSauvolaBinarizeTiled`, `pixRotateOrth`) are not in this tree.
- The network input normalisation (`Input::PrepareLSTMInputs`, `NetworkIO::FromPix`) is not described here. `src/lstm/lstmrecognizer.cpp:358` is cited only as the call site.
- The resolution estimate uses `line_size`, the median text-partition height, not the x-height.
