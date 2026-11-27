# ane-workloads

Three benchmark scripts for an Apple Silicon Mac. Each runs a workload that Core ML or
Vision can place on the Neural Engine (the neural-network accelerator in Apple chips, which
programs reach only through those frameworks), and measures it twice: with the GPU idle,
and while the GPU decodes with a 27B language model. The question is which useful work the
Neural Engine can do beside a busy GPU, how well, and at what cost to either side.

**Status: written, not yet run.** The scripts need a Mac (Oskar's M5 Max, 128 GB, macOS 26).
They were written in a Linux container, where the parts that do not need a Mac were run;
"What was checked" below says which.

| script | workload | quality measure | speed measure |
|---|---|---|---|
| `whisper_ane.py` | Whisper large-v3-turbo: WhisperKit (Core ML encoder and decoder), whisper.cpp with a Core ML encoder and a CPU decoder, mlx-whisper on the GPU as the contrast | WER on LibriSpeech test-clean | realtime factor; first load apart from later loads |
| `ocr_ane.py` | Apple Vision: `VNRecognizeTextRequest` (accurate) and `RecognizeDocumentsRequest` (macOS 26, through `docs_ocr.swift`) | Omni Parse Bench score, same pages and grader as `indra-browser`, `ocria-pack` and `nemotron-parse-pack` | ms per page |
| `crossenc_coreml.py` | the `ane-context-filter` student (an Ettin-17m cross-encoder that scores a task against a transcript chunk), converted to Core ML at fixed shapes | agreement with the PyTorch model | ms per (task, chunk) pair, CPU + Neural Engine against CPU only; where Core ML places each operation |

## Run

On the Mac, plugged in, Energy Mode set to Automatic or High Power. Each script refuses
battery power and Low Power Mode unless given `--force`, and writes the chip, macOS
version, power source and energy mode into `env.json` beside its results.

```sh
git clone --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments
cd experiments
git sparse-checkout set _lib ane-workloads bonsai-metal-bench ocria-pack nemotron-parse-pack
cd ane-workloads
brew install cmake python@3.12 whisperkit-cli    # plus the Xcode command line tools

python3.12 crossenc_coreml.py                                   # stock Ettin-17m, random head
python3.12 crossenc_coreml.py --weights ~/path/to/student17-fp16.pt   # the trained student
python3.12 whisper_ane.py                                       # 40 windows, three engines
python3.12 ocr_ane.py --grade                                   # all 2,937 pages, both arms, scored
```

Each script makes its own venv under `work/` (gitignored) and writes
`results/<timestamp>-<chip>/<whisper|ocr|crossenc>/` with `summary.md`, `summary.json`,
`env.json` and `raw/`. Commit that directory, or paste `summary.md` back.

Shorter runs to check that everything starts:

```sh
python3.12 crossenc_coreml.py --batches 1 --seconds 3 --no-gpu-busy
python3.12 whisper_ane.py --windows 4 --only whisperkit --no-gpu-busy
python3.12 ocr_ane.py --sample 40 --busy-pages 10
```

The student's weights are private (`ane-context-filter/models/student17-fp16.pt` on the
private `ctxfilter-data` branch of `oaustegard/claude-workspace`). Fetch that file to
somewhere outside this checkout and pass its path. Without `--weights` the script runs
stock `jhu-clsp/ettin-encoder-17m` with a random head: placement and timing are the same,
the scores mean nothing.

Downloads, roughly: the GPU load's GGUF (about 6 GB, shared with `bonsai-metal-bench/work`
if that benchmark has run), Whisper models (about 6 GB over the three engines), LibriSpeech
test-clean (346 MB), Omni Parse Bench pages (1.4 GB).

## The GPU-busy condition

`common.py` runs back-to-back `llama-bench` decode runs (`-p 0 -n 512 -r 6`) on Bonsai 2.5
27B PQ2_0 with the PrismML llama.cpp build, both taken from `bonsai-metal-bench` (it calls
that script's `build_fork` and `hf_download`, so nothing is built or downloaded twice). The
first run finishes alone and gives the GPU's own decode rate. The workload then runs during
the later ones, and each result carries the GPU's rate alone and during the workload.

`llama-bench` reloads the model for every run, so the GPU idles for a few seconds between
runs. Each result reports `gpu_busy_fraction`, the share of its measurement that fell
inside a run and past the first `--gpu-settle` seconds of it (default 20, a guess at model
load plus warm-up: check it against the load time in `raw/gpu-load.json`).
`--llama-bench` and `--gpu-model` substitute another build or GGUF; `--no-gpu-busy` skips
the condition.

## What each script does

### `whisper_ane.py`

Audio: consecutive utterances of one LibriSpeech test-clean chapter are concatenated into
windows of at most 29 s (Whisper's encoder always processes 30 s, so a short clip costs as
much as a full one); 40 windows spread evenly over the set, about 16 minutes of audio.

Per engine: one invocation on one window first (the first Core ML load on a machine compiles
the model for the Neural Engine and macOS caches the result), then for each GPU condition
one window alone and all windows in one process. Reported: WER with `jiwer` after Whisper's
English text normaliser on both sides, realtime factor from the engine's own stage timings
with the load excluded, the same from wall-clock (all windows minus one window), and the
load times.

- **WhisperKit**: `whisperkit-cli transcribe --model large-v3-v20240930_turbo
  --audio-encoder-compute-units cpuAndNeuralEngine --text-decoder-compute-units
  cpuAndNeuralEngine --chunking-strategy none --concurrent-worker-count 1 --report`.
  Timings come from the per-file report JSON (`timings.fullPipeline`, `modelLoading`).
- **whisper.cpp** at commit `79b1191`, built with `-DWHISPER_COREML=1`, run with `-ng
  -bs 1 -bo 1` (greedy, like the other two). Upstream loads the Core ML encoder with
  `MLComputeUnitsAll`, which leaves Core ML free to choose the GPU; the script changes that
  one line in `src/coreml/whisper-encoder.mm` to `MLComputeUnitsCPUAndNeuralEngine` before
  building. `--whispercpp-units all` builds upstream's version instead. The encoder is the
  prebuilt `ggml-large-v3-turbo-encoder.mlmodelc.zip` from `ggerganov/whisper.cpp` on
  Hugging Face, so the Python 3.11 conversion step is not needed.
- **mlx-whisper** 0.4.3 with `mlx-community/whisper-large-v3-turbo`, fed decoded audio so
  it needs no ffmpeg.

"First load" is a compile only if this model has never been loaded on this machine. The
script records whether its model folders existed before it started; it cannot clear the
OS cache.

### `ocr_ane.py`

Pages: Omni Parse Bench (`datalab-to/omni_parse_bench`, 2,937 pages, pinned revision). PDF
pages are rasterised with pdfium at 200 DPI, as `indra-browser/run/prerender.py` does, in
colour. The timed part is Vision's perform call on a decoded image in a warm process.

- **text arm**: `VNRecognizeTextRequest`, accurate, language correction on, automatic
  language detection, through pyobjc. It returns lines with boxes. `adapt_text` cuts
  Vision's line order into paragraphs (a gap over 0.7 of a line height, a line that starts
  above the previous one, or no horizontal overlap) and emits one text block per
  paragraph. It produces no tables, so table tests fail by construction.
- **docs arm**: `RecognizeDocumentsRequest`, which Apple's documentation lists from macOS
  26.0. It is a Swift struct with no Objective-C counterpart, so `docs_ocr.swift` is
  compiled with `swiftc` and fed image paths over a pipe. `adapt_docs` writes paragraphs,
  tables as HTML with row and column spans, lists, and the title as a heading.

The full page set runs once with the GPU idle and that run is graded. GPU busy re-reads 150
of the same pages and compares their times with the idle run's (and counts pages whose
output changed). `--device ne|gpu|cpu` asks Vision for a compute device on every stage
that offers it; the default leaves the choice to Vision and records what it supports.

Scoring uses `ocria-pack/run/grade.py` with the benchmark's scorer
(`omni-parse-bench` 1.8.0). `--grade` installs the scorer in a venv and runs it on the Mac,
paired against the published Indra OCR and Tesseract rows. Without `--grade`, commit the
results directory and grade it in a hub session inside `scripts/jail.sh`. Raw Vision output
is kept in `<arm>/raw.jsonl.gz`; `python3 ocr_ane.py --readapt results/<run>/ocr` rebuilds
the page JSONs from it on any machine, so the two adapters can be changed without the Mac.

Numbers to set a result beside, all on this benchmark's headline (mean of the text, tables
and layout pass rates): Indra OCR fp32 with math off 61.51 on all 2,937 pages
(`indra-browser`), 67.80 with math on; OCRia-nano-1 67.66 against the published Indra row's
71.70 on 2,544 pages (`ocria-pack`); the Nemotron Parse arms on a 250-page sample
(`nemotron-parse-pack`, `--pages ../nemotron-parse-pack/run/sample250.txt` selects it). The
published results have no Apple Vision row.

### `crossenc_coreml.py`

The student is `train.py`'s `Scorer`: ModernBERT encoder (Ettin-17m: 7 layers, width 256),
the [CLS] vector through a Linear, GELU, LayerNorm, Linear head, at most 384 tokens. The
script rebuilds it as `ScorerANE` in the layout that kept `ettin-ane`'s model on the Neural
Engine: activations `(B, C, 1, S)`, every linear layer a 1x1 convolution, attention one
head at a time, the sequence padded to 384 with an additive -1e4 on padded keys, RoPE
tables and the local-attention mask as constants, the head on the [CLS] column only.

It then (1) checks the rewrite against the Hugging Face model in PyTorch on padded batches
and against the same model run unpadded, and stops if either differs by more than 1e-4 in
the logit; (2) converts with coremltools at fp16 for batch 1 and batch 16, 384 tokens,
fixed shapes; (3) on macOS, for `CPU_AND_NE` and `CPU_ONLY`: counts operations per device
with `MLComputePlan` and lists those not on the Neural Engine, loads twice, compares the
output with PyTorch, and times `predict` for 15 s per configuration, GPU idle and busy.

**Two norm variants are converted and timed.** The residual stream of Ettin-17m reaches
about 1,100 after centring, and the square of that is past fp16's largest value (65,504).
"plain" is the textbook norm; coremltools recognises it and emits 16 `layer_norm`
operations, the form `ettin-ane` ran on an M1 with matching answers. "prescaled" divides
the centred input by a constant power of two (1/64 here, set from sample pairs) before
squaring, which leaves 4 `layer_norm` operations and 12 written out as `reduce_mean` and
`rsqrt`. Evaluated literally in fp16 in PyTorch, plain is off by 0.38 in the logit and
prescaled by 0.003. Whether Core ML's own `layer_norm` has that problem on the Neural
Engine, and which variant stays on it, is for the Mac run to show.

The parity check uses invented (task, chunk) pairs in the student's input format. The real
pairs come from session transcripts and are private; `--pairs file.jsonl` (`{"task",
"chunk"}` per line) uses them without anything being committed.

## What was checked in the container (2026-11-27)

- **`crossenc_coreml.py` ran end to end up to the conversion**, in a venv with the pinned
  versions (torch 2.7.0, transformers 4.57.6, coremltools 9.0). Stock weights, 28 pairs of
  50 to 384 tokens: largest logit difference from the Hugging Face model 1.1e-6 padded and
  1.3e-6 against the unpadded model. With the private student weights: 1.6e-6 and 2.9e-6.
  All four packages converted and saved (706 operations plain, 790 prescaled, 32.8 MB
  each). `results/20261127-1717-x86_64/crossenc/` is the stock run.
- **`MLComputePlan`, `load_duration_in_nano_seconds` and the other coremltools names** were
  read from the installed coremltools 9.0 source.
- **Audio windows and WER**: the LibriSpeech download, window building and the
  `jiwer` + `whisper-normalizer` scoring ran. The pinned whisper.cpp commit was fetched and
  the line the script patches is present exactly once.
- **OCR adapters and grading**: synthetic "Vision" output built from the text layer of 40
  benchmark PDFs went through `adapt_text`, `adapt_docs` and `--readapt`, and
  `ocria-pack/run/grade.py` scored both run folders with omni-parse-bench 1.8.0 inside the
  hub's jail, beside the published rows. That checks the file formats and the grading
  path, not Vision.
- **GPU load manager**: `GpuLoad` ran against a stand-in `llama-bench` (a script that sleeps
  and prints llama-bench's JSON): baseline, busy fraction, rate during a window, stop, and
  the path where the binary is missing.

## Checked against upstream sources, not run

The starting notes for this experiment were an unverified model summary. Each claim was
checked against the source named here; corrections are marked.

| claim | source | finding |
|---|---|---|
| WhisperKit CLI flags | `argmaxinc/argmax-oss-swift` `Sources/ArgmaxCLI/TranscribeCLIArguments.swift`, tag v1.1.1 (the Homebrew formula's version) | **Corrected.** The flags are `--audio-encoder-compute-units` and `--text-decoder-compute-units`, not `--encoder-compute-units`. `--report` writes `<audio name>.json` and `.srt` into `--report-path`. `--audio-path` repeats. |
| WhisperKit model name | `argmaxinc/whisperkit-coreml` file list | **Corrected.** There is no `large-v3-turbo`; large-v3-turbo is `openai_whisper-large-v3-v20240930`, and `_turbo` is Argmax's variant of it (`large-v3-v20240930_turbo`). `large-v3_turbo` is the full large-v3. |
| `whisperkit-cli serve` | README, `ArgmaxCLI.swift` | The server is compiled only with `BUILD_ALL=1`; the script uses `transcribe`. |
| whisper.cpp `-DWHISPER_COREML=1`, `--no-gpu`, `-oj` | README, `examples/cli/cli.cpp` at `79b1191` | Confirmed. |
| "only the encoder is on the ANE" | `src/coreml/whisper-encoder.mm` | **Corrected.** The encoder is loaded with `MLComputeUnitsAll`; nothing pins it to the Neural Engine. Hence the patch. |
| Core ML encoder needs a Python 3.11 conversion | `models/download-coreml-model.sh`, Hugging Face file list | A prebuilt `ggml-large-v3-turbo-encoder.mlmodelc.zip` exists. |
| mlx-whisper | PyPI 0.4.3 wheel source | `transcribe(audio, path_or_hf_repo=...)` takes an array; a file path needs ffmpeg. |
| `RecognizeDocumentsRequest` availability | Apple documentation JSON for `vision/recognizedocumentsrequest` | macOS 26.0. Members used by `docs_ocr.swift` (`document.paragraphs`, `tables`, `lists`, `title`, `Table.rows`, `Cell.rowRange`/`columnRange`/`content`, `List.Item.itemString`/`markerString`, `boundingRegion.boundingBox.toImageCoordinates(_:origin:)`) are each documented. |
| `setComputeDevice:forComputeStage:` | Apple documentation; pyobjc-framework-Vision 13.1 metadata | macOS 14.0 on `VNRequest`. pyobjc carries metadata for `supportedComputeStageDevicesAndReturnError:` and `performRequests:error:`. |
| coremltools conversion on Linux | run here | Works; prediction and `MLComputePlan` need macOS. |

## What only the Mac can show

- Every timing, every placement, every WER and benchmark score.
- Whether `docs_ocr.swift` compiles. It was written from the documentation with no Swift
  compiler at hand. If `swiftc` rejects it, `ocr_ane.py` records the error and runs the
  text arm alone.
- Whether the pyobjc calls behave as written: none of them ran. The compute-device
  selection is wrapped so a failure is recorded and the run continues with Vision's choice.
- The order in which `RecognizeDocumentsRequest` returns paragraphs relative to tables and
  lists, and whether `paragraphs` repeats the text inside tables. `adapt_docs` assumes it
  may and drops paragraphs whose centre lies in a table or list. The raw output is kept so
  the adapter can be corrected and rerun.
- Whether Vision's requests run on the Neural Engine at all on this machine: the script
  records the devices each request reports as supported, which is what Vision exposes.
- Whether the fp16 Core ML model agrees with PyTorch on the Neural Engine, for either norm
  variant.
- `llama-bench`'s flags as `bonsai-metal-bench` uses them (that benchmark is also unrun),
  and whether 20 s is the right settle time.
- Watching the Neural Engine directly. The notes suggested `sudo powermetrics --samplers
  ane_power,gpu_power,cpu_power`; I could not confirm the sampler names without a Mac, and
  the scripts do not depend on it.
- The contents of the prebuilt whisper.cpp encoder zip (the script checks that unzipping
  produced `ggml-large-v3-turbo-encoder.mlmodelc` and says so if not).
