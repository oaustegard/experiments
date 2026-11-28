# RESULTS: What the Neural Engine can do beside a busy GPU, on an M5 Max

The Neural Engine is the neural-network accelerator in Apple chips; programs reach it only through
Core ML and Vision. This experiment ran three workloads that those frameworks can place there (a
small cross-encoder that scores a task against a transcript chunk, Whisper large-v3-turbo speech
recognition, and Apple Vision OCR) on one M5 Max (128 GB, macOS 27.2.1, plugged in, Energy Mode
Automatic). Each ran with the GPU idle and then while the GPU decoded Bonsai 2.5 27B with
llama.cpp, so each result says how well the workload does and what it costs the GPU. First run of
[`crossenc_coreml.py`](crossenc_coreml.py), [`whisper_ane.py`](whisper_ane.py) and
[`ocr_ane.py`](ocr_ane.py); one machine, so differences under about 5% are noise.

## Answer

Yes for the cross-encoder, mostly yes for Whisper, usable but weak for OCR. The cross-encoder lands on the Neural Engine (703 of 707 operations) and scores a pair in 0.96 ms, 2.7x faster than the CPU. <!-- @evidence[[results/20261127-1914-Apple-M5-Max/crossenc/summary.json#models.plain-b1.placement.CPU_AND_NE.ops.MLNeuralEngineComputeDevice]] -->
With the GPU decoding, it slows 5 to 9% and the GPU's decode rate drops 7 to 10%; the same model on the CPU slows 44 to 89%.
WhisperKit transcribes at 31.7x realtime with the GPU idle and 27.3x with it busy, at 2.00% word error rate. <!-- @evidence[[results/20261127-1929-Apple-M5-Max/whisper/summary.json#engines.whisperkit.idle.speed_x_realtime]] -->
whisper.cpp is faster idle (59.5x) but falls to 25.7x busy because its CPU decoder slows 2.6x.
Vision OCR runs at 4.9 pages per second idle and takes 20 to 45% longer per page with the GPU busy, but scores 39.8 on Omni Parse Bench against 78.8 for the published Indra OCR row. The loss is in the recognizer, not in our conversion to the benchmark's format. The documents arm's score on all 2,937 pages is 0.398 of 1. <!-- @evidence[[results/20261127-2006-Apple-M5-Max/ocr/opb.json#summaries.docs.score]] -->

What would change this: another workload size, or a power sampler instead of Core ML's plan for placement.

## Findings

1. **The cross-encoder is on the Neural Engine and 2.7x faster than the CPU.** Trained student,
   plain norms: 703 of 707 operations on the Neural Engine, 73% of estimated cost at batch 1 and
   97% at batch 16; the four on the CPU are `greater_equal`, `add`, `select` and `gather` (mask
   and embedding lookup). 0.958 ms per pair against 2.589 on the CPU at batch 1, 1.062 against 1.813
   at batch 16. Both norm variants stay on the Neural Engine, including the 16 `layer_norm`
   operations. CPU-only time at batch 1: 2.589 ms per pair. <!-- @evidence[[results/20261127-1914-Apple-M5-Max/crossenc/summary.json#models.plain-b1.timing.idle.CPU_ONLY.ms_per_pair]] -->
   (`results/20261127-1914-Apple-M5-Max/crossenc/summary.json`, `models.*`)
2. **Core ML on the Neural Engine agrees with PyTorch, for both norm variants.** Largest probability
   difference 3.7e-4 (batch 1) and 8.6e-4 (batch 16) for plain norms, 3.1e-4 and 1.1e-3 for
   prescaled. The fp16 overflow worry (residual stream near 1,100 squared) shows up in PyTorch's
   literal fp16 (plain off by 0.41 in the logit, prescaled by 0.002) and not in Core ML's
   `layer_norm` on the Neural Engine, so the plain variant is enough here. These pairs are invented,
   so this is agreement with PyTorch, not the student's accuracy. PyTorch's literal fp16 error for plain norms: 0.41 in the logit. <!-- @evidence[[results/20261127-1914-Apple-M5-Max/crossenc/summary.json#torch_parity.plain.fp16_max_abs_logit_diff]] -->
   (`results/20261127-1914-Apple-M5-Max/crossenc/summary.json`, `models.*.timing`, `torch_parity`)
3. **A busy GPU costs the Neural Engine model about 5 to 9%, and costs the CPU model 44 to 89%.**
   Batch 1 plain: 0.958 to 1.043 ms (+8.9%) on the Neural Engine, 2.589 to 3.734 (+44%) on the CPU.
   Batch 16: 1.062 to 1.111 (+4.6%) and 1.813 to 3.434 (+89%). The busy batch-1 Neural Engine figure is 1.043 ms. <!-- @evidence[[results/20261127-1914-Apple-M5-Max/crossenc/summary.json#models.plain-b1.timing.busy.CPU_AND_NE.ms_per_pair]] -->
   (`results/20261127-1914-Apple-M5-Max/crossenc/summary.json`, `models.*.timing.busy`)
4. **The Neural Engine model costs GPU decode 7 to 10%.** Three runs (stock, student, student with
   layer-wise weights), batch 16, window long enough to overlap a finished `llama-bench` run:
   45.5 to 40.8, 45.0 to 41.8, 45.1 to 41.7 tok/s. The "during" rate covers a whole `llama-bench` run
   including its model load, and the busy fraction was 0.71 to 0.81, so the true slowdown is
   somewhat larger. Most other windows were too short to produce a rate. The stock run's figure: 40.8 tok/s. <!-- @evidence[[results/20261127-1906-Apple-M5-Max/crossenc/summary.json#models.plain-b16.timing.busy.CPU_AND_NE.gpu_during_tok_s]] -->
   (`results/20261127-1906-Apple-M5-Max/`, `1914` and `1921`, `crossenc/summary.json`)
5. **WhisperKit holds up under a busy GPU; whisper.cpp does not, because of its CPU decoder.**
   Realtime factor idle to busy: WhisperKit 31.7x to 27.3x, whisper.cpp 59.5x to 25.7x, mlx-whisper
   (GPU) 64.7x to 53.5x. In whisper.cpp the CPU decoder went from 9.7 s to 25.2 s over the 982 s of
   audio (2.6x), the Core ML encoder from 5.5 s to 10.3 s (1.9x). GPU decode fell from 45.4 to 39.1
   tok/s while it ran. Word error rate 2.00% for all three; mlx-whisper 2.58% idle and 2.33% busy.
   whisper.cpp busy: 25.7x realtime. <!-- @evidence[[results/20261127-1929-Apple-M5-Max/whisper/summary.json#engines.whispercpp.busy.speed_x_realtime]] -->
   (`results/20261127-1929-Apple-M5-Max/whisper/summary.json`, `engines.*`)
6. **Vision OCR is fast and scores well under half of the best published row.** All 2,937 pages:
   182.6 ms per page median for the text arm (`VNRecognizeTextRequest`), 316.4 for the documents arm
   (`RecognizeDocumentsRequest`). Headline score 29.1 and 39.8 against 78.8 (Indra OCR) and 5.9
   (Tesseract). Documents arm: text 25.2, tables 20.3, layout 73.9; the text arm produces no tables
   by design and scores 0.0 on them. Documents-arm idle median: 316.4 ms per page. <!-- @evidence[[results/20261127-2006-Apple-M5-Max/ocr/summary.json#arms.docs.idle.ms_median]] -->
   (`results/20261127-2006-Apple-M5-Max/ocr/summary.md`, `opb.json`)
7. **The OCR loss is in the recognizer, not in the conversion.** For plain-text tests, the gold text
   is found in Vision's raw output in 47.4% of tests (text arm) and 38.2% (documents arm), and in our
   converted markdown in 47.5% and 36.7%; the conversion lost 6 and 60 of 3,684 tests. For table
   cells (with named neighbours), 88.7% are in the text arm's raw output (and score 0 for lack of
   structure) and 75.3% in the documents arm's. Vision found a table on 525 of the 635 pages that
   have table tests. The match is a lenient substring, so these are upper bounds on the score.
   A failing page shows the mechanism: Vision's table grid was converted faithfully, with digits
   misread (`（742）` for 74.2) and headings mangled. The conversion lost 60 documents-arm tests. <!-- @evidence[[followups/ocr_split.json#arms.docs.present.converter_lost]] -->
   (`followups/ocr_split.json`)
8. **A busy GPU slows OCR by 20 to 45%.** Median ms per page on the same 150 pages, idle to busy:
   text 186.2 to 269.7, documents 325.0 to 389.7. Documents-arm GPU decode 45.1 to 43.8 tok/s
   (busy fraction 0.67). Output was identical on all 150 text pages and on 148 of 150 documents pages.
   Documents-arm busy median: 389.7 ms. <!-- @evidence[[results/20261127-2006-Apple-M5-Max/ocr/summary.json#arms.docs.busy.ms_median]] -->
   (`results/20261127-2006-Apple-M5-Max/ocr/summary.json`, `arms.*.busy`)
9. **The first WhisperKit load on this machine is a 64 s compile; once cached it loads in under a second.**
   WhisperKit first invocation: 64.2 s of model loading, then 0.8 s in the full run. <!-- @evidence[[results/20261127-1734-Apple-M5-Max/whisper/summary.json#engines.whisperkit.first_invocation.load.modelLoading]] -->
   (`results/20261127-1734-Apple-M5-Max/whisper/summary.json`, `engines.whisperkit.first_invocation`)

## Method

The GPU-busy condition ([`common.py`](common.py)) runs back-to-back `llama-bench` decode runs
(`-p 0 -n 512 -r 6`) on Bonsai 2.5 27B PQ2_0 with the PrismML llama.cpp fork; the first run, alone,
gives the GPU's own rate, and the workload runs during the later ones. Each result carries the GPU
rate alone and during, and the share of the measurement inside a run (`gpu_busy_fraction`).
`crossenc_coreml.py`: the Ettin-17m cross-encoder student rebuilt as `ScorerANE` (1x1 convolutions,
one attention head at a time, padded to 384), converted at fp16 for batch 1 and 16, two norm
variants, timed 15 s per configuration for `CPU_AND_NE` and `CPU_ONLY`, placement from
`MLComputePlan`. `whisper_ane.py`: 40 LibriSpeech test-clean windows (982 s), three engines,
word error rate with `jiwer` after Whisper's normaliser. `ocr_ane.py`: all 2,937 Omni Parse Bench
pages through two Vision requests, scored by the benchmark's scorer (omni-parse-bench 1.8.0) beside
the published Indra OCR and Tesseract rows; GPU-busy re-reads 150 pages. Student weights are private
and were read from outside the repository.

## Log

### Round 1: smoke runs, then the full runs, 2026-11-27

Smoke runs (`20261127-1733`, `1734`, `1741`) first, then crossenc stock, student and student with
layer-wise weights (`1906`, `1914`, `1921`), whisper (`1929`), a 250-page OCR run (`1954`; headline
26.1 text arm and 40.8 documents arm) and the full 2,937-page OCR run (`2006`). Plugged in and
Automatic Energy Mode throughout.

What broke. The `MLComputePlan` placement call failed on the first smoke run: the compiled model
was freed before it was read. One line in `crossenc_coreml.py` now keeps the model object alive.
`--grade` crashed on the first OCR run because the sparse checkout did not include `ocria-pack`
(the README lists it); adding it fixed it, and the run was repeated. Hugging Face rate-limited the
dataset check with HTTP 429 once; the client retried by itself. "GPU alone to during" is `None`
where the workload's window was shorter than one `llama-bench` run.

### Round 2: follow-ups, 2026-11-27

whisper.cpp's two timings disagreed under a busy GPU: 0.0389 from the engine's stage timings and
0.021 from wall clock. The wall figure subtracts a one-window reference run, and that run took
19.5 s instead of the 1.6 s it takes idle, for a reason this experiment did not find. The 40-window
wall time (39.6 s, minus 1.3 s load) matches the summed stage times (38.2 s), so the engine figure,
25.7x realtime, is the consistent one and the wall figure is dropped.

The converter-or-recognizer question was settled by `followups/ocr_split.py`, which looks up every
gold test in Vision's raw output and in our converted markdown (Finding 7).
