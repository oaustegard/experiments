# pdf-text-extractor: streaming + parallelism throughput

## Goal

Verify that adding page-level streaming to
`web-utilities/pdf-text-extractor.html` (a) actually surfaces pages before the
whole PDF is done and (b) that a bounded worker pool over pdf.js gives real
throughput wins, not just paper-parallelism.

## Setup

- Local static server (`python3 -m http.server 4001`) rooted at the
  oaustegard.github.io repo, so the modified extractor page is served as-is.
- Playwright + headless Chromium. All requests to
  `cdnjs.cloudflare.com/.../pdf.min.js`, `.../pdf.worker.min.js`, and the
  `?url=...` PDF are intercepted and answered from `vendor/` so the run has no
  external network dependency.
- Three PDFs, all arXiv:
  - `test.pdf` — 8 pages (short model card)
  - `big.pdf` — 15 pages ("Attention Is All You Need")
  - `bigger.pdf` — 75 pages (GPT-3 paper)
- Each PDF x concurrency in {1, 2, 4, 8}, extracted end-to-end via the URL API
  (`#url=…&concurrency=…&format=markdown`). Test asserts total elapsed,
  page ordering, and captures a "pages visible in the output DOM" timeline.

## Results

### Wall-clock (end-to-end, includes pdf.js worker init + fetch)

| PDF          | conc=1  | conc=2  | conc=4  | conc=8  |
|--------------|---------|---------|---------|---------|
| test (8 p)   | 1.31 s  | 1.11 s  | 1.10 s  | 1.08 s  |
| big (15 p)   | 1.45 s  | 1.46 s  | 1.28 s  | 1.29 s  |
| bigger (75 p)| 4.31 s  | 3.49 s  | 2.39 s  | 1.87 s  |

### Extraction-only (progress line, excludes worker init)

| PDF          | conc=1  | conc=2  | conc=4  | conc=8  |
|--------------|---------|---------|---------|---------|
| test (8 p)   | 33.7    | 37.4    | 46.9    | 49.4    |
| big (15 p)   | 27.6    | 30.8    | 39.9    | 40.6    |
| bigger (75 p)| 23.9    | 30.3    | 57.2    | 90.7    |

(pages/second, higher is better)

### Streaming behavior (pages rendered vs t_ms) — `bigger` at conc=1

```
296:0 1062:1 1217:9 1375:18 1535:25 1690:30 1858:35 2020:39 2195:43
2387:47 2578:51 2738:54 2899:57 3075:60 3281:63 3446:66 3638:69
3876:72 4076:74 4229:75
```

First page visible at t≈1s (after pdf.js worker cold-start + PDF fetch). Then
pages arrive every ~150-200ms in strict order, all the way to the end. The
"stare at a spinner" UX is gone: for a 75-page paper, the user starts reading
at ~1s instead of ~4s.

## Findings

1. **Streaming works and is worth doing.** Time-to-first-page is the same
   across concurrency settings (~1 s, dominated by worker cold-start + fetch);
   the win is that pages 1..n become readable while the tail is still being
   extracted, cutting perceived latency roughly in half for anything > 20 p.

2. **Parallelism gives real throughput on non-trivial PDFs.** On the 75-page
   GPT-3 paper, extraction-only throughput scales 23.9 → 30.3 → 57.2 → 90.7
   pages/s from conc=1 to conc=8 — a **3.8× speedup**. Wall-clock drops
   4.31s → 1.87s (**2.3×** including fixed worker-init cost).

3. **Small PDFs plateau early.** Once the pool has more workers than pages
   (or the total work is <1s of extraction), extra concurrency does nothing —
   the 8- and 15-page PDFs converge by conc=2/4. That matches expectation:
   parallelism is buying pipelining, and there's nothing to pipeline once
   everything is in flight.

4. **Page ordering is preserved.** Every run passed `ordered=true`; the
   `nextToFlush` cursor pattern (buffer out-of-order results, emit contiguous
   head) works as intended.

5. **Default of 4 is a good compromise.** On typical papers it captures most
   of the win with less memory pressure than 8. Power users can pin
   `&concurrency=8` in the URL for large PDFs; the tool reports elapsed
   seconds and pages/second at the end of each run so A/B is one line change.

## Reproduce

```bash
cd /home/user/oaustegard.github.io
python3 -m http.server 4001 > /tmp/srv.log 2>&1 &

cd /home/user/claude-workspace/experiments/pdf-streaming-test
# vendor/ already contains pdf.min.js, pdf.worker.min.js, and the three PDFs
node test_streaming.js
```
