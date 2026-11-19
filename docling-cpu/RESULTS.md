# Document OCR on a CPU-only server: granite-docling, granite-vision-4.1-4b and MinerU2.5-Pro

Asked (Oskar, 2026-11-19): run an OCR model efficiently on a CPU-only server, where memory is what
matters. First IBM's granite-docling-258M. Then IBM's granite-vision-4.1-4b, after a colleague found
docling dropped a title block on a scanned PDF that the 4B read well. Then the most capable OCR model
that runs readily on a CPU, leaving out Datalab's Surya OCR 2.5 because its licence limits commercial
use above $5M revenue.

## Result

| arm | weights on disk (language model + vision encoder) | pages | median s/page | p90 s/page | pages that looped (see Method) | peak memory |
|---|---|---:|---:|---:|---:|---:|
| granite-docling-258M bf16 | 332 + 190 MB | 20 | 32 | 159 | 3 of 20 | 0.87 GB |
| granite-vision-4.1-4b Q4_K_M | 2,100 + 1,162 MB | 8 | 599 | 612 | 5 of 8 | 6.74 GB |
| MinerU2.5-Pro-2605 Q8_0 | 531 + 709 MB | 20 | 126 | 166 | none | 3.70 GB (2.32 GB with its cache off) |

<!-- @evidence[[results/summary.json]] -->

All three ran under llama.cpp (commit `9a080f4`, CPU build) on the same 4 cores, one page at a time.
MinerU2.5-Pro is the one to use on a CPU server. No page looped, it read the scanned and rotated
pages the others got wrong, and it takes about 2 minutes a page on 4 cores with a peak of
about 2.3 GB once llama-server's prompt cache is turned off (3.7 GB after 20 pages with it on). granite-docling's median page is 4 times faster (32 s against 126 s) and it uses a
quarter of the memory, but 3 of its 20 pages fell into repetition loops, and it garbles old scans
and rotated pages. granite-vision-4.1-4b is the wrong tool for whole pages: it has no full-page
transcription task (it was trained for charts, tables and key-value extraction), 5 of its 8 pages
generated at least about 3,300 tokens of looping output, and such a page takes about 10 minutes on
a CPU.

**No accuracy scores here.** These 20 pages show speed, memory and failure modes. Accuracy comes from
[Omni Parse Bench](https://huggingface.co/datasets/datalab-to/omni_parse_bench), a document-parsing
benchmark of 16,288 yes/no tests on 2,937 pages, scored 0-100, published by Datalab (the company
behind Surya). The sibling experiment [`granite-docling/`](../granite-docling/RESULTS.md) scores
docling and the 4B on it with Kaggle GPUs. Datalab's leaderboard gives MinerU2.5-Pro 87.1, from its
own run of the model, not ours.

## What the pages showed

The pages are Omni Parse Bench pages, named by the benchmark's suite and source. Three of the 20
(every page's output is in `results/pages/<arm>/`):

- **An 1880s Red Cross letterhead (`olmo_old_scans-74`).** docling garbled the header ("CHESTER A.
  ANTHUR", "George Kenan, Thurera") and ended in a run of "1 1 1 1". The 4B and MinerU both read the
  title, "The American Association of the Red Cross", and the body text correctly. This is the
  failure the colleague reported for docling on a scanned PDF.
- **A page rotated 270° (`rotation_skew-rs0700454e_rot270`).** docling read the rotated lines as
  vertical fragments ("AUTOPSY REPORT", "NO", "EVIDERIES") and looped until its context ran out.
  MinerU detected the rotation and returned the autopsy report with its toxicology tables. The 4B
  never reached this page.
- **A conference paper page with tables (`olmo_table_tests-66aa22ac`).** The 4B returned one 4-row
  table, 127 characters, and dropped the rest of the page. docling and MinerU returned the full page.

Two of the 4B's looping outputs are 523,659 characters each: about 128 characters a token over its
4,096-token limit, so runs of whitespace tokens. The sibling's GPU run of the same prompt saw the
same loops. It also saw the 4B wrap its answer in an `<md>` tag and put every line in a one-column
Markdown table, which a consumer would have to strip. Both come from the model and the prompt, not
from the CPU build or the 4-bit weights.

## Memory

docling's 0.87 GB and the 4B's 6.74 GB are the peak resident memory of one `llama-mtmd-cli` process
per page, with an 8,192-token context. For the 4B that is 2.1 GB of 4-bit weights, 1.16 GB of fp16
vision encoder, about 0.65 GB of fp16 attention cache, and about 2.8 GB of vision-encoder working
memory for a page split into many 384-pixel tiles.

MinerU runs as one long-lived `llama-server`. Its resident memory was 1.91 GB once loaded and climbed
page by page to 3.70 GB after 20 pages. Most of that growth is llama-server's prompt cache, which
keeps past requests in RAM so a repeated prefix can be reused; MinerU's requests share little. On the
same first 4 pages with `--cache-ram 0`, the peak went 2.25, 2.27, 2.31, 2.32 GB, against 2.29,
2.38, 2.49, 2.58 GB with the cache on, at the same speed (141, 108, 174, 163 s against 136, 102,
177, 165 s). Run it with `--cache-ram 0`; the remaining growth, 12-39 MB a page over those 4 pages,
was not followed further.

## Speed

On 4 cores the 4B spends about 34 s before its first output token (18 s in the vision encoder, 14 s
reading the image and prompt in) and then writes about 7-9 tokens a second, so a page that runs to
its 4,096-token limit takes about 10 minutes. docling spends about 15 s encoding a page as 512-pixel
tiles and then writes DocTags (its own markup of text, tables and layout boxes) quickly; its loop
pages filled the 8,192-token context and `llama-mtmd-cli` stopped with "failed to find a memory
slot". MinerU makes one request to find the page's layout blocks and then one request per block,
20-40 requests a page.

## Method

**Pages.** A seeded sample of 60 of Omni Parse Bench's 2,937 pages (`random.Random(0).sample` of the
benchmark's page list), of which the first 20 were run, covering 17 of its suites. PDFs rendered with
pdfium at 200 DPI, every page scaled to at most 2,048 pixels on the long edge.

**Runs.** `run/sweep.py` runs `llama-mtmd-cli` once per page for docling and the 4B: greedy decoding,
4 threads, an 8,192-token context, at most 8,192 new tokens for docling and 4,096 for the 4B. docling
gets its trained prompt, "Convert this page to docling."; the 4B gets "Transcribe this document page
to Markdown. Include every piece of text exactly as written, in reading order, including headers,
title blocks and footers. Write tables as HTML." `run/mineru_sweep.py` starts one `llama-server` (4
threads, one request at a time) and drives it with MinerU's own client, `mineru-vl-utils` 2.0.9
(`MinerUClient(backend="http-client").two_step_extract`). That client sets its own decoding per
request, including repetition penalties, so MinerU did not decode greedily; its `json2md` makes the
Markdown. Peak memory is `ru_maxrss` of each `llama-mtmd-cli` process, and the server's `VmHWM` for
MinerU. Every page writes one JSON file, and a rerun skips pages already written. Everything from
llama.cpp and MinerU ran inside `scripts/jail.sh` of claude-workspace (no network, no secrets, user
`nobody`).

**Counting loops.** docling: the page's DocTags do not end in `</doctag>` (all 3 are pages that
filled the context). The 4B: the page took over 450 s, which at its decode speed means at least about
3,300 tokens; llama-mtmd-cli printed no token counts, so this is inferred from time. MinerU: no output has a
3-word sequence repeated twice in a row, and no line occurs more than 3 times apart from the `\[`
and `\]` lines around display equations.

**Weights.** IBM's `granite-docling-258M-GGUF` (bf16 + f16 vision encoder) and
`granite-vision-4.1-4b-GGUF` (Q4_K_M + f16 vision encoder); mradermacher's
`MinerU2.5-Pro-2605-1.2B-GGUF` (Q8_0 + Q8_0 vision encoder).

**llama-server settings for MinerU.** Use `--special`: MinerU's layout parser reads `<|box_start|>`
and `<|ref_start|>` tokens, llama-server strips special tokens by default and ignores the client's
request to keep them, and without the flag every page parsed to 0 blocks. Avoid `--image-min-tokens
1024`, which llama.cpp's log recommends for Qwen-VL models: it inflates every small block crop to
1,024 image tokens, about 80 s of prefill each.

**Caveats.** 20 pages, and 8 for the 4B: its arm was stopped at 10-14 minutes a page. Two of its 8
pages ran while I was testing MinerU on the same cores; both were re-run alone and their clean times
are the ones reported.

## Log

### Round 1 (2026-11-19)

Built llama.cpp in the jail; docling bf16 on 20 pages; a 4B Q4_K_M test on two pages (a letter came
out clean, a tax form as `<md> | | |` repeated); a MinerU test on one page (0 blocks until
`--special`; 80 s per block with `--image-min-tokens 1024`, dropped); the 4B Q4_K_M on 8 pages (a Q8_0
arm dropped at Oskar's request); MinerU Q8 on 20 pages; a 4-page MinerU rerun with `--cache-ram 0`
to see where its memory growth comes from.
