# RESULTS: native small embedders vs EmbeddingGemma 2 on the Bluesky task

Run 2026-10-20 on Oskar's MacBook Air (M1, 8 GB, CPU onnxruntime 1.31, 4 threads).
Script: `run/native.py`; numbers: `results/native.json`.

## Question

#102 and #103 asked whether a small model can embed queries into EG2's space. This asks the
prior question: does a ~23 MB model with its own index (docs and queries both) do the task
well enough to replace EG2 in the browser?

## Answer

Nearly. **mdbr-leaf-ir** (23.9 MB int8, 768-d) finds the source post in the top 10 for
0.889 of the 1,498 held-out synthetic queries using its own 2-bit remex codes (200 B/doc).
EG2 q4 (174 MB) scores 0.918 at 192 B/doc. The paired difference is -0.029 [-0.045, -0.015].
That is close to #103's 4-layer EG2 cut (29 MB, 0.90, -0.019). The two were not compared
paired, and their CIs overlap.

| Arm | Model MB | dim | B/doc | hit@10 (rx2) | minus EG2, paired | hit@1 | real-query top-10 agreement with EG2 |
|---|---|---|---|---|---|---|---|
| EG2 q4 (committed codes) | 174 | 768 | 192 | 0.918 [0.903, 0.932] | — | — | — |
| leaf-ir int8 | 23.9 | 768 | 200 | 0.889 [0.872, 0.905] | -0.029 [-0.045, -0.015] | 0.732 | 0.213 |
| leaf-mt int8 | 24.0 | 1024 | 264 | 0.828 [0.809, 0.847] | -0.089 [-0.107, -0.071] | 0.652 | 0.209 |
| MiniLM-L6 int8 | 23.7 | 384 | 104 | 0.823 [0.803, 0.842] | -0.095 [-0.114, -0.077] | 0.656 | 0.215 |

fp32 doc vectors add at most +0.007 hit@10 over 2-bit codes for every arm (fp32 leaf-ir 0.893).

## Findings

1. **The retrieval-tuned variant matters more than anything else measured.** leaf-ir beats
   leaf-mt by 0.06 hit@10. They share an architecture, tokenizer and size. Earlier
   benchmarks (bekko, remax_kb) used only leaf-mt.
2. **The 2-bit codes cost almost nothing here,** at most 0.007 for any arm.
3. **Small models build the whole index cheaply.** 61,686 posts in 104-113 s on the M1
   CPU, about 580 docs/s. Query latency is about 1 ms median, single query, 4 threads.
   EG2 q4 needs WebGPU (no wasm GatherBlockQuantized), while these run on wasm/CPU.
4. **Small models and EG2 rank very differently on real queries.** Their top-10 lists share
   about 0.21. That is lower than the distilled students (0.48-0.59), which were trained to
   imitate EG2. Without relevance labels for real queries, this cannot say which ranking is
   better.

## Caveats

- Haiku wrote the synthetic queries from single posts, so they likely share words with
  their source. That favours any model and shrinks gaps. A harder or human-labelled query
  set could separate leaf-ir from EG2 more.
- The EG2 row recomputes 0.918 from the committed codes. #102 reports 0.919 for the same
  arm, and the difference is the bootstrap mean.
- Latency was measured on an M1 CPU, not in a browser.
