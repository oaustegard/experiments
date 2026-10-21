# embed-coord state (coordinator-maintained; board = issue #104)

Updated 2026-10-20 evening. Metric: hit@10 on 1,498 synth_test queries over 61,686 posts, source post in top 10. Never mix families.

## ASYM — EG2 docs kept (committed q4 2-bit codes, 768-d index), only the query encoder varies. EG2 q4 = 0.9192
| arm | download | hit@10 | real overlap | owner |
|---|---|---|---|---|
| EG2 q4 (full) | 174 MB | 0.919 | 1.000 | anchor |
| EG2 vocab-pruned 32K | 102.9 MB | 0.917 | | eg2-query-student |
| layer cut k=8, q4 | 44.1 MB | 0.911 | 0.650 | eg2-layer-student |
| k=8, 3-bit table | ~41 MB | 0.910 | 0.649 | vocab |
| k=4, q4 | 29.0 MB | 0.907 | 0.630 | eg2-layer-student |
| k=4, 3-bit table | ~25 MB (measuring) | 0.906 | 0.624 | vocab / eg2-layer-student |
| k=4, 2-bit table | ~24 MB | 0.903 | 0.607 | vocab |
| k=2, q4 | 21.8 MB | 0.8965 | 0.577 | eg2-layer-student |
| k=2, 3-bit table | ~18 MB (measuring) | 0.898 | 0.572 | vocab / eg2-layer-student |
| k=2, 2-bit table | ~16.5 MB | 0.893 | 0.554 | vocab |
| distilled leaf-mt int8 | 24.4 MB | 0.794 | | eg2-query-student (plateau; dead end) |
Elbow at k=4 (k=4 vs k=2 +0.011 hit@10 paired; k=8 vs k=4 +0.004 n.s.). 3-bit table is free on hit@10; QAT does not beat quantizing after training. Runs on onnxruntime-web wasm via standard-op unpack (no GatherBlockQuantized); latency not yet measured.

## SYM — each arm on its own remex 2-bit codes, corpus re-encoded by the arm. EG2 = 0.918
| arm | size | hit@10 | owner |
|---|---|---|---|
| snowflake-arctic-embed-m-v1.5 (teacher) | 109M params | 0.903 | ettin |
| leaf-ir (k16000 t4 w8 graph) | 16.7 MB | 0.893 | ettin; warm 36 ms |
| leaf-ir, runtime-free int8 (own kernel) | 23.76 MB | pending | wasm |
| ettin-32m-distil-gte-v2 (off the shelf) | 32.2M | 0.858 | ettin |
| leaf-mt / all-MiniLM-L6 | | 0.828 / 0.823 | ettin |
| Ettin-17m best off-the-shelf (capemox) | 17.2 MB t8-w8; warm 9.4 ms | 0.777 | ettin |
| Ettin-17m distilled from arctic | | running, ETA ~23:45 EDT | ettin |
Student ceiling vs arctic docs: leaf-ir ASYM-arctic 0.898 (n.s. vs SYM 0.893). ORT wasm runtime = 3.07 MB brotli fixed cost on top of any ORT graph.

## Cross-family read (anchor-relative only)
At ~17-19 MB: EG2 k=2 (ASYM) is -0.021 from EG2; leaf-ir (SYM) is -0.025. Differences: ASYM keeps EG2 docs (768-d codes), SYM re-encodes the corpus but at 384-d halves the index. Decide on total download = model + index.

## Open
- wasm: SYM rx2 hit@10 + MB with/without ORT for leaf-ir fp32/int8/wasm-i8, leaf-mt, minilm; needs a free Mac slot for scoring + uncontended timing.
- ettin: distillation chain on MPS (lock held); general-English vocab k16000 build; screen of gte-tiny, bge-micro-v2, granite-small-r2.
- eg2-layer-student: build 3-bit / 2-bit standard-op unpack in run/export.py, measure bytes, run wasm; then browser latency on a real device.
- Latency for the ASYM builds: nobody has measured them in a real browser yet (headless SwiftShader numbers are void).
