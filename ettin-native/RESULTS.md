# RESULTS: Ettin as an in-browser query embedder, download vs retrieval

Run 2026-10-20/21 on Oskar's M1 MacBook Air (8 GB; MPS for training, CPU onnxruntime for
scoring, Chrome wasm for timing), as team member `ettin` of the embedding board
(issue #104). Task, split and metric are eg2-native-small's: 61,686 AI-list Bluesky posts,
1,498 held-out Haiku-synthetic queries, hit@10 = the query's source post in the top 10, each
arm scoring its own remex 2-bit index (SYM). EG2 q4 scores 0.918 on the same index.

![Pareto](results/pareto.png)

## Answer

The frontier for general-purpose embedders is Ettin-17m at 10 to 13 MB and leaf-ir above that.
Ettin-17m, distilled from Arctic-m-v1.5 and then tuned on in-domain query/post pairs, ties
EG2's hit@10 at 13.2 MB total download, EG2's own browser download being about 180 MB.
It queries in 9 ms on one wasm thread and starts in about 0.2 s once the bytes are cached.
leaf-ir with the same tuning is the next point: 19.1 MB at 0.942.

| arm (in-domain tuned unless noted) | download MB | hit@10 [95% CI] | vs EG2, paired | real-query top-10 overlap with EG2 | real-query cos to fp32, p05 / min | wasm warm ms |
|---|---|---|---|---|---|---|
| Ettin-17m, 2-bit table | 10.3 | 0.883 | −0.035 | 0.202 | 0.925 / 0.844 | 9.0 |
| Ettin-17m, 3-bit table | 12.2 | 0.910 | −0.008 | 0.230 | 0.981 / 0.971 | 9.5 |
| **Ettin-17m, int4 table** | **13.2** | **0.918 [0.903, 0.931]** | **+0.000** | 0.229 | 0.992 / 0.982 | **9.0** |
| Ettin-17m, int8 table | 19.5 | 0.918 | +0.000 | 0.231 | 0.999 / 0.999 | 9.2 |
| **leaf-ir, int4 table** | **19.1** | **0.942** | **+0.024** | 0.257 | 0.996 / 0.993 | 24.4 |
| leaf-ir, int8 table | 25.6 | 0.939 | +0.021 | 0.259 | 1.000 | 24.2 |
| leaf-ir, stock int8, untuned (wasm #29) | 19.2 | 0.889 | −0.029 | 0.225 | — | 24 (ORT) / 10.8 (leaf-wasm) / 5.0 (leaf-gpu) |
| EG2 q4 | ~180 (WebGPU only) | 0.918 | — | 1 | — | — |

Download = model + tokenizer.json at brotli-11, plus 3.07 MB for the onnxruntime-web wasm
runtime. Matmuls in every row are weight-only int8 per output column with codes held to ±63.
"Paired" is a bootstrap over the 1,498 queries. Ettin int4 minus leaf-ir int4 is
−0.024 [−0.035, −0.013]. Sources: `results/onnx_{e17c,leafirc}.json`,
`results/browser.json`, `results/pareto.json`, per-query arrays in `results/*_perquery.json`.

## Against the 4-layer EG2 cut (eg2-layer-student, release eg2-layer-onnx-20261021)

Paired on the same 1,498 queries; per-query arrays from both experiments. The cut is ASYM: it
embeds queries into EG2's own index, so the corpus is still encoded by full EG2 offline. Ettin
is SYM, using its own index at the same 200 B/doc.

| | download (brotli + 3.07 MB ORT) | hit@10 | minus k=4 cut, paired | real-query top-10 overlap with EG2 | per query |
|---|---|---|---|---|---|
| EG2 k=4 cut, 3-bit table (`model_t3w`, what image search ships) | 25.8 (22.7 without ORT, WGSL engine) | 0.907 | — | 0.62 | 38.8 ms ORT wasm; 0.7 ms WGSL on Metal |
| EG2 k=4 cut, q4 (`model_q4w`) | 29.5 | 0.907 | — | 0.63 | — |
| Ettin-17m tuned, int4 table | 13.2 | 0.918 | +0.011 [−0.005, +0.025] | 0.23 | 9.0 ms ORT wasm |
| Ettin-17m tuned, 3-bit table | 12.2 | 0.910 | +0.003 [−0.013, +0.017] | 0.23 | 9.5 ms ORT wasm |
| Ettin-17m tuned, 2-bit table | 10.3 | 0.883 | −0.025 [−0.041, −0.008] | 0.20 | 9.0 ms ORT wasm |
| leaf-ir tuned, int4 table | 19.1 | 0.942 | +0.035 [+0.021, +0.048] | 0.26 | 24.4 ms ORT wasm |

For text search over these posts, Ettin-17m with an int4 or 3-bit table statistically ties
the cut. It does so at half the download and about 4× faster on wasm (9 vs 38.8 ms). Its
index is also cheap to rebuild: Ettin encodes the 61,686 posts in about 90 s on the M1 CPU.
The cut wins where staying in EG2's space matters. Its real-query rankings overlap EG2's
0.62 against Ettin's 0.23. It can query an EG2 index that is already built, including the
multimodal image index (though text→image hit@10 drops from 0.803 to 0.563, wasm #62). On a
GPU, the WGSL engine makes it the fastest arm here at 0.7 ms. leaf-ir tuned beats both
(+0.035 over the cut) at 19.1 MB.

## Findings

1. **No off-the-shelf Ettin embedder is competitive.** Screened 17M/32M/68M community
   fine-tunes score 0.23 to 0.86. The best 17M, capemox pretrain-50m, scores 0.777, and the
   best overall, much1na 32m distil-gte, scores 0.858. leaf-ir scores 0.893. Source:
   `results/screen.json`.
2. **LEAF-style distillation alone falls short at 17M.** From the MLM base at lr 3e-4,
   epoch 1 reached 0.503. From capemox at lr 1e-3, epochs 1 to 3 reached 0.674, 0.770 and
   0.802, still rising (ASYM-arctic 0.832, doc cosine to the teacher 0.894). The teacher
   itself scores 0.903. This matches the board's fixed-space plateau lesson.
3. **In-domain contrastive tuning does most of the work, for any model.** MNRL on 23,985
   synth_train pairs (posts disjoint from the test pool, asserted) ran 2 epochs at about
   5 min per epoch. It moved Ettin from 0.802 to 0.921, and leaf-ir, the control, from 0.893
   to 0.937. Real-query overlap with EG2 also rose (0.174 to 0.230, 0.225 to 0.258), so the
   gain is not only Haiku style. The Ettin vs leaf-ir gap shrank from −0.091 to −0.016.
4. **Most compression is free.** Weight-only int8 matmuls lose nothing measurable; ORT's
   dynamic int8 does lose (median cos 0.93, batch-dependent). Codes held to ±63 cut brotli
   bytes about 13% at no measured cost (wasm #49). An int4 table also costs nothing measured.
   A 3-bit table costs −0.008 and a 2-bit table −0.035; these use eg2-layer-student's unpack,
   vendored verbatim in `run/lowbit_vendor.py`. Every variant runs on onnxruntime-web wasm with
   standard ops only, and browser vs Python vectors agree at cos ≥ 0.9998.
5. **Vocab pruning is not general-purpose, and is worse on Ettin's byte-level BPE.** The
   16K-token prune still scores 0.879 (Ettin) and 0.939 (leaf-ir) on synthetic queries. But
   real MS MARCO queries fall to cos p05 0.75, min 0.51 (Ettin) and p05 0.81, min 0.33
   (leaf-ir). This held even with general-English counts mixed into the kept set. At similar
   bytes, the 2-bit full-vocab table is far more robust (min 0.84). The hollow points in the
   plot mark the pruned arms.
6. **Ettin-17m's advantage is shape.** It has 3.9M transformer parameters plus a
   12.9M-parameter token table, against leaf-ir's roughly 11M plus 11.7M. The table quantizes
   well and the transformer is small. So it is 6 MB lighter than leaf-ir at equal precision
   and 2.7× faster per query on wasm (9 vs 24 ms on ORT; leaf-wasm's hand-written kernels
   reach 10.8 ms and WebGPU 5.0 ms for leaf-ir). Cold start, once cached, is 0.17 to 0.25 s
   for both, so on a fresh visit the download dominates.

## Caveats

- Training and test queries come from the same Haiku prompt, so hit@10 after tuning likely
  overstates real-query quality. Real-query overlap with EG2 (no labels) is reported beside
  every row. A labelled real-query set remains the missing evaluation.
- The corpus posts are in the distillation texts (transductive, teacher vectors only, no
  labels). The leaf-ir control did not get that distillation step, only the MNRL step.
- The EG2 layer cuts in the plot are ASYM (queries into the EG2 index, eg2-layer-student
  numbers) and are drawn hollow. They are a different metric family from these SYM arms.
- Timing was taken on an M1 Air in Chrome, uncontended, and with files served from localhost.
  Network time is excluded from the cold-start figures.

## Reproduce

`run/screen.py` → `run/teacher.py` → `run/distill.py e17-cap --base capemox/ettin-encoder-17m-pretrain-50m --epochs 3 --lr 1e-3`
→ `run/contrast.py e17-cap-c out/students/e17-cap/epoch3` (and `leafir-c MongoDB/mdbr-leaf-ir`)
→ `run/export.py <ckpt> <name> t4-w8L63 t3-w8L63 ...` → `run/eval_onnx.py <name>` → `web/bench.html`
→ `run/pareto.py`. Models stay in `out/` (not committed).

## Context

This was one slice of a multi-session embedding effort coordinated on a shared board.
The whole picture (which session holds what, the SYM vs ASYM metric split, the shared
cpu.lock protocol and every cross-session finding) lives in
[`embed-coord/STATE.md`](../embed-coord/STATE.md) and issue #104.
Sibling slices: `eg2-native-small` (native small-embedder baseline this builds on),
`leaf-wasm` (runtime-free leaf-ir in wasm/WebGPU), `eg2-query-student` and
`eg2-layer-student` (ASYM query encoders into EG2's space), and
`embeddinggemma-vocab-prune` (token-table quantization and pruning).
Large artifacts (student checkpoints, ONNX builds, teacher vectors) are listed in
[`ARTIFACTS.md`](ARTIFACTS.md); they are not in git.
