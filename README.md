# Experiments

Random vibe-experiments — one-off explorations and data products. Each
subdirectory is self-contained: scripts, data, and a results file
(`RESULTS.md` for pipeline runs, `README.md` for build/recipe artifacts,
or the artifact itself when it speaks for itself, like an HTML page).

**Trust conventions.** [`ANCHORS.md`](ANCHORS.md) registers every published
constant in use with its *covered range*, because a range gap is invisible from
inside a green run. Each experiment should carry an `ERRORS.md` (what was wrong,
how it was caught, which direction it pushed the conclusion — the base rate is
the most useful calibration number about a body of work) and a `recheck.py`
(a sub-5-minute fixture that checks the prose against the artifacts, so the
writeup and the data cannot drift apart between full rebuilds).
`remex-vs-higgs-ablation/` carries all three and is the reference shape.

Results are reported as they came out. Several of these are negative
results, one is a correction of earlier rejected work, and one turned
out to reproduce a 2004 paper rather than extend it — all labelled as
such rather than quietly dropped.

Migrated from `oaustegard/claude-workspace/experiments/`, which is a
session-boot repo and was the wrong home for 37 research projects.

## Index

| Experiment | Started | Status | Results | Origin |
|---|---|---|---|---|
| [`rasp-numeric-select/`](rasp-numeric-select/RESULTS.md) | 2026-09-05 | **done — a numeric `select_at` for RASP: one attention head of head_dim 2 attends to a computed integer address through the parabolic key `(2j, -j^2)`, winner/runner-up gap exactly 1, and the compiled weights carry no compile-time length bound** | [`RESULTS.md`](rasp-numeric-select/RESULTS.md) + `rasp_ns.py` + `compile_ns.py` + `programs.py` + `margin.py` + `test_rasp_ns.py` + `gap_vs_beta.png` | Oskar, extending the LAC line (`oaustegard/llm-as-computer`) after `torchlean-lac/`: RASP/Tracr, ALTA and B-RASP[pos] all reach a position through a categorical predicate over one-hot keys, so a computed address costs one residual dimension per reachable position. `select_at(addr)` reaches it by key geometry instead. Three programs a categorical compiler cannot express without a bounded position table — `y[i] = x[2i+1]`, `y[i] = x[x[i]]`, `y[i] = x[i - x[0]]` — compile to 13, 11 and 19 residual dimensions and match the interpreter exactly at n = 3, 8, 32, 128, 257 and 1000, having never been shown a length; the same gather through a categorical position table costs 55 dimensions at n_max = 8 and 391 at n_max = 64 and is wrong above its n_max. Softmax attention matches average-hard attention within 0.5 from beta 3.5 (`gather`), 3.0–6.25 (`chase`, rising with n because the read values do) and ~9 (`shift`, whose range gate costs about 6 extra units of beta). numpy only, 34 tests, ruff clean. |
| [`torchlean-lac/`](torchlean-lac/RESULTS.md) | 2026-09-05 | **done — LAC's parabolic addressing proved in TorchLean's spec layer, and its float32 capacity ceiling (exact through address 4096, fails from 4097) proved over TorchLean's `FP32` arithmetic; the hard-argmax read cannot reach the graph IR** | [`RESULTS.md`](torchlean-lac/RESULTS.md) + `LAC/Core.lean` + `LAC/Check.lean` + `LAC/Capacity.lean` + `capacity_numerics.py` + `recheck.py` + `ERRORS.md` | Oskar, after the TorchLean survey: *"Can we describe/craft LAC in TorchLean?"* Five lemmas about `score j i = 2*j*i - j*j` elaborate against Mathlib, including `score_gap` (adjacent integer addresses differ by at least 1 in score); `paraKey`/`paraQuery`/`Mem C`/`rowScores` elaborate against a fully built `NN` (4352 jobs, clean). `#print axioms` gives the three standard Lean axioms. Blocked: `NN/IR/Graph.lean`'s `OpKind` has no argmax, gather, or one-hot, so no autograd and no IBP/CROWN on hard-argmax LAC; and `Spec.Tensor.argmax` has no characterization lemma anywhere in `NN/` (the three argmax theorems in the tree are all about `argmaxClassifier`), so read-exactness needs the loop invariant proved first. The binary32 capacity ceiling (`j^2 > 2^24`) is proved in `LAC/Capacity.lean` against `FP32 = NF binaryRadix fexp32 rnd32`: `read_exact_below_ceiling` (every intermediate of the float32 score is exact for `j ≤ i ≤ 4096`) and `read_ties_above_ceiling` (addresses `i-1` and `i` score identically for `4097 ≤ i ≤ 5792`, the first binade), plus `not_both_representable_above_ceiling` for every binade and four bit-level `IEEE32Exec` instances closed by kernel `decide`. numpy float32 confirms the pipeline fails on exactly 4097–5793 and then only sporadically. |
| [`eml-prove2me/`](eml-prove2me/README.md) | 2026-09-05 | **done — 11 EML witness theorems verified on Prove2Me; lower bounds posted open; mission proposal drafted** | [`README.md`](eml-prove2me/README.md) + `Definitions/` + `Theorems/` + `Solutions/` + `mission_description.md` + `scripts/` + `published.json` | Oskar, after the Anthropic FLT formalization post: *"Try the EML table as a Prove2Me mission. Sign yourself (or me if needed) up for an account."* Account `muninn` registered under his email; `EmlComplexity` definition (Tree, size, real eval, valid, Attains, Complexity) published; the eleven real-branch witnesses from eml-sr `benchmarks/eml_complexity.md` (e:1, e-1:2, 0:3, e-2:7, -1:8, 2:9, ln2:12, 3:14, 1/2:17, -3:20, 4:21) proved by exhibiting the enumeration's tree and rewriting with `log_exp`/`exp_log` under pre-proved positivity facts, all ACCEPTED by the server's Lean (v4.33.1, Mathlib 0df444a); `complexity_two`, `not_attains_four_below_twenty_one`, `complexity_four` posted as open theorems; proposal `9a4eba3f` with 13 milestones awaits the human confirm-and-submit step. Lean tactic lessons in the README. |
| [`static-code-embed/`](static-code-embed/RESULTS.md) | 2026-09-03 | **done — no cheap adaptation of a static table reaches rg, let alone bekko; fusing one with rg lowers rg** | [`RESULTS.md`](static-code-embed/RESULTS.md) + `scripts/` + `results.json` + `data/vocab_*.json` | Oskar, on the zvec-grep evaluation: *"isn't part of the attraction ease of fine tuning?"* then *"Run it!"*. Same n=59 sklearn harness as `bekko-embedding-bench`; rg reproduces 0.596/0.682 exactly and bekko-a25m re-encodes to 0.651/0.706. Sibling of `potion-code-quant/` (same day, separate session; that one covers quantization and index compatibility, this one adaptation) — its vanilla potion r@5 0.436 is reproduced here independently. Five adaptations of potion-code-16M-v2, all paired against vanilla: MNRL on 4,223 docstring→code pairs (val loss 1.64→1.14, r@5 0.435, Δ −0.001); +12,276 corpus identifiers as whole-word rows, sum-initialised so epoch 0 equals vanilla (0.439), then MNRL (0.418 / r@10 0.563, n.s.); the same with mean-initialised rows **0.334, p=0.021 worse** — one row replaces k pieces and long identifiers lose k-fold pooling weight; a hand-rolled Model2Vec distillation of bekko-a25m (23.6k tokens, PCA 92%, SIF) 0.249, +MNRL 0.350; `model2vec.distill.distill()` of the same teacher (268,850 rows) **0.052**, +MNRL 0.135. bekko-a25m vs potion **+0.214 r@5, 26/2, p<0.001**; every static arm vs rg p≤0.004; potion RRF with rg 0.537 vs rg 0.596. Static encode 6 s vs bekko 20.6 min for 11,439 chunks. Caveats: identifier-poor stratum n=1; one learning rate for all arms; ast chunking only. |
| [`potion-code-quant/`](potion-code-quant/RESULTS.md) | 2026-09-03 | **done — potion trails bekko-a8m by 0.16 r@5; no static table is a drop-in query encoder for a transformer-built index** | [`RESULTS.md`](potion-code-quant/RESULTS.md) + `run.py` + `distill.py` + `fit_table.py` + `specter2_fit.py` + `check_centered.py` + `results_*.json` | Oskar: test the Model2Vec static code model (minishlab/potion-code-16M-v2, a 63k x 256 fp16 token table, mean-pool) against our embedders and see if its output quantizes; then, can a static table produce embeddings compatible with an index a transformer built, including a remax/remex-quantized SPECTER2 set. On `bekko-embedding-bench`'s n=59 sklearn file-discovery task: **potion fp16 r@5 0.436 vs bekko-a8m 0.595 (Δ −0.159, p<0.001)**; bekko's 96 B remex 2-bit sidecar (0.590) beats potion's uncompressed 512 B vector. potion encodes 80x faster (1.3 vs 104 ms/query) and its own output quantizes about like bekko's: remex 2-bit −0.015 (n.s.), remax 1-bit −0.075 (p=0.10); Model2Vec's native int8 table is free and its `dimensionality=` truncation equals a post-hoc slice. **Token-level Model2Vec distillation of bekko (pca_dims=None) is not space-compatible**: cosine 0.26 to the teacher, a 0.957 common component across all student vectors, r@5 0.017 querying bekko's index. **Fitting the table by ridge regression to the teacher's sentence vectors** gets cosine 0.76 on held-out code (0.60 on NL queries) and r@5 0.36 against bekko's index (teacher 0.60; remex 2-bit index 0.39, remax 1-bit 0.24), 9 MB table. **SPECTER2** (remax bench 10k set, seed 99): the fitted 26 MB table reads cosine 0.96 raw but **0.71 after centering** (SPECTER2's shared mean gives raw pairwise cosine 0.85), sign agreement 0.75 after the index's rotation, and **r@10 0.288 against the teacher's remax 1-bit index where the teacher gets 0.645**; float index 0.264, remex 2-bit 0.117. The quantizer on the index side is not the limiting factor; the student/teacher residual is. |
| [`embedding-inversion/`](embedding-inversion/RESULTS.md) | 2026-09-01 | **done — a vec2text-shaped inverter on bekko-a8m behaves as the paper describes (verifier selection, then one large correction round, then convergence) and at 40k pairs / t5-small / 4 vCPU recovers the exact string 2.4% of the time from the float vector and 0.9% from the 384-bit sign code; not retrieval, not an inverter either, a paraphraser that lands on topic** | [`RESULTS.md`](embedding-inversion/RESULTS.md) + [`ERRORS.md`](embedding-inversion/ERRORS.md) + [`recheck.py`](embedding-inversion/recheck.py) + `results_float.json` / `results_bin1.json` | Oskar: *"How would you craft a reverse embedding model?"*, then *"implement that PoC"*. Pre-registered four predictions before the first stage ran; three held, and the one that decides whether this is an inverter (beat nearest-training-string by 20 points of exact match) failed in both arms. Zero-step base scores cosine 0.55 on its own training set and 0.547 on dev, so the shortfall is underfitting: vec2text's base starts near 0.9 after 5M pairs, this one at 0.55 after 40k. The arm trained on the sign code loses to the float-trained arm even at matching the code's own bits (69.2% vs 71.8%). Exact match is zero past 10 words in both arms. Also recorded: a container restart kills a `nohup` driver about a minute after the turn ends, and a harness-tracked `Monitor` keeps the container up between turns; a `Monitor` with `persistent: true` still dies at 30 min. |
| [`ms13-k4/`](ms13-k4/RESULTS.md) | 2026-09-01 | **done — theorem: `R_max(4) = 4/5` exactly (Conjecture 12.2 and Q7′ at k=4)** | [`RESULTS.md`](ms13-k4/RESULTS.md) + [`bbk.py`](ms13-k4/bbk.py) + [`splits2.py`](ms13-k4/splits2.py) | Reopens `ms13-campaign`'s open question at the `k` it called a 2,070-hour no-go. The tree census is unnecessary: maximal row-set types are the split systems of binary trees on `2k` leaves with leaves paired into chords (Buneman), 4 shapes × 105 pairings at `k=4`, 14 maximal types, reproduces the campaign's `k=3` census exactly. Fail-first branching in the campaign's own exact rational B&B (next rounding = fewest live (row, side) options at the parent LP optimum) proves `R ≤ 4/5` per type in 3–64 s where lexicographic order had not finished type 0 after 900k LPs. All 14 maximal types have `R = 4/5` exactly, each attained at **unit** demands; unequal demands never beat Doerr at `k=4`. Lower bounds re-verified by a code-disjoint evaluator; two-sided calibration at `v=3/4` finds margin-1/20 witnesses on all 14. Hand lemma: the spider type is `≤ k/(k+1)` for all demands. Open: `k ≥ 5`, a proof (the gap is named: weighted rows lose the two-valued error structure that Hoffman–Kruskal + Carathéodory needs). |
| [`bts-coordinates/`](bts-coordinates/RESULTS.md) | 2026-09-01 | **negative (mechanism), headline withdrawn on its own null** | [`RESULTS.md`](bts-coordinates/RESULTS.md) + [`PLAN.md`](bts-coordinates/PLAN.md) | Transplant of the growing-feature-set mechanism in Large Discovery Models (arXiv:2608.15669 §4.2) onto the between-the-spokes cross-field prior-art problem, against the SLT objection in memory `a8b97f70` that novelty on a new representational direction is unreachable by flat embedding search. Retro-eval targets from the ms13 campaign (Doerr 2004, cross-field; MSW25, same-field) over arXiv-built pools of 1101 and 1241 titles, one encoder across all arms so only the basis changes. **The growing-coordinate arm does not beat its own frozen ablation** (one win each over four informative comparisons, sign test p = 1.0; both arms also beaten by a static ranking that costs zero reads). The first pass's headline — a blind-named axis ranking the target 5th of 1101 — **was destroyed by the null control the pre-registered adversarial pass demanded**: 12 random pool titles used as axes give a median best-of-12 rank of 10, so best-of-k is an order statistic. Surviving effect is narrow: mean-over-named-probes beats both the raw query and random probes in 5/5 configurations, i.e. multi-probe query expansion, which METHODS.md records losing three times before. Also found: paraphrasing only the target's title moves its rank by up to 383 places, larger than every measured effect; a fabricated negative control outscores the real target (0.7719 vs 0.7295), so absolute similarity says nothing about whether an answer exists; and issue #179's own P1 test case leaks the answer's vocabulary, so a stripped variant was written. Upstream of all of it, a subagent with **zero tool calls**, given the problem with all cross-field vocabulary removed, returned the target's verbatim title as its top query in under two minutes. Infra: arXiv keyword search works again (it killed PR #180 in July) and HF weights are reachable from CCotw. |
| [`hypothetical-classification/`](hypothetical-classification/RESULTS.md) | 2026-08-31 | **done — the pattern beats every model-free baseline (0.564 vs 0.417 acc@1 on WANDS) and loses to shipping the vocabulary by 14 points (0.701); the prompt must anchor on the vocabulary's REGISTER, never on novelty, or a Haiku subagent that obeys scores a fifth of the no-model control. Shipped: [muninn-utilities#127](https://github.com/oaustegard/muninn-utilities/pull/127), [claude-skills#782](https://github.com/oaustegard/claude-skills/pull/782)** | [`RESULTS.md`](hypothetical-classification/RESULTS.md) + [`ERRORS.md`](hypothetical-classification/ERRORS.md) + [`recheck.py`](hypothetical-classification/recheck.py) | Oskar: *"Don't classify. Hallucinate!"* ([softwaredoug.com, 2026-08-10](https://softwaredoug.com/blog/2026/08/10/hypothetical-classifications)) — *"it just takes a class corpus, a cheap model and a cheap embedder"*. Measured on WANDS (860 labels, 468 queries, one gold label) and on Muninn's own tag vocabulary (1,273 tags, 250 memories). The source post's novelty-anchored prompt is the single largest variable and is wrong: `gemini-3.5-flash-lite` half-ignores *"novel, never-seen-before"* and writes `Salon & Styling Chairs`, while a Haiku 4.5 subagent obeys it, writes `Hydraulic Styling Thrones`, and scores **0.100 acc@1 against a 0.500 no-model control**. Register-anchored, the same subagent takes the best acc@3 of any arm (0.750). A boundary claim — that the pattern halves against direct embedding on long documents — was published into both downstream PRs from the novelty prompt and then withdrawn: under the register prompt that arm goes 0.200 → 0.500, past the control, and the union of both rankings reaches 0.676/0.848/0.876. Also measured: batching 40 items per call is free at 1/17 the input tokens, char-ngram TF-IDF snaps nearly as well as MiniLM (0.528 vs 0.564) and beats it outright where documents contain their own label words, and a `general-purpose` Haiku subagent spends **32,539 tokens** to output the word `ok`. On the in-browser question: Pleias `Monad` (57M) and `Baguettotron` (321M) package fine (35 MB and 236 MB at q4f16, `onnx-community` builds) and earn nothing — 0.425/0.400 acc@1 as label writers and 0.325/0.350 as likelihood rerankers, against a **0.500** no-model control, with the gold label in the encoder's top-10 for 82.5% of queries. What the cheap model contributes is a prior over how taxonomies name things, which is the first thing a shrinking model loses. A fully client-side classifier is real and it is the encoder alone: `gte-small` int8 is **33 MB** and scores 0.455 acc@1 / 0.594 acc@3 with no API call. |
| [`hyde-recall/`](hyde-recall/RESULTS.md) | 2026-08-31 | **negative — HyDE query expansion over Muninn's FTS5 corpus is a wash at matched depth (0.263 vs 0.250 R@10, n=80); plain `recall(n=40)` scores 0.525 for free and dominates every arm** | [`RESULTS.md`](hyde-recall/RESULTS.md) + [`ERRORS.md`](hyde-recall/ERRORS.md) | Built on a misreading of the request above (HyDE, arXiv 2212.10496, rather than hypothetical classification) and kept because it closes a question `METHODS.md` records as open. HyDE's filter is a dense encoder's lossy bottleneck; BM25 has none, so a hallucinated term either matches nothing or drags in an off-topic document. Generating without corpus exemplars scores **below** the baseline (0.175 vs 0.250) — the same register failure the sibling experiment measures at scale. The confidence gate `nl2sh-dense` flags as unmeasured is now measured: term-coverage gating reaches 0.300 against a 0.338 oracle ceiling and still loses 5 queries. Third consecutive loss for query expansion on this account's corpora, after `nl2sh-dense` and `muninn-rm3`. The real finding is that Muninn's default recall depth of 10 leaves half the reachable targets unretrieved at zero cost. |
| [`halo-ccotw/`](halo-ccotw/RESULTS.md) | 2026-08-24 | **done — HALO's engine, WASM sandbox and trace tooling all run in a CCotw container; the RLM needs an `OPENAI_API_KEY` the environment does not carry, and the multi-session corpus it wants does not exist because release writes are 403 for this session type** | [`RESULTS.md`](halo-ccotw/RESULTS.md) + [`cc_to_halo.py`](halo-ccotw/cc_to_halo.py) + [`validate_dataset.py`](halo-ccotw/validate_dataset.py) | Oskar: *"Assess the viability of a solution like this in the Claude Code on the web environment"* — [context-labs/halo](https://github.com/context-labs/halo). Measured rather than reasoned about: `pip install halo-engine` is clean, the bundled deno 2.7.14 and ripgrep 15.1.0 wheels run, and the Deno+Pyodide sandbox boots numpy/pandas/pydantic in 7.5s cold (24 MB of deno cache, ~7s per invocation since each run is a fresh subprocess). POST to OpenAI and OpenRouter `/v1/chat/completions` returns the provider's own 401, not a proxy 403, so a key supplied through the environment is all the engine needs; there is no way to route its calls through the harness, since the Agent tool is a model tool call rather than an HTTP endpoint a subprocess can reach. `cc_to_halo.py` converts Claude Code transcripts to OpenInference spans — one 11-minute session became 161 spans with all three of the index's health counters at zero, and `view_trace` correctly reports it oversized at 391 KB against a 150 KB budget. Two conversion traps: durations are derived (each record has one timestamp, so LLM-span duration is an upper bound while tool spans are true wall-clock), and prompt tokens are three Anthropic fields summed — counting only `input_tokens` reports 106 for a session that processed 13.7M. The corpus is the real blocker: `persist-transcript.sh` has archived nothing since 2026-04-05, because `/tmp/.workstation-booted` is gone and, underneath that, the proxy returns 403 on every release write and Contents-API PUT for this session type. [claude-workspace#246](https://github.com/oaustegard/claude-workspace/pull/246) fixes the discovery and makes the 403 audible. |
| [`coherence-remex/`](coherence-remex/RESULTS.md) | 2026-08-24 | **done — one mechanism in `daniloc/coherence` is worth having (the meta-oracle, which reads a test's own AST and refuses a `via test` claim whose test loops a hand-written list); the surrounding apparatus is not. Adopted into `remex` as a live trial: [remex#80](https://github.com/oaustegard/remex/pull/80)** | [`RESULTS.md`](coherence-remex/RESULTS.md) + [`ERRORS.md`](coherence-remex/ERRORS.md) + [`recheck.py`](coherence-remex/recheck.py) + `artifacts/` | Oskar: *"Read this full thread"* ([daniloc.xyz, 2026-08-22](https://bsky.app/profile/daniloc.xyz/post/3mtofpi2pxc2y) — models write fine code at file level; the failure is complexity across files, swamped by volume no human reads; cites Conant & Ashby on regulators), then *"attempt implementation against one of our repos to assess it with a live use case. Maybe remex?"*. Tool ground-truthed rather than read: builds clean on Node 22, 936/936 tests pass, `verify` on its own repo is 83 claims all green over a 643-decision dogfood ledger — but **66 of 66 of its own boundary claims use `via guard`**, the escape hatch that skips the meta-oracle, which its own Known Limits calls "a laundering channel for hand-lists dressed as guards". Zero `via test`, zero `parity`, zero `conforms to` in its own specs, so the flagship check is unit-tested and never exercised through its own enforcement path. That made a trial on foreign code the only assessment available. On `remex` it ran on Python out of the box (shipped tree-sitter grammar, no adapter) and immediately flagged `test_pq_and_npz_round_trip_rotation` — `@pytest.mark.parametrize("rotation", ["haar", "rht"])` against a three-member `ROTATION_CODES` — as *"iterates a LITERAL domain… a sampling oracle, not totality"*. Four perturbations, each reverted: `"hadamard2": 3` into `ROTATION_CODES` with no construction, and `"hadamard2"` into `Quantizer.ROTATIONS` alone, **both leave the entire 267-test suite green** while only the live-domain oracle goes red by name; the other two are co-detected and labelled as the weaker refutations they are. Adoption cost is a refactor, not a spec edit: the second literal domain had no registry to loop — the packable widths were spelled five times as `bits in (5, 6, 7)` across three modules plus twice as `[1, 2, 3, 4, 8]` in tests — so closing the claim meant extracting `SUPPORTED_BITS` in `remex` itself. Also found wrong: the **parity arm false-fails a correct oracle** that aliases its domain (`for name in sorted(persistable)` refused; `for name in ROTATION_CODES` accepted), against a README that claims the analyzer never false-fails; and `redundancy` found two disagreeing README benchmark tables while missing the five-site source duplication the claim machinery had just caught. Ends at 7 claims/7 green/3 anchored invariants each with a recorded refutation, 288 Python tests passing. Verdict: transplant the LIVE/LITERAL/NO-ITERATION classification into `verifying-claims`; leave the 40+ subcommands, the 217 KB README (~54k tokens it tells agents to read) and the doctrine/gyroscope/premise-lease vocabulary. |
| [`hyparam-survey/`](hyparam-survey/README.md) | 2026-08-24 | **done — hypvector is remax_kb's architecture in a Parquet container, and the one thing it has that we do not is a build-time IVF whose cells are physically contiguous on disk, so a reader range-fetches 28 of 112 clusters instead of downloading the index** | [`README.md`](hyparam-survey/README.md) + [`NOTES.md`](hyparam-survey/NOTES.md) + [`ERRORS.md`](hyparam-survey/ERRORS.md) + [`results.json`](hyparam-survey/results.json) + [`evidence/`](hyparam-survey/evidence/) (nine cited source passages, re-derivable from the sha256-pinned npm tarball) + [`recheck.py`](hyparam-survey/recheck.py) | Oskar: *"This org's work is seriously impressive! Spelunk and pick up tooling/references that may aid us in our work. The vector search on parquet part is maybe something to consider for remax-kb?"*. Surveyed 27 public repos; the vector library is npm-only (`hypvector@0.2.2` ships unminified `src/`, the GitHub repo is private). Ran the sweep independently on 50k synthetic 384-dim vectors: their `constants.js` claim that residual misses are a `rerankFactor` limit and not a `probe` limit reproduces — scanning all 112 clusters instead of 28 moved recall@10 from 50% to 49% — and their default `rerankFactor: 10` reaches only 50% on this corpus, 93% at 100. `remex.IVFCoarseIndex` already partitions a corpus but is deliberately data-oblivious and keeps an 8-byte-per-vector permutation; hypvector reorders rows physically and stores per-cluster counts instead. |
| [`model-register-drift/`](model-register-drift/RESULTS.md) | 2026-08-23 | **done — Opus 5 is third-cleanest of six models on `declaude_lint.py` and roughly twice the next model's rate once the register entries regex cannot reach are counted by hand** | [`RESULTS.md`](model-register-drift/RESULTS.md) | Oskar: *"people are not all that happy with your Opus 5 substrate's attitude and personality... I find its idiolect insufferable"*. Six bare CCR sessions (`create_session`, explicit model id, no `source_url`, so no repo, no boot, no identity) wrote the same 700–900 word post with no voice instruction from anyone. The Agent tool could not run this — its `model` parameter takes four aliases and cannot address 4.8 or 4.6. Scored twice: `declaude_lint.py` normalised per 1000 words, then every one of the 42 entries by hand. The two rankings invert. Opus 5 sits at 5.04 tics/1k mechanically (3rd of 6) and 25.1 violations/1k adjudicated (1st, next is 16.4), with 85% of them in the aphorism/verdict family — entries 3, 7, 12, 13, 37, 38, 39, which `SKILL.md` already names as the third its regexes miss. Four of six headers are verdicts; six paragraphs end on a quotable line. Within the Opus line the adjudicated rate rises 11.4 → 15.3 → 25.1; the Sonnet line does not (16.4 → 13.4). n=1 per model, one prompt, one judge — only the Opus 5 gap is wider than adjudicator noise. |
| [`nl2sh-cli/`](nl2sh-cli/README.md) | 2026-08-23 | **done — the shippable half of the nl2sh line as a CLI: hybrid search over shell documentation with no model at all, plus an optional generator through ten backends** | [`README.md`](nl2sh-cli/README.md) + `nl2sh/` (`search.py`, `backends.py`, `cli.py`, `config.py`, `vendor/`) | Oskar: *"let's create a product that optionally invokes a model, either running locally via ollama/llm.cpp/mlx etc, or remotely. If no model is chosen the user just gets the hybrid search"*. `backend='none'` is the default and the product — ranked documentation, no weights, no network, no key — rather than a degraded mode. Local: ollama, llamacpp, llama-server, LM Studio, mlx, transformers. Remote: any OpenAI-compatible endpoint, Anthropic, Gemini. Remote costs no dependency because all of it is HTTP and urllib speaks it, so the extras cover only in-process runtimes. Every probe returns `Availability(ok, detail)` with an actionable reason instead of a bare `False`, the same rule `compose_layers._tag_exists` was fixed for. Defaults come from the measurements: `instantiate_anchored` because it wins every column at 1B, with a warning rather than a silent switch when a model's name looks small, since that prompt collapses to 0.146 at 270M. `prompts.py` and `extract_params.py` are vendored with pinned hashes because every quoted number came from those exact bytes. Search reaches 0.506 and not the research's 0.555: the query adapter that separates them gained +0.184 on the 207 utilities it trained on and **lost** 0.039 on unseen ones, so it stays out. Nothing here executes a command; it prints one. |
| [`caps-emphasis/`](caps-emphasis/RESULTS.md) | 2026-08-23 | **done — capitalising a directive does nothing on Baguettotron: mid-sentence CAPS is +0.003 log-odds [-0.147,+0.168], and violation rate is 42/43 in every surface form including no directive at all; where CAPS does move the number it tracks the extra tokens capitalising costs, not the case (+0 token bin CI spans zero). Markdown bold looked 20x stronger, but most of that was entropy and the rest reverses sign when the bold is moved into the reasoning register, where SYNTH actually puts it. Corpus reason: SYNTH holds one genuine capitalised directive in 22,100 documents; bold outnumbers emphatic caps 23:1. Largest effect measured was not typographic — restating the constraint in the reasoning register cuts the ironic rebound by ~0.8 log-odds** | [`RESULTS.md`](caps-emphasis/RESULTS.md) + [`Q5_CORPUS.md`](caps-emphasis/Q5_CORPUS.md) + [`METHOD.md`](caps-emphasis/METHOD.md) + [`PRIOR_ART.md`](caps-emphasis/PRIOR_ART.md) + [`ERRORS.md`](caps-emphasis/ERRORS.md) | [#45](https://github.com/oaustegard/experiments/issues/45) — does writing part of a prompt in capitals change what a model does, and by what mechanism |
| [`tc-interference-weights/`](tc-interference-weights/RESULTS.md) | 2026-08-22 | **done — review: three checkable problems in Anthropic's interference-weights note, all re-derived from its own figure data** | [`RESULTS.md`](tc-interference-weights/RESULTS.md) + [`check_claims.py`](tc-interference-weights/check_claims.py) + [`mirror.sh`](tc-interference-weights/mirror.sh) | Oskar: *"get this in its entirety, including the charts, then review"* — [Turner, Wu & Batson, *Characterizing interference weights in a tiny language model*](https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html), Transformer Circuits Thread, 21 Aug 2026. A 1L transformer plus a transcoder, expanded into 331M virtual weights, scored by **Fisher effectiveness** (2nd-order KL, all 331M weights over 537M tokens) and **helpfulness** (mean loss change under ablation, 7,765 sampled weights over 1B tokens). Its genuinely new contribution stands: `" IN "`→`" utions "` is the first interference weight demonstrated inside a trained transformer with a loss measurement attached — largest raw virtual weight from that token, target never once follows source in training, ~3 OOM below top effectiveness, negative helpfulness. Three problems, each reproducible from `shared/data/figures/*.js` via `check_claims.py`. **(1) The stated pruning exception names the wrong family.** Text: Fisher beats raw `\|w\|` "for every density and individual weight family (except for negative Features→Logits weights)". `threshold_data.js` says Fisher wins that family at all 19 densities by 3–10×; the family where it actually loses is **Tokens→OV→Logits negative-only**, worse at 8/19 densities and up to **2.22×** across density 0.05–0.40. **(2) The main conclusion does not survive the paper's own effect-size appendix.** The Discussion rests "the model is still dense in this basis" and "no saliency scheme will perform much better" on 47.6% of weights having positive mean helpfulness. Its ROPE appendix, population-weighted over all 331M: at ε→0, **15.1% positive / 64.3% uncertain**; at the paper's own yardstick ε = budget/N_total = 1.5e-8 nats/token, **90.1% practically zero and 2.9% positive** — ~9.6M weights, roughly 3× the transformer's parameter count, landing next to its own helpfulness-mass estimate (2.43% density holds 90% of positive mass). Three routes converge on 2–3%; only the sign-of-the-mean route reaches the Discussion, which never cites ROPE. **(3) Helpfulness and Fisher effectiveness are the same quantity on the tail.** Expanding the paper's own formula to second order gives `helpfulness(w) = −w·∂L/∂w + fisher(w) + O((sw)³)` — the Optimal Brain Damage saliency it cites in related work but never connects to its headline. `fisher ≥ 0` always and grows as `w²` while the gradient term grows as `w`, so weights above a crossover in `\|sw\|` are *forced* to measure as helpful. Measured on the helpful branch: **log-log slope 0.89–1.06, r 0.956–0.990, median h/fisher 0.58–1.22 in all six families over 6+ OOM**. So "the model puts its most effective weights in helpful directions" is partly an identity between the two metrics, not solely evidence about training pressure — and it inverts the claim that "a sharper metric has little room to improve", since the first-order term the two metrics *don't* share is exactly what separates helpful from harmful. **Checked and clean:** the 7,765 sample is family-stratified (~1,210–1,379 each) while families range 16.8M–104.9M weights; population-weighting moves h>0 only 47.6% → 48.4%, so the headline is not biased by that. Pruning numbers verified (70% pruned → +0.0107 nats, 85% → +0.0702). **Minor:** the dead fraction of one sample is reported three ways — 12.7% (table), 13.1% (histogram), 13.07% (implied by the mass figure's counts). **Artifact:** `mirror.sh` reproduces a 107 MB offline mirror — main page, 21 interactive figures with data and vendor bundles, the 34-page figure gallery, the 6.2 MB `feature_vis` page. Two assets 403 on the live site (`shared/data/token_splits.js`, `shared/data/token_vocab_split.js`, Feature 157 figure), broken upstream. |
| [`avo-supervisor-specter2/`](avo-supervisor-specter2/RESULTS.md) | 2026-08-22 | **done — best bits=4/haar/seed=2026/two-stage(300) R@10=0.7630 vs bits=4/rht/seed=42 baseline 0.7510 (+0.012), 30 candidates, 3 strategy classes, 2/3 plateau switches; the Stop-hook loop itself did not run — this session is a child session with no supervisor hook wired, so candidates ran via a manual driver applying the same plateau rule instead** | [`RESULTS.md`](avo-supervisor-specter2/RESULTS.md) + `run_candidates.py` + `ledger.json` | [claude-workspace#233](https://github.com/oaustegard/claude-workspace/issues/233) — first real search run off the supervisor infrastructure assessed there |
| [`avo-supervisor-assessment/`](avo-supervisor-assessment/RESULTS.md) | 2026-08-22 | **done — go: exit-2 Stop hook blocks and injects verbatim (tested live), hook wall clock 38 ms, single-candidate remex fitness 1.4–1.7 s, in-repo JSON ledger beats Turso 0.2 ms vs 0.5–0.7 s; Fable 5 session with `create_session(model=…)` makes the three-arm ablation spawnable** | [`RESULTS.md`](avo-supervisor-assessment/RESULTS.md) + `artifacts/` | [claude-workspace#233](https://github.com/oaustegard/claude-workspace/issues/233) — assess AVO-style supervisor loop before building |
| [`nl2sh-instantiate/`](nl2sh-instantiate/RESULTS.md) | 2026-08-21 | **done — the bake-off in [`MODELS.md`](nl2sh-instantiate/MODELS.md) revises this headline: a zero-shot Gemma 3 1B beats the fine-tuned 270M on every column (routing 0.799 vs 0.610, usable 0.793 vs 0.470), and the instantiation framing WINS at 1B (0.848 vs 0.799) once the model stops imitating the source-line format. The 270M finding below stands, scoped to 270M:  framing the task as substitution does not move routing on a 270M model (p=0.76) but collapses token-repeat loops 0.183 → 0.049, and execution scoring puts the real functional accuracy at 0.055 against a 0.427 routing headline** | [`RESULTS.md`](nl2sh-instantiate/RESULTS.md) + `prompts.py` + `run_gen.py` + `train.py` + `score.py` + `funceq_ext.py` + `funceq_alfa.py` + `alfa_prep.py` + `bench.py` | Issue [#52](https://github.com/oaustegard/experiments/issues/52), stage 2: retrieval finished at 0.555 gold-in-sources against an 0.640 oracle ceiling, so the loss is the generator's. The issue argues the job is **instantiation** — the user's literals substituted into a documented example — because §6 of `nl2sh-dense` measured an exemplar worth +0.189 routing while the *choice* of exemplar was worth zero. Oskar: *"let's start with the 270 million Gemma three"*. Model, sources, distractors, decode and seed held fixed; only the instruction varies. **Zero-shot the substitution prompt loses 0.146 to 0.500, and the loss is a format artifact**: on 0.774 of rows the model answers in the shape of the source lines it was shown, bullet included. One epoch erases that entirely (0.774 → 0.000), the same shape as stage 1's 0.026 → 0.706. **Past training the prompts route the same** — 23 wins to 20, p = 0.76 — **and separate on garbage**: loops fall 0.183 → 0.049 and *usable* gains +0.092 (31–16, p = 0.040). `nl2sh-selfhist/MODELS.md` had named degeneracy the real ceiling after `repetition_penalty=1.3` bought a similar reduction at a cost of 0.118 routing; this buys it free. **The published benchmark reproduces the direction on rows built to execute** (`westenfelder/NL2SH-ALFA`, MIT, arXiv:2502.06858): 0.854 → 0.866 routing, 0.144 → 0.085 degeneracy on the non-`find` slice — after subtracting its **0.393 always-`find`** prior, which eats most of the 0.911 headline. **And the first functional number for this line of work is 0.055**: with the `funceq` fixture built from every path the gold commands name, 36 of 164 cyber rows are decidable and 0.250 of those are equivalent — `utility_ok` overstates by ~8x, corroborating the issue's hand-read estimate of "nearer 0.05". The cyber corpus caps at 0.22 coverage no matter how wide the fixture; ALFA is where execution decides. **Recovered from a lost container**: the session that ran the grid wedged on a `run_in_background` poll loop and was reclaimed with everything uncommitted — scripts and tables came back from its transcript and re-ran to identical numbers. |
| [`gemma-proxy-tuning/`](gemma-proxy-tuning/RESULTS.md) | 2026-08-20 | **open — vocabulary precondition holds, logit scale does not; experiment not run** | [`RESULTS.md`](gemma-proxy-tuning/RESULTS.md) + [`tokenizer_diff.py`](gemma-proxy-tuning/tokenizer_diff.py) + [`tokenizer_diff.json`](gemma-proxy-tuning/tokenizer_diff.json) | Oskar, after establishing that speculative decoding cannot merge two models: "how similar are Gemma 3 and Gemma 4? Could we fine-tune the smallest Gemma 3 and apply it to the Gemma 4 ~30B?" Proxy-tuning (Liu et al., COLM 2024) and emulated fine-tuning (Mitchell et al.) are the same equation, `softmax[s_base + alpha*(s_expert - s_antiexpert)]`, and both need the three logit vectors to share an index and a scale. **Index: yes.** Both tokenizers are 262,144-piece BPE with a **byte-identical** 514,906-rule merge list; 255,938 ids (97.63%) carry the same token and **zero ordinary text tokens moved** — all 6,187 that did are special or reserved. The disagreeing ids form exactly two contiguous ranges, `[46,106]` (61 `<unusedNN>` shifted by 7) and `[255999,262143]` (6,145 Gemma 3 image/turn tokens replaced by Gemma 4 `<|tool>`/`<|think|>`/`<|channel>`), so ids 107..255,998 are identical throughout; six probe strings across English, Python, Norwegian, math, Japanese and emoji encode to identical id sequences under both. That is the constraint that killed cross-family transfer in `monad-specdec` (157 of 8,192 shared ids), and here it costs two slice assignments. **Scale: no.** Every Gemma 4 config sets `final_logit_softcapping: 30.0` (31B, 31B-it, 12B, E2B, 26B-A4B); no Gemma 3 config has it (Gemma 2 did, Gemma 3 dropped it for QK-norm). Gemma 4 emits `tanh(l/30)*30` while a Gemma 3 delta is unbounded and linear, so adding them degrades worst on the tokens the base is surest about; invert with `atanh(clamp(l/30))*30` first, which also rules out the top-k-logprob black-box case. Checkpoints exist for all three slots (`gemma-3-1b-pt` base/tuned pair, `gemma-4-31B` **base** not `-it`, per Liu et al.'s CodeLlama degradation) at ~6% extra FLOPs vs 20% for the paper's 7B/70B. **Untested by anyone:** every published result transfers a delta across *scale within one pretraining run*; this crosses a generation. Shared vocabulary is necessary, not sufficient, and `alpha` is where that shows up. |
| [`needle-depth-growth/`](needle-depth-growth/README.md) | 2026-08-20 | **done — the surgery is 30 lines and lossless; the training is pretraining, not a fine-tune** | [`README.md`](needle-depth-growth/README.md) + [`grow.py`](needle-depth-growth/grow.py) | Oskar, reading the Cactus paper next to `needle-bsky`: "can a model like needle be *retro trained* to add MORE layers, à la Monad?". **Mechanically yes, and verified.** Needle 2's stack is an `nn.scan`, so every per-layer tensor carries a leading axis of `num_layers` — block weights inside the scanned collection, MHC lane parameters as explicit `(L, ...)` arrays — and nothing downstream hardcodes 27 (`decode.py` sizes the KV cache from `cfg.num_layers`, `export.py` loops over it). Growing is a concat along axis 0 plus a config bump. `grow.py` does it with identity-initialised blocks and the grown model's logits are **byte-identical** (`max |delta| = 0.000e+00`) at **27 → 31** (+4.40M, +9.7%) and **27 → 48** (+23.12M, +51.1%). The identity recipe is four values: `attn_gate = -30` (sigmoid underflows to exactly 0 in fp16, so the scan body's `y = block(u) - u` is 0), `hadamard_mlp/d3 = 0`, `mhc_a_res = 0`, `mhc_b_res = 40*I` (Sinkhorn maps that to identity to ~1e-17; the shipped `_res_identity_init` uses `4.0*I`, which leaves ~1.8% off-diagonal mass and is an initialisation, not an identity). Appending at the *end* also preserves the `arange(L) % 4` lane assignment for every existing layer and leaves the Engram sites at 2 and 15 — a middle insertion would shift lanes and must come in multiples of 4. **What it does not buy:** a grown model is the old model exactly until the new layers are trained, and that is pretraining — the shipped trainer is LoRA-only over a fixed target list, `needle-bsky` had 800 templated rows against 4.4M newly-added parameters, and the paper's own ablation puts depth at iso-param on a U-shape with a **20-layer optimum at this width** (8/20/32/48L → 2.168 / 2.1343 / 2.153 / 2.175 nats) with Needle 2 already shipping at 27. It also moves the representation under the never-retrained 8,192-parameter confidence head, the same cost every other stack edit here has carried. **Premise correction:** Monad is 64x256 by pretraining design, not by retro-fitting layers, so it is not a precedent for growth — and `monad-bsky` measured that shape at this task losing 0.481 to Needle's 0.611. Untested: whether `export.py`/`quantize.py` emit a working `.cact` at non-standard depth. |
| [`needle-tool-naming/`](needle-tool-naming/RESULTS.md) | 2026-08-20 | **done — negative: tool names and tool descriptions are two near-equal channels, and better names buy nothing** | [`RESULTS.md`](needle-tool-naming/RESULTS.md) + [`PREREG.md`](needle-tool-naming/PREREG.md) + [`ERRORS.md`](needle-tool-naming/ERRORS.md) + [`recheck.py`](needle-tool-naming/recheck.py) + `results_*.json` | Oskar, after reading Cactus's [attention-only paper](https://arxiv.org/abs/2607.18363) (which is Needle 2's architecture — the shipped class is literally `SimpleAttentionNetwork`): "is it worth further experimentation?". `needle-bsky` left `profile` at 0.250 and `identity` at 0.333 in every arm, unmoved by a schema rewrite *and* a fine-tune — even though the discriminating fact was already in the description verbatim (`get_profile` says "follower count"; the query "how many followers does pfrazee.com have" still returns `get_followers` at 0.80). That is the **opposite** of the failure the paper predicts, which localizes the SAN deficit to *low-context* tokens. Hypothesis: names outrank descriptions. Six variants over the same 18 tools, same 62 queries, same scoring code, predictions committed before the first run. **The hypothesis failed**: `names-only` 0.444 vs `desc-only` 0.407, predicted gap ≥0.15, measured **0.037 at p=0.82**. What replaced it is cleaner — the two channels are **near-equal and partly redundant**: deleting descriptions costs 16.7pp (**p=0.035**), deleting names costs 20.4pp (**p=0.019**), deleting both leaves **0.074** against a 0.056 chance floor, and against that floor names are worth 0.370, descriptions 0.333, both together 0.537 — a quarter of each carried by the other. **Rule-written names buy exactly nothing** (`separated` = `canon` = 0.611 flat and 0.778 oracle, p=1.00 both) **and cost the gate**: confidence separation collapses 0.191 → 0.101 because mean confidence on *wrong* calls rises 0.392 → 0.480. Name capture is nonetheless real and visible where it acts — mechanically rotating names onto neighbours moves `profile` **0.250 → 0.750**, with "how many followers does pfrazee.com have" → `get_profile` @ 0.81 and "look up the account jay.bsky.team" → `get_profile` @ 0.90, descriptions unchanged by one byte — it is just too small (4 queries of 54) to surface as a main effect. Mirror case in the same arm: `resolve_identity` wearing the name `check_network_outage_status` is still chosen correctly on its description alone, at confidence 0.584 → 0.167 — conflict keeps the answer and destroys the belief. **The two heads read different things**: retrieval cost (oracle − flat) is 0.056 with opaque names and real descriptions but **0.185** with real names and no descriptions and **0.278** with rotated names, so the contrastive head reads descriptions and is damaged more by a wrong name than by a missing one — though `needle-bsky`'s +26pp `auto`→`tuned` rewrite survives the oracle at +20.4pp, so most of *that* win was decode, not retrieval. **Answers the deployment question directly: no, there is no naming route to declaring all 18** — retrieval costs 16.7pp under the best naming available and `separated` reproduces that gap to three decimals, leaving the already-measured ladder (regex→≤5 at 0.722/316 ms, regex-only at 0.833/0.022 ms, two-model agreement at 0.880) unchallenged. Side benefit: stripping all 18 descriptions cuts the median turn 808 → 532 ms, the only lever here that trades the other way. **Harness and version check**: `canon` reproduced `needle-bsky`'s `tuned-min` **exactly** — 0.611, identical calls and confidences — on `cactus-needle` 2.0.7 against that experiment's 2.0.6. Also censused the shipped checkpoint while scoping this: **45,211,383 params**, of which **18.6% are Engram hash-indexed n-gram tables** (a parametric store the paper never mentions, added back after the paper concluded a SAN lacks one), and LoRA reaches **28.31M of 45.21M** — `out_proj`, the paper's write path, **is** adapted, so the fine-tune negative was never a frozen-write-path story. Caveats: n=54 routable (one query = 1.85pp), the `profile` flip is 4 queries, `separated` is one author's rule applied once, and descriptions were stripped rather than degraded. |
| [`nl2sh-dense/`](nl2sh-dense/RESULTS.md) | 2026-08-20 | **done — a 25.6 MB encoder plus a page-level index lifts gold-in-sources 0.262 → 0.390 (p=0.0003); query reformulation is a clean negative; and the old 34-row eval would have called the whole thing a regression** | [`RESULTS.md`](nl2sh-dense/RESULTS.md) + `sample_cyber.py` + `cyber_nl_ext.json` + `results_*.json` | Issue [#48](https://github.com/oaustegard/experiments/issues/48): the on-device shell helper is retrieval-bound — Gemma 3 270M routes 0.706 with the gold page in context and 0.206 with real BM25, which surfaces it 26% of the time. **The eval was extended first.** `sample_cyber.py` drew 149 more commands from the same Zenodo/UCI corpus under the same tiered protocol and `gen_nl.py` wrote their NL, taking the independent eval from 34 to **164 leak-free requests over 132 distinct gold utilities** (constant prior 0.012). That turned out to be the load-bearing step: on the original 34 rows the winning retriever *lowers* routing 0.206 → 0.147 while raising sources 0.235 → 0.382, and on the 130 new rows both rise — the old eval's headline would have been that better retrieval makes the system worse. **A 23.5 MB encoder matches a 164.5 MB one.** all-MiniLM-L6 int8 scores 0.341 in sources against bekko-a8m's 0.354 (one query on 164) at a seventh the disk, so the issue's 157-MB footprint worry does not bind. **Page-level indexing is the free half of the win**: grouping 31,169 chunks into 6,397 pages lifts BM25 alone 0.262 → 0.323 with no encoder at all, and composed with the dense arm reaches **0.390, p = 0.0003, 27 wins to 6**. But *feeding* whole pages to the model does nothing (0.159 vs 0.165) — the index wants pages, the prompt does not. **Reformulation lost**, confirming `muninn-rm3`'s prediction: RM3 costs 0.036 and dense-PRF costs 0.030, even though dense-PRF surfaces `fcrackzip` on the exact query the issue quotes. **The abstention gate's `margin >= 5` fails because 5 is in BM25 score units**, not because it is a difference — a quantile-set absolute margin transfers as well as `top2/top1` does; the ratio's advantage is needing no calibration sample. RRF is the wrong substrate for the gate (margin AUC 0.47–0.53, a coin flip) because it discards score magnitudes. **End to end the retrieval gain is decisive and the routing gain is not**: 0.128 → 0.165/0.183, p = 0.26/0.11 at n=164. **A follow-up answers 'should we fine-tune the embedder' with no**: one identity-initialized `d x d` matrix on frozen query vectors — 4,588 NL2Bash pairs, 40 seconds on 4 CPU cores, 4.2 MB, document vectors untouched — takes gold-in-sources 0.384 → **0.463** (p=0.024) and routing 0.128 → **0.201** (p=0.058), but **the whole gain is on the 207 utilities the training data covered** (+0.184 seen, **−0.039** unseen), and a rank-64 adapter with 16x fewer parameters reproduces the same split — so the limit is utility coverage, not capacity, and a fine-tune would hit the same wall with more room to memorize. **The largest lever turned out to be the corpus, not the retriever**: a flash-lite pass over all 6,397 pages adding goal-level phrasings (Pleias-Redline-style) takes BM25 alone 0.311 → **0.427** and end-to-end routing 0.128 → **0.226** (p=0.0052), composing with the adapter for **0.555 / 0.250** — validated against a human-authored control (+0.098, p=0.0001) because both the eval's NL and the corpus's came from Gemini, and localised by a fidelity split (+0.224 on pages whose generated intents added vocabulary the page lacked, +0.086 where they only echoed it). Ships as a greppable handbook with the tldr-pages CC-BY-4.0 attribution this repo had been missing. |
| [`nl2sh-retrieval/`](nl2sh-retrieval/RESULTS.md) | 2026-08-19 | **done — the 350M model works after 25 minutes of fine-tuning (0.923 where a constant scores 0.000), retrieval is now the bottleneck, and an adversarial pass returned OVERSTATED on the first draft of the retrieval numbers** | [`RESULTS.md`](nl2sh-retrieval/RESULTS.md) + [`EXTRACTION.md`](nl2sh-retrieval/EXTRACTION.md) + `results_*.json` | Oskar's six-component architecture — LLM-composed regex, a small model for intent, regex parameter extraction, man pages with remax-kb-style hybrid indexing, ICL grounding, and error logging — built and gated overnight. **The gate is the result.** [Pleias-RAG-350M](https://huggingface.co/PleIAs/Pleias-RAG-350M) is trained to quote sources literally, which is exactly the operation `monad-bsky` measured its 56M sibling failing (51% identifier copying), so the bet was that converting *generation* into *extraction* rescues a tiny model. Handed the gold utility's tldr example every time: **0 usable commands in 40**, **verbatim rate 0.000**, 6.5 s on 4 CPU cores. It quotes descriptions into cited prose, never a command — `monad-bsky`'s zero-shot result one generation later. It degrades with source count (6/8 at 3 sources, 4/8 at 5, worse at 15), so **k≈3** is an architectural constraint. **Two harness findings cost a false negative**: a prompt assembled from the special-token list alone gives a **0.000 parse rate that mimics incapacity** (it must end with `<|language_start|>`), and **pre-filling the reasoning scaffold is 9x** — 61.2 s → 5.4 s — because ~700 tokens of preamble precede the answer span. **Retrieval fails as built**: BM25 over 31,169 chunks / 4,698 utilities scores **0.233 recall@5 on the deployment slice**. The cause is diagnosable — *"find files bigger than 100MB"* returns `oneliner`, `rmlint`, `blkdiscard`, because 7,232 tldr pages bury the ~400 classic utilities, which is **`xr`'s documented cross-repo confusability in a new domain**. Scoping to `$PATH`-installed utilities (4,698 → 569) nearly doubles recall@1 on an identical sample, and an oracle scope bounds the headroom at 0.533. **An adversarial verifier recomputed everything and returned OVERSTATED**: the naive `all` row (0.280@5) *loses to a query-independent frequency prior* (0.625@1) because of the find skew, **34.7% of NL2Bash prompts name the gold utility literally** (0.586@5 vs 0.117@5 for the rest), and the quoted latency timed the wrong computation. **Parameter extraction works** — 0.971 precision held out — but only **37.4% of requests contain every operand**, and only **53% of correct extractions are a whole command token**, so spans feed a template rather than assembling argv. **Fine-tuning answers what the gate left open**: 600 rows, one epoch, **25.5 minutes on 4 CPU cores** takes it to **0.950 overall and 0.923 on the non-`find` slice** — scored against an *always-`find`* constant prior of 0.675, because 27 of the 40 gold utilities are `find` and an earlier draft of this row nearly reported the gain against zero. The zero-shot failure was **output shape, not capability**, replicating `monad-bsky`'s 0.000 → 0.481 on the 56M sibling. **But the premise that selected this model did not survive**: verbatim rate is **0.000 before and after** — it was chosen for literal source-quoting and what fine-tuning installed was command *generation*, so the non-RAG ablation is unrun and any 350M base might do as well. **And Oskar's pushback was right**: Claude's rule iteration is strictly monotone (McNemar **125–0**, **39–0**; Gemini's round 1 broke 62 rows and round 3 broke 193) — but the round-3 test never ran, because family A saturated at round 2 with zero errors, and iteration does not transfer: **wild sat at 0.540 across all three rounds**. |
| [`nl2sh-scoping/`](nl2sh-scoping/README.md) | 2026-08-19 | **done — the terminal-helper problem is utility selection over a long tail, not flag composition; and man pages are the coverage backbone, not the fallback** | [`README.md`](nl2sh-scoping/README.md) + `results.json` + `doc_corpus.json` | Oskar, after the `gh-mcp-regex-fit` cascade: *"how about a model + regex combo for something much bigger: bash + zsh commands"*. Scoping measurement run **before** building, to choose between a cascade (needs a thick head) and retrieval (needs a long tail the model does not know). Shell history was unavailable — a week-old laptop — and the substitute is arguably the better corpus: **a helper is asked about what you would look up, not what you type most**, and NL2Bash was scraped from forums and tutorials. One caveat governs everything: **60.3% of NL2Bash leads with `find`**, so it is a correctness corpus and not a usage distribution; quote the non-find column. **The head is thin and the tail is long**: top 10 utilities cover only **29.1%** of non-find requests, you need ~50 for 70%, and 176 utilities appear exactly once — a weak case for rules that *answer* and a strong one for rules that *narrow*. **The difficulty is utility selection, not flag composition**: 72.6% of non-find commands carry at most one flag. **A self-correction worth the entry**: the first draft argued for `tldr` over man pages, conflating *RAG over man pages* with *putting a man page in the context window*. Measured properly, whole pages run to 47k tokens but `.TP` option entries are **median 56, 93.4% under 350**, and tldr covers **96% of the top-50 utilities against only 50% of the used-once tail** — where the requests and the model's ignorance both are. Both corpora, tiered by whether a chunk is quotable or needs composition. Unused lead: **SYNOPSIS is a grammar**, the same object `needle-bsky` compiled constrained decoding from. |
| [`nl2sh-selfhist/`](nl2sh-selfhist/RESULTS.md) | 2026-08-19 | **done — an eval where neither side is Claude's drops the fine-tuned model from 0.92 to 0.62; a real 16k-command corpus with a 0.189 constant prior replaces NL2Bash's 0.603 find-skew** | [`RESULTS.md`](nl2sh-selfhist/RESULTS.md) + `corpus_probe.py` + `gen_nl.py` + `run_independent_eval.py` + `results_*.json` | Oskar: *"you've issued hundreds of commands tonight, you could use your own history"* then *"SURELY there are bash history logs to be found online"*. This session's 289 Bash calls proved unusable as a benchmark (26 general-shell, ~15 of one 'print lines X-Y' shape, and the description field is not persisted), but the search turned up the **Zenodo/UCI hands-on cybersecurity training corpus** (record 8136017, CC-BY-4.0): **16,065 real bash commands, 696 utilities, constant prior 0.189 (`ls`)** against NL2Bash's 0.603 (`find`). Documentation coverage on that real distribution: **87.7% of invocations, 24.4% of utilities, 9.8% of the used-once tail**, with the uncovered tail mostly undocumentable (`ll` a shell alias at 114 uses, `./ssh2john.py` a local script, `mfsconsole` a typo) — arguing for reading shell config and `$PATH` over a bigger corpus. Found a builder bug: **379 tldr alias pages dropped as stubs** rather than resolved to their target (`whoami`->`id`), a nearly-free +1.2-point fix. **The capstone is the eval that finally has neither side authored by the model under test**: the cyber corpus supplies real commands, and `gen_nl.py` has **Gemini write the request** for each (instructed not to name the utility; 4 of 38 leaked). The fine-tuned RAG model routes **0.618 leak-free (n=34)** against its **0.923 on the NL2Bash gate** — a **0.30 drop** that is the cost of templated phrasing, the find-heavy distribution, and NL2Bash naming the answer 34.7% of the time. Failures are distractor-utility hallucination (`pgmbentley`, `calligrastage`) and abstention. **This closes the eval-authorship problem that ran through the whole thread**: every prior number measured against a self-authored or utility-naming eval is an upper bound, not a capability. Still utility-routing, not functional equivalence, which would be lower again. |
| [`gh-mcp-regex-fit/`](gh-mcp-regex-fit/RESULTS.md) | 2026-08-18 | **done — fitted routing rules lose to hand-written ones; the answer is neither, it is a cascade: precise rules first, a *scored* fallback that can also abstain second (+0.136 wild accuracy at zero abstention cost)** | [`RESULTS.md`](gh-mcp-regex-fit/RESULTS.md) + `results*.json` + [`wild.jsonl`](gh-mcp-regex-fit/wild.jsonl) | Oskar, after `monad-bsky`: "what other tool use and routing use cases ought to sit behind a trained regex heuristic like that of regex_only.py" — then "build the fitting harness against the GitHub MCP catalog". `monad-bsky` left its 20 regex rules hand-written after reading the eval's failures, so 0.833 was fitted to an unknown degree. This searches for the rules instead, on the real **58-tool GitHub MCP catalogue** (50 with upstream schemas, **79 routing targets** once the seven `method`-enum dispatchers are counted), built from `github/github-mcp-server`'s own committed schema snapshots. `fit.py` induces an ordered decision list by greedy precision-constrained covering (CN2/RIPPER shape) over structural cues, catalogue-vocabulary tokens and IDF-weighted schema-overlap features, abstaining rather than falling back. **Result is negative and clean: every fitted arm loses.** Best fitted reaches **0.239** on a held-out phrasing family and **0.351** on hand-authored queries against **0.546 / 0.486** for rules written by hand (McNemar p=1.7e-44 on family B, p=0.058 on the 74-row wild set); no regularisation knob — Laplace scoring, min-coverage 8, dropping bigrams — moves it. The diagnosis is vocabulary, not entity memorisation: the learned rules are sensible (`tok:diff -> get_diff`) but never fire on *"what code does this PR actually change"*, whereas a human writes `\b(diff|patch|changeset)\b` for words no training query contained. **Three things transfer.** The cues-only arm covers **4.9%** of training rows and scores 0.013 wild, so `monad-bsky`'s stated boundary — argument-shape catalogues yes, intent catalogues no — is real and takes one minute to test before committing to a design (`owner`/`repo` appear in 40 of 50 tools here, so the dominant cue discriminates nothing). The catch-all ablation isolates the parent's refusal collapse exactly: abstention **0.867 -> 0.000** on all three splits for **+0.014** accuracy. And `context_probe.py` finds the number that undercuts every regex-router figure in this repo — generated queries carry their `owner/repo` 61-77% of the time, hand-authored ones **13.5%**, with all-required-arguments-extractable falling 0.66 -> **0.15**, because real requests say "go ahead and merge it". Bonus replication: two *deterministic* routers agreeing gate at **0.867** precision over 0.203 coverage against 0.667 ungated, the `synergy.py` shape at microseconds instead of 11x latency. **Second pass** (Oskar: *"do the others improve anything? and up the stack there's good old spaCy"*) added BM25, spaCy and a sentence-encoder arm behind a common `arms.py` interface, built by three parallel subagents and re-verified through the writeup's own eval. **The cascade is the result**: hand rules first, thresholded encoder where they abstain, gives wild **0.622** against 0.486 for the rules alone and 0.500 for the destructive catch-all — **+0.136 accuracy with abstention unchanged at 0.867**, at a 0.088 ms median because 82% of requests never reach the encoder. **BM25 is a shortlister, not a router**: 0.635 top-1 but **0.851 at k=5 / 0.919 at k=10** over 79 targets at 0.03 ms, which is how not to pay `needle-bsky`'s ~750 ms sixth-tool penalty. **A negative document beats a threshold** for abstention (+0.400 for -0.014, the exact mirror of the catch-all). **spaCy is a clean negative**: every arm loses to a 20-line IDF schema-overlap control at 250x the latency, and the decisive measurement is the zero-lexical-overlap slice where vectors score **0.000** (n=77) — they move "has anyone approved it" from unreachable to rank 21 of 79, findable but not routable. **Stemming and lemmatisation lose** because in an API catalogue grammatical number is semantic (`branches` idf 4.37 -> `branch` 2.76; `tag`/`tags` merge `get_tag` into `list_tags`) — though stemming *helps* recall@10, so stem to shortlist, not to decide. **Two first-pass corrections**: a zero-parameter BM25 ranker drops 0.611 -> 0.200 across the same families, so family B is adversarial toward *schema vocabulary* and overstates generalisation loss, and hand-authored queries outscore it on every schema-reading arm; and BM25 at 0.005-0.028 ms is *faster* than the regexes, so latency is no longer part of their case. Agreement gates in proportion to independence: hand ∧ encoder reaches 0.351/**0.923** against hand ∧ fitted 0.203/0.867. **Third pass** (Oskar: *"you realize YOU wrote them and YOU are a model"* — then CF gateway credentials arrived): the clean-room rig ran, with **two independent compilers** — Gemini 3.7 Flash via the CF AI Gateway, and a **Claude subagent given one spec file and forbidden the repo** (isolation verified from its transcript) — both executing through the same executor so authorship is the only variable. **The clean room beats the contaminated arm**: Claude having seen no split scores **0.540** wild against 0.486 for the rules written with the eval in view, so contamination was worth **+0.042 on family B and −0.054 on wild** — disqualifying, but pointing the wrong way. **Compiling is a model-capability cliff**: gemini-2.5-flash-lite 0.000, 2.5-flash 0.013, 3.7-flash 0.176, Claude 0.540 — and the gap survives every procedure control, an explicit breadth instruction (coverage *falls* 0.216→0.149), an 8x call budget yielding **224 rules, 2.8 per target, more than Claude's 154** (0.176→0.162, p=1.00), and two rounds of supervision (0.419, still under Claude's zero-shot). **The first pass's fitting result was not its algorithm**: hand a frontier model the same 237 labelled rows and it overfits *harder* than the greedy learner (0.050 vs 0.239 on family B), below the 0.161 of the arm shown no queries at all — supervision must arrive as **errors** (0.219), and iteration peaks at two rounds then loses 0.123 *in sample* at three. **The live arm reinterprets the whole table**: fitted on nothing it scores **0.532 / 0.532 / 0.568** across the three splits every other arm spreads 3–16x on, so those gaps are authorship and supervision, not difficulty — quote wild. **And the deliverable is the cascade**: compiled rules in front of the live model reach **0.770** wild against **0.568** for the model alone while avoiding **58.4%** of its calls, because the model declines 40% of routable requests at 0.955 precision and the rules answer exactly those — complementary abstention, not better rules. |
| [`monad-specdec/`](monad-specdec/RESULTS.md) | 2026-08-18 | **done — negative: Monad drafting for Baguettotron runs at 0.90x baseline at best; depth sets decode latency, so a 5.7x smaller model is only 2.1x faster** | [`RESULTS.md`](monad-specdec/RESULTS.md) + [`specdec.py`](monad-specdec/specdec.py) + [`results.json`](monad-specdec/results.json) + [`analysis.json`](monad-specdec/analysis.json) + [`depth_scaling_both.json`](monad-specdec/depth_scaling_both.json) | Oskar: "could we use monad as a speculative decoder for baguettotron?". Both PleIAs, both `LlamaForCausalLM` on SYNTH, but different tokenizers — 7,397 of Monad's 8,192 tokens exist as strings in Baguettotron's 65,536-piece vocabulary and only 157 share an id — so drafts are verified by string-level exact match (Timor et al. 2025) rather than against target logits. The loop is lossless: every run is token-identical to plain greedy. It is also slower at every draft length: **0.90x at γ=1, 0.42x at γ=8**. Three causes compound. (1) **Latency tracks depth, not parameters**: Monad is 64 layers to Baguettotron's 80 and decodes at 57.0 ms vs 119.9 ms, so c = 0.476 rather than the 0.18 its parameter count suggests. Truncating each layer stack gives 0.804 ms/layer at width 256 against 1.264 ms/layer at width 576 — a ratio of 1.57 where a compute-bound decode would show 2.25² = 5.06, so fixed per-layer overhead dominates and a narrower model of equal depth is barely cheaper. (2) **A smaller vocabulary needs more draft steps**: 3.25 chars/token vs 4.12 means 1.27 draft steps per target token, lifting the effective c to 0.602. (3) **Acceptance lands just under break-even**: 0.546 measured against a 0.602 break-even at γ=1. Monad is a *good* predictor — Baguettotron's own first-N layers agree with its full stack on only 2.1%/7.3%/25.0% at 20/40/60 layers — it is just not a cheap one. A viable draft would be ~24 layers at width 256 sharing the target's tokenizer (c ≈ 0.21, ~1.3x at γ=2), assuming acceptance holds, which it would not fully. CPU, 4 threads, fp32, batch 1, 5 prompts. | Follow-up [`RESOURCES.md`](monad-specdec/RESOURCES.md) prices the two obvious next moves. **4-bit: already shipped and nearly free of benefit** — PleIAs' official GGUF ladder has Q4_0/Q4_K_M/IQ4_XS, and on this CPU Q4_0 buys 1.19x over Q8_0 while Q4_K_M is *slower* (0.97x), because weight bandwidth is only 19-23% of decode time (fp32->bf16 gives just 1.13x). Switching torch->llama.cpp buys 3.9x at unchanged precision, so the runtime is the lever, not the bit width. **A drafter: worth building** — an EAGLE-style head (FC 2h->h + one decoder layer, reusing the target's embedding and LM head) measures **c = 0.059** against Monad's 0.476, with only **4.2M trainable params** (1.3% of target). 78% of that draft step is the 65,536-wide LM head, so cutting the draft vocab to 16k drops c to 0.019 and the projection to 2.2-2.8x at alpha 0.6-0.7 (8,192 types cover 98.75% of occurrences on English technical prose). Untested below 1.7B, but AngelSlim's Qwen3-1.7B EAGLE-3 shows 1.69x with the series' *highest* acceptance, so the risk is overhead, not acceptance. Training cost extrapolates to a few single-GPU hours; the harvest is the bottleneck, and at 192 tok/s on 4 cores a 50M-token harvest is 72h, which needs a rented GPU rather than CCotw. Oskar: "can we base eagle around asd-ste100?" — **no**, in [`STE.md`](monad-specdec/STE.md). STE's approved dictionary is the CLOSED half of an deliberately OPEN language: the STEMG excludes technical names/verbs by design and lets them contain otherwise-unapproved words, so STE bounds the function words a drafter already gets right and leaves the content words it misses. A ~900-word list expands to **3,296 BPE token types** (3.66/word after casing and leading-space forms) and only **54.3%** of corpus BPE tokens are plain alphabetic words at all. It would land between the 8,192 and 4,096 rows of the draft-vocab sweep where returns are already flat (c 0.011 -> 0.010, ~0.03x of projected speedup), versus an 8k head derived by counting tokens for free and with no license. STE-shaped text is also NOT lower-entropy under Baguettotron: 3.01 nats vs 2.79 for general prose at identical top-1 prob, though greedy hit rate is 17% better (0.457 vs 0.392); Python code beats both at 2.12 nats / 0.620. Corpora are 72-198 hand-written tokens, so this rules out a large effect, not a small one. Also corrects RESOURCES.md: the 1.3x ceiling bounds WEIGHT BANDWIDTH only and the flat quant ladder indicts llama.cpp's DEQUANT kernels, not 4-bit — T-MAC's lookup-table kernels attack the arithmetic term and report 4-5x on 3B CPU. remex hit the opposite on the same idea in retrieval (`search_adc`: 152 ms/query at 39 MB vs cached dequant 3.9 ms at 192 MB, 39x SLOWER — a memory optimization, not a latency one). At width 576 per-layer cost is fixed-overhead-dominated (1.57 vs 5.06), so T-MAC likely transfers poorly here. **Quant x drafter interaction measured** (Oskar: "did you test EAGLE with baguettotron - and which quant?" — NO: no trained head exists, so no acceptance number anywhere; the modules are randomly initialized and TIMED, valid for cost only, and c=0.059 was torch fp32 with the GGUF ladder a separate measurement). c does NOT survive quantization, because an EAGLE draft step is 80-84% vocabulary projection while quantization speeds up the 80-layer stack. torch, both sides dynamic-int8: target 121.9->102.1 ms (1.19x) but draft 6.02->5.94 ms (1%), so **c 0.049 -> 0.058, +18%**, projected 2.36x -> 2.28x. llama.cpp sharpens it: Baguettotron ties embeddings so there is NO separate output.weight — `token_embd.weight` IS the LM head, and it stays **Q8_0 in Q4_0 and Q4_K_M alike** (in Q4_0 it is the lone Q8_0 tensor among 560 Q4_0 ones), so the draft step's dominant term is byte-identical across quants while the target drops 28.3->23.9 ms. Quantization and speculative decoding are SUBSTITUTES here, and a reduced draft vocab matters MORE on a quantized target. Also corrects the Q4_K_M anomaly: only **40 of 561** quantized tensors are actually Q4_K (rest Q5_0/Q6_K/Q5_1), so it is a ~5-bit build, which is why it is larger than Q4_0 and slower. **A head was then actually built and trained** — [`DRAFTER.md`](monad-specdec/DRAFTER.md), answering Oskar's "what is a trained head, and why can't you do it without a GPU?". Turns out IT DOESN'T NEED ONE: harvest 250k positions in 9.7 min at **431 tok/s** (correcting the 192 tok/s in RESOURCES.md, measured under core contention), train the 4.2M-param head (FC 1152->576 + one decoder layer, borrowing the target's tied+frozen embedding/LM head) for 12 epochs in 11 min at 0.51 s/step on 4 CPU cores. Acceptance goes **0.0000 untrained -> 0.4095 teacher-forced in-domain -> 0.207 free-running**, against EAGLE's published 0.74-0.79. **DATA, NOT COMPUTE, IS THE BINDING CONSTRAINT**: more passes saturate at 0.427 by epoch 12 then decline while train loss keeps falling 1.87->1.43 (memorization), but more data still pays ~+0.05 per doubling (56k/112k/225k tokens -> 0.274/0.310/0.374). EAGLE's recipe is ~200x this harvest; at 431 tok/s that is 32h of forward passes and ~58 GB, which outlives a session — so the GPU buys wall-clock and disk, not FLOPs. End-to-end greedy specdec with the head is lossless (every run token-identical) but still **0.96x at gamma=1**, projecting 1.15x once the draft head keeps its own KV cache. Caught a real bug worth 53% of acceptance: feeding a transformer layer one position with no history leaves its attention nothing to attend to (gamma=1 read 0.135 until the head saw its own context). Domain mismatch explains the rest — the Python prompt scores 0.333 vs 0.09-0.23 for prose, and the training corpus was this repo's markdown and .py files. **RETRAINED IN DISTRIBUTION — the drafter now BEATS baseline.** Rebuilt the corpus on **PleIAs/SYNTH rendered through the model's own chat template** (4:1 with wikitext-103), since Baguettotron is a reasoning model whose assistant turn opens with `<think>` — the first run's repo-markdown corpus was out of distribution on both counts. Harvested 2.72M tokens at 431 tok/s into 11 shards. Data scaling at 4 epochs on an identical held-out set: **975k -> 0.364, 1.97M -> 0.401, 2.72M -> 0.412**, about **+0.037 per doubling**, holding at every epoch-matched pair. (The earlier 0.4095 is NOT comparable — different held-out set; a scaling curve is only valid inside one eval distribution.) **End-to-end greedy specdec, lossless (every run token-identical): 1.092x measured at gamma=1, 1.052x at gamma=2**, projecting 1.32x/1.22x once the draft head keeps its own KV cache (it currently re-runs its whole prefix each round, so the measured column is pessimistic). On the OLD raw-prose prompts the same head gets alpha 0.265 and 0.986x — prompt distribution alone is worth 0.386 vs 0.265. gamma>2 still degrades faster than published EAGLE because the head trains on next-token prediction only, with no feature-regression loss and no training-time feedback. Depth bet verdict: 80 layers is WHY a 1-layer head wins (a cost ratio no wide-shallow model of the same size offers) and also why Monad failed (latency tracks depth); the narrow 576 width + 65,536 vocab makes the output projection worth 10.7 layers by params / 3.85 by time vs 0.65 for Llama-2-7B, so reduced draft vocab is the largest remaining lever. No GPU at any point. [`HANDOFF.md`](monad-specdec/HANDOFF.md) ports it to Apple Silicon with a streaming trainer that removes the 58 GB storage ceiling; SYNTH holds ~0.88B tokens so data is no longer the limit. |
| [`monad-bsky/`](monad-bsky/RESULTS.md) | 2026-08-18 | **done — a fine-tuned 56M generalist reaches ~2/3 of a purpose-built 45M tool-caller; the gap is transcription, not choice** | [`RESULTS.md`](monad-bsky/RESULTS.md) + [`ERRORS.md`](monad-bsky/ERRORS.md) + [`params.json`](monad-bsky/params.json) + [`recheck.py`](monad-bsky/recheck.py) + `results_*.json` | Oskar: "do the same but fine-tune Pleias' Monad on the same task". Direct continuation of `needle-bsky` — same 18 Bluesky tools, same 62-query eval, same scoring code (imported, not copied), same 800 training rows from the same generator and seed. [Monad](https://huggingface.co/PleIAs/Monad) is 56M params, 64 layers x 256 hidden, trained on SYNTH, with no decode grammar and no confidence head. **Zero-shot it routes nothing** — 0/54 routable, 0/62 parseable, it analyses the instruction instead of answering it. A full fine-tune (3 epochs, 108 min on 4 CPU cores) takes it to **0.481 routable at epoch 2** against Needle's 0.611 base and 0.722 best config; paired McNemar vs Needle-LoRA **p=0.043**, vs Needle two-stage **p=0.0037**. **The gap is transcription, not routing**: over the 41 eval arguments that appear verbatim in the query, Monad reproduces **0.512** against Needle's 0.780 base / 0.902 LoRA — `austegard.com` comes back as `afethew.com`, `jetstream` as `jetforek` — and *more training makes copying worse* (0.561 at one epoch). **The obvious explanation is wrong and was retracted before publishing**: both models carry 8,192-piece vocabularies and segment these strings identically (111 vs 109 pieces over ten identifiers), so the cause is the training objective — Needle's base weights, never exposed to this data, already copy at 0.780. It also invents undeclared tool names on 6.5-14.5% of queries, which a decode grammar makes impossible by construction. Constructive fix measured: keeping Monad's tool choice and refilling arguments by regex lifts args 0.296 -> 0.370 against a 0.444 ceiling, and what it cannot fix is free-text search terms, which have no structure to extract. Two invented numbers caught in one draft (Needle's vocab size, its layer count) — see `ERRORS.md`. **Synergy (`synergy.py`, eight combinations, pure post-processing over both experiments' committed rows): one works.** Where the two models independently name the same tool, that answer is right **0.880** across 0.455 coverage, against **0.741** for Needle's own confidence head at matched coverage — and the two signals compose (0.929 at 0.255). Agreement needs no confidence head, which is exactly what fine-tuning Needle destroys; the price is running both, 11x latency. **Clean negative: calibration does not transfer** — Needle's confidence separates its own correctness (0.584 right / 0.392 wrong) but is flat-to-inverted for Monad (0.486 / 0.532), and worsens with threshold. The other five combinations do not pay for a second model: union ceiling 0.806 vs 0.710, name-snapping buys ~3.7pp not 14, split roles (Monad chooses / Needle transcribes) reaches 0.407 args against 0.685 for Needle doing both, fallback rescues 1-2 queries, per-category dispatch tops out at 0.758 and is fitted. **Four follow-ups (`cascade.py`, `classifier.py`, `regex_only.py`).** A **retry cascade** (Needle confidence, then agreement, then escalate) reaches **0.613 coverage at 0.842 precision** where the confidence gate alone gives 0.323/0.800 — cascading beats any single gate. Rewriting the ask adds 1-3 of 24 escalated queries and lowers precision; the english→Monad→Needle pipeline is unavailable because Monad corrupts handles *inside its own think trace*. Scoring Monad as an 18-way **classifier** over the declared names removes hallucinations and yields a softmax confidence but drops routable to **0.241** from 0.481 (verified not a harness bug via independent forward passes). And the null model nobody had measured: **20 regex rules with no model route at 0.833 routable, 0.022 ms** — beating Needle's two-stage 0.722 *and* its oracle ceiling 0.778, holding at **0.824 on unseen template queries**. The models' remaining edge is refusal (0.625 vs 0.183) and a confidence score. |
| [`needle-bsky/`](needle-bsky/RESULTS.md) | 2026-08-18 | **done — schema wording is worth +26pp; the confidence gate only works if you declare no optional arguments** | [`RESULTS.md`](needle-bsky/RESULTS.md) + [`ERRORS.md`](needle-bsky/ERRORS.md) + [`params.json`](needle-bsky/params.json) + [`recheck.py`](needle-bsky/recheck.py) + `evalset.jsonl` + `results_*.json` | Oskar, overnight: "implement Cactus Needle in your compute environment and have it set up as an interface in front of some tools — maybe the Bsky tools in muninn-utilities and/or the ATProtoing skill". [Needle 2](https://cactuscompute.com/needle) is a 45M-parameter, 14 MB tool-calling model for phones and microcontrollers; this puts it in front of an 18-tool Bluesky read surface drawn from the `browsing-bluesky` and `atprotoing` skills, with a CLI (`route` / `ask` / `repl`) and a 62-query eval set. **Base-model top-1 routing is 61–70%** across a 2×2 of schema wording × argument arity; only the wording contrast is significant (`auto` 0.444 → `tuned` 0.704 on 54 routable queries, paired McNemar **p=0.0072**). **Schema arity moves the confidence gate**: declaring an optional argument the query does not license makes the model fill it anyway, and since the head scores the whole call, a correct routing decision lands at **0.0004**. Drop optional arguments and the same gate becomes monotone and usable — 38% coverage at 0.762 precision, 20% at 0.909, 13% at 1.000, against `tuned`'s 2% coverage for the same precision. Confirmed on a **single-tool catalogue** where misrouting is impossible: 30 queries, confidence falls 0.199 → 0.111 → 0.068 as unlicensed arguments are declared, sign test p=0.043 / 0.043 / **0.00032**. Two more numbers for anyone deploying it: an **oracle five-tool catalogue is worth +11 to +17pp** (retrieval, not selection, is where most remaining errors are — best measured arm 0.815), and **declaring a sixth tool costs 3.6× the per-turn latency** (284 ms → 1034 ms) and then nothing more out to 18, because retrieval is a fixed per-turn cost above five. `tool_index_path` does not help — it caches tool embeddings, not the per-turn query embed. **Acting on both**: splitting the 18 into five groups of ≤5 and routing in two steps works, but only if stage 1 is not a model — a Needle turn over group descriptions scores **0.370** routable (24pp *worse* than the flat 18) while ~20 lines of regex over structural cues scores **0.722** (+11pp over flat, against a 0.778 five-tool ceiling). Needle's contrastive retrieval head is much better at picking 5-of-18 than Needle-the-model is at picking 1-of-5 categories. Also found: the engine holds one global session per process, so alternating agents re-runs `needle_init` every turn and a loaded `.cact` can never be unloaded. **The LoRA arm is a negative**: 800 templated rows and ~2h of CPU moved routing 0.611 → 0.667 (paired McNemar **p=1.0**), left `profile` (0.25) and `identity` (0.33) — the two categories the templates specifically covered — completely unmoved, regressed off-topic refusal 0.625 → 0.375, and replaced every confidence score with `None`, since fine-tuning does not update the confidence head. **Extraction is a second negative**: across 22 attempts over live and constructed posts, every single one scored below 0.05 confidence (max 0.0434) — at any threshold that makes the routing gate useful, all of it escalates. Route with this model; do not extract with it. |
| [`orchestrated-coding-pareto/`](orchestrated-coding-pareto/RESULTS.md) | 2026-08-16 | **done — orchestration arms never activated; token economics, not accuracy, set the frontier** | [`RESULTS.md`](orchestrated-coding-pareto/RESULTS.md) + [`ERRORS.md`](orchestrated-coding-pareto/ERRORS.md) + [`params.json`](orchestrated-coding-pareto/params.json) + [`recheck.py`](orchestrated-coding-pareto/recheck.py) + `tasks/` + `data/` | Continuation of `luna-onprem-tco` (PR #37): is a big orchestrator driving a fleet of Luna-class workers Pareto-optimal for coding? Built a 14-task bank (precise specs, hidden pytest suites validated against references before any model saw them) in three escalating tiers, run one-shot at haiku/sonnet/opus tiers in the CCotw Workflow harness with per-arm output-token metering, plus two orchestration arms (haiku+raw-test-feedback, opus-diagnoses→haiku-fixes) seeded from haiku failures. **Quality saturated everywhere: haiku 14/14 = opus 14/14, sonnet 13/14** (accepted `1.0.0-01` as semver) — the ceiling survived two difficulty escalations ending in a 20-opcode stack-VM and a character-exact table formatter, so **both orchestration arms went vacuous: zero haiku failures to orchestrate over**. The measured story is verbosity: **haiku emitted 6.7× opus's output tokens** (280,717 vs 42,019; 11× on tier-3), which more than cancels its 5× per-token discount — **haiku-solo $0.101/task vs opus-solo $0.079/task at equal quality**; at Anthropic prices the cheap tier is Pareto-dominated by frontier-solo. Repricing haiku's measured token profile at Luna direct ($0.20/$1.20) gives **$0.024/task (4.2× cheaper than opus-solo)**, DeepSeek-Flash $0.0057 (13.8×) — the thesis lives or dies on the fleet tier's sticker price and verbosity discipline, not on orchestration structure. Tie-back to PR #37: an orchestrated fleet is batchable (Luna batch halves it again) but at ~20k output tokens/task, saturating the 7.64 B-token/night self-host break-even needs ~380k tasks/night — self-hosting still doesn't pencil. Caveats carried in the writeup: all tasks are single-module fully-specified stdlib work (the saturation claim does NOT extend to ambiguous/multi-file/long-context coding); the harness's `effort: medium` has no documented mapping on Haiku 4.5, so some verbosity may be tunable; Luna prices inherit PR #37's secondary provenance. Tier-1's difficulty was misjudged and caught mid-run by early grading after Oskar questioned it — disclosed as an adaptive extension; 3 of 14 hidden suites had authoring bugs caught by reference validation before any model ran. **Follow-up (same day, Oskar's effort question):** re-run at `effort: low` cut haiku's tokens only 26% (verbosity is intrinsic, not the knob) but broke the ceiling — 12/14 — which finally activated the orchestration arms on the 2-failure seed: **raw pytest feedback and opus-diagnosis both went 2/2 in one round**, so the measured orchestrator premium over mechanical test feedback is zero quality at +$0.017/task; the best cheap pipeline (haiku-low + test-retry, 14/14, $0.080/task) ties opus-solo at Anthropic prices and wins 4.1x at Luna prices. |
| [`mdbr-leaf-mt-bench/`](mdbr-leaf-mt-bench/RESULTS.md) | 2026-08-16 | **done — no swap of the remax_kb default; leaf-mt-int8 ties bekko-a8m at 5.2x smaller** | [`RESULTS.md`](mdbr-leaf-mt-bench/RESULTS.md) + [`ERRORS.md`](mdbr-leaf-mt-bench/ERRORS.md) + [`recheck.py`](mdbr-leaf-mt-bench/recheck.py) + `results_*.json` | "Another embedding model to evaluate": [MongoDB/mdbr-leaf-mt](https://huggingface.co/MongoDB/mdbr-leaf-mt), 23M params / 1024-d, distilled from mxbai-embed-large-v1, **#1 on MTEB v2 (Eng) ≤30M** — run through `bekko-embedding-bench`'s Part B harness (179-chunk blog + 179-chunk sklearn-AST self-retrieval, incumbents re-encoded on the same splits so every verdict is paired). **The billing does not transfer: jina v5 nano q4 wins both distributions**, decisively on code R@1 — **0.888 vs 0.581, Δ −0.307, 2 wins/57 losses, p<1e-5** — the same cell, with nearly the same margin, that settled the bekko verdict; blog R@10 −0.067 (p=0.036), and iso-byte truncation widens the gap (leaf@64 vs jina@64 code: −0.251). **Where it does land: the compute-bound rung.** The int8 export (23.7 MB) is a paired statistical tie with bekko-a8m in all four cells (all p>0.07) at **5.2x smaller** and 8.0 vs 10.8 ms same-session 1-vCPU query — the smallest credible remax_kb embedder measured in this family, 19.2x faster per query than jina. Export quirk worth knowing: **leaf's int8 is faster than its own fp32** (7.3 vs 15.4 ms) with no significant retrieval cost — unlike bekko, whose transformer-int8 ships `_not_recommended` — while **q4 is dominated on every axis** (slower than fp32 at 1 thread, only export with a directional code dip). Cross-run anchors all reproduced exactly (jina 0.631/0.978, bekko 0.575/0.888, corpus 11,380 chunks/674 files at sklearn `7cb1868aa`); 30/30 recheck. Part A deliberately not run (an encoder that loses to jina on code has no path to moving "dense ties grep"); asymmetric teacher-doc mode (mxbai docs + leaf queries, the card's strongest configuration) untested — the natural next pass for an offline-index budget. **Codec follow-up (`pareto.png`): remax does not beat the card's own plain sign bits** — at the shared 128 B, blog 0.503 vs 0.547 (n.s.), remex 1-bit level with vendor binary on both dists, so the rotation/centering machinery buys nothing on a model whose quantization robustness was distilled in; **quantize-before-truncate reproduces on a second model** (remex 2-bit @1024 beats the fp32 MRL floor d=64 at equal bytes, +0.073 blog p=0.015 / +0.117 code p<1e-4, and ties the uncompressed 4096 B vector); the Pareto frontier is composition — binary-asym d=512 at 64 B hits full-fp32 quality (64x compression free), MRL-fp32 is dominated everywhere; and kb-k-sweep's "dims beat stacks" inverts within remax here (d=512 k=2 > d=1024 k=1 at 128 B). 40/40 recheck. |
| [`luna-onprem-tco/`](luna-onprem-tco/RESULTS.md) | 2026-08-15 | **done, two passes — API wins at 800 seats by 2.7x; one RTX 5090 wins above ~4 h/day of flat-out generation** | [`RESULTS.md`](luna-onprem-tco/RESULTS.md) + [`ERRORS.md`](luna-onprem-tco/ERRORS.md) + [`model.py`](luna-onprem-tco/model.py) + [`hourly.py`](luna-onprem-tco/hourly.py) + [`params.json`](luna-onprem-tco/params.json) + [`recheck.py`](luna-onprem-tco/recheck.py) | Asked to price the raw electricity of running a GPT-5.6-Luna-equivalent locally in Montgomery County MD against Luna's API price. **The literal question has a clean answer that decides nothing.** Electricity at EIA's MD commercial rate (16.4 ¢/kWh) is **$0.0035 per million input tokens** on a GB200 NVL72 — Luna direct ($0.20/M) is **57×** that and even Luna *batch* ($0.10/M) is **29×**. But electricity is **3.1%** of the cost of owning the hardware, so the ratio is load-bearing for nothing. **The binding constraint is a memory floor.** Capability parity is set by measured index, not parameter count (OpenAI publishes none, and the "27 B dense" figures in circulation are unreliable): **DeepSeek V4 Pro 0813 at AA index 53 vs Luna max 52**, MIT-licensed — which means 1.6 T params, ~800 GB NVFP4, and a **~$450 k 8×B200 node minimum**, bought whole at 5% utilisation as at 85%. At 800 office seats (55% daily-active, staggered 07:00–18:30, 2.5 h nightly batch) three usage intensities give peak utilisation of **5% / 17% / 85%** and API bills of **$6.1 k / $29.3 k / $153 k** against **$279 k** self-hosted — API wins by 3.5× / 2.7× / 1.2×. **The overnight batch cannot rescue it, and not for a reason that depends on any usage estimate**: fully saturated for 2.5 h the box emits **1.98 B input tokens/night** where break-even needs **7.64 B** — short by 3.9× at 100% saturation, across the entire 20–35% MFU sensitivity band. The window where the box could run flat out is also the window where the API is half price; the two compound. **It flips on token intensity per node, never on seat count** — scenario B stays API-side at 6,000 seats because past one node each added seat buys capex as fast as savings, while scenario C flips at ~2,000. Two errors in the model itself were caught and fixed structurally, both flattering self-hosting: a verdict of "self-host" returned at **845% peak utilisation** (one node's cost vs a nine-node bill; now `nodes = ceil(peak)`), and a *serving* power floor applied to *parked* hours (+35% annual kWh). Spin-off finding: **rack benchmarks do not transfer to single nodes** — 8×B200 decode costs **$0.083/M output** against the NVL72 rack's **$0.014/M** on identical GPUs, 5.8×, because MoE decode scales with NVLink-domain size, while prefill is flat. **Caveat that dominates all others:** the session's egress proxy blocked `WebFetch` and `curl` to every primary source (openai.com, eia.gov, pepco.com, artificialanalysis.ai, inferencex.semianalysis.com), so **every constant came from a search-engine summary rather than the page**; `params.json` tags each row's confidence and `recheck.py` (104 checks) fails if one lacks a source. The workload scenarios are authored, not sourced — the published per-seat figures found sit ~7× above even the heavy case and were rejected as not credible, which is the single judgement call most worth challenging. **Second pass (2026-08-16), single GPU:** same price book, 1/1000th the scale — one RTX 5090 (600 W, $4,700 street median in an August-2026 shortage) running **Qwen3.8-27B** (released 14 Aug, Apache 2.0, 27.78 B **dense**), a far closer capability match than V4 Flash was at **SWE-Bench Pro 61.7 vs Luna's 62.7**. Electricity is **$0.18/hr** at Maryland's EIA-corrected residential 22.2 ¢/kWh against **$0.90–1.49/hr** of Luna for the same output — **5–11×**, not 57×, because a consumer card serving one stream costs **$0.19–0.26 per million output tokens** against the 8×B200's $0.083 and the rack's $0.014. **It answers the opposite way from the fleet model, for the same reason**: break-even is **4.2–7.6 h/day of flat-out generation** at street prices (2.3–4.2 at MSRP-era prices), so the shortage roughly **doubles** the break-even and the used-GPU market outweighs electricity, model choice and rate schedule combined. Two premise corrections generalise: a quoted **180–200 tok/s is 1.56× the hard `bandwidth ÷ weight_bytes` decode ceiling** (121 tok/s; ~97 at 80% MBU), reachable only via this model's multi-token prediction, speculation or batching — kept as a branch rather than corrected away, with a roofline-respecting 95 tok/s branch beside it; and **prefill and decode contend for one card**, turning 190 tok/s into a sustained 123–167 once `1/(1/decode + fresh_ratio/prefill)` is applied. Price-book finding worth carrying alone: Luna's cache **writes** cost 1.25× uncached input, so **caching only pays above a 21.7% hit rate** — below it, re-sending is cheaper. Two further errors logged, one flattering self-hosting (prefill contention omitted in the scratch pass) and one overstating a correction ("1.9× the ceiling" quoted against the 80%-MBU figure, not the ceiling). `recheck.py` now runs **154** checks. |
| [`ttt-embed-quantized/`](ttt-embed-quantized/RESULTS.md) | 2026-08-14 | **done — artifact committed; fp32 nDCG@10 0.7152, inside the expected band** | [`RESULTS.md`](ttt-embed-quantized/RESULTS.md) + [`ERRORS.md`](ttt-embed-quantized/ERRORS.md) + [`encode.py`](ttt-embed-quantized/encode.py) + [`recheck.py`](ttt-embed-quantized/recheck.py) + `data/{Dm,Q}.npy` + `data/meta.json` | One-time SciFact corpus encode for the TTT-Embed x remex/remax quantization experiment ([#33](https://github.com/oaustegard/experiments/issues/33)), so claude.ai — <2 docs/s on 1 core, detached jobs reaped after ~100 s — never pays for it again. Not a hypothesis test: the deliverable is `Dm.npy` (5183, 256) fp32 + `Q.npy` (300, 256) fp32 + `meta.json`, encoded with jina-v5-nano `model.q4.onnx` @ `v5-nano-8a7f00aa` (SHA256-verified) at the 2026-07-08 codec eval's exact settings — `dim=256`, `max_length=384`, `title + ". " + text`, `Document: `/`Query: ` prefixes, last-token pool, truncate-then-L2-normalize — so that eval's fidelity numbers carry over. **fp32 nDCG@10 = 0.7152** (R@10 0.8346, R@100 0.9483), inside the issue's 0.60–0.72 band near the top; **14.7 min on 4 vCPU** at 5.9 docs/s. Re-scored cold by `recheck.py` through a deliberately disjoint code path (`sorted()` over Python floats, explicit `math.log2` DCG) reproducing **0.715232 vs 0.715232** to <1e-9, 17/17 checks, with negative controls that collapse to 0.003–0.004 — a sanity check that cannot go red is not evidence. **The issue's HF-CDN warning did not reproduce**: all three files landed first try on `us.aws.cdn.hf.co`, the host the spec calls un-allowlisted, confirming `bekko-embedding-bench`'s per-environment reading — allowlist state is a fact about a container, not about a host, and the retry loop is kept only for the claude.ai path. **The pinned encoder is knowingly superseded** — the mirror's own `PERFORMANCE.md` and this repo's `METHODS.md` both say the authors' upstream q4 is smaller *and* more faithful — and was used anyway, because comparability with the prior eval requires identical weights, not better ones; flagged rather than silently upgraded. **Prior art was found and deliberately declined**: `rotation-decorrelation` already caches a `jina_scifact_corpus.npy` for this exact corpus and embedder, but it is corpus-only and its settings are unrecorded, so reuse would have risked a silently non-comparable matrix that every shape check would have passed. **Caveats:** 27% of docs hit the 384-token cap, so better than a quarter of the corpus is encoded from a prefix; no fp32-vs-q4 fidelity was re-measured on SciFact (the 0.975 cosine is inherited from NFCorpus/muninn); and `pytrec_eval` does not build here, so both scorers share an author. |
| [`subagent-messaging/`](subagent-messaging/RESULTS.md) | 2026-08-12 | **done — 4 of 5 documented claims hold; the reply rule does not** | [`RESULTS.md`](subagent-messaging/RESULTS.md) + [`ERRORS.md`](subagent-messaging/ERRORS.md) | Does the Claude Code `SendMessage`/`ListAgents` tool pair behave as its description says? Live test rather than reading: Opus 5 parent, Haiku 4.5 `general-purpose` peer instructed to report every envelope verbatim, 3 agent runs, ~121k subagent tokens. **The reply rule is wrong.** Both the tool doc and the harness footer appended to every delivered message say to reply by copying the incoming envelope's `from` into `to`; for subagents that value is the agent *type*, so the send returns `No agent named 'general-purpose' is reachable`. Two `general-purpose` peers emit two identical unusable `from` values — the attribute cannot distinguish senders even in principle, and the agentId from the spawn result is the only handle. **Envelopes are asymmetric**: `<agent-message from=…>` parent-side, bare `<system-reminder>` subagent-side, so a subagent cannot route a reply by inspecting what it received. **`ListAgents` does not exist inside a subagent** — `ToolSearch("select:ListAgents")` returns `No matching deferred tools found`, not an unloaded schema — so peers cannot be discovered from below and the topology is a **star through the main conversation, not a mesh**; peer-to-peer coordination requires the parent to hand out ids at spawn. Confirmed as documented: delivery **enqueues to the receiver's next tool round and never interrupts** (probe 2 landed after a `sleep 20` finished, not during it), and **resume-on-send works with context intact** — but it is **undetectable by the resumed agent** (asked directly, it reported no gap or restart marker, "reads as one continuous conversation") and costs **a full agent turn, ~40k tokens, each time**. Also observed: the harness **neutralizes instruction-shaped tags** in agent→parent output (`<` → `<\`) and relabels them as findings rather than instructions. **Prior art, two passes.** Account-local found nothing (zero `sendmessage` hits across claude-workspace, scoped `xr` nothing above 0.371); account-wide `xr` surfaced `claude-skills/orchestrating-agents` at 0.539, a different mechanism (API agent pools) that **carries a factual defect** — v0.5.0 tells the reader the native runtime *lacks* inter-agent messaging and never mentions either tool. **The published pass demoted two findings**: cross-session messaging shipped 2026-08-07 in v2.1.224 with a thorough [official page](https://code.claude.com/docs/en/cross-session-messaging) that states the queue-never-interrupt model almost verbatim, and [claudefa.st](https://claudefa.st/blog/guide/agents/persistent-subagents) already documents resume-with-intact-context and a better-measured cost chain (199k→324k over eight rounds). Still unpublished anywhere found: the `from` attribute, the envelope asymmetry, absent `ListAgents` in subagents, and resume being undetectable by the agent. **And it gained a contradiction the account pass could not**: [claude-code#48160](https://github.com/anthropics/claude-code/issues/48160) (closed as duplicate) and [ruflo#2028](https://github.com/ruvnet/ruflo/issues/2028) (open) both report that subagents can receive but **cannot originate** `SendMessage` — this peer originated three sends to `main` successfully with no `AGENT_TEAMS` flag, lacking `ListAgents` instead, the exact inverse. Either fixed since, or environment-specific; CCotw vs local terminal is the uncontrolled confounder. Closest published match to the addressing finding is [claude-code#42999](https://github.com/anthropics/claude-code/issues/42999) (closed as not planned), where a user-assigned *name* fails silently while the id works — adjacent but distinct, since `from` is not a name and fails loudly. **Caveats:** parent↔subagent only, one session; true cross-session peers (`<cross-session-message>`), Remote Control, named teammates, and `Workflow`-spawned agents all untested, and the `from`-as-address rule may well hold for them. |
| [`lowbit-scan-crossover/`](lowbit-scan-crossover/RESULTS.md) | 2026-08-09 | **done — positive; the reported scale gate is a 4.1 ms constant, and bit planes beat the shipped kernel 2.4–5.2x** | [`RESULTS.md`](lowbit-scan-crossover/RESULTS.md) + [`fit.py`](lowbit-scan-crossover/fit.py) + [`layout.py`](lowbit-scan-crossover/layout.py) + `roofline.py` + `arms.py` + `steelman.py` + `xover.py` + `hamkern.c` | Challenge the inevitability of BLAS beating low-bit storage at small corpora, per memory `dab41dd6` ("compression is SCALE-GATED and below ~150k rows its win is negative"). **It is not a scale gate.** `dab41dd6`'s own table fits `t = 4.108 ms + 32.40 ns·n`; the same numpy expression here fits `−0.78 ms + 33.98 ns·n` — **per-row cost agrees to 5% across the two machines, and the entire crossover is the constant.** `n* = a/(b_f32 − b_ham)` recovers the reported gate at **~68,000**, derived rather than interpolated between two rows. The generalisable form: in `t(n) = a + bytes·n/(BW·eff)`, `n` is a multiplier identical for every kernel, so **two kernels can cross only if one has `a > 0`** — a reported crossover with `a ≈ 0` on both sides is an artifact of the two `n` values bracketing it, and `fit.py` is the check. Nothing here crossed at any n from **100 to 1e6**, cold or warm, at any ISA from SSE4.2 up, single-query or batch-1024 (4.2x at batch 1024, the narrowest point). **Where the numpy time goes:** `np.bitwise_count` is fine at 14.7 GB/s; **`.sum(axis=1)` over a 4-wide inner axis is 1.9 GB/s and 62% of the kernel** — a reduction *shape*, not bit-packing. Storing the words as contiguous **bit planes** recovers **5.2x** at k=256 and **2.4x** at the shipped d=512·k=4 config, pure numpy, no compiled dependency; the compiled kernel is **37x** over BLAS warm, 19x cold, versus the 1.7x the shipped idiom gets. **Both disconfirming arms failed:** an adversarial `challenging` pass named AVX-512 VPOPCNTDQ as load-bearing and cold cache as untested — rebuilt at `-march=x86-64-v3`/`v2` with **zero `vpopcnt` in the object** it costs **6%**, and L3-evicted it is still 19x. So the original result is **not** attributable to the 1 vCPU container's core, cache, or instruction set. **Also corrects the correction:** `dab41dd6` refuted `remax-hamming-speedup`'s "beats BLAS at every N" using a *different configuration* — k=256 (4 words/row) vs the shipped d=512·k=4 (32 words/row) — and the narrow-reduction pathology is specific to the former; the shipped claim reproduces here (2.58x vs its published 2.43x at N=50k). **Caveats:** one machine; ARM/NEON untested; single-threaded; latency only, not latency-at-fixed-recall; the 4.1 ms constant's *cause* is unidentified, only its existence and size; and nothing transfers to the remex ADC path, whose 13.5x the family model predicts is gather latency — flat in `n` as `901e3c06` reports, but the bit-width leg is untested. **Prior art — the C kernel is a rediscovery.** `remax/src/remax/_native.py` already ships a `__builtin_popcountll` Hamming scan that `remax.packing.hamming_distances` already dispatches to, whose docstring already reports **25–35x over the NumPy path** and already identifies the 100k–1M cache falloff; `remax/core.py` and `QUERY_PATH_SPEED.md` already record that a gather cannot use the popcount kernel (38–45x). Externally it is textbook — faiss's `HammingComputer32` is four `uint64`s XORed and popcounted, and what gcc emits at v3 is Muła/Kurz/Lemire's Harley-Seal. `hamkern.c` is kept only as a fallback-free roofline reference. **The mandated account-wide `xr` check was skipped** — run afterwards it returns `remax/packing.py` at rank 1 and `_native.py` at rank 8, or rank 2 with `-r remax`, in 175 ms warm. A first attempt raised `ModuleNotFoundError: remex` and was written up as "`xr` is unavailable in this container" instead of fixed with `pip install remex onnxruntime tokenizers` (under a minute; `remex` is on PyPI from the same author). **An ImportError in a mandated check is a missing dependency, not a broken check** — that wrong diagnosis reached METHODS.md, RESULTS.md and a PR body before being caught. Logged in the duplication map, with the install recorded under Environment gotchas. **What survives as new:** the constant-term reconciliation, the `.sum(axis=1)` narrow-axis diagnosis (the `bitwise_count` path, not the LUT path `_native.py` analyses), the bit-plane numbers, the 6% `-march` measurement, and **a wiring gap — `remax_kb/_hamming.py` already imports from `remax.packing` but never calls its native dispatch, so the compiled kernel is one import away from the shipped scan.** **Method note:** a first pass labelled `inversion` was retrofitted onto a result already in hand and did not fire; the finding came from `family traversal` run properly afterward. |
| [`account-index-corpora/`](account-index-corpora/RESULTS.md) | 2026-08-09 | **done — qualified; account-wide PR bodies +4.6% (a floor), tombstones +7.1% but 94% deleted data** | [`RESULTS.md`](account-index-corpora/RESULTS.md) + [`clone_depth.py`](account-index-corpora/clone_depth.py) + `corpora_scoped.json` + `results.json` | Should the account-wide index carry the two corpora that measured as wins per-repo — deleted files (`history-tombstone-index`, 0/6 -> 6/6 on mechanism) and merged PR bodies (`pr-decision-log`, 6/8 -> 8/8 on rationale)? `claude-workspace#197` names the risk correctly as **size, not answer quality**, since an off-class corpus already measured inert rather than harmful. Size is the half that costs seconds instead of a 22-minute sharded encode, so it was answered alone, via a new `account.py corpora` that chunks through the real build path and never loads the encoder. Re-run over all 65 repos on a runner in 2 min 18 s: **PR bodies +4.6%** (1,953 chunks over 1,444 merged PRs) and **tombstones +7.1%**, against a tree of 42,578 chunks that matches the published manifest exactly — so `corpora` is measuring the real index, not an approximation. The PR number is a **floor**: 12 of 65 repos returned HTTPError on `/pulls` while cloning fine, so the PAT reads contents but not pull requests on them, dropping claude-workspace's own 154 merged PRs out of the total. Both 3-repo estimates were wrong in opposite directions (tombstones 11.8% -> 7.1%, PRs 3.2% -> 4.6%) without moving the verdict. On 3 repos first: **PR bodies +3.2%** (419 chunks, 261 merged PRs), and the account meets the condition `METHODS.md` records for believing the remax result transfers — median body 1,577–3,197 chars, 11 effectively empty. **Tombstones: +11.8% nominal and not worth it.** The first run said **+564%** — 74,822 chunks against a 13,257-chunk tree — because a deleted file gets no `stat()` and no `rglob`, so every filter `hcindex.discover` applies to the tree (extension, `skip_dirs`, `skip_names`, `exclude`, the 1 MiB cap) has to be reapplied by hand; without them a 767,692-line deleted embedding dump enters a corpus the live index refuses. Filtered it is 47x smaller, but claude-workspace still contributes **1,484 tombstone chunks against a 232-chunk working tree**, ~94% of it sub-1 MiB JSON data dumps and ~23 chunks of actual prose and source. Two account-scale effects the per-repo experiment could not see: the relocation guard must compare **across repos** (the 2026-07-28 migration deletes in one repo and lands in another — 540 files skipped only because the check was widened), and candidates are restricted by basename to keep it linear. **`--depth 50` resolved**: it cost nothing (6.5s vs 7.3s vs 7.4s full, summed, no consistent sign) *and* would not have worked — `git log --diff-filter=D` sees only the grafted window, so depth 50 found **2 of muninn-utilities' 18** deletions; coverage is a function of commit rate, not of anything anyone chose. Now depth 1, or full history when tombstones are on. **Caveats:** this measures *size only* — the answer-quality half still needs a full encode and a benchmark that does not exist; and the PR total is understated until the PAT gains pull-request read on the 12 repos it cannot currently list. |
| [`account-routing-tier/`](account-routing-tier/RESULTS.md) | 2026-08-06 | **done — qualified; 87-90% @k=3 of 9 repos, not safe as a default** | [`RESULTS.md`](account-routing-tier/RESULTS.md) + [`run.py`](account-routing-tier/run.py) + `results.json` | Can a small always-loaded index of per-repo summary cards route a query to the right partition, so a whole-account index can live as per-repo release assets fetched on demand? The failure mode is unforgiving: a flat index that ranks badly still contains the answer further down, while a coarse tier that routes wrong makes it **unreachable** and returns a confident result from the wrong repo. 9 repos on disk, 25,899 fine chunks; **the coarse tier is 0.1-0.35% of it**, so storage was never the constraint. 30 queries about *internals* only (`ascii_fold`, the CSR builder, NVFP4 dequant, sklearn's CSR `indptr`) so cards must route on similarity to a summary that does not contain the answer; gold is an oracle (flat RRF over all chunks) rather than hand labels. **Content cards beat front-matter cards at every k with 2.5x fewer chunks** (@3 80% vs 73%, 26 vs 64 chunks); both together reach **@1 53%, @3 87%, @5 97%**. The diagnosis came from a *failed* fix: 4 of 8 initial misses wanted sklearn-bench whose card lacked a README because CARD_FILES held only `README.md` and scikit-learn ships `README.rst` — fixing that moved recall@1 **47% -> 43%** and the same four queries still missed. scikit-learn's README has **zero** occurrences of "gradient boosting", "one-hot", "cross validation", "sparse" or "estimator" against 296 and 311 files in-tree; **front matter states identity, routing needs inventory**. Splitting large repos into per-directory cards to fix a 350x card-capacity imbalance (sklearn 0.033 card-terms/chunk vs 11.8) **backfired** — @1 53% -> 47% for 3.7x the cards — because ranking a repo by its *best* card makes more cards more draws, inflating a split repo's maximum for reasons unrelated to relevance. **Verdict:** 13% of queries land in no fetched partition at k=3, silently; k=5 fetches over half the partitions and defeats the point at this scale. Usable only behind **confidence-gated escalation** — route to top-3, widen if the best fine score is weak — which turns a silent wrong answer into latency. **What broke:** a confidently wrong diagnosis (above); the harness sat inside the corpus it measures for the **fourth** time here, after being diagnosed in `code-index-duplication` and guarded in `hybrid-code-index` — knowing a failure by name did not prevent reproducing it twice more; the oracle is not reliable gold (one query routed correctly to `claude-container-layers` at rank 1 and was scored a miss because flat search disagreed), so these are agreement-with-flat numbers, not accuracy; and self-pollution is structural, since `experiments` is both an indexed repo and where these writeups live. |
| [`pr-decision-log/`](pr-decision-log/RESULTS.md) | 2026-08-05 | **done — positive; 8/8 vs tree's 6/8, and the three corpora are orthogonal** | [`RESULTS.md`](pr-decision-log/RESULTS.md) + [`run.py`](pr-decision-log/run.py) + `prs.json` | Are PR descriptions worth indexing as a searchable decision log? Code says **what**, commit messages say **what changed**, PR bodies say **why** — including what was rejected. Proposed as an alternative to hunk-level change indexing, and the volume argument alone favours it: 43 merged remax PRs is 87 chunks, **+12% over the tree**, where hunks would be thousands of near-identical neighbours (the pollution that cost `repo-index` 20% of its corpus). Eight "why" questions written from **CLAUDE.md's claims** rather than PR text, scored as marginal value over a tree whose CLAUDE.md *already* documents decisions. **tree 6/8 -> tree+PRs 8/8.** The two gains are cases where the tree carries the outcome but not the reason: `[PR #65] Consolidation: −4,879 lines` for why the bench harness left the wheel (CHANGELOG.md ranks first in both arms and records only *that* it happened), and `[PR #61] Restore rotations_ assignment via write-through setter` at rank 1. **Tombstones add exactly nothing on rationale (6/8 -> 6/8)** — which is the more useful finding: the three corpora are orthogonal, tree answering *what*, tombstones *how did the deleted thing work*, PRs *why*, and a corpus aimed at the wrong question class is inert rather than harmful, so they stack. **Caveats:** the generalization threat is severe and was recorded before running — every remax PR is Claude-authored, median body **2,727 chars**, none empty, where most repos have one-line or blank bodies; this measures "PR bodies are worth indexing *when written like this*". n=8, one repo. One gold list held the bare substring `"PR #"` matching any PR chunk — fixed, and outcome-neutral (that query hits via CLAUDE.md at rank 1 in *both* arms), but it is the third answer-key defect of this shape in this line of work. **The real objection is architectural:** PR bodies are not in git, so indexing them makes network access and a token a hard dependency of a full rebuild — every other corpus here comes off the filesystem, which is why the indexer runs offline and in CI. A real version needs a cache with a staleness policy that degrades to tree+tombstones rather than failing the build. |
| [`history-tombstone-index/`](history-tombstone-index/RESULTS.md) | 2026-08-05 | **done — positive; 12/12 fused vs 5/12 working-tree-only** | [`RESULTS.md`](history-tombstone-index/RESULTS.md) + [`run.py`](history-tombstone-index/run.py) + `results.json` | Does indexing *deleted* code add anything over a repo that already documents its rejections? A current-state index structurally cannot hold code that no longer exists — but that is only valuable if the **knowledge** left with the code. `remax` is the hard case: its CLAUDE.md mandates *"a measured rejection is an asset — delete the driver, never the record"*, so removed apparatus leaves a prose writeup in `bench/results/*.md`. Tested against `remax` (144 commits, 10,042 deleted lines, **17 true deletions**) because this repo cannot answer it — 73 commits and **zero** files deleted-and-never-restored. Recovered each dead file at its last living revision, headed with its removing commit's subject, and scored RRF(dense, stored-BM25) over three corpora. **The convention works — for existence: 5/6 from prose records alone.** **It cannot work for mechanism: 0/6.** A record is prose about a *verdict*; "the encoder, its CSR-builder and a BEIR benchmark were all built" does not tell you the signature, the batching, or what the tests asserted — that left with the file, and no writeup discipline short of pasting the code retains it. Tombstone-only scores 4/6 + 6/6; **the union is 12/12, strictly better than either arm**, for **+19% corpus**. Arms are complementary, not competing. **What broke:** relocations look exactly like deletions — five `src/remax/bench/*` files were *moved* to `bench/*`, not deleted, and indexing them inflated the tombstone corpus 27% and manufactured a false current-only mechanism hit (a *live* `crossover.py` satisfying a query whose gold was its deleted path). A current-state index scoring on a mechanism-only query was the tell that the answer key was wrong. Fixed by detecting relocation **by content** (>50% of non-trivial lines present in a live file), not by path or basename — basename would have wrongly dropped `src/remax/bench/__init__.py`, which is genuinely gone. The correction *strengthened* the result (1/6 -> 0/6). **Caveats:** n=12, one repo, queries written by someone who knew the answers, gold matched on filename substrings, and only whole-file deletions are indexed — removed hunks inside surviving files are probably the larger population and are untested. The defensible product claim is narrower than "index your git history": **index what was removed and never came back**; everything else in history is a near-duplicate of content already indexed. |
| [`hybrid-code-index/`](hybrid-code-index/RESULTS.md) | 2026-08-05 | **done — hybrid wins 24/24; two assumptions refuted** | [`RESULTS.md`](hybrid-code-index/RESULTS.md) + [`hcindex.py`](hybrid-code-index/hcindex.py) + `bench.py` + `bench_incremental.py` | `repo-index` was a markdown sidecar with a code afterthought and a single dense arm; the target is a general-purpose **hybrid** code indexer. Six arms over three query classes (rediscovery / keyword / duplication), each scored against an answer key that already existed for another purpose. **`rrf(dense, stored-BM25)` scores 24/24**, beating bm25 alone (23), dense alone (22) and every rg variant. **Stored BM25 beats ripgrep as the lexical arm decisively** (23/24 vs 17/24), and the gap is almost entirely duplication (8/9 vs 3/9): ripgrep returns a *set*, and 'find me a file like this one' is a ranking question no amount of term-counting recovers. **Adding a third arm HURTS** — `rrf(all 3)` drops to 22/24, because RRF is unweighted so a weak arm votes as loudly as a strong one. More retrieval arms is not monotonically better. **The .json dilution was refuted:** 79% of the corpus is generated results data and it is *inert* (24/24 with and without), so no build-time exclusion is warranted — unlike the `outputs/` model-generation case, where near-duplicate *prose* competed directly with real answers. Volume does not predict pollution; **similarity to real queries does.** Storage is not free though: BM25 postings inflate to 6.36 MB / 138k terms with JSON vs ~1 MB without. **Rebuild cost forced incremental:** a full build is 537 s, vs 0.2 s for a one-file change (**2735x**), and the incremental result is **verified bit-identical** (max delta 0.000e+00) rather than an approximation — safe because the encoder is per-chunk independent and remex is data-oblivious. **BM25 cannot be incrementalized the same way** (IDF shifts for every term on any insert), which generalizes: any component fitted *on* the corpus — PCA, k-means, ITQ, PQ codebooks, IDF — breaks the equivalence that makes incremental safe. Incremental does *not* fix the committed-blob cost: 1.00 MB dense + 6.36 MB postings per rebuild is ~1.5 GB of git history at 200 rebuilds, which wants the artifact published as a release asset instead. **Not done:** graduating the winning arm into `ask.py`, moving the artifact out of git, a second-repo check, and history/tombstone indexing (untestable here — 73 commits and zero files deleted-and-never-restored). |
| [`code-index-duplication/`](code-index-duplication/RESULTS.md) | 2026-08-05 | **done — positive; shipped into `repo-index`** | [`RESULTS.md`](code-index-duplication/RESULTS.md) + [`run.py`](code-index-duplication/run.py) + `results.json` | Should [`repo-index/`](repo-index/README.md) index `.py` as well as `.md`? This repo's own `bekko-embedding-bench` already measured dense retrieval as **not** beating grep at NL->code *localization* (r@5 0.656 vs 0.596, n=59, ns) — so the usual reason to index code is a measured non-reason here. But localization is not the failure this repo has with code: `METHODS.md`'s **duplication map** records three independent reimplementations of one bench harness plus three more near-identical pairs, all found by hand after the fact. That map is an answer key written before this experiment existed, for an unrelated purpose. Scored leave-one-out hit@5 over 831 flat 60-line windows from 190 `.py` files (flat not AST, because AST-vs-flat was noise at p=0.424 in the earlier bench): querying with a file's own text finds a documented sibling **9/9 at ranks 1-3**, and **content-only scores identically to with-path-header**, so it is content matching and not filename matching — the confound that would have made the headline hollow. NL-description queries get 8/9; grep handed the most distinctive `def` name out of the query file gets 8/9. **Dense ties grep, it does not beat it** (n=9), but the arms need different things: grep's needs a draft containing a distinctive name, the NL arm needs no draft at all. The single NL 'miss' is the index returning `_lib/pipeline.py`/`_lib/textnorm.py` — which is where that code was *extracted to*, per the same map, so it is the better answer against a stale key. **Three process failures, all self-inflicted and all in the measurement:** (1) `run.py` embeds its own NL queries verbatim and so retrieved itself, top-5 on 4 of 9 — excluding it moved content-only NL 6/9 -> 8/9; (2) the first number described excluding only the query *file*, a configuration the shipped tool does not use, so `--file` and the harness were both changed to exclude the query's *directory* before any number was recorded; (3) after adding `.py` to `repo-index`, keyword agreement looked like it fell 10/10 -> 7/10, but **all three 'regressions' were the index returning the definition instead of a prose mention** (`ascii_fold` -> `_lib/textnorm.py`, `GRID_VERSION` -> `grids.py`) and the grep arm was still restricted to `*.md` — a baseline scoped narrower than the system under test reports improvements as regressions. Matched arm: 9/10, rediscovery unchanged at 5/5. Shipped: `repo-index` indexes `.md`+`.py` (0.14 -> 0.27 MB) and grew `--file`. |
| [`bekko-embedding-bench/`](bekko-embedding-bench/RESULTS.md) | 2026-08-04 | **done — split verdict; one prior reversed, one default upheld** | [`RESULTS.md`](bekko-embedding-bench/RESULTS.md) + `instances.json` + `recheck.py` | Handoff [claude-workspace#185](https://github.com/oaustegard/claude-workspace/issues/185): benchmark **hotchpotch/bekko-embedding-v1** (a8m/a25m, 384-d Matryoshka mmBERT) for two separate decisions. **Part A: the reversal did NOT survive a bigger sample — n=6 → n=59.** The first run said bekko beats identifier `rg` at r@5 0.806 vs 0.667 and cleared the pre-registered gate. Re-mined to **n=59** (630 PRs harvested, 97 candidates with live gold), dense/a8m lands at **0.595 against grep's 0.596 — a dead tie** — and *worse* at r@10. **No dense-vs-grep comparison is significant in any of the four cells.** The gate now passes or fails depending on which cell you pick, which is itself the finding. The identifier-poor stratum is **still n=1 of 59** — 10x the sample bought zero additional instances, independently corroborating the ~0.3% base rate. **So the 2026-07 retirement of the semantic tier stands; the n=6 reversal was noise.** Two things do survive: **a25m > a8m is real** (+0.061, 13 wins to 1, **p=0.0018**), which *reverses* the n=6 call that a25m doesn't earn its cost; and **RRF(rg, dense) is directionally best in every cell** (r@10 up to 0.762 vs grep's 0.682, bootstrap CI excluding zero but sign test p=0.09) — suggestive, not established. Cost at n=59: dense 200k tokens vs `rg -l` 315k, i.e. dense is now 1.6x *cheaper*. **A code-trained encoder does not rescue it either**: `jina-embeddings-v2-base-code` (161M, 768-d, 30 languages) scores r@5 **0.630** against general-text bekko-a25m's **0.656** — it *loses* to the general encoder, at **6x the encode cost** (612 MB / 61.9 min vs 124 MB / ~10 min), with no comparison significant. The obvious confound was ruled out: path-only retrieval (no code content at all) scores 0.304/0.370, so the code body is genuinely carrying signal — the specialization just adds nothing on top of it. Every arm clusters within noise of grep, and **RRF fusion is the only thing that consistently helps**. **Part B is a regime choice, not a dominance.** Official **jina v5 nano q4 wins 11 of 12 iso-byte cells** and owns the top quality rung — but bekko-a8m encodes a query **12.9x faster on 1 vCPU** (11.3 vs 146.4 ms, 11.2x tokens/s), which is the entire design point of a 7.7M-*active*-parameter model and which an iso-byte table cannot see. The measured ratio matches the ~12x FLOPs ratio (4x384x1152 vs 12x768x3072), so it is architectural, not a q4 artifact. Result is an **iso-quality ladder**: bekko-a8m up to blog R@10 0.575 at 11.3 ms, a25m to 0.598 at 35.0 ms, jina alone above 0.60 at 146.4 ms. Keep jina when quality-bound or compute is amortized; take a8m when the reader is a 1-vCPU container or the corpus is large. **End-to-end through `remax_kb.read.KB.search`, though, only 2.3x of that 12.9x reaches the reader** — a ~50-60 ms constant (`_stacked_simhash_encode` rebuilding k Haar rotations by QR per query, from manifest params that cannot change) is **87% of bekko's query**. Caching it per opened index is one line, verified to give identical codes *and* hits, and restores **11.6-15.1x** — a finding about remax_kb, not about bekko. Swapping the projection does **not** fix it: every option is a per-query *construction* cost of 14-76 ms against a ~6 ms encode, and remax_kb v2's default `srht` is the slowest of them (1.4-3.0x slower than Haar at every dim), deliberately, because it is seed-only and bit-for-bit reproducible by a JS reader. remax's own `rht_rotation` at rounds=2 does reproduce its documented 1.5-1.8x -- a different function. Projection choice is a portability decision; the latency is a caching decision. **Still no remax_kb swap made** — the call is the deployer's, and the code-distribution gap (0.983 vs 0.888) is where bekko was advertised strongest. **Matryoshka trimming vs quantization:** quantization wins directionally at every budget — against the vendor floor d=64 (256 B, R@10 0.520), remex 1-bit @384 is 48 B at 0.564 — and the **vendor's own HAKARI table agrees** (binary@384 −12.93% vs truncation-to-64d's −17.51%). **But a paired-McNemar audit at the end found this corpus cannot establish it: 179 chunks from 11 blog posts, where one query is 0.56 pp, and SEVEN OF EIGHT headline claims are noise** — including the one I led with (remex 2-bit beats the uncompressed vector: +0.011, p=0.625). Only *truncation-to-d=64 costs recall* survives (p=0.009). The direction is consistent and matches the vendor's much larger eval; the demonstration is theirs, not mine. The **one overwhelming result** is Part B's: jina beats bekko on **code-distribution R@1, +0.168, 31 discordant wins to 1, p<1e-5** — that, not the twelve correlated iso-byte cells, is the real basis for 'do not swap'. Compute went to the wrong arm: **78 min encoding 41,500 sklearn chunks** for a 6-instance code-search benchmark, while every embedding-quality conclusion rode on 179 chunks encoded in seconds. Earlier passes also quoted an off-spec d=12 tier (**strawman, retracted**) and mis-priced shared structure (**over-correction, retracted** — remex's codebook is 28 B and the rotation is seed-derived). **The R@50 ceiling (26/179)** is partly harness artifact, has a 7.3% true shared floor, and is recovered by **BM25 (14/26) and RRF (best overall, R@10 0.615)** — *not* by query expansion (3/26, and R@50 drops), reproducing the repo's `muninn-rm3` negative. **2-bit beats 1-bit in all 8 cells**, so bekko is a **Jina-side** embedder and the SPECTER2 one-bit-beats-two result does not transfer. Confirmed the artifact facts independently (404.3 MiB → 124.1 MiB at cosine 0.99992 to its own fp32, holding on both distributions) and **failed to reproduce the card's 5.5x OpenVINO-over-ORT claim** (20.1 vs 21.8 ch/s on 4 vCPU). |
| [`remex-vs-higgs-ablation/`](remex-vs-higgs-ablation/RESULTS.md) | 2026-08-02 | **done — mixed; 2 of 4 pre-registered predictions failed, and one published mechanism refuted** | [`RESULTS.md`](remex-vs-higgs-ablation/RESULTS.md) + `tables.md` + `gate.log` + `axes.png` / `marginals.png` / `seeds.png` | Issue [#8](https://github.com/oaustegard/experiments/issues/8): does **remex** (exact fp32 norm + dense Haar rotation + scalar Lloyd-Max) buy anything for retrieval-index compression over the **QuIP# -> HIGGS -> TurboQuant** lineage (randomized Hadamard + per-block scale + Gaussian-MSE-optimal grid)? Full **2x2x2 factorial**, 11 arms x 6 bit widths x **4 corpora** (d=100/768/784/1024) x 5 rotation seeds x 2 metrics, scored against fp32 exact search rather than qrels. **Only axis C moves.** Rotation is null (-0.0004 recall@10) and norm handling is null (+0.0007); the codebook is an order of magnitude larger (+0.0082 cosine / +0.0112 IP), peaking at **+0.035 recall@10 at 2-3 bits** and decaying to zero by 8 bits, with the effect ~2x larger at d=100 than at d=768/1024. **Failed predictions:** (1) the RHT was predicted 10-100x *faster* at d=768-1024; **corrected 2026-08-01 to ~parity** (1.2x slower at 768, 1.07x at 1024, 3-4x *faster* at 4096-8192, crossover d~1024) after the FWHT was made BLAS-bound -- the original "13-21x slower" measured a butterfly doing two full-array copies per stage against one tuned sgemm, i.e. the implementation, not the transform; (2) exact-norm was predicted to win under inner product and does not, partly because BGE-family encoders are trained under cosine so their raw norms barely vary (CV 1.4-2.7% vs GloVe's 20%) -- axis B is close to moot on modern encoders. **The practical reversal:** counting the shared codebook, the vector arm costs 52.5 B/vector at 4 bits on a 20k-vector index against a 50 B payload, so remex at 6 bits (81 B true, R@10 0.965) beats HIGGS-like at 4 bits (112.5 B true, 0.893) on bytes *and* recall; the vector arm needs ~350k vectors to amortize. **Axis B closed 2026-08-05:** the pending `fmnist784` sweep (raw pixels, norm CV 31%, d=784) ran; axis B is flat there too (+0.0005 cosine / +0.0009 IP), so exact-norm storage is null even off cosine-trained encoders — and the published *mechanism* for the 1-bit remex win was **refuted** (it predicts the effect fades as norm spread grows; fmnist has the most spread and the biggest remex win, and reverses under cosine where the norm is divided out). Two further mechanisms were measured and refuted; the effect is now carried as MEASURED but UNEXPLAINED. **Process:** the two-sided calibration gate caught Lloyd-from-random-init producing grids **worse than scalar** at 6-8 bits, and a scheduled adversarial review then found five more blocking defects -- a stale codebook served by a cache keyed on the problem rather than the method (8-bit vector arm 87% worse than scalar), a Lloyd-Max MSE identity evaluated off the fixed point (+16% at 8 bits, in the direction that makes the gate *more* permissive), a 'provably no worse' guarantee that was argued rather than enforced, a block/sub-vector divisibility bug hitting only the HIGGS-like arm, and a gate that never certified the grids behind any glove result. Scoring `q.xhat` without dividing by `||xhat||` also manufactured a fake 1-bit axis-C reversal -- **the same pattern was present in `jina-remex-vs-remax/score_fidelity.py`, a portable-code entry, and is now fixed there too.** |
| [`lattice-representation-hypothesis/`](lattice-representation-hypothesis/RESULTS.md) | 2026-07-31 | **done — negative result** (opening thesis refuted by the experiment's own adversarial + WordNet arms) | [`RESULTS.md`](lattice-representation-hypothesis/RESULTS.md) + [`THEORY.md`](lattice-representation-hypothesis/THEORY.md) + [`fca.py`](lattice-representation-hypothesis/src/fca.py) + `noise_reversal.png` | A *Paper Skygest* Bluesky post pointing at [arXiv:2603.01227, "The Lattice Representation Hypothesis of LLMs"](https://arxiv.org/abs/2603.01227) (LLM embedding geometry encodes an FCA concept lattice; meet/join as half-space operations). Hypothesis: the concept algebra has a **broken join** — FCA's meet extent is a bare intersection (exact under half-spaces) while its join extent is the *closure* of a union, and the paper's Definition 7 writes that join as a literal set union, contradicting its own Appendix B. Measured at scale: **0 meet phantoms across 9,615,370 concept pairs**, join overshoot mean 0.60. **Then the experiment's own arms killed it.** (i) Definition 7's "conic hull" clause makes the join *exactly* `R(Y_A n Y_B)` under linear independence — verified 96/96 by an independently written Minkowski-sum LP, strict only under conic dependence (43/96, 34/96). (ii) The "phantoms" are not errors: they *are* the least upper bound, and the gold label in the paper's own task (join of {dog, wolf} is `canine`, which contains foxes). (iii) **The premise was mathematically wrong**: `meet = objs_of(B1 u B2)` and `join = objs_of(B1 n B2)` are *both* plain half-space intersections (0 identity violations over 30 configs x 8 seeds); the join needs **fewer** constraints (0.83 vs 5.55). The only non-representable object is the plain set union, which isn't a lattice operation — and it is indeed recovered worst of the three. The one measured effect — join degrades more slowly than meet under probe error on Jaccard (5% flip: 0.939 vs 0.835) — carries a **12x target-size confound**, flips sign on symmetric-difference error in one of two contexts (cross: join 0.110 vs meet 0.028), and its size-controlled version is reproduced *more strongly by random-direction controls*; the proposed "closure absorbs noise" mechanism is contradicted by a **negative** overshoot-vs-error correlation (-0.44). **Surviving caveats, aimed at the paper's setting:** WordNet noun hypernymy gives an extremely thin lattice (**15 concepts** from 150 objects x 13 attributes, **0/78 cross-cutting attribute pairs**, 66% of meets empty, ~42% of joins the top element); **learned attribute directions can be exactly antipodal** (mutual coherence **1.0000**, since `living_thing` and `artifact` are complementary), so the canonical form's linear-independence assumption is not secured by `d >> |M|` and is never checked in the paper; hypernym trees are the **worst** case for closure gaps, not a benign one (0.93 non-closed / 0.71 overshoot vs 0.000 for a chain); and **Eq. 6's meet orientation looks inverted** (max beats min in 96.5% of trials) though the join half failed to replicate (59%) against the reviewing agent's claimed 99.5%. Forensics from two independently-prompted readers: Figure 4's numbers are printed nowhere in the paper (recovered identically to 3 d.p. from its SVG bar geometry), **12 of 45 Table 2 cells have F1 below both their own precision and recall**, and no released code computes Figure 4. Held-out probes do fit the LRH threshold model (AUC 0.87-0.98 with glosses, controls at chance), with a large fit-on-all inflation caught by the split (join Jaccard 0.901 -> 0.501). Negative results kept in the record: Arm C's meet metric was vacuous by construction (my bug) and its probes were indistinguishable from random directions. |
| [`svgview/`](svgview/README.md) | 2026-07-30 | working on Linux; Windows build never run on a real machine | [`README.md`](svgview/README.md) + [`src/`](svgview/src/) | [andri.dk on Bluesky](https://bsky.app/profile/andri.dk/post/3mrewq7fcsc2j), arguing that launching a full browser to render PDF or SVG is "bonkers insane" — narrowed in his own reply to systems doing it internally. Tested the SVG half by building the alternative: a native Windows-first viewer wrapping `resvg`, ~600 lines. Measured here: **4.8 MiB executable, 12 MiB resident, 16 ms exec→window, 20 ms parse+render to a 1000 px PNG**. Verdict: he is right about SVG and `resvg` had already done the hard part; the argument does *not* transfer to PDF, where the honest options are wrapping Chrome's own PDFium or accepting `hayro`'s coverage gaps. Windows compiles in CI; the file dialog, icon, and association scripts are unverified. |
| [`erdos-gyarfas/`](erdos-gyarfas/README.md) | 2026-07-28 | open problem; partial results | [`README.md`](erdos-gyarfas/README.md) + [`note.html`](erdos-gyarfas/note.html) + [`tutte_coxeter_lemma.py`](erdos-gyarfas/src/tutte_coxeter_lemma.py) | this session |
| [`ms13-campaign/`](ms13-campaign/SUMMARY.md) | 2026-07-24 | closed (compute exhausted; open maths recorded) | [`SUMMARY.md`](ms13-campaign/SUMMARY.md) (academic writeup) + [`NOGOS.md`](ms13-campaign/NOGOS.md) (ledger) + [`BLOGPOST.md`](ms13-campaign/BLOGPOST.md) | [issue #169](https://github.com/oaustegard/claude-workspace/issues/169): campaign against Morell–Skutella Conjecture 1.3 (two-sided unsplittable-flow rounding). **No counterexample.** Produced instead: a **reduction** showing 1.3 restricted to 2-path instances *is* a linear-discrepancy question on **network matrices with demand-scaled columns** — a connection neither literature appears to draw (full-text greps of TVZ/Swamy/MSW25 find no mention of Doerr, "linear discrepancy" or "totally unimodular"); a **theorem** settling that question for k=3 (`R = 3/4` exactly, unconditional, census complete through m=10, exact branch-and-bound on the 2 maximal classes); and a ledger of 20 refuted families/claims **including two conjectures of our own** (12.1 refuted at k=4, the staircase conjecture at k=8). Rediscoveries correctly identified as such: the tightness gadget is Morell–Skutella Fig. 3, the equal-demand bound is Doerr 2004. Enumeration shown dead by arithmetic (~2,070 h at k=4). Open: Q7′ (column-scaled Doerr bound, general k). Methodology notes: a false-positive "counterexample" caught at the certificate gate when two independent verifiers turned out to share one blind spot; every over-claim had the same shape (clean at k≤6, false at larger k). |
| [`ssuf-beta/`](ssuf-beta/RESULTS.md) | 2026-07-24 | done (scoped) | [`RESULTS.md`](ssuf-beta/RESULTS.md) + [`engine.py`](ssuf-beta/engine.py) + [`calibration.py`](ssuf-beta/calibration.py) + [`family.py`](ssuf-beta/family.py) | [claude-workspace#165](https://github.com/oaustegard/claude-workspace/issues/165): quantitative hunt for the SSUF cost-preserving violation constant β* following the Goemans/DGG conjecture disproof (16/15 < β* ≤ 2). Built an exact-rational β* engine (breakpoint-enumerated convex-hull membership LP via sympy's tested simplex, after a hand-rolled one failed its own sanity test). **Calibration against the real Rybin instance was blocked** — no arXiv writeup exists, and this session's WebFetch can't reach the source X thread (HTTP 402, no working mirror) — so calibrated instead against a fully hand-derived, independently-constructed triangle-conflict instance (β\*=1/2, exact match) and swept a parametrized generalization of it — that family's β\* has supremum exactly 1 (approached, never attained or exceeded), a clean negative result short of even the original refuted β=1 bound. Literature gate confirmed TVZ's planar +2·d_max bound and ring-loading bounds (1.1D/1.3D) from primary sources. Honest scope cut: no claim here reproduces or exceeds β\*=16/15; gadget search, ms13 engine sweep, and the ring-loading secondary target were not attempted. |
| [`discrepancy/`](discrepancy/RESULTS.md) | 2026-07-24 | done (D(4) + n≥17 deferred) | [`RESULTS.md`](discrepancy/RESULTS.md) + `growth.png` | [issue #166](https://github.com/oaustegard/claude-workspace/issues/166): certified discrepancy lower-bound records — Komlós per-size K(n) + Beck–Fiala small-t exact values, max-min engine with exact certificates. Literature gate first (logged on the issue): Kunisky's K ≥ 1+√2 record reframed Target A to per-size records; 2025 Bansal–Jiang resolution of Beck–Fiala for t ≥ log²n left small-t exact values open. **Beck–Fiala: D(2)=2, D(3)=3 exactly** — the CEGAR SAT search *rediscovered the Fano plane from scratch* and proved it's the minimum-ground-set (n=7) witness; D(2)=2 (triangle minimal); D(4,n≤9)=3 with D(4)∈{3,4,5} open (PG(2,3) computed weak: disc 2). Small-t truth sits on **D(t)=t**, far below 2t−3. **Komlós: certified rational per-size records K(3)≥1.571, K(4)≥1.731 (beats Kunisky's own n=4 tree matrix 1.707), K(5)≥1.785, K(7)≥1.830**; proved the "exact Δ vs published δ" gap on Kunisky's family is empty (Δ=δ, one-line proof) before wasting compute on it. All records verified by an independent second code path (`verify_certificates.py`); calibration gates G1–G9 green first. |
| [`woodall/`](woodall/README.md) | 2026-07-23 | phase 1 (verifier+calibration done, search deferred) | [`README.md`](woodall/README.md) | [issue #163](https://github.com/oaustegard/claude-workspace/issues/163): SAT/MIP dijoin-packing verifier for Woodall's conjecture τ=3 counterexample search (Goemans-fall follow-on). Built a CEGAR SAT verifier (python-sat/cadical+glucose) for ν(D,u) and brute-force τ; transcribed Schrijver's (D1,u1) counterexample directly off a 600dpi render of Figure 6 in the Feofiloff survey PDF (not from memory). **Calibration gate 1 (Fact 7.1, ν=1/τ=2) passes exactly**, after catching and fixing a real transcription bug (one dashed arc misread) via cross-checking the graph's derived 3-fold symmetry and the paper's "4 critical cuts" claim against an initial read that gave only 3 and a spurious ν=2. Generalized D1 into a ring-of-length-2i family from its own orbit structure and validated the paper's odd/even parity claim (i=3,5 counterexample, i=2,4 not) exactly. A first light random-search pass (16,000 trials) found 0 candidates, confirming unstructured random generation is a weak filter-pass rate (0.15%) for τ≥3 — the real search needs the structured null-arc-resolution and Williams-catalog generators, explicitly deferred (not silently dropped) along with gate 2 (Cornuéjols-Guenin D2/D3 calibration). |
| [`pdf-streaming-test/`](pdf-streaming-test/RESULTS.md) | 2026-07-05 | done (shipped) | [`RESULTS.md`](pdf-streaming-test/RESULTS.md) + [`test_streaming.js`](pdf-streaming-test/test_streaming.js) | "Speed up the UX of `austegard.com/web-utilities/pdf-text-extractor` by streaming pages one-at-a-time; also check if parallelising improves throughput." Added a bounded worker pool + streaming display (pages append to the output pane in strict page order via a `nextToFlush` cursor as they complete) and a URL API knob (`&concurrency=1|2|4|8`, default 4). Playwright harness over three arXiv PDFs (8/15/75 p), routes-intercepted so no external network. **Streaming works** (75-page paper: first page at t≈1s, then every ~150-200ms in order) and **parallelism gives a real win on non-trivial PDFs** — the 75-page paper drops 4.31s → 1.87s wall-clock (**2.3×**, extraction throughput 23.9 → 90.7 pages/s, **3.8×**) going conc=1 → 8; small PDFs (<20 p) plateau at conc=2 since there's nothing left to pipeline through pdf.js's single worker thread. Default of 4 keeps the memory tax down while capturing most of the win. Ordering preserved across every run. Shipped in [oaustegard/oaustegard.github.io PR (branch `claude/pdf-extractor-streaming-ijudhv`)](https://github.com/oaustegard/oaustegard.github.io/pulls). |
| [`atproto-pad-login/`](atproto-pad-login/RESULTS.md) | 2026-07-04 | done | [`RESULTS.md`](atproto-pad-login/RESULTS.md) + [`pad_login.mjs`](atproto-pad-login/pad_login.mjs) + `04_connected.png` | "you can log into ATProto as muninn — try this pad URL" → Drove `austegard.com/bsky/pad.html` end-to-end as **muninn.austegard.com** via Playwright/Chromium: opened the shared-pad URL, clicked *Join*, filled the login dialog with `$MUNINN_BSKY_HANDLE` + `$MUNINN_BSKY_APP_PASSWORD`, waited for `rtc.login()` to establish the PDS session and `rtc.connect(peerDid)` to send its WebRTC knock. **Result: status flipped `Local only · 0 peers` → `Connected · 0 peers`** (session up; the invited peer just isn't online sharing, so no data channel — a one-sided join can't complete). **Load-bearing finding along the way — headless Chromium via `$HTTPS_PROXY` returns `net::ERR_CONNECTION_RESET` on every HTTPS goto unless launched with `--ssl-version-max=tls1.2`.** Curl and Node fetch work; Chromium doesn't. NetLog shows the CONNECT tunnel returns 200 and Chromium's TLS 1.3 ClientHello (~1700 B, ECH extension included) goes out — then the socket is reset (`SOCKET_READ_ERROR net_error=-101 os_error=104`) before any TLS response. `--ignore-certificate-errors`, `ignoreHTTPSErrors: true`, and even `--disable-features=EncryptedClientHello` don't help — only pinning TLS 1.2 does. So the session's egress-gateway MITM terminator can't complete a modern Chromium TLS 1.3 handshake. Side finding: the docs claim the browser NSS trust store is pre-loaded with the proxy CA; ground truth is that `/root/.pki/nssdb` was **empty** — `certutil -L` returned only the header. Adding `/root/.ccr/agent-proxy-ca.crt` with `certutil -A -t 'C,,'` is needed for the proxy-terminated cert to verify after the RESET is worked around. |
| [`session-relay/`](session-relay/RESULTS.md) | 2026-07-02 | done | [`RESULTS.md`](session-relay/RESULTS.md) + [`relay.py`](session-relay/relay.py) + [`README.md`](session-relay/README.md) | "How might this be accomplished?" → [Joshua Shew's Bluesky post](https://bsky.app/profile/joshuashew.bsky.social/post/3mnx2urp3f22n): one Claude session built a relay service so two other sessions could chat and share inbox best practices. **Reproduced with ZERO deployed infrastructure — the shared Turso DB is the relay**: one `relay_messages` table (channel, sender, AUTOINCREMENT seq as cursor) + a ~190-line CLI (`init/post/poll/wait/history`, 5xx backoff). Live run: two concurrent agents with disjoint seeds (muninn-a: `docs/architecture.md` lifecycle; muninn-b: experiment conventions) negotiated a joint 5-item "experiment survival checklist" over channel `hub-coord` — **4 messages, one round-trip each way, ~75s, clean CONSENSUS/ACK termination**, with verifiable two-way knowledge transfer (final checklist contains facts neither seed had alone) and genuine critique (b fact-checked a against Error Patterns, restructured its draft). Protocol findings: table-global seq leaks cross-channel counts (cursor discipline handles it); shell quoting is the ergonomic tax (mandate the `post ... -` stdin path); front-loaded opener + explicit delegation → 1-round convergence. Transport is container-agnostic (HTTPS to Turso) — works across CCotw/Claude.ai session boundaries, unlike the `/tmp`-files original. Caveat: for two Muninn sessions the marginal value over shared memory is *live negotiation*, not knowledge transfer. |
| [`omnigent-library-eval/`](omnigent-library-eval/RESULTS.md) | 2026-06-29 | done | [`RESULTS.md`](omnigent-library-eval/RESULTS.md) | "Is this anything you could make use of?" → [omnigent-ai/omnigent](https://github.com/omnigent-ai/omnigent), the open-source **meta-harness** (5.3k★, 18 days old, alpha `0.3.0.dev0`, Apache-2.0) that orchestrates 11 vendor harnesses (Claude Code/Codex/Cursor/Pi/…) with policies + sandboxing + multi-device. **Verdict: NOT a claude-workspace dependency.** It's a competing harness — a server+CLI that wants to OWN orchestration/model-routing/sandboxing, all of which CCotw already provides; the Python SDK is a *client to a running omnigent server*, not an importable lib; it can't even run here (no model key by design, wants Node 22/tmux/bwrap). **But three borrowable wins:** (1) **free** — `omnigent/spec/skill_sources.py` reads the SAME `SKILL.md` format with a `claude` family, so Muninn's whole skill library loads with zero porting; (2) the **Polly orchestration prompts** — though the cross-vendor *review* discipline they encode Muninn can ALREADY run in CCotw today (Gemini reviewer via the Cloudflare AI Gateway, `gemini_generate()`); omnigent only uniquely adds cross-vendor *implementer* agents (Codex/Cursor/Pi as full coding harnesses), a narrow win; (3) the **ALLOW/DENY/ASK three-level policy engine** (`docs/POLICIES.md`: `blast_radius`/`spawn_bounds`/`cost_budget`) as a reference design if Muninn ever formalizes its `settings.json` permissions. Rec: keep on radar as a place to run Muninn-the-agent *outside* CCotw; don't take the dep. No code executed (alpha server, no key). |
| [`q4-official-vs-ours/`](q4-official-vs-ours/RESULTS.md) | 2026-06-28 | done | [`RESULTS.md`](q4-official-vs-ours/RESULTS.md) + [`run_batched.py`](q4-official-vs-ours/run_batched.py) + [`results.txt`](q4-official-vs-ours/results.txt) | [remax_kb#23](https://github.com/oaustegard/remax_kb/issues/23): run the q4 head-to-head claude.ai couldn't (egress blocks both weight hosts) — does the model authors' own `onnx/model_q4.onnx` (137.8 MB, HF Optimum) match/beat our `JinaQ4ONNXEmbedder` build (~170 MB, remax_kb#14) on retrieval fidelity to fp32, deciding whether to upload ours to HF? NFCorpus, 2058 docs/100 queries, 4 vCPU. **Result: ours is DOMINATED — official wins every axis at once: smaller (138 vs 170 MB, −19%) AND strictly more faithful to fp32 on all four metrics (nDCG@10 0.4291 vs 0.4250, per-doc cos 0.976 vs 0.974, recall@10-vs-fp32kNN 0.870 vs 0.862, Spearman ρ 0.980 vs 0.976).** The hoped-for edge — our int8 embedding-table mop-up holding where Optimum's generic q4 degrades — is refuted; Optimum handles the EuroBERT Gather at least as well. **Decision: do NOT upload ours; close the loop. `embedders.py` now steers users to the official asset.** Side findings: docstring's "cosine 0.975 to fp32" holds (ours 0.9743 on this larger subsample); q4 is NOT faster than fp32 on CPU (~13–14 min each — int4 dequant offsets size; "~2x faster decode" didn't reproduce); patched the bench's one-shot `encode(all_docs)` OOM (~26 GB attn Expand) with numerically-identical mini-batching. Reinforces the prior-art-check lesson: the upstream repo already shipped a smaller, better q4. |
| [`jina-remex-vs-remax/`](jina-remex-vs-remax/RESULTS.md) | 2026-06-27 | done | [`RESULTS.md`](jina-remex-vs-remax/RESULTS.md) + [`score_fidelity.py`](jina-remex-vs-remax/score_fidelity.py) + [`fidelity.png`](jina-remex-vs-remax/fidelity.png) | "Can we apply the **remex** (not remax) quantization optimizations to our Jina q4 embedding — as a practical compressed-Jina format, not necessarily beating remax at a byte budget?" First clears up the **two q4s**: q4-the-*model* (`JinaQ4ONNXEmbedder`, weight quant, fp32-parity float output) is orthogonal to remex/remax, which compress the *vectors*; remex replaces the remax 1-bit step, not the q4 model. Recall-vs-qrels **saturates both ways** (muninn fp32 ceiling 0.90/1.00; NFCorpus fp32 floor 0.241 — embedder-limited), so switched to the saturation-proof metric: **fidelity to fp32's own ranking** (recall@k vs fp32-kNN chunk-level, Spearman ρ, recon cosine — remax's own bench metric). **Result: remex (rotation+Lloyd-Max scalar) ≫ remax (1-bit SimHash) at every byte budget** — NFCorpus n=120: remex ρ 0.92–1.00 vs remax 0.63–0.74; **remex 4-bit @ d768 (384 B) near-lossless (ρ 0.998, R@10-vs-fp32 0.96); remex 2-bit @ 192 B (ρ 0.978) crushes remax d768/k2 @ 192 B (0.741); remex 1-bit @ 96 B beats every remax config.** Headline: **bits beat stacks** — graded magnitude (Lloyd-Max) dominates more sign-bits (stacked SimHash); full-dim-low-bit > truncated-dim-high-bit. remex is the data-oblivious Haar rotation our own ITQ-rejection work (remax#46) endorsed. **Rec: ship remex into remax_kb as the near-lossless/mid-byte codec (default 4-bit @ d768); needs a new SPEC binarizer type + ADC scan path (numpy+scipy), additive not drop-in.** Caveats: muninn n=5; norms excluded from B/row (Jina unit-norm). |
| [`remax-hamming-speedup/`](remax-hamming-speedup/RESULTS.md) | 2026-06-26 | shipped (PR) | [`RESULTS.md`](remax-hamming-speedup/RESULTS.md) + [`bench.py`](remax-hamming-speedup/bench.py) + [`latency.png`](remax-hamming-speedup/latency.png) | [remax_kb#15](https://github.com/oaustegard/remax_kb/issues/15): the 1-bit Hamming scan (`_hamming.hamming_scan`, popcount **LUT gather**) was ~10× slower than a BLAS float-cosine search at small N, forfeiting the latency half of the 1-bit format's promise. Benchmarked the issue's candidates head-to-head (LUT vs `bitwise_count` u8/u64 vs ±1 BLAS matmul vs float cosine) over N=600→1M, single-thread BLAS. **Winner is the cheapest candidate (approach 1): `np.bitwise_count` over a uint64 view of the XOR — ~10× over the current LUT and faster than BLAS float cosine at *every* N** (e.g. N=10k: 0.58 ms vs LUT 7.83 ms vs cosine 2.08 ms), zero-copy, codes stay bit-packed (256 B/row). The ±1 matmul (approach 2) is 2–6× slower *and* costs 8–32× RAM (OOM at 500k); compiled SIMD (approach 3) is unnecessary. Issue's success criterion (Hamming ≤ cosine at N≥10k without losing storage) **exceeded — holds at all N**. **Shipped: `_hamming.py` + `read_v2.py` swap to a shared `_popcount_rows` fast path with a numpy<2.0 LUT fallback (numpy≥1.24 floor preserved) + remax-free regression test ([remax_kb PR](https://github.com/oaustegard/remax_kb/pulls)).** Bit-for-bit exact vs the old LUT across 8 realistic `(dim,k)` widths incl. non-multiple-of-8 byte rows. |
| [`lfm25-230m-verify/`](lfm25-230m-verify/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](lfm25-230m-verify/RESULTS.md) + [`run.py`](lfm25-230m-verify/run.py) | "test this Bluesky claim — RUN the model" ([@sungkim post](https://bsky.app/profile/sungkim.bsky.social/post/3mp4zqketts25)) on Liquid AI's LFM2.5-230M. Spec-checked the HF card, then *loaded and ran the weights* on container CPU to verify the runtime-checkable claims. Confirmed: 229.7M params, `Lfm2ForCausalLM`/`model_type=lfm2`, open weights, coherent CPU generation at 29.2 tok/s (fp32, 4 vCPU — the slow path vs card's quantized 42/213 tok/s). Training claims (19T tokens, distill-from-350M) aren't runtime-observable. Minor nuance: post/card say 32K context, loaded config carries 128K positional ceiling. Verdict: post accurate. |
| [`lfm25-embedder-remax_kb/`](lfm25-embedder-remax_kb/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](lfm25-embedder-remax_kb/RESULTS.md) + [`lfm25_embedder.py`](lfm25-embedder-remax_kb/lfm25_embedder.py) | Follow-on from `lfm25-230m-verify`: if the 230M *runs* locally, does Liquid's `LFM2.5-Embedding-350M` work as a third remax_kb embedder — in-process CPU, no 847 MB Jina download, no Gemini API key? Wrote an `LFM25Embedder` (sentence-transformers, CLS-pool, 1024-d, query:/document:) on remax_kb's `Embedder` protocol. One real fix: the model's bidirectional remote code predates transformers' `seq_idx` shortconv kwarg (5.12.1 `TypeError`) — patched at our layer, not the model's cached files. Tiny-corpus 1-bit pipeline: 3/3 topical top-3 (matches Jina torch). **Full-float head-to-head on the 73-post muninn corpus: LFM2.5 0.73/0.83 R@5/R@10 — *below* lexical 1.00/1.00 and Jina v5-nano 0.90/1.00, despite being the larger model; embeds slowly (1.1 chunks/s fp32, ~19.5 min).** Verdict: viable where in-process/no-key/no-network is the hard constraint and retrieval is tolerant; Jina still wins on quality. Closing analysis: remax's 1-bit step is corpus-global (can't quantize up front); per-chunk-memory levers = fp16 buffer, streaming Welford mean, dim-truncation (non-MRL caveat), and int8 model quant for RAM+speed. |
| [`muninn-rm3/`](muninn-rm3/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](muninn-rm3/RESULTS.md) + [`bench.py`](muninn-rm3/bench.py) | The platform-less, agent-less site-search floor (no model to host, no Claude to expand the raw query). Pure RM3 vs plain BM25 on the muninn corpus (5-query phase0 gold). **RM3 is a dud — identical R@5, *hurts* R@10 (1.00→0.90 whole-doc); skip it. But plain BM25 (whole-doc) = 0.833/1.00 — ties Jina-q4→remax (0.833/1.00) at ZERO inference/agent/per-query cost, all in the Worker.** Only the agent-expansion path (1.00, unavailable to site search) beats it. Caveat: these are in-vocab acceptance queries; the residual Gemini buys is *vocabulary-divergent* (paraphrase) queries — BM25/RM3 can't bridge that lexical gap, no platform-free option does. Recommendation: drop Gemini for plain in-Worker BM25 IF muninn searches are keyword-ish; keep Gemini only if paraphrase-heavy. Decision input: actual query mix. |
| [`muninn-embedder-bakeoff/`](muninn-embedder-bakeoff/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](muninn-embedder-bakeoff/RESULTS.md) + [`jina_remax.py`](muninn-embedder-bakeoff/jina_remax.py) + [`embed_one.py`](muninn-embedder-bakeoff/embed_one.py) | Test our special-case **Jina→remax** as a Gemini replacement for muninn search (+ how it's practical). On the muninn corpus (5-query phase0 gold): **Jina-q4 full-float = 0.900/1.000 (identical to fp32 — q4 is free); Jina-q4→remax 1-bit at d=512/k=4 (256B) = 0.833 R@5 / 1.00 R@10** (near the float ceiling; the shipped d=256/k=8 default is the weak config at 0.667 — dims beat stacks here too). So Jina→remax is a credible Gemini replacement; lexical (1.00) still edges R@5. **Practicality: indexing is free (offline CI pack with Jina), only the online query-embed is hard** — Worker can't host 170MB (10MB bundle/128MB mem), Workers AI has no Jina, so serve queries from a **CF Container/endpoint running q4** (replaces the paid per-query Gemini call with cheap fixed compute; q4 is what makes that light). Migration = corpus re-pack (Jina space, d=512/k=4) + query-embed service swap; KV 1-bit mechanism unchanged. Sidebar: CF-native Workers-AI BGE alts underperform (bge-large 0.733/bge-base 0.667). Caveats: n=5; no live Gemini (Jina-0.90 float is the reference); embeddinggemma-300m HF-gated. |
| [`rotation-decorrelation/`](rotation-decorrelation/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](rotation-decorrelation/RESULTS.md) + [`sweep.py`](rotation-decorrelation/sweep.py) | "How can I trust either end of the ITQ pendulum? Embedder-specific? Test the decorrelation angle." Controlled study, remax/bench metric (self-retrieval recall@10 vs float32 kNN), pure numpy on precomputed caches: **SPECTER2 (specialized) + Jina-v5 (general)**, k-ladder {1,2,4,8}, simhash/itq/decorr, **in-corpus vs transfer**, 3 seeds. **Resolution: the pendulum is 3 interacting axes** — (1) k: ITQ wins k=1, decays/reverses on the ladder (#46's mechanism reproduced); (2) protocol: ITQ in-corpus overfits (gap ~0.02–0.03 SPECTER2, ~0.04–0.05 Jina; simhash/decorr zero gap); (3) embedder: ITQ's edge is 3× bigger on general Jina (+0.048 k=1) than specialized SPECTER2 (+0.015) — partly embedder-specific but mostly overfit. **My NFCorpus 'win' = perfect storm (general × in-corpus × k=1); #46 = specialized × transfer × ladder.** Honest config (transfer+ladder, k=8): simhash beats itq on BOTH (SPECTER2 −0.029, Jina −0.015). **Decorrelation (α-mix) is a wash** — ties simhash everywhere, no overfit gap. Verdict: keep parameter-free SimHash (#46 upheld, now for general embedders too); the open lead doesn't pay. |
| [`recall-per-byte/`](recall-per-byte/RESULTS.md) | 2026-06-25 | **corrected (re-derived rejected work)** | [`RESULTS.md`](recall-per-byte/RESULTS.md) + [`sweep.py`](recall-per-byte/sweep.py) | Generative-thinking move (random stimulus "river") reframed compaction toward *information-per-stored-bit (the rotation)*. Recall-per-byte bake-off on cached NFCorpus fp32 Jina vectors (600 docs/120 q): remax StackedSignBit vs SimHash vs **ITQ** vs **PQ**. Apparent headline (ITQ/PQ@16B beat shipped remax@256B) **does NOT hold**: ITQ was already tested rigorously and rejected — **remax#46/PR#47 (closed unmerged, SPECTER2 n=10k): learned ITQ loses to centered SimHash at every ladder rung, deficit grows with k**. My k=1/600-doc "win" is the exact *in-corpus overfit artifact* #46 diagnosed (transfer rotations beat in-corpus ones); I never tested the stacked ladder where ITQ definitively loses. PQ not novel either (remex/TurboQuant owns codebook compression; my 256-centroid book on 600 docs flatters it). **Lesson: recall() prior art before claiming a win.** One genuinely open lead (from #46's parting question): a *decorrelating joint multi-rotation* objective — diverse-across-stacks, better-than-random per-stack — validated on SPECTER2 with transfer + ladder, explicit kill criterion. |
| [`jina-int8-remax_kb/`](jina-int8-remax_kb/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](jina-int8-remax_kb/RESULTS.md) + [`quantize.py`](jina-int8-remax_kb/quantize.py) + [`bench.py`](jina-int8-remax_kb/bench.py) | Pivot from quantizing the weak LFM2.5 to quantizing the *strong* embedder: int8-quantize Jina v5-nano's ONNX export to cut its 847 MB download. `quantize_dynamic(QInt8)` → **212 MB (4.0× smaller)**. Head-to-head on the 73-post muninn corpus (same `stage_b.py` methodology): **fp32 0.90/1.00 @ 8.3 ch/s vs int8 0.83/1.00 @ 16.7 ch/s** — 4× smaller, 2× faster, R@10 untouched; the R@5 dip is one query (Q3) sliding past rank 5. fp32 reproduces the prior 0.90/1.00 exactly (harness validated). **int8 Jina dominates the LFM2.5 local option on every axis** (R@5 0.83>0.73, 212<919 MB, 16.7≫1.1 ch/s). Standing rec stays lexical 1.00/1.00; int8 Jina is the real-vector fallback. **Follow-up (NFCorpus, 600 docs/120 qrel-queries, full-float + 1-bit recall): overturns int8.** Per-tensor dynamic int8 is domain-fragile — 0.445 per-doc cosine to fp32 on medical abstracts (vs 0.83 R@5 on muninn tech text; probe rules out seq-length: 0.44@256 vs 0.41@512). **Blockwise 4-bit `q4` matches fp32 (0.975 cosine, cos R@10 0.241 vs 0.242, 1-bit R@10 0.208 vs 0.222) at 170 MB < int8's 212 MB**; q2 too far (0.73). Embedding-table workaround: MatMulNBits leaves EuroBERT's ~400 MB Gather fp32 (naive int4=465 MB > int8), so int8-mop-up the embedding → q4=170/q2=141 MB. 3-bit unsupported (ORT {2,4,8}). **Answer to "is int8 the floor before 1-bit?": no — 4-bit blockwise is smaller, more faithful, domain-robust; floor is 4-bit; the int8 embedding table is now the size-dominant cost.** Recommended quantized Jina = q4, not int8. **SHIPPED: q4 integrated into remax_kb as `JinaQ4ONNXEmbedder` ([oaustegard/remax_kb#14](https://github.com/oaustegard/remax_kb/pull/14)) — embedder + deterministic `scripts/build_q4_onnx.py` + gated test; `model.q4.onnx` hosted (SHA-verified) on the jina-v5-nano-mirror release; fp32-parity query cosine 0.977; opt-in experimental.** |
| [`kb-packer-web/`](kb-packer-web/build_packer.py) | 2026-06-25 | shipped (PR) | [`build_packer.py`](kb-packer-web/build_packer.py) + [`kb-packer.html`](kb-packer-web/kb-packer.html) | Browser KB-packer tool for austegard.com/ai-tools/ — drag files → BM25 index → download an installable `<name>.skill`, fully client-side (no upload/model/network). Generator inlines the `creating-kb` build core + vendored runtime (search.js/search.py/bundle_SKILL.md) into one self-contained HTML matching the ai-tools single-file convention; browser-built `.skill` byte-identical to the Node CLI's. E2E-tested in headless Chromium. [oaustegard.github.io PR #253](https://github.com/oaustegard/oaustegard.github.io/pull/253). Moved here out of claude-skills (#714 closed — skills repo isn't for hosted web apps). |
| [`lexical-kb-phase0/`](lexical-kb-phase0/RESULTS.md) | 2026-06-25 | done | [`RESULTS.md`](lexical-kb-phase0/RESULTS.md) | The load-bearing study: does agent-expansion + BM25 match embeddings on the real muninn corpus (73 posts)? **Yes, with a bounded residual.** Chunk-size sweep confirms the null hypothesis (whole-doc = 500-char on recall, 17× fewer chunks, R@5 up). Head-to-head vs the full-float Jina ceiling: lexical 1.00/1.00 vs embedding 0.90/1.00 mean R@5/R@10 on in-vocab queries (lexical ties/edges). Paraphrase frontier: agent expansion ties/beats embedding on 4/5; the 5th is what embeddings buy — a vocabulary-divergent relevant post expansion missed. |
| [`lexical-kb/`](lexical-kb/RESULTS.md) | 2026-06-25 | skill built+tested | [`RESULTS.md`](lexical-kb/RESULTS.md) | Embedding-free portable KB: BM25 over a precomputed index, semantic layer moved to the consuming agent (query expansion at search time). Ships as a self-contained `.skill` zip (SKILL.md + search.py + index.json + chunks.jsonl), pure stdlib, no model/network. Makes remax_kb#12 (pure-JS packer) trivial by deleting the embedding half. Tiny-corpus test validated correctness (BM25/RM3/filter) + caught/fixed a substitutive-expansion bug (expansion is now additive over the raw query). Quantitative lift → Phase 0 (muninn corpus). |
| [`kb-k-sweep/`](kb-k-sweep/RESULTS.md) | 2026-06-22 | done | [`RESULTS.md`](kb-k-sweep/RESULTS.md) | recall@10 over a (dim,k) grid on the real Mac-search corpus (Gemini `gemini-embedding-001`, 1779 chunks). Answers CROSSOVER's open follow-up on a non-SPECTER2 embedder, then sweeps dims. **Headline: at a fixed byte budget, dimensions beat stacks** — shipped dim=256/k=8 is Pareto-dominated by dim=512/k=4 (+5.9 R@10, same bytes) and dim=768/k=2 (smaller AND better). dim=512/k=1 matches shipped recall at ¼ the size. int8 rotations then make 768/k2 smaller than baseline (remax_kb#10, muninn#213). |
| [`anything-to-text/`](anything-to-text/RESULTS.md) | 2026-06-22 | done | [`RESULTS.md`](anything-to-text/RESULTS.md) + [`build.py`](anything-to-text/build.py) | "Anything to Text SPA for austegard.com/web-utilities, composite of two uploaded apps, re-brand from vendor to my site" — merged in-browser OCR (PaddleOCR) + Whisper transcriber into one drop-zone-routed page; engines kept verbatim in separate ES modules, reskinned to Grouch's Workshop. [oaustegard.github.io PR #247](https://github.com/oaustegard/oaustegard.github.io/pull/247) |
| [`dc-mall-timelapse/`](dc-mall-timelapse/RESULTS.md) | 2026-06-18 | done | [`RESULTS.md`](dc-mall-timelapse/RESULTS.md) | "timelapse of the DC mall (WAMO) webcam since May 1" — EarthCam HOF endpoint only serves newest ≤50 stills (no pagination); May 1 unreachable. Built ~4-day clip (Jun 8→12) from what's public. `imageio-ffmpeg` = static ffmpeg in CCotw. |
| [`python-lsp-stress/`](python-lsp-stress/RESULTS.md) | 2026-06-15 | done | [`RESULTS.md`](python-lsp-stress/RESULTS.md) | validate pyright 1.1.410 (PR #124) — `pyright-langserver` LSP + batch checker against Django (2,922 `.py`); references 2,490 hits in 3.9s, whole-pkg batch in 22s |
| [`qat-cpu-demo/`](qat-cpu-demo/RESULTS.md) | 2026-06-06 | done | [`RESULTS.md`](qat-cpu-demo/RESULTS.md) + [`qat_vs_ptq.png`](qat-cpu-demo/qat_vs_ptq.png) + [`inference/`](qat-cpu-demo/inference/README.md) | "could we run a smaller version of Gemma 4 QAT locally, GPU-less?" — toy-scale QAT-vs-PTQ reproduction on a char-LM (CPU), plus the real Gemma 4 E2B Q4_0 checkpoint clocked at ~20 tok/s on CPU |
| [`memory-redundancy-probe/`](memory-redundancy-probe/RESULTS.md) | 2026-06-04 | done | [`RESULTS.md`](memory-redundancy-probe/RESULTS.md) | "is the memory store a landfill?" probe + curate/consolidate review; spun out of reviewing arxiv 2606.03787 |
| [`optimizing-skills-retro/`](optimizing-skills-retro/RESULTS.md) | 2026-05-29 | done | [`RESULTS.md`](optimizing-skills-retro/RESULTS.md) + [`proposed-patch-optimizing-skills.md`](optimizing-skills-retro/proposed-patch-optimizing-skills.md) | retro test of `optimizing-skills` gate on the down-skilling v1.2.0 edit |
| [`skillopt-skill/`](skillopt-skill/README.md) | 2026-05-29 | done (ready to port) | [`README.md`](skillopt-skill/README.md) + [`optimizing-skills/SKILL.md`](skillopt-skill/optimizing-skills/SKILL.md) | SkillOpt review (arXiv:2605.23904) → new skill |
| [`haiku-assessment/`](haiku-assessment/RESULTS.md) | 2026-05-26 | done | [`RESULTS.md`](haiku-assessment/RESULTS.md) + [`GUIDE.md`](haiku-assessment/GUIDE.md) | Haiku 4.5 vs. down-skilled-Haiku probe |
| [`spoke-branch-cleanup-2026-05-25/`](spoke-branch-cleanup-2026-05-25/RESULTS.md) | 2026-05-25 | done | [`RESULTS.md`](spoke-branch-cleanup-2026-05-25/RESULTS.md) | one-shot ops |
| [`phase-a-bridges/`](phase-a-bridges/RESULTS.md) | 2026-05-23 | done | [`RESULTS.md`](phase-a-bridges/RESULTS.md) | [#90](https://github.com/oaustegard/claude-workspace/issues/90), [PR #93](https://github.com/oaustegard/claude-workspace/pull/93), [PR #96](https://github.com/oaustegard/claude-workspace/pull/96) |
| [`te-bridges/`](te-bridges/RESULTS.md) | 2026-05-24 | done (4 runs) | [`RESULTS.md`](te-bridges/RESULTS.md) + per-run | [#97](https://github.com/oaustegard/claude-workspace/issues/97), PRs [#98](https://github.com/oaustegard/claude-workspace/pull/98)/[#99](https://github.com/oaustegard/claude-workspace/pull/99)/[#100](https://github.com/oaustegard/claude-workspace/pull/100)/[#101](https://github.com/oaustegard/claude-workspace/pull/101) |
| [`specter2-gap-issue-87/`](specter2-gap-issue-87/) | 2026-05-23 | done | embeddings JSON | [#87](https://github.com/oaustegard/claude-workspace/issues/87) (remex#69 phase-0 gap fill) |
| [`muninn-kb-issue-76/`](muninn-kb-issue-76/README.md) | 2026-05-13 | done (reproducible) | [`README.md`](muninn-kb-issue-76/README.md) (build recipe) | [#76](https://github.com/oaustegard/claude-workspace/issues/76) |
| [`reviews/`](reviews/) | 2026-04-21 | done | the `.md` files are the results | freeform repo reviews |
| [`snooker-break/`](snooker-break/snooker-break.html) | 2026-05-10 | done | [`snooker-break.html`](snooker-break/snooker-break.html) (interactive) | spike — apex vs Murphy break strike |

## Per-experiment notes

### `needle-depth-growth/` — growing a shipped 45M checkpoint deeper

A one-afternoon check on whether Needle 2's depth can be raised after the fact.
The answer splits cleanly in two, which is why it is worth writing down.

The mechanics are easier than they look. Because the layer stack is scanned
rather than unrolled, every per-layer weight already lives in one array with a
layer axis, and adding depth is `np.concatenate`. Four values make a new block
the exact identity, so a grown checkpoint reproduces the original bit for bit —
verified at +4 and +21 layers.

The training is not a fine-tune, and that is the whole obstacle. New layers
start contributing nothing by construction; making them contribute means
full-parameter training on pretraining-scale data, which the shipped LoRA
trainer cannot express and 800 templated routing rows cannot supply. The paper's
own depth ablation also points the other way at this width.

So the useful form of the answer is: growth is not blocked by the checkpoint
format, it is blocked by the corpus.

### `needle-tool-naming/` — names versus descriptions in a 45M router

`needle-bsky` and `monad-bsky` both ended at the same wall: two categories that
nothing moved. This asks why, with a pre-registered design rather than a
demonstration — the predictions were committed to git before the first variant
ran, and most of them are wrong.

The idea was that Needle 2 routes on the tool's *name* and skims its
description. It does not. Names and descriptions turn out to carry about the
same amount, and to overlap by about a quarter: strip either one and the router
loses roughly a fifth of its accuracy, strip both and it is at chance.

What survives is narrower and more useful than the hypothesis. Improving names
is not a lever — a systematic rewrite matched the original to three decimals and
made the confidence head worse, which for this model is a straight regression.
Wrecking names *is* a lever, and the one category that moves shows the mechanism
plainly: put the word "follower" on the profile tool and the follower-count
query follows it, at 0.81 confidence, with the description untouched.

The deployment answer, which is what the question was actually about: there is
no naming route to declaring all 18 tools. The retrieval head's 16.7-point cost
is not a naming problem, and the fixes remain the deterministic ones the two
sibling experiments already measured.
### `nl2sh-dense/` — closing the retrieval bottleneck the previous three opened

`nl2sh-retrieval` and `nl2sh-selfhist` ended on one number: the fine-tuned Gemma
3 270M routes 0.706 when handed the right documentation page and 0.206 when BM25
picks it. Issue #48 scoped four retrieval items against that, and this directory
runs all four.

The first move was not any of them. Every number in those issues rests on 34
leak-free requests, where one query is 0.029, so `sample_cyber.py` drew 149 more
commands from the same Zenodo/UCI corpus and `gen_nl.py` wrote their natural
language through the same prompt and the same Gemini model — 164 leak-free
requests over 132 distinct gold utilities. On the original 34, the retriever
this directory recommends *lowers* end-to-end routing while raising retrieval;
on the 130 new rows both rise.

What worked: a dense arm fused with BM25, and ranking whole documentation pages
instead of individual examples. Together they take gold-in-sources from 0.262 to
0.390 (p = 0.0003) with a **25.6 MB** ONNX encoder, and a 23.5 MB one performs
the same — the 164.5 MB encoder `xr` vendors is not needed for a phone-sized
tool. What did not: query reformulation in either form, and feeding the model
whole pages rather than one example.

Two follow-up questions from Oskar closed out the directory. Asked whether the
embedder should be fine-tuned, the measurement said no. A linear adapter on frozen query vectors — 4.2 MB, 40 seconds — beats
every arm above it (0.463 gold-in-sources, routing 0.201), and all of that gain
sits on the 207 utilities NL2Bash covers, with a small loss on the 4,491 it does
not. Cutting the adapter's capacity 16-fold reproduces the same split, so the
limit is training coverage rather than model size, and a fine-tune would buy the
same head and the same tail at far greater cost.

Asked whether BM25 could rank pages while the encoder ranks chunks — coarse to
fine — the granularity table says yes in principle: pages are worth +0.061 to
BM25 and −0.054 to MiniLM's dense arm, so a shared granularity suits whichever
arm it happened to favour. Measured, the mixed cell is the best one for MiniLM
(0.366 vs 0.341) and an exact tie for leaf-mt, at p = 0.45. The second half of
that question turned out to have a sharper answer: **which example accompanies a
retrieved utility does not matter, and having one does.** Under oracle sources,
a names-only prompt routes 0.451, the arbitrary first example 0.640, and the
example a relevance pass picks 0.640 exactly. The model reads the documentation
and takes a fixed benefit from an exemplar rather than a graded benefit from a
better one, so the fine stage has no headroom and everything between 0.640 and
1.000 belongs to the generator.

Asked whether a flash-lite pass could preprocess the corpus for a small model to
reason over — pointing at Pleias' Redline, which reports most of its effort went
into converting its source document rather than into its 321M model — the answer
is the largest lever in the directory. Enriching all 6,397 pages with a
normalized summary, goal-level "use when you want to" phrasings and a
disambiguation line, examples kept verbatim, took gold-in-sources from 0.311 to
**0.427** on BM25 alone and end-to-end routing from 0.128 to **0.226**
(p = 0.0052) — beating the trained adapter while training nothing, and composing
with it for **0.555 / 0.250**. 2h05 at concurrency 2, one refusal in 6,397 pages.
Because a Gemini model wrote the eval's language and a Gemini model wrote the
corpus's, a 300-row control with human-authored English was built first: the lift
survives there (+0.098, p = 0.0001). Splitting on whether a card's intents were
reachable from the original page shows the mechanism — pages whose intents merely
echoed them gained +0.086, pages whose intents added vocabulary they lacked
gained **+0.224**, starting 0.126 behind and finishing level. The enriched corpus
ships as an artifact via `handbook.py`, with tldr-pages CC-BY-4.0 attribution
that this repo had been missing.

The abstention gate's diagnosis changed on measurement. `margin >= 5` fails
across corpora because 5 is a number in BM25 score units fitted once on one
corpus, not because a difference is the wrong shape — set the same margin by
quantile and it transfers as evenly as `top2/top1`. What the ratio buys is not
needing a calibration sample at deployment.

### `nl2sh-instantiate/` — the generator, and the difference between an answer and a loop

Stage 1 ended with retrieval finished and the generator not: 0.555 gold-in-sources
against an 0.640 oracle ceiling. Issue #52 argues the remaining loss is not free
generation but **instantiation** — the user's literals substituted into a
documented example — and rests that on §6 of `nl2sh-dense`, where an exemplar is
worth +0.189 routing under oracle sources while the *choice* of exemplar is worth
exactly zero. A model copying a template it was handed, badly. If that reading is
right, asking for the copy explicitly should beat asking for free generation on
the same weights.

It does not, and the first run said so for the wrong reason. Zero-shot the
substitution prompt scored 0.146 against 0.500 — but on 0.774 of rows the model
was not attempting a command at all. It answered `- go — Go to my home
directory`: the shape of the source lines it had just been read, bullet and
em-dash included. That is output-slot confusion, not a verdict on substitution,
and the fix is a control rather than an argument — append a bare `Command:` cue
to *both* conditions and re-run the grid. The lesson generalises past this
directory and is in `METHODS.md`.

One epoch of stage 1's own recipe erases the imitation completely, 0.774 to
0.000, which is the third time this line of work has watched a small instruct
model go from unusable to competent on format alone (0.026 → 0.706 in
`nl2sh-retrieval`, 0.000 → 0.481 in `monad-bsky`). Past training the two prompts
are indistinguishable on the metric the issue set out to move: 23 wins to 20,
p = 0.76.

What they do separate on is garbage. Token-repeat loops — the `mv -t X /usr/ -f X
/usr/ -f …` outputs that route correctly and do nothing — fall **0.183 to
0.049**, and *usable*, which requires both, goes 31 to 16, p = 0.040, +0.092.
`nl2sh-selfhist/MODELS.md` had already named degeneracy "the real ceiling to chip
at" after `repetition_penalty=1.3` bought a 3.7x reduction at a cost of 0.118
routing. Framing the task as substitution buys the same reduction at no routing
cost, which is a better answer to the issue's question than the one it asked for.

Two things the issue asked for and got. The published NL2SH benchmark
(`westenfelder/NL2SH-ALFA`, MIT, arXiv:2502.06858) reproduces the direction on
rows built to be executed — routing 0.854 → 0.866, degeneracy 0.144 → 0.085 —
once its **0.393 always-`find`** prior is subtracted from the 0.911 headline.
And `funceq`, blocked in stage 1 because the sandbox had none of the files the
commands name, unblocks far enough to produce the first functional number for
this whole line of work: with the fixture built from the gold commands' own
paths, 36 of 164 cyber rows are decidable and 0.250 of those are equivalent —
**0.055 over all**, against a 0.427 routing headline. `utility_ok` overstates by
about eight times, which is what #52 estimated by reading every stage-1 output by
hand. The cyber corpus cannot do better than 0.22 coverage under any fixture:
48 golds need tooling absent from the container by design, and 53 gold-or-
prediction rows hit `funceq`'s deny list.

The directory is also a recovery. The session that ran all of this finished the
grid and then waited for its last background job with a `run_in_background` poll
loop, which suspends the turn rather than freeing it; it never answered again and
the container was reclaimed with the scripts, both checkpoints and every
`results_*.json` uncommitted. Everything except the checkpoints came back from
the transcript and re-ran to identical numbers. That failure is also in
`METHODS.md`, because the cheap prevention — detach with `nohup`, commit each
artifact as it lands — is worth more than the recovery was.

### `nl2sh-scoping/` and `nl2sh-retrieval/` — a terminal helper, scoped then gated

`gh-mcp-regex-fit` ended with a cascade that beat live inference by 0.202. The
obvious next target was much bigger: natural language to shell commands. These
two directories scope that problem and then gate it.

The scoping pass answers what shape it is, and corrects itself once. Utility
selection over a long tail, not flag composition — the top ten utilities cover
29% of non-find requests and 72.6% of commands carry at most one flag. Its first
draft argued for `tldr` pages over man pages on the grounds that a man page is
enormous, which conflated retrieval-over-man-pages with putting-a-man-page-in-
context. Measured properly: whole pages reach 47k tokens, but option entries are
median 56 and 93% fit in 350, while tldr covers 96% of the top fifty utilities
and only half the used-once tail. Man pages are the coverage backbone.

The build pass gates the model component. Zero-shot it fails outright:
Pleias-RAG-350M, given the right source every time, produced zero usable
commands in forty and answered with cited encyclopedic prose, because that is
its job. Twenty-five minutes of CPU on six hundred rows fixed it — 0.923 on the
slice where a constant answer scores nothing — so the failure was output shape
rather than capability, which is what `monad-bsky` found on its 56M sibling.

The instructive part is what did *not* hold. That model was chosen because it is
trained to quote sources literally, to fix exactly the identifier-copying
failure `monad-bsky` measured. Its verbatim rate was zero before fine-tuning and
zero after: it generates the command rather than copying it. The result stands
and the reasoning that picked the base model does not.

Retrieval fails separately and for a legible reason: 7,232 tldr pages covering
the modern CLI bury the four hundred classic utilities, so *"find files bigger
than 100MB"* returns `oneliner` and `blkdiscard`. That is `xr`'s documented
cross-repo confusability arriving in a new domain, and scoping to installed
utilities recovers much of it.

The most useful thing here may be the verification. Every measurement made by
one agent was recomputed by another instructed to break it, and the retrieval
numbers came back **OVERSTATED** — the headline figure loses to a
query-independent frequency prior, and a third of the recall that exists comes
from prompts whose English names the answer. The corrections are stated inline
rather than the originals quietly replaced.

### `gh-mcp-regex-fit/` — searching for the routing rules instead of writing them

`monad-bsky` ended with twenty hand-written regex rules beating a 45M
tool-calling model on an 18-tool Bluesky catalogue, at four orders of magnitude
less latency — and with the honest caveat that the rules were written after
reading that eval's failures. This builds the harness that fits them instead,
against the 58-tool GitHub MCP catalogue as actually served to a CCotw session.

The answer is no. A greedy precision-constrained decision list reaches 0.984 on
the family it was fitted to and 0.239 on a held-out phrasing family, where
hand-written rules score 0.696 and 0.546. The cause is not entity memorisation —
the pools were made disjoint to rule that out — but vocabulary: a fitter learns
the surface forms it was shown, and a person writing `\b(diff|patch|changeset)\b`
is supplying alternates the data never contained. The only intervention that
helped adds features computed against the schema at inference time rather than
learned, which is the same shape of fix for the same reason.

Three results outlive the negative one. Fitting on structural cues alone covers
4.9% of training rows, which turns `monad-bsky`'s qualitative boundary into a
one-minute pre-commitment test. A catch-all fallback takes abstention to zero on
all three splits and buys 0.014 accuracy, isolating the parent's 0.500 -> 0.183
refusal collapse to exactly that rule. And hand-authored queries carry the
`owner/repo` their tool requires 13.5% of the time against 61-77% for generated
ones — so every regex-router number in this repo, including 0.833, was measured
on requests structurally richer than the ones a deployed router meets.


**Second pass — the rest of the lexical stack, and one rung up.** BM25, spaCy and
a sentence encoder, measured on the same three splits behind a common arm
interface. The answer to "does anything beat the regexes" is a cascade rather
than a replacement: precise rules first, a *scored* fallback that can itself
abstain second, for +0.136 wild accuracy over the rules alone at zero abstention
cost and a 0.088 ms median. The parent experiment's catch-all was not the wrong
idea, only the wrong implementation of it — a fallback needs a score it can
decline on.

Two findings generalise past this catalogue. Stemming and lemmatisation *lose*,
because in an API catalogue grammatical number is semantic — plural names a list
endpoint — and normalising it deletes the most discriminative token; but they
help recall@10, so stem to shortlist and not to decide. And spaCy's general
vectors score 0.000 on the zero-lexical-overlap slice: what a human supplies
writing `\b(diff|patch|changeset)\b` is domain synonymy, a fact about this API
rather than about English.

It also corrected the first pass twice: a zero-parameter ranker shows the same
family-A-to-B collapse as the fitted list, so that split is adversarial toward
schema vocabulary rather than purely diagnostic of overfitting; and BM25 is
faster than the regexes, so speed is no longer part of the deterministic layer's
case.

**Third pass — the clean room actually ran.** Two compilers that never saw the
eval write the rules instead: Gemini 3.7 Flash through the Cloudflare gateway,
and a Claude subagent handed one spec file and forbidden this directory. The
question the thread started from gets a yes — a model reading a catalogue and
emitting regexes produces a router scoring 0.540 on hand-authored requests at
0.077 ms, unsupervised, which is 95% of what the same model achieves answering
every request itself at 1,267 ms.

Two corrections fall out. The contamination the second pass worried about is
worth about ±0.05 and points the wrong way: rules written with the eval in view
score higher on the template family and *lower* on realistic requests. And every
A/B/wild gap in this writeup is authorship rather than difficulty, because an arm
fitted on nothing scores 0.532 / 0.532 / 0.568 across the three splits that every
other arm spreads three- to sixteen-fold on.

What is worth keeping is the cascade. Compiled rules in front of the live model
reach 0.770 against the model's own 0.568 while removing 58% of its calls — not
because the rules are better, but because the model declines 40% of routable
requests at 0.955 precision and the rules answer exactly those. Complementary
abstention is the mechanism, so the two standalone accuracies do not predict it,
and it only appears above a compiler-capability threshold: three cheaper Gemini
tiers score 0.000, 0.013 and 0.176 where Claude scores 0.540, and that gap
survives an explicit breadth instruction, an eightfold call budget producing more
rules per target than Claude wrote, and two rounds of supervision.

### `monad-bsky/` — Monad at the needle-bsky routing task

`needle-bsky` measured a purpose-built 45M tool-caller. This runs PleIAs Monad,
a 56M generalist reasoning model, at the identical task: same tools, same 62
queries, same scoring code imported from the sibling, same training rows.

Zero-shot Monad emits no parseable call in 62 attempts. Fine-tuned it routes
about half of them, which is real capability transfer from 800 examples and two
hours of CPU, and still loses to Needle at every configuration.

The reason is not what it looks like. Sorting the errors separates routing
mistakes from transcription mistakes, and Monad's are mostly the second kind: it
reproduces an identifier from the request correctly 51% of the time against
Needle's 78–90%. `austegard.com` comes back as `afethew.com`. The tempting
explanation — a small prose-trained vocabulary shattering identifiers — is
measurably false: both models carry 8,192 pieces and segment these strings the
same way. What differs is that Needle was built around span-copying and Monad
was not, and three epochs did not install it.

Keeping Monad's tool choice and filling arguments with a regex recovers most of
the argument gap, which is the same shape as this repo's other finding about
small models: let the model choose, let deterministic code handle the strings.

Eight ways of combining the two were evaluated over the committed rows. One is
worth having: when both models independently name the same tool the answer is
right 88% of the time, where Needle's own confidence head manages 74% at the
same coverage — and agreement works without a confidence head, which is what
fine-tuning removes. Calibration itself does not transfer between the models at
all.

### `needle-bsky/` — a 14 MB router in front of the Bluesky reads

Cactus Needle 2 (45M parameters, 14 MB, built for microcontrollers) declared over
the 18 read tools in `browsing-bluesky` and `atprotoing`, measured on 62
natural-language queries. Base top-1 is 61–70%. Rewriting the schemas for the
router rather than reusing the existing docstrings is worth +26 points and is
the only contrast that survives a paired test at this n.

The confidence head behaves in a way that carries to other toolsets. Needle
scores the full call, so an optional argument the query never
licensed — a `limit` the model invents because the schema offers one — drags a
correctly-routed call to 0.0004 and flattens the act/escalate gate. Declaring
only required arguments restores a monotone precision/coverage curve, and a
single-tool control where misrouting cannot happen reproduces the effect on its
own (p=0.00032 across two added arguments).

Two deployment numbers: five tools per agent is a different latency regime from
six (284 ms vs 1034 ms per turn, then flat to 18), and a five-tool catalogue
containing the right answer beats the full 18 by 11–17 points, so retrieval is
where most of the remaining error lives.

Acting on that splits the catalogue into five groups of ≤5 and routes in two
steps, which works only when stage 1 is deterministic. A Needle turn over group
descriptions scores 0.370 routable, 24 points below simply declaring all 18;
a regex over structural cues scores 0.722, above the flat arm and most of the
way to the five-tool ceiling, at 316 ms against 1187 ms. The contrastive
retrieval head is good at picking 5-of-18; the decoder is bad at picking 1-of-5
abstract categories.

Two arms came back negative and are reported as such. A LoRA fine-tune on 800
templated rows did not move routing (p=1.0), did not touch the two categories
its templates covered, and cost the confidence head outright. Extraction over
post text never cleared 0.05 confidence in 22 attempts, in three different
schema shapes.

### `orchestrated-coding-pareto/` — the fleet didn't need the orchestrator

Follow-up to `luna-onprem-tco`. Five arms, 14 reference-validated hidden-test
tasks, three difficulty tiers. Haiku 4.5 one-shot matched Opus 5 at 14/14
everywhere — including a stack-VM interpreter and a character-exact table
formatter — so the orchestration arms (test-feedback retry; opus-diagnose →
haiku-fix) had zero failures to work on. What did separate the arms was output
volume: haiku emitted 6.7× opus's tokens, cancelling its price advantage
outright at Anthropic rates. At Luna-class rates the fleet wins by ~4×. The
orchestrator-fleet question reduced to a procurement question plus an open
eval-design question (the regime where an orchestrator could earn its premium —
under-specified, multi-file work — needs a non-unit-test grader). Effort follow-up: `low` cut haiku's tokens only 26% but cost two task passes, and the retry arms it finally activated showed raw test feedback fixing everything opus diagnosis fixed — zero orchestrator premium at n=2.

### `luna-onprem-tco/` — the ratio that decided nothing

**Second pass (2026-08-16)** added `hourly.py`: the same price book at 1/1000th
the scale, one RTX 5090 running Qwen3.8-27B. It answers the *opposite* way —
self-hosting wins above ~4 h/day of flat-out generation — for exactly the same
reason the fleet model said no. Neither turns on $/token; both turn on what
fraction of a fixed capacity gets filled. Two premise corrections came out of
it that generalise: check a quoted tok/s against `bandwidth ÷ weight_bytes`
before building on it (190 was 1.56× the hard ceiling, reachable only via
multi-token prediction or batching), and remember that prefill and decode
contend for one card, which turned 190 tok/s into a sustained 123–167. Plus a
price-book result worth knowing on its own: Luna's cache writes cost 1.25×
uncached input, so **caching only pays above a 21.7% hit rate**.

A procurement question that arrived pointed at the wrong variable, and a record
of noticing rather than of being right first.

The ask was to compare electricity against an API price. That comparison is
easy, defensible, and inert: local inference costs **$0.0035 per million input
tokens** in electricity at Montgomery County commercial rates, and the API
charges 57× that. Every part of it is true. None of it decides anything,
because the electricity line is **3%** of the cost of owning the machine. What
decides it is that matching a closed model's capability forces an open model of
a particular size, which forces a minimum GPU count, which you buy entire
regardless of how much of it you use — and 800 office seats use 17% of it in
their busiest hour.

Two turns of user-supplied constraints (seat count, work pattern, batch window,
and a correction that the quoted price was Bedrock's rather than OpenAI's, with
a batch tier at half) inverted the conclusion without changing a single piece
of the arithmetic. That is the entry in `METHODS.md` worth having: **compute a
marginal cost's share of total cost before reporting its ratio to a price.**

The model is stdlib-only, ~40 constants in `params.json` each carrying a source
and a confidence tag, and `recheck.py` runs 104 checks in ~4 s — including
negative controls asserting that *free electricity does not flip the verdict*,
which is the finding stated as a test. Two bugs in the model were caught by
those checks and by a seat sweep, both flattering self-hosting, both fixed
structurally rather than annotated.

The provenance is uniformly weak and the writeup says so in its own section:
the session's egress proxy blocked every primary source, so the figures are
search-engine summaries of pages nobody in this session could open.

### `ttt-embed-quantized/` — the committed SciFact matrix

A data product, not a study. claude.ai can encode a *query* in-session but not a
*corpus*: jina-v5-nano q4 measured there at <2 docs/s on 1 core with a 13 s
session load, and detached jobs die silently after ~100 s, so 5,183 documents is
45–85 min that cannot be run to completion. CCotw can, in **14.7 min at 5.9
docs/s on 4 vCPU**, so the matrix is committed and the encode never happens
again.

The settings are the point. They are pinned bit-for-bit to the 2026-07-08 codec
eval — `dim=256` Matryoshka, `max_length=384`, `title + ". " + text`, asymmetric
`Document: `/`Query: ` prefixes, last-token pool at `mask.sum(-1) - 1`, truncate
*then* L2-normalize — because that is what lets the earlier fidelity numbers
carry over to the downstream remex 8/4/2/1-bit and remax k=8/4/2 sweeps without
re-measuring them. Two consequences followed from taking that seriously:

The pinned encoder is **knowingly the worse one**. The mirror's `PERFORMANCE.md`
and `METHODS.md` both record that the model authors' upstream q4 beats this
repo's `model.q4.onnx` on every axis. It was used anyway — comparability with a
prior eval is a constraint on *weights*, not just on hyperparameters, and
silently substituting a better encoder would have quietly voided the reason the
artifact exists.

And a cache that already existed was **found and declined**.
`rotation-decorrelation` carries a `jina_scifact_corpus.npy` for this exact
corpus and embedder. It is corpus-only, and nothing records its encode settings.
Had it happened to be 256-dim, every structural check would have passed and the
mismatch would have been silent. This is the inverse of the failure this repo
usually guards against: the standing rule is to search for prior art before
building, and here the correct outcome was to find it and not use it. *Found* and
*reusable* are different findings, and provenance is what separates them.

The sanity check is a real check: `recheck.py` re-scores the committed files cold
through a deliberately disjoint implementation and reproduces nDCG@10 to <1e-9,
and its two negative controls collapse to ~0.003, which is the part that makes
the agreement mean anything.

**A concurrent second encode turned this into a replication.** Run in a
container where every HF host was 403, it rebuilt SciFact from the upstream
AllenAI release, verified BEIR's four published cardinalities, and registered
the one gap counts cannot cover — the document *strings*.
`crosscheck_allenai.py` closed it: the rebuild matches `BeIR/scifact` on
**4128/5183 documents and differs on 1055 (20.4%)**, because AllenAI's abstract
sentences keep trailing whitespace at structured-abstract section boundaries
that BEIR normalised away. The two artifacts scored **0.7152 and 0.7067** — both
inside the pre-registered band, so the band certified a corpus it could not see
was wrong. Wherever the input strings were identical the vectors were
*bit*-identical (4137 docs, 300/300 queries, zero
same-string-different-vector), so the encode path is deterministic and the whole
gap is text.

### `account-index-corpora/` — what tombstones and PR bodies cost the account index

The two corpora that measured as wins on one repo — deleted files
(0/6 → 6/6 on *how did the removed thing work*) and merged PR bodies (6/8 → 8/8
on *why was it done this way*) — are still not in the account-wide build.
[`claude-workspace#197`](https://github.com/oaustegard/claude-workspace/issues/197)
frames the open question exactly right: the risk is **size**, not answer quality,
because `hybrid-code-index` already measured an off-class corpus as inert rather
than harmful. Size costs seconds; answer quality costs a 22-minute sharded
encode and a benchmark nobody has written. So size was answered on its own.

The instruction to measure before shipping paid for itself immediately. The
first tombstone run reported **74,822 chunks against a 13,257-chunk tree** —
1.7× the entire live account index, from three repos. Not sampling noise: a
deleted file gets no `stat()` and no `rglob`, so every filter
`hcindex.discover` applies to the working tree has to be reapplied by hand, and
none of them were. A 767,692-line deleted embedding dump walked into a corpus
the live index refuses at its 1 MiB cap. **Any corpus that does not come from
`discover()` has to re-implement `discover()`'s exclusions, or it will index
precisely what the tree was configured to keep out.**

Filtered, tombstones are 47× smaller and still not worth turning on here:
claude-workspace contributes 1,484 tombstone chunks against a 232-chunk working
tree, ~94% of it sub-1 MiB JSON data dumps, ~23 chunks of prose and source. The
corpus that claims to answer *how did the deleted thing work* is mostly deleted
machine-generated data — a doubling of the tree's own crowding problem rather
than a new defect. PR bodies come out the other way: **+3.2%**, on an account
whose median PR body (1,577–3,197 chars) matches the profile the original win
was measured on.

Two things only visible at account scale. The relocation guard has to compare
**across repos** — the 2026-07-28 migration deletes 37 projects in one repo and
lands them in another, and a per-repo guard would re-import all of it labelled
*gone*. And `--depth 50` was doubly wrong: it cost nothing (7.3 s vs 6.5 s at
depth 1, summed over three repos, with no consistent sign — and account-wide,
65 repos at full depth took 68 s against the 89 s on record at depth 50), *and*
it would not have worked if used, because `git log --diff-filter=D` sees only the grafted
window — it found 2 of muninn-utilities' 18 deletions. Shallow-clone deletion
coverage is a function of the repo's commit rate, which is not a property anyone
chose.

Both corpora ship off by default, decided in `plan` and carried in `plan.json`
rather than passed as a flag to each job: a sharded rebuild has three
independent processes rebuilding the corpus and matching rows by content hash,
so a flag that reached one and not the others would not error — the merge would
silently re-encode the account.

### `bekko-embedding-bench/` — does a code-capable encoder change the search verdict?

Two decisions from one handoff ([claude-workspace#185](https://github.com/oaustegard/claude-workspace/issues/185)),
and they came out opposite ways.

**Part A reverses a standing verdict.** `searching-codebases`' semantic tier was
retired in 2026-07 after two runs in which it never beat identifier `rg`. But
that tier was **TF-IDF**, so the replication falsified TF-IDF, not dense
retrieval. Swapping in a neural encoder flips it: bekko beats `rg` at r@5
**0.806 vs 0.667**, and on the identifier-poor instance — one that describes its
bug derivationally and points at code by line-number URL, pasting no identifier —
`rg` scores **0.000** against bekko's 0.667. The pre-registered gate passes. The
honest recommendation is **fusion**, not replacement: RRF(rg, bekko) hits 0.889
because the two arms fail on disjoint instances.

Two things the run refused to let stand. The handoff's prior that *chunk
boundaries matter more than the encoder* **does not hold** — both axes are inside
noise (+0.028 and 0.000 on r@5) and both are small next to the +0.139
dense-vs-grep gap. And the flattering token number is not the right one: dense
looks 12.8x cheaper than `rg`'s full line output, but `rg -l` returns exactly
what a *file*-discovery metric scores and costs **half** what dense does.

**Part B needed a second axis before it had an answer.** On quality per byte the
incumbent wins outright — jina v5 nano q4 takes 11 of 12 iso-byte cells, worst
for bekko on the *code* distribution it was advertised for (0.983 vs 0.888). That
was the whole verdict until Oskar pointed out it prices bytes and ignores
compute, which is the entire design point of a 7.7M-**active**-parameter model.
Measured: bekko-a8m answers a query in **11.3 ms on 1 vCPU against jina's
146.4 ms** — 12.9x, matching the ~12x FLOPs ratio, so architectural rather than a
q4 artifact. The honest output is an **iso-quality ladder** (a8m ≤ 0.575 blog
R@10 at 11.3 ms; a25m to 0.598 at 35.0 ms; jina alone above 0.60 at 146.4 ms),
and the choice is regime-dependent. It also revises a claim made here earlier:
regime A from `7cecfd94` was already available on *size* with the incumbent, but
"never needed bekko" was too strong — a 1-vCPU reader paying 146 ms/query is a
different product from one paying 11 ms.

**Composition, answering "does quantization beat truncation at a fixed byte
budget":** yes, decisively. remex d=384 @ 2-bit costs **96 B at R@10 0.609**
against full fp32's **1536 B at 0.598**. Spend the budget on coordinates, not
bits. And 2-bit beats 1-bit in **all 8** cells, so bekko sits on the **Jina**
side of the one-bit-beats-two inversion, not the SPECTER2 side.

**Process note.** The instruction was to reuse the prior 7 instances; that was
not satisfiable, because only 2 were ever recorded by number and the harness has
now been rebuilt three times. The code was never the expensive part — the
*instance set* was. It is committed this time (`instances.json`), which is the
only reason a fourth rebuild would produce comparable numbers.

### `svgview/` — is a browser really the wrong tool for rendering SVG?

Prompted by [a Bluesky post](https://bsky.app/profile/andri.dk/post/3mrewq7fcsc2j)
calling browser-based PDF/SVG rendering "bonkers insane." The claim is testable,
so it got tested: build the lightest native SVG viewer that is actually usable
and see what it costs.

It costs very little, because `resvg` had already done the hard part. The whole
viewer is ~600 lines over `resvg` + `winit` + `softbuffer`, and the numbers are
one to two orders of magnitude below a browser-based equivalent: 4.8 MiB
executable, 12 MiB resident, 16 ms from `exec` to a window on screen. Static SVG
is a solved problem outside the browser and has been for years.

The argument does **not** generalize to PDF, which is what the original post
actually lumped together. PDF is thirty years of accretion — forms, JBIG2/JPX,
seven shading types, CID fonts, encryption — and the two honest Rust options are
wrapping PDFium (which *is* Chrome's PDF engine, just without the browser around
it) or accepting `hayro`'s documented gaps. The elephant gun exists because that
particular pigeon fights back.

Two process notes worth more than the code:

- The first version of the GUI smoke test counted distinct colours per
  screenshot and reported `ok` for all seven key bindings while **none** of them
  were reaching the window — with no window manager on the virtual display,
  nothing had assigned input focus. Both the bug and the useless check are in
  `METHODS.md`.
- The app icon is generated by rasterising `assets/icon.svg` with svgview
  itself, so artwork and icon cannot drift. Costs about 40 lines of Python.

Not done: no interactive Windows testing at all, so the `Ctrl+O` dialog,
embedded icon, GUI-subsystem behaviour, and the file-association scripts are
written-but-unproven. No macOS attempt.

### `erdos-gyarfas/` — does every min-degree-3 graph have a power-of-two cycle?

Open since 1995; Erdős expected it false and offered $100 for a proof, $50
for a counterexample. What holds up: a reformulation of the conjecture as a
growth race between two integer sequences (it is false for cubic graphs iff
`f(k) <= 2^(k+1) - 1`, where `f(k)` is the smallest cubic graph avoiding
`4..2^k`); a barrier proposition showing no bound on the *number* of distinct
cycle lengths can ever settle it, since the target set has size `log2 n`; a
one-line corollary of Bondy–Vince killing every residue-class construction;
and a sharpening of Carr (arXiv:2605.22844) from 4/7 to 2/3 using a corollary
that paper proves and never applies.

Also a genuine gap in Exoo (arXiv:1403.5636): the supporting lemma for
`f(5) <= 450` asserts every 8-cycle of the Tutte–Coxeter graph contains two
consecutive outer-Hamiltonian edges. It has 90 such cycles and 10 violate it.
`src/tutte_coxeter_lemma.py` is self-contained — run it.

The computational half — complete censuses at orders 24 and 26 — reproduces
Markström (2004) graph-for-graph rather than extending it. That was not the
intent: it was built on the belief that the order-26 count was unknown, which
was wrong, and the paper was a search away. Kept as independent verification
of a 2004 computation nobody had checked since, and as a caution.

Frontier: a cubic counterexample needs 54 <= n <= 62, even, avoiding cycles
of length 4, 8, 16 and 32.

### `ms13-k4/` — Q7′ and Conjecture 12.2 at k = 4

Single session, 2026-09-01. `ms13-campaign` left one clean question open —
is the linear discrepancy of a network matrix with demand-scaled columns
bounded by Doerr's unit-box constant `1 − 1/(k+1)` — and had proved it only
at `k = 3`, calling `k = 4` a 2,070-hour census. Two changes made `k = 4` a
seven-minute job. The census is replaced by Buneman's theorem: a row is a
split of the `2k` chord endpoints, compatible split systems are trees, every
tree refines to a binary tree with all endpoints at leaves, and `R` is
monotone in the row-set, so the maximal types are binary trees on `2k` leaves
with a perfect pairing (4 shapes × 105 pairings), and the enumeration
reproduces the campaign's `k = 3` census exactly. The prover keeps the
campaign's exact rational B&B and changes only the branching order to
fail-first. Result: every one of the 14 maximal four-chord types has
`R = 4/5` exactly, attained at unit demands, so column scaling does not beat
Doerr at `k = 4`. The writeup also records a six-line proof of the unit
bound (Hoffman–Kruskal integrality plus Carathéodory on the floor/ceil
polytope) and names the step the weighted case lacks.

### `ms13-campaign/` — kill Morell–Skutella Conjecture 1.3 (campaign)

Issue #169, multi-session; state in `WORKLOG.md` + `NOGOS.md` (the ledger is
the deliverable). Three sessions, no counterexample, and the negative results
compounded into a structure theory: path-closed 2-path instances are
out-trees (Lemma 4, from the post-mortem of a false positive that the
certificate gate caught); on that class the conjecture is exactly signed
discrepancy over the tree's totally unimodular network matrix (Lemma 5);
equal demands are provably safe (Lemma 6, Hoffman–Kruskal); and 85% of
enumerated instances are already closed by MSW25's series-parallel theorem
via a K4-minor criterion (NG-9), with the Rybin instance landing exactly on
the K4 boundary. The swept live set plateaus at −4/9·d_max. Kill-path B is
parked (β* ladder self-limits at 7/6 against the 2 needed). What's left is
one clean open question — linear discrepancy of a network matrix with
demand-scaled columns — which subsumes every instance in the class and has
neither a theorem nor a counterexample in the literature.

### `discrepancy/` — certified Komlós + Beck–Fiala lower-bound records

Issue #166's two-target discrepancy play, run with the ms13 max-min skeleton
and a hard "exact certificates only" rule: floats screen, integers decide
(ℤ[√2] pairs for the Kunisky family, Fractions for rationalized search
records, SAT UNSAT + exhaustive enumeration for set systems). The mandated
literature gate reshaped both targets before compute: Kunisky's asymptotic
K ≥ 1+√2 made "beat disc > 1" dead on arrival (per-size K(n) records are the
honest deliverable), and the 2025 Bansal–Jiang line resolved Beck–Fiala
asymptotically while leaving small-t exact values unpublished. Two negative
results caught early saved the day: Δ = δ on Kunisky's finite instances
(provable in one line — the planned "compute exact Δ" deliverable was
vacuous), and PG(2,3) has disc 2 (no classical D(4)=4 shortcut). The CEGAR
search rediscovering the Fano plane as the *provably minimal* D(3)=3 witness
is the flagship result; the per-size Komlós records (n=4 beating Kunisky's
own tree matrix at n=4 is the structural surprise) are the second. Search
lesson recorded in RESULTS: restart diversity dominates iteration depth on
this piecewise-linear landscape (n=7 record moved 1.726 → 1.830 on a lucky
seed batch; the n=4 record was never re-found in 24 later restarts).

### `session-relay/` — inter-session Claude coordination over Turso

Reproduces the Bluesky relay pattern (two Claude sessions chatting to share
best practices) without deploying anything: the shared Turso DB is the
rendezvous, so the relay is one table plus a small CLI. Validated with a live
two-agent run — disjoint knowledge seeds, blind negotiation over the channel,
clean CONSENSUS/ACK handshake, and a joint deliverable neither side could have
written alone. `relay.py` is reusable as-is for cross-session coordination
(CCotw ↔ Claude.ai included); see RESULTS.md for the protocol lessons
(cursor discipline, stdin posting, front-loaded openers).

### `remex-vs-higgs-ablation/` — does remex beat the QuIP#/HIGGS lineage for retrieval?

Follow-on to `jina-remex-vs-remax/`, commissioned as issue #8. remex and the
**QuIP# → HIGGS → TurboQuant** lineage differ on three axes at once, so a
head-to-head can only say *which* wins. This runs the full **2×2×2 factorial** —
rotation (dense Haar vs randomized Hadamard) × norm handling (exact fp32 norm
out-of-band vs per-block scale in the payload) × codebook (scalar Lloyd-Max vs
Gaussian-MSE-optimal multi-dimensional grid) — so the difference can be
attributed to an axis rather than to a method.

**Only axis C moves.** Pooled over **4** corpora (d=100/768/784/1024) × 6 bit
widths × 5 rotation seeds: rotation −0.0004/+0.0005 recall@10, norm handling
+0.0007/+0.0010, codebook **+0.0082 (cosine) / +0.0112 (inner product)**. The
codebook effect peaks at **+0.035 at 2–3 bits**, decays monotonically, and is
gone by 8 bits — and it is about twice as large at d=100 as at d=768/1024,
which is what the scalar-vs-vector gap should do as coordinates get closer to
i.i.d. Gaussian.

**Axis B is now measured rather than argued (2026-08-05).** The first two runs
registered `fmnist784` — ANN-benchmarks fashion-mnist-784, raw pixels, norm
CV 31% — as the corpus that could actually read axis B, then shipped without
sweeping it, grading the conclusion *ARGUED, one corpus*. It has now been swept:
axis B is flat there too (+0.0005 cosine, +0.0009 IP) despite the largest norm
spread of the four, so **exact-norm storage buys nothing as a main effect even
off cosine-trained encoders**. The same run *refuted the published mechanism*
for the one place remex does win — the 1-bit reversal. That story predicted the
effect should fade as norm spread grows; fmnist784 has the most spread and the
biggest remex win (−0.047 under IP), and it reverses under cosine too, where
‖x̂‖ is divided out and norm noise cannot reach the ranking. Two further
candidates (non-Gaussian rotated marginals; block-correlated VQ residuals) were
measured and refuted. What survives is a **decoupling**: at 1 bit the vector arm
wins reconstruction MSE *and* projected score-error variance on every corpus and
still loses recall on 5 of 8 corpus×metric cells. The effect is MEASURED and
**UNEXPLAINED** — recorded as such rather than given a fourth story.

**Two of the four pre-registered predictions failed.** The randomized Hadamard
was predicted 10–100× *faster* to apply at d=768–1024; corrected, it is **at parity**,
because numpy runs the dense rotation as one BLAS `sgemm` and the FWHT as a
Python loop over strided slices. The ratio does move the right way with
dimension (50× at d=100 → 1.5× at d=8192), so the asymptotics are visible, but
the crossover is far past any retrieval dimension. And exact-norm was predicted
to win under inner product; it does not — partly because BGE-family encoders
are *trained* under cosine, so their raw norms barely vary (CV 1.4–2.7% against
GloVe's 20%) and inner-product retrieval is nearly the same problem as cosine.

**The result most likely to change a decision** is one the scheduled adversarial
review forced into the writeup: the headline tables exclude the rotation and the
codebook as "shared across the index", which is the convention both lineages use
and is *not symmetric*. remex's shared cost is a d×d rotation; the vector arm
additionally carries a codebook up to 1 MiB. On a 20,000-vector index at 4 bits
that is **52.5 B/vector against a 50 B payload**, and the recall-per-byte
ordering reverses: remex at 6 bits (81 B true, R@10 0.965) beats HIGGS-like at
4 bits (112.5 B true, 0.893) on bytes *and* recall. The vector arm needs roughly
**350,000 vectors** before its codebook amortizes below 5%.

So remex loses on distortion in the 2–3 bit regime and should not be defended
there — but its distinctiveness was never distortion. It is numpy-only,
calibration-free, data-oblivious, carries a 2 KiB side
table instead of 1 MiB, and wins on true bytes-per-vector below a few hundred
thousand documents.

**Rerun 2026-08-02 under the [`gating`](https://github.com/oaustegard/claude-skills/tree/main/gating)
skill.** The question that run asked was not "does the calibration gate pass"
but "can it go red." Audited (`audit.py`), three of its checks could not fail in
the way that mattered: the rotation check compared the two arms to each other
and nothing else, so replacing **both** with the identity still passed; the
axis-C check asserted an inequality with no margin, which a vector arm doing no
vector quantization clears by sampling noise; and the published-table anchor
stops at 5 bits while the sweep runs to 8. Mutation testing (91 mutants,
55 survivors) found the gate scored codebook *contents* and never ran the
*encoder* — mutating the decision boundaries, the nearest-neighbour k, and the
axis-B norm-mode branch all left it green. The rebuilt gate (`gate.py`, 166
checks, 5 known-bads, 14 stated coverage limits) now blocks the sweep on a
non-zero exit instead of being a documented step, and three of its own new
checks went red on first run — including one that turned an *attainable*
optimum into a failure, because a Hadamard transform maps a coordinate spike to
exactly ±1/√d. Conclusions are unchanged: `glove100` and `nfcorpus1024`
reproduce to four decimals, `arxiv768` shifts because its abstracts are a fresh
draw.

**Process is most of the value here.** The two-sided calibration gate ran
*before* the sweep and immediately caught Lloyd-from-random-init converging to
grids **worse than the scalar quantizer** at 6 and 8 bits — which would have
shipped as "scalar wins axis C at high rate", exactly the wrong conclusion the
issue warned about. Fix: seed Lloyd from the scalar product grid, and keep the
unrefined product grid as a candidate. A scheduled adversarial review then found
five further blocking defects, two confirmed by direct measurement before
anything was changed: a **stale codebook** served by a cache keyed on the
problem `(m, K)` rather than on the method (the 8-bit vector arm was 87% worse
than scalar), and a **Lloyd-Max MSE identity evaluated off the fixed point**
(+16% at 8 bits — and Max (1960)'s table stops at 5 bits, so the published-value
check could never have caught it, while the inflated baseline made the gate
*more* permissive). Also: a "provably no worse than scalar" guarantee that was
argued rather than enforced (Lloyd is monotone in *training* distortion, but
selection happens on held-out), a block/sub-vector divisibility bug that
corrupted only the HIGGS-like arm and so confounded axes B and C, and a gate
that never certified the grids behind any glove result. Separately, scoring
`q·x̂` without dividing by `‖x̂‖` manufactured a fake 1-bit axis-C reversal,
because the 1-bit scalar quantizer's reconstruction norm is constant *by
construction* and pays no penalty for norm error.

Gate anchors: Max (1960) table 1 reproduced to the printed digits at 1–4 bits,
E8's normalised second moment to 3.7e-4 relative, a tuned ball-shaped E8
codebook (QuIP#'s codebook family) that the trained grid must beat by ≥1%, and
HIGGS §4.3's own stated envelope (grid dimension p ∈ [1,5], size n ∈ [9,4096]) —
which the grids here meet or exceed at every bit width.

### `jina-remex-vs-remax/` — remex as a practical compressed-Jina vector format

Oskar asked whether the **remex** (rotation + Lloyd-Max scalar quant, the
TurboQuant lib) optimizations could apply to "our Jina q4 embedding," loosened to:
can remex be a practical compressed-Jina format on its own, not just a byte-budget
winner over remax? Two prerequisites cleared first: (1) **remex ≠ remax** — two
different spokes (remex = multi-bit scalar quant; remax = 1-bit centered SimHash,
what remax_kb ships). (2) **The two q4s** — q4-the-model (`JinaQ4ONNXEmbedder`,
weight quant, outputs fp32-parity floats) is orthogonal to remex/remax, which
compress the output *vectors*; remex would replace the remax 1-bit step.

The measurement was the real work. Recall-vs-human-qrels **saturates from both
ends** — muninn ceilings (fp32 R@5/R@10 0.90/1.00, every remex config ties it),
NFCorpus floors (fp32 R@10 0.241, embedder-limited, everything piles at ~0.24).
Both hide quantization damage because they conflate embedder quality with codec
fidelity. Fix: score each code against **fp32's own ranking** (recall@k vs
fp32-kNN at chunk level, Spearman ρ, reconstruction cosine — remax's own bench
metric). That de-saturates cleanly.

Result (NFCorpus, n=120): **remex dominates remax at every byte budget.** remex
ρ 0.92–1.00 vs remax 0.63–0.74. remex 4-bit @ d768 (384 B) is near-lossless
(ρ 0.998, recon 0.9953, R@10-vs-fp32 0.96); remex 2-bit @ 192 B (ρ 0.978) beats
remax d768/k2 at the same 192 B (0.741); remex 1-bit @ 96 B beats *every* remax
config. **Bits beat stacks**: graded magnitude (Lloyd-Max) beats more sign-bits
(stacked SimHash) at equal bytes; and full-dim-low-bit beats truncated-dim-high-bit
(2b/d768 192 B > 4b/d512 256 B). remex's rotation is data-oblivious random Haar —
the kind `recall-per-byte`/`rotation-decorrelation` endorsed when they rejected
learned ITQ (remax#46). Recommendation: ship remex into remax_kb as the
near-lossless/mid-byte codec (default 4-bit @ d768), alongside (not replacing)
1-bit; costs a new SPEC binarizer type + ADC scan path (numpy+scipy), additive.
Caveats: muninn n=5 (directional); Jina vectors unit-norm so remex per-row norms
excluded from B/row (honest parity with remax). Assets (q4 ONNX, muninn.kb,
NFCorpus) gitignored — regenerable via the run scripts.

**Reconciliation with `one-bit-beats-two` (Oskar flagged the apparent
contradiction):** the founding remax result — 1-bit *beats* 2-bit — is real and
reproduced here on SPECTER2 (this harness: 1/2/4/8 = 0.642/0.496/0.742/0.974 vs
blog 0.635/0.501/0.731/0.971), but is **embedder-specific**. My first hypothesis
(a Matryoshka bit-shaving artifact) was *falsified*: `reconcile.py` shows both
Matryoshka and independent codebooks monotone on Jina, and `reconcile_specter2.py`
shows the reversal in *both* on SPECTER2 — so it's the embedder, not the codebook
construction. SPECTER2 (specialized, tight clusters) reverses; Jina v5-nano
(general, isotropic) is monotone. So remax_kb's 1-bit choice was right for
SPECTER2 but flips for the Jina-based muninn deployment. `reversal.png`.

### `remax-hamming-speedup/` — make the 1-bit Hamming scan beat BLAS float matmul

remax_kb#15: the storage win of 1-bit codes (12× smaller per vector) was proven,
but the *latency* win wasn't — the reader's `hamming_scan` used a per-byte
popcount **LUT gather** (`POPCOUNT_LUT[xor].sum(axis=1)`) that micro-benched ~10×
slower than an equivalent BLAS float-cosine search at small N. The issue listed
four candidates cheapest→heaviest; `bench.py` times them all on the muninn
micro-bench shape (d=512·k=4 → 256 B/row) over N=600→1M with BLAS pinned to one
thread, asserting each Hamming kernel returns the *identical* top-k to the LUT.

**The cheapest candidate wins outright.** Approach 1b — `np.bitwise_count`
(numpy≥2.0, hardware POPCNT) over a **uint64 view** of the contiguous XOR (8× fewer
elements before the reduction) — is **~10× faster than the current LUT and faster
than BLAS float cosine at every N _in this configuration_** (d=512·k=4 → 256 B/row,
32 uint64 words), zero-copy, codes stay bit-packed. The ±1 BLAS
matmul (approach 2) ranks identically (`q·D = nbits − 2·Hamming`) but is 2–6×
slower than the popcount path *and* must un-pack the corpus to int8/fp32, forfeiting
the storage win (OOM at N=500k on 15 GB); a compiled SIMD kernel (approach 3) is
unnecessary when pure numpy already beats BLAS. Issue success criterion met and
exceeded — the win holds at all N, not just ≥10k.

**Config-scoped, and superseded on the kernel** — see
[`lowbit-scan-crossover/`](lowbit-scan-crossover/RESULTS.md). The "every N" claim
was later contested by a measurement at **k=256 (32 B/row, 4 words)**, a different
configuration: `.sum(axis=1)` over a 4-wide inner axis runs at 1.9 GB/s against a
32-wide axis's 3.3 GB/s, so the narrow config loses to BLAS below ~68k while this
one does not. Both measurements are correct about their own shape. Separately,
approach 1b is **not** the fastest pure-numpy kernel: storing the words as
contiguous **bit planes** (SoA) beats it 2.4x here and 5.2x at k=256, and a
compiled kernel — ruled "unnecessary" above — is 37x over BLAS, not 2.6x.

Shipped to remax_kb: `_hamming.py`'s `hamming_scan` and `read_v2.py`'s duplicate
popcount both delegate to a shared `_popcount_rows(xor)` — fast path when
`np.bitwise_count` exists (uint64 view when row bytes %8==0, else uint8), per-byte
LUT fallback for numpy<2.0 so the `numpy>=1.24` floor is untouched. Added
`tests/test_hamming.py` (remax-free, 20 cases) proving bit-for-bit equivalence to
the frozen LUT across 8 `(dim,k)` widths including non-multiple-of-8 byte rows,
plus top-k order, distance bounds, and the validation guards. Reproduce with
`OPENBLAS_NUM_THREADS=1 python3 bench.py && python3 plot.py`.

### `kb-packer-web/` — browser KB packer for austegard.com/ai-tools/

The user-facing front end of the lexical-KB line: a single-file web tool where
you drop files and download an installable `<name>.skill` — pack → download →
install in any skill-capable agent → query. Fully client-side (the embedding-free
design is what makes a pure-browser packer possible). `build_packer.py` is a
generator that inlines the `creating-kb` build core (`lexkb-web.mjs`, exports
stripped) plus the vendored shipped runtime (`search.js`/`search.py`/
`bundle_SKILL.md`, snapshot from claude-skills main) into one self-contained HTML
matching the ai-tools `<github-toc>`-listed convention — re-run it to re-sync the
snapshot. A browser-built `.skill` is byte-identical to a `creating-kb` Node build;
verified end-to-end in headless Chromium (stage files → build → download → query
with the shipped search.py). Lives on austegard.com (oaustegard.github.io PR #253),
not in claude-skills — the skills repo is for skills, not hosted web apps (the
earlier in-repo SPA, #714, was closed). `vendor/` holds the pinned input snapshot.

### `lexical-kb-phase0/` — does agent-expansion + BM25 match embeddings?

The study that could have killed the embedding-free KB. On the real
muninn.austegard.com corpus (73 posts, the corpus behind the published embedding
`muninn.kb`), three stages: (1) chunk-size sweep — confirms lexical tolerates big
chunks (whole-doc 73 chunks = 500-char 1237 chunks on recall, R@5 even up); (2)
head-to-head vs the full-float Jina ceiling — lexical ties/edges on in-vocabulary
queries (1.00/1.00 vs 0.90/1.00 R@5/R@10); (3) paraphrase frontier — agent
expansion ties or beats embedding on 4 of 5 lay-phrased queries, recovering gaps
raw BM25 loses (P1: 0.50→1.00), with one honest residual (P5) where a
vocabulary-divergent relevant post was found only by the embedding. Verdict:
competitive with embeddings, deleting the model/asset/embed-passes, at the cost
of a bounded recall residual on conceptually-related-but-lexically-divergent docs
— the precise answer to "what did embeddings buy?". Methodology note: gold judged
from post descriptions (not body terms) to avoid pre-favouring lexical; embedding
is the float ceiling (deployed 1-bit codes would score lower), making it a
conservative test. Closes Phase 0 of the lexical-kb / creating-kb line.

### `lexical-kb/` — embedding-free portable KB as a `.skill`

The "what if we drop embeddings entirely and let the agent expand the query?"
thread. The embedding model's only job in retrieval is bridging the
query↔document vocabulary gap; an agent in the loop does that with current
context. So: precompute a BM25 inverted index, ship it as a self-contained
`.skill` zip (`SKILL.md` + `search.py` + `index.json` + `chunks.jsonl`, pure
stdlib), and put the semantic step in the SKILL.md protocol — the agent emits
`{core, expand}` weighted terms at query time. `search.py` also carries an RM3
pseudo-relevance fallback for non-expanding consumers and `--filter` over
structured metadata. Deleting the embedding half is what makes remax_kb#12
(pure-JS browser packer) trivial: no quantizer, no projection, no fp-fragile
Python↔JS bit-identity — chunker + BM25 + zip writer, all exact-integer.

Tiny-corpus test (Federalist + Gettysburg) validated the full lifecycle and
caught a real bug: substitutive expansion (curated synonyms *replacing* the
user's words) sent a gold doc to score 0. Fix: expansion is now strictly
additive — the raw query contributes at a floor weight beneath core/expand, so
a synonym can lift but never drop a literal match. The corpus is too topically
disjoint to show ranking lift (raw BM25 wins at rank 1 on any shared term); the
quantitative study — chunk-size sweep + recall vs the embedding `muninn.kb` — is
Phase 0 on the real muninn corpus.

### `kb-k-sweep/` — remax stack-count `k` vs recall on the Mac-search corpus

Settles whether the shipped `k=8` over-provisions bits. CROSSOVER answered this
on SPECTER2 (flat curve, k=1 ≈ 88% of k=8); this reruns on the production
embedder — Gemini `gemini-embedding-001` via the CF AI Gateway — over the real
1779-chunk muninn index. Part 1 (`sweep.py`): curve steeper at low `k` on Gemini
(k=1 = 0.585 R@10, 82% of k=8's 0.713); truncation 768→256 is the dominant loss.
Part 2 (`dim_sweep.py`) sweeps the (dim,k) grid against fixed float-768 GT and
settles it: **at a fixed byte budget, dimensions beat stacks at every budget.**
Shipped dim=256/k=8 (256 B, 0.713) is Pareto-dominated — dim=512/k=4 = 0.772 at
the same bytes, dim=768/k=2 = 0.781 at 25% smaller, dim=512/k=1 = 0.697 at ¼ the
size. Recommendation: re-pack at 512/k4 or 768/k2 (free re-binarization, no
re-embed). Embed-once/re-binarize: 18s corpus, ~9s for the whole 30-cell grid.
Part 3 (`build_both.py`/`verify.py`) builds real .kbi artifacts and verifies on
40 real RETRIEVAL_QUERY queries: dense R@10 256/k8=0.585 → 512/k4=0.698 →
768/k2=0.723; no Worker code change needed (JS reader is data-driven). Part 4
(`int8_rotations.py`): the rotation sidecar (`k·dim²·4` B, corpus-independent,
dominates a small .kbi) quantizes f32→int8 for a 4× shrink at 0.24% bit-flips /
zero recall loss — making 768/k2+int8 both higher-recall AND smaller (~2.6 MB)
than the shipped 256/k8 (3.6 MB). Part 5: int8 shipped as a backward-compatible
remax_kb SPEC_v2 change ([remax_kb#10](https://github.com/oaustegard/remax_kb/pull/10),
v0.2.0) + worker reader/build flag ([muninn#213](https://github.com/oaustegard/muninn.austegard.com/pull/213));
the real JS worker reader validated bit-identical to the Python encoder on an
int8 .kbi under node 22. Live deploy gated on merging both PRs.

### `dc-mall-timelapse/` — EarthCam WAMO cam, "since May 1"

Requested a timelapse of the Washington Monument (WAMO) cam back to May 1 via
EarthCam's `gethofitems.php` "index". Ground-truth probing killed the premise:
that endpoint only ever serves the **newest ≤50** Hall-of-Fame stills —
`start`/`date_start`/`date_end`/`last_item_id` are all ignored (tested every
combination), `length` caps at 50, `fullcount:75000` is decorative. The two
archive VODs (`events/dc/WAMO.mp4`, `archives/4356/backup.mp4`) are rolling
~60-minute clips. Deep date-selectable archive is premium, no open API. So
May 1 (~7 weeks back) is unreachable for free. Built `wamo_timelapse.mp4` from
the only public history — 50 frames, **Jun 8→12 2026**, 8 fps — and delivered
it honestly labeled. Reusable tooling find: `pip install imageio-ffmpeg`
provides a static **ffmpeg 7.0.2** binary, the clean way to get ffmpeg in a
CCotw container (apt's archive 404s). Full table in `RESULTS.md`.

### `python-lsp-stress/` — pyright LSP against a large codebase

Validated PR #124's pyright 1.1.410 (base layer) end-to-end against Django
(2,922 `.py` files). `lsp_probe.py` is a stdlib-only LSP client that drives
`pyright-langserver` over stdio: `initialize` advertised all 16 navigation
providers in 0.19s; `references` on the `Model` class returned 2,490 hits
across the tree in 3.9s; `workspace/symbol` returned 3,817 in single-digit
seconds. The batch `pyright` checker analyzed the 907-file `django` import
closure in 22s with structured JSON output. Cross-file navigation scales;
the Python LSP is production-ready for large-codebase work. Full numbers in
[`RESULTS.md`](python-lsp-stress/RESULTS.md).

### `qat-cpu-demo/` — QAT vs PTQ, GPU-less, in 2 minutes

Asked whether we could run a smaller version of Google's Gemma 4 QAT locally on
the GPU-less container. Built a faithful toy-scale reproduction of the post's
central claim: a 2-layer char-level transformer (393 K quantizable weights)
trained end-to-end on 4 CPU cores in 2m13s, no GPU/network/external-data. QAT =
fake-quant (per-channel symmetric round-and-rescale) in the forward pass +
straight-through estimator in the backward; PTQ = the same rounding applied to a
finished fp32 model. **Result:** at the 2-bit floor (where the Gemma stack uses
2-bit for token generation), PTQ blows perplexity 1.30→5.21 (+3.91, 4× worse)
while QAT recovers to 1.31 at the identical 0.098 MB footprint. int4/int3 PTQ
don't degrade at this scale, so their "recovery %" is noise — honest caveat in
RESULTS.md. Verdict: the technique runs locally trivially. **Follow-up
(`inference/`):** also pulled and ran Google's *actual* checkpoint —
`gemma-4-E2B-it-qat-q4_0-gguf` (3.35 GB, 4.63 B params / 2 B effective), built
llama.cpp CPU-only, **20.5 tok/s generation / 182 t/s prompt on 4 cores**, no
GPU, ~3.5 GB RAM. Bottleneck was tok/s as predicted, not RAM or gating (repo
ungated). The `.gguf` + llama.cpp build are gitignored (heavy, regenerable).
**Part 3 (`inference/` + `layers/Containerfile.local-llm`):** turned the
"when is local inference actually worth it" answer into a reusable opt-in
container layer — `llama-cpp-python` + baked EmbeddingGemma 300M / Gemma 3 270M
(~650 MB cached), with helpers `scripts/llm_local.py` (`embed`/`score`/`generate`)
and `scripts/fetch-model.sh`. Validated all four capabilities on the pip wheel
incl. Gemma 4 load (no from-source CLI needed). Key finding: the high-level
`echo`+`logprobs` scoring path is pathologically slow on Gemma's ~262k vocab
(>2 min/call); reading raw logits + numpy softmax over the target token cuts it
to ~150 ms.

### `memory-redundancy-probe/` — is the memory store a landfill?

Spun out of reviewing arxiv 2606.03787 (surprise-gated robot episodic memory):
before considering a measured storage gate for Muninn's own episodic memory,
checked whether a storage-quality problem actually exists. It doesn't, the way a
gate would fix. Across 1,747 active memories, lexical near-duplicates are ~1%
(exact) to 7.7% (loose TF-IDF) — but that floor undercounts the running-topic
digest family, where TF-IDF is known-bad (memory `517a2f07`: 0.05–0.13 for
obviously-dup content). The real issues: write-side leaks (6 `"Valid"` stub
memories, exact write-twice dups, 14 `"Skipped zeitgeist"` no-op logs some at
priority 1), session-log accumulation (185 `SLEEP`/`FLY SESSION` memories,
10.6%), and a curation pass that doesn't do what it claims — `consolidate` is
tag-bucket concatenation and `curate`'s advertised textual-overlap dedup is
unimplemented. Also surfaced and corrected a live confabulation: the store has
**no embeddings** (removed v0.13.0/v2.0.0), but Muninn asserted `consolidate`
did embedding clustering before reading the code. Correction stored as
scar-tissue (`4bfb05fa`). Verdict: fix write hygiene + the curate dedup, prune
session logs; do **not** add a surprise gate.

### `optimizing-skills-retro/` — running the gate on its first real case

Retroactive test: ran the `optimizing-skills` validation gate (v0.1.0, #677)
against the `down-skilling` v1.2.0 edit (#674), using the haiku-assessment
voice-rewrite confabulation as the triggering-failure check. Held `best` =
v1.1.0 (four edits reverse-applied) vs `candidate` = v1.2.0; two Sonnet authors
each compiled a Haiku prompt from one SKILL version; each prompt run on Haiku ×5.

**Verdict: the gate reproduces SHIP.** Candidate strictly beats best on the
triggering failure (architectural hallucination 0% vs 60%; original n=20 had the
un-anchored author at 95%). But two of the four edits did the work
(source-anchoring, BAD/GOOD, model-the-silence); the **length-calibration edit
did not transmit** — both arms 0/5 in the 60–90 range, candidate even shorter.
The retro also exposed two gaps in `optimizing-skills` itself: (1) collapsed
multi-criterion pass/fail masks a real win behind an unrelated tie — score on the
triggering-failure criterion; (2) n=1 author per arm lets author variance
dominate — sample ≥2 authors when the artifact is Agent-compiled. Proposed fixes
in `proposed-patch-optimizing-skills.md` (must clear the gate before shipping).

### `skillopt-skill/` — `optimizing-skills`, a skill distilled from SkillOpt

Output of reviewing SkillOpt (microsoft/SkillOpt, arXiv:2605.23904) and reading
its code. A new skill, `optimizing-skills/`, that encodes the paper's
discipline — minus the training harness — for revising existing skills:
held-out check set, two-tier `best`/`candidate` validation gate (ship only on a
strict improvement), bounded edits (~4), failure-first reflection, impact
ranking, a protected core, and `remember()`-backed cross-revision memory.
Staged here because this session is scoped to `claude-workspace`; to go live it
must be copied into `oaustegard/claude-skills` so it auto-mounts. See
`README.md` for the port step and `optimizing-skills/references/` for the
SkillOpt mapping + Agent-tool scoring recipe.

### `haiku-assessment/` — Haiku 4.5 vanilla vs. down-skilled, three tasks

Probe testing whether down-skilling actually closes the reliability gap
Oskar perceives with Haiku 4.5. Three tasks (triage, code review, voice
rewrite) × 2 conditions (vanilla, down-skilled) dispatched to Haiku via
parallel sub-agents. Started as n=1 sketch, scaled to **n=20 per cell
(120 dispatches)**, then **broadened to 6 task archetypes at n=5**
covering JSON extraction, changelog writing, filter/sort, NL→gh CLI,
copyediting, and a calibrated-examples rerun of the voice task.

Down-skilling delivered uniform improvement on triage (94%→100%
accuracy, 0%→100% schema discipline), code review (100%→100% bug
detection, 100%→0% unsolicited fix-section drift), JSON extraction
(5 different schemas → 1 schema), filter/sort tie-breaking (40%→100%),
and NL→gh CLI (5/5 invented non-existent flags → 5/5 correct
commands). On voice rewrite with the original (un-calibrated) examples,
it caused architectural hallucination in 19/20 runs; with calibrated
examples (source-anchored, length-matched, tagged BAD example), the
hallucination dropped to 0/5.

Writeups: `RESULTS.md` (n=1) + `n20/RESULTS-n20.md` (n=20) +
`n20/broader/RESULTS-broader.md` (6 archetypes). Guide: `GUIDE.md`.
Proposed patch to the down-skilling skill:
`n20/broader/proposed-skill-patch.md`.

### `spoke-branch-cleanup-2026-05-25/` — abandoned-branch sweep across 28 spokes

One-shot cleanup: 186 non-default branches inventoried, 183 deleted, 3
kept (one open PR, two Oskar-authored commits). Reusable audit script
in `audit-branches.sh`; raw data in `branch-audit.tsv`, deletion log in
`delete-log.tsv`. Follow-up noted: `muninn.austegard.com`'s
`update-tools-index` workflow creates daily branches but the
`gh pr create` step isn't firing.

### `phase-a-bridges/` — math × cs-theory bridge MVP (Phase A)

End-to-end smoke test of the bridge-discovery pipeline on a ~1000-paper
math + cs-theory corpus. Five stages: corpus → SPECTER2 embed →
cross-axis scan → rerank → bridge-attempt with Claude. Two production
runs (`data/`, `run2/`) plus a tier-1 organic-rerank pass (`tier1/`).
RESULTS at the top level.

### `te-bridges/` — theory → empirical bridge MVP

Asymmetric successor to phase-a. Theory papers anchor, empirical papers
get cross-axis scanned and reranked. Four sub-runs evolved the
methodology:

- `RESULTS.md` — original uniform-sampling MVP (path A)
- `anchor_run/RESULTS.md` — path C, anchor-mode (paused mid-run)
- `anchor_run_filtered/RESULTS.md` — anchor + citation/author/abstract-mention filters
- `path_c_cross_domain/RESULTS.md` — cross-domain extension with twin-diagnostic

Verdict at the top-level RESULTS; later runs are follow-up evidence.

### `specter2-gap-issue-87/` — SPECTER2 embeddings for unindexed papers

Gap-fill for three papers not yet in Semantic Scholar's precomputed
SPECTER2 index (Sawin 2605.20579, OpenAI companion 2605.20695, Lenstra
1986). `embed.py` is the current entry point; output is
`sawin_lenstra_specter.json`. The `*.legacy.json` and `specter2_run.py`
are the earlier root-level versions, preserved for traceability.

### `muninn-kb-issue-76/` — full-corpus `muninn.kb` build

Reproducible build script for the `muninn.kb` artifact published in
[muninn.austegard.com#136](https://github.com/oaustegard/muninn.austegard.com/pull/136).
See `README.md` for inputs, steps, and determinism guarantees.

### `reviews/` — repo reviews

Freeform comparative / single-repo reviews. Each `.md` is the
deliverable; no separate `RESULTS.md`.

- `lac-vs-transformer-vm.md` — LAC vs. transformer-vm comparative review
- `openmythos.md` — review of `kyegomez/OpenMythos`

### `snooker-break/` — apex vs Murphy break-strike simulator

Interactive HTML simulator. Open `snooker-break.html` in a browser.

## Adding a new experiment

0. **Grep [`METHODS.md`](METHODS.md) first.** It is the ledger of portable
   methods, environment gotchas and negative results from every experiment
   here. Two experiments in this repo have already re-derived a result a
   sibling had documented; that file exists so it stops happening.
1. `mkdir <short-name>/`
2. Put scripts, data, and a `RESULTS.md` (or rename if the README is the result) inside.
3. Add a row to the table above and a section under "Per-experiment notes."
4. Add anything regenerable (logs, checkpoints, large caches) to `.gitignore`.
5. When it ends, add what you learned to `METHODS.md` — anything that would
   change what a *different* experiment does.

Shared helpers live in [`_lib/`](_lib/) — path resolution, retry/backoff,
atomic JSON checkpointing, Unicode folding. Reach for them rather than
re-implementing; add to them only once a second experiment needs the code.

If an experiment grows scripts worth reusing more broadly, lift them into
the relevant spoke repo — but the data products and a results writeup stay here.
