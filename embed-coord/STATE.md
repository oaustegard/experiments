# embed-coord state (handoff snapshot, 2026-10-21 ~09:30 EDT)

Coordinator hands off to the "Embedded Embedder" project coordinator. Board for cloud sessions = issue #104. The local board (file mailbox + `cpu.lock`) lives on Oskar's Mac under `embed-coord/board/`; its full log is in `board-log-20261021.txt` here.

## Rules that held (they worked, keep them)
1. **Two metrics, never mixed.** `SYM` = each arm on its own remex 2-bit codes (corpus re-encoded by the arm; EG2 = 0.918). `ASYM` = EG2's committed q4 2-bit doc codes, only the query encoder varies (EG2 = 0.919). Anchor-relative comparison only; total download = model + index (+3.07 MB brotli for onnxruntime-web).
2. **Report real_test overlap (1,000 MS MARCO tech/AI) beside synth_test hit@10**; in-domain-trained arms inflate synth hit@10 (style match). Per-query arrays in `results/<arm>_perquery.json`, synth_test file order, for paired bootstraps inside a family.
3. **One heavy job at a time on the Mac** (`cpu.lock`, covers MPS and WebGPU jobs; stated RAM need; swap was 4-5 GB of 6-7 GB). Handoff by writing RELEASED. Never quote latency measured under another job.
4. **Charter**: a finding relayed by coord or a peer is from the same goal: act on it, reply `ok <topic>`; disagreements to coord, decided within a turn; no silent finishes. (`CHARTER.md`)
5. **Wire size**: report brotli-11 MB beside raw; per-column int8 compresses worse than per-tensor.

## Results (details in each experiment's RESULTS.md)
**ASYM (keep EG2 docs):** EG2 q4 174 MB 0.919. Layer cut k=8 q4 44.1 MB 0.911; k=4 q4 29.0 MB 0.907 (3-bit table 25.4 MB 0.9065; 2-bit 22.7 MB 0.904); k=2 q4 21.8 MB 0.8965 (3-bit 18.2 MB 0.898; 2-bit 15.5 MB 0.890). Elbow at k=4. Sub-4-bit tables run on ORT-web wasm via standard-op unpack (`eg2-layer-student/run/export.py --table-bits`). Distilled-student plateau (leaf-mt 0.794). Release `eg2-layer-onnx-20261021` (zips on R2). Pruned-vocab EG2 loses OOD (69% of general-English queries tokenize intact at 32K).
**SYM (own codes):** arctic-m-v1.5 0.903; leaf-ir 0.893 (wasm int8 0.889). In-domain MNRL on 23,985 synth_train pairs: leaf-ir 0.937, Ettin-17m 0.918 (real_test overlap with EG2 only 0.26 / 0.23). Frontier, total incl. ORT: Ettin-17m int4 13.2 MB 0.918; 3-bit 12.2 MB 0.910; 2-bit 10.3 MB 0.883; leaf-ir int4 19.1 MB 0.942 (Ettin-vs-leaf-ir paired -0.024). Browser warm: Ettin 9.0 ms, leaf-ir ORT-web 24 ms, leaf-wasm 10.8 ms, leaf-gpu (WebGPU, no runtime) 5.0 ms; leaf-gpu corpus 61,686 posts in 102 s. See `ettin-native/RESULTS.md`, `ettin-native/results/pareto.png`, `leaf-wasm/`.
**Image demo (`bsky-image-search`, owner wasm):** 16,878 images, EG2 q4 vision budget 70, remex 2-bit index ~4.2 MB gz; alt-text->image hit@10 0.803 (2-bit) with full EG2 text. **The layer cuts do not transfer to text->image**: k4 0.563, k2 0.513 hit@10 (`results/eval_cuts.json`). Demo stays on the hosted EG2 embedder.

## Open
1. **Retrain the cut on alt-text queries** (eg2-layer-student offered; COORD #5 on #104 has the constraints): image-level 80/20 split (wasm owes the split ids + training query texts + their EG2 q4 text vectors on R2/embstore), keep prompt `task: search result | query: `, check caption-vocabulary coverage of the 32K cut, mix Bluesky targets to avoid text->text regression, report both. Needs Oskar to nudge that cloud session.
2. **Uncommitted work on Oskar's Mac** (needs his go): ettin-native/, leaf-wasm/, eg2-native-small/, bsky-image-search/ (large `out/`), embed-coord/board/.
3. Not yet run: hit@10 of leaf-wasm at int8 level L63/L31; Ettin general-English vocab k16000 was dropped (BPE pruning is bad, see ettin RESULTS).
4. Cloud sessions cannot be messaged from local sessions; only Oskar's nudge or issue comments reach them.

## Sessions (as of handoff)
Local (Claude Desktop, Mac): ettin = `local_0fb5b8cc…` (done, idle); wasm = `local_2efa69ca…` (owns leaf-wasm, bsky-image-search); mteb = `local_657342ff…` (done). Cloud: eg2-query-student `session_01SZizDG…` (done), eg2-layer-student `session_0168St6r…`, vocab `session_01QXYoSq…`.
