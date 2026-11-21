# Projects Search: local semantic search on the Neural Engine

Searches the write-ups (`.md`, `.txt`, `.rst`) under `~/Projects` by meaning plus keywords. Everything runs on this Mac.

- **Embedder:** MongoDB/mdbr-leaf-ir (6-layer BERT, 23M parameters, 768-d), rewritten in the Neural Engine layout (`ane_bert.py`) and converted to Core ML fp16. Documents use the 256-token model on the Neural Engine; queries use a 32-token model on the CPU.
- **Index:** files are split at headings into passages of at most 215 tokens, each embedded with its `path — heading` line. Semantic and keyword rankings are fused by reciprocal rank.
- **Storage:** a [remax_kb](https://github.com/oaustegard/remax_kb) v2 knowledge base, written by `kb.py` through `KBWriter.sync` (content-addressed, so only new or changed passages are embedded). `index/projects.kbi` holds the codes (remex 4-bit over the first 256 dimensions, 128 bytes per passage), the chunk map and the bm25s keyword table; `index/projects.kbc/` holds the passage text. The `.kbi` also carries `remex/rotation.f32` and `remex/centroids.f32`, two extra entries that let a reader without numpy score the codes; the Python `remax_kb` reader ignores them and opens the file as usual. Needs `remex` v0.6.0: remax_kb 0.5.0 fails against remex 2.x (rotation default changed from haar to rht).

```bash
PY=~/Projects/ane-search/.venv/bin/python
$PY index.py            # re-index; only changed passages are embedded (--full to redo all)
$PY server.py "query"   # search from the terminal
PORT=8791 $PY server.py # the page, at http://127.0.0.1:8791
$PY convert.py 256 32   # rebuild the Core ML models; needs torch 2.5.1, transformers 4.51.3 and safetensors added to the venv
```

Measured 2026-11-16 on the M1 Air: 1,601 files -> 19,469 passages; first index 90 s at 1.98 ms per passage; re-index with nothing changed 17.5 s; index 30 MB of vectors; a query embeds in 3-4 ms and ranks in 7-30 ms.

Skipped: `data/ out/ results/ node_modules/ .git/` and similar, the vendored Core2 SDK, `claude-skills/plugins` (generated copies), and `.txt` files that are data dumps rather than prose. Edit `SKIP_DIRS` in `common.py`.

Measured 2026-11-17 after moving to remax_kb: 2,052 files -> 23,277 passages; index 36 MB on disk (17 MB `.kbi` + 19 MB text) against 73 MB before; re-index with nothing changed 12 s. Against the previous float16 ranking over all 768 dimensions, the 4-bit x 256 codes keep 0.77 of the top 10 in the top 10 and 0.99 in the top 50 (100 queries; agreement with the old ranking, not relevance). Keyword matching now uses bm25s' tokenizer: `response_model` is one token and single characters are dropped.

Relevance, measured 2026-11-17 (`bench/`): 236 passages sampled at random, one search query written for each by Haiku from the passage text (so the queries lean keyword-rich), then the rank of that passage. Hybrid as shipped: first result 0.61, top 5 0.85, top 10 0.92. Same test with full float 768-d vectors: 0.63 / 0.85 / 0.93. Semantic only: 0.57 / 0.81 / 0.90 with the 4-bit x 256 codes, 0.60 / 0.84 / 0.92 with float 768. Keyword only: 0.58 / 0.81 / 0.89. Differences under about 0.03 are within noise at this sample size. `bench/prep.py --embed` rebuilds the float reference (5 minutes); passage numbers in `bench/queries_0.json` are row numbers and stop being valid after a compaction.

Not done: no file watcher (re-run `index.py`), source code is not indexed.

## Menu-bar app

`app/` is a native menu-bar app (Swift, no Xcode project) with a global shortcut, Control-Option-Space. It embeds the query with the bundled 32-token Core ML model and ranks against the knowledge base in `index/` in-process, with its own reader for the `.kbi`/`.kbc` format (stored zip, chunk map, 4-bit codes, bm25s tables): no server and no Python at search time. `index.py` (cron, hourly) still builds the index; the app reloads it when it changes.

```bash
app/build.sh                                   # builds app/build/Projects Search.app
cp -R "app/build/Projects Search.app" ~/Applications/
"…/MacOS/ProjectsSearch" --search "query"      # headless: JSON results
"…/MacOS/ProjectsSearch" --batch queries.txt   # headless: one JSON line per query
"…/MacOS/ProjectsSearch" --selftest "query" /tmp/out   # drives the real panel, writes /tmp/out.png and .json
defaults write com.austegard.projects-search hotKeyCode -int 49        # change the key (virtual key code)
defaults write com.austegard.projects-search hotKeyModifiers -int 6144 # Carbon modifiers: cmd 256, shift 512, option 2048, control 4096
```

Checked 2026-11-17: the Swift WordPiece tokenizer matches the `tokenizers` library on 10 of 10 test queries, and the top-20 results from the Swift reader match the Python reader (`remax_kb` decoding the same file) on 104 of 104 queries, including with a tombstoned row present. Query embed about 1.5 ms warm, rank about 6 ms over 23,277 passages. The shortcut is registered without error but could not be triggered by a synthetic key press, so it was not exercised automatically.
