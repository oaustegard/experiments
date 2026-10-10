# mojo-port-haiku results — 2026-10-10

## Answer

Can Haiku 5.5 port established, slow, pure-Python libraries to Mojo 1.1
extension modules behind their original Python API, and do the ports pay
off? Yes, all three on the first attempt: one agent per library, 19–30
minutes each, $2.97 input-side in total.

| port | held-out correctness | speed vs pure Python, from Python |
|---|---|---|
| difflib.SequenceMatcher | 600/600 fuzz cases, 57/57 CPython `test_difflib` | 27.9× char diffs, 3.2× `unified_diff`, 1.2× `get_close_matches` |
| snowballstemmer (en, de, ru, fr) | 100,608/100,608 words | 41–63× batch, 0.81–1.32× PyStemmer's C |
| yake 0.7.3 | 540/540 doc×config cases at 1e-9 relative | 3.6× (97 → 27 ms per doc) |

The speedup follows how much work each Python call carries. Batch stemming
and long diffs gain 28–63×. Many tiny calls (`get_close_matches`) gain 1.2×.
yake is capped near 4.3× by segtok tokenization, which stays in Python.

The snowball agent wrote a Python→Mojo translator for Snowball's generated
modules instead of porting by hand. Pointed unchanged at all 36 languages,
it translated 27 and 23 of those compiled. All 23 match the Python reference
exactly on Snowball's vocabularies, 19 of them languages the agent never
saw. The other 13 fail loudly, at translation or compile time; none gives
wrong stems.

What would change this: libraries built on class hierarchies and callbacks
(Mojo 1.1 has no dynamic OO), or a truly hidden held-out split.

## Findings

1. **Three of three ports pass held-out grading on the first attempt.**
   Each was rebuilt from its sources in a clean copy before grading. difflib
   600/600 fuzz cases plus 57/57 of CPython 3.13.16's `test_difflib` run
   with the class swapped in, snowball 100,608/100,608, yake 540/540 (r1,
   `runs/r1/grade-*.json`).
2. **Snowball reaches C speed.** Batch stemming takes 0.58–0.88 µs/word
   against 23.9–49.1 µs in pure Python and 0.54–0.89 µs in PyStemmer's C
   (cache off): 0.81× (english) to 1.32× (russian) of C. Per-word
   `stemWord` calls gain only 4.6–11.6×, because each pays the crossing
   (r1 bench).
3. **A Haiku-written translator generalizes to unseen languages, with loud
   failures.** 36 languages: 27 translate, 23 compile, 23/23 exact on up to
   33,000 vocabulary words plus 3,000 generated words each, at 11–57×.
   Rejections are the translator's own assertions (9); compile failures are
   runtime methods it never needed (`insert`, `slice_to`) and Bool/Int typing
   (4). The only change was the language list and a class-name rule for
   `dutch_porter` (r1-unseen, `translate_report.json`, `unseen_check.json`).
4. **The boundary sets the ceiling.** Measured before the run: ~3.3 µs per
   Mojo call, ~1.5 µs per string crossing in and out (13× Python's own
   `str.upper`), against 0.19 µs per word for a UTF-32 buffer round trip.
   All three agents used buffers. difflib still gains only 1.2× on
   `get_close_matches`, thousands of short matchers, one call each (setup).
5. **Agents found numeric traps the task did not name.** Mojo 1.1's
   `std.math.log` on Float64 is off from libm by up to 1.1e-10 relative
   (`log(10)`, rechecked here), which breaks a 1e-9 score match. The yake
   agent called libm through `external_call` instead. It also found that
   jellyfish's Rust `jaro_similarity` compares grapheme clusters on
   non-ASCII, unlike its Python fallback, and approximated UAX #29 to match
   (r1 yake report).
6. **My checker dropped tests silently.** Replacing `difflib.SequenceMatcher`
   before `DocTestSuite(difflib)` runs makes doctest skip the class (its
   `__module__` is no longer `difflib`), so the upstream run fell from 57 to
   48 tests with nothing failing. Fixed by collecting the doctests first with
   `extraglobs`. The difflib port passed all 57 (r1 grading).
7. **Mojo 1.1 broke 1.0's spellings**, so the agents were given Modular's
   current skills: `UnsafePointer` → `Pointer`, `MutExternalOrigin` gone,
   `p[i]` → `p[unsafe_offset=i]`, `@export` init needs `abi("C")` (setup).

## Method

**Targets.** CPython 3.13 `difflib.SequenceMatcher`; snowballstemmer 3.1.1
pure-Python stemmers (english, german, russian, french), never
`snowballstemmer.stemmer()`, which delegates to PyStemmer's C when that is
installed; yake 0.7.3 `KeywordExtractor` with `lemmatize=False`.

**Shape of a port.** A package (`mdifflib`, `msnowball`, `myake`) with a
thin Python shim and `kernel.mojo` built by `mojo build --emit shared-lib`
to `_kernel.so`. The package may not import the library it ports at
runtime. For difflib, the API surface and cheap methods
(`get_opcodes`) may stay in Python. For yake, segtok tokenization stays in
Python.

**Agent kit** (`kit/`, `harness/new_run.py`). The task, Mojo 1.1 notes with
the measured boundary costs, pointers to Modular's skills (github.com/modular/skills
at b9b3a8e), and a starter package that builds. One Haiku 5.5
`general-purpose` subagent per target, with prompt `harness/PROMPT.md`
tagged `[no-context]` so the parent's transcript is not appended. Tools were
limited to Bash, Read, Edit, Write, Grep and Glob through
`guard_subagent_scope`. The planned ladder (an informed Haiku retry on
failure) was never needed.

**Oracles** (`oracle/`), each self-tested against the reference (pass) and
a planted bug (fail):
- snowball: dev = even-indexed half of each Snowball `voc.txt`; test = odd
  half plus 6,500 generated words per language (grafted suffixes, random
  strings, case-mixed, edge tokens), compared with the pure-Python stemmer.
  The pure-Python stemmers match `output.txt` 100% on all four.
- difflib: differential fuzzer, 600 cases per split with different seeds.
  Strings, line lists, int tuples and mixed hashables; isjunk variants;
  autojunk on and off; lengths across the 200 threshold; `set_seq1`/`set_seq2`
  reuse. Plus CPython's `test_difflib` with the class replaced, run in
  claude-workspace's jail.
- yake: 30 dev and 85 test markdown docs (this repo's own READMEs and
  RESULTS, frozen) plus 5 edge texts, under 6 configs. Keyword lists must
  match in order, with scores within a relative 1e-9; a near-tie swap is
  forgiven.

**Grading** (`harness/grade.py`) copies the workdir without build outputs,
runs the agent's `build.sh`, then the test-split oracle there.
**Benchmark** (`bench/bench.py`): wall clock from Python, minimum of 5
reps after a warm-up, PyStemmer with its cache off. **Cost**: input-side
dollars from the subagent transcripts with `commit0-swarm/harness/cost.py`.

**Held-out caveat.** The test split was hidden by instruction only. The
snowball and yake agents both ran `--split test` themselves; both passed it
on their first run of it.

## Log

### r1 — 2026-10-10, one Haiku 5.5 agent per target

Asked (Oskar): "go for the difflib though I am more interested in the
snowball stemmer and yake". Three agents launched 18:09Z in parallel on a
4-vCPU container.

| target | wall | tool calls | $ input-side | Mojo lines | shim lines | grade |
|---|---|---|---|---|---|---|
| difflib | 19.3 min | 63 | 0.64 | 389 | 383 | 600/600 + 57/57 |
| snowball | 20.6 min | 101 | 1.17 | 4,397 (generated by an 858-line translator) | 101 | 100,608/100,608 |
| yake | 30.0 min | 87 | 1.16 | 1,120 | 225 | 540/540 |

Benchmarks at grading (`--reps 5`):

| workload | Python | Mojo port | speedup |
|---|---|---|---|
| difflib, 40 char ratios of ~3,000 chars | 0.454 s | 0.016 s | 27.9× |
| difflib, `unified_diff` over 85 line-list pairs | 0.045 s | 0.014 s | 3.2× |
| difflib, 150 `get_close_matches` over 6,000 words | 2.211 s | 1.779 s | 1.24× |
| snowball english, per word | 27.4 µs | 0.67 µs (C 0.54) | 40.9× |
| snowball german | 49.1 µs | 0.88 µs (C 0.89) | 55.6× |
| snowball russian | 23.9 µs | 0.58 µs (C 0.77) | 41.0× |
| snowball french | 49.1 µs | 0.78 µs (C 0.79) | 62.9× |
| yake, per 12 KB doc (40 docs) | 97.5 ms | 26.8 ms | 3.6× |

Design choices the agents made on their own:
- **difflib:** b2j in CSR form; code points interned in a Mojo hash table;
  one int64 context block of buffer addresses; `quick_ratio` kept in Python
  on the C `_count_elements` helper.
- **snowball:** lowered the generated code's `raise labN()` / `except labN`
  gotos to a jump flag; one heap-allocated stemmer per language;
  output-buffer overflow retried from the shim.
- **yake:** numpy-exact pairwise summation for mean and std; a stable merge
  sort; Unicode predicates via a per-character table built in Python, since
  Mojo 1.1's stdlib has none beyond ASCII.

Known gaps the agents reported:
- **difflib:** `find_longest_match` raises on negative indices, where the
  reference silently wraps; junk extension compares ids rather than `==`,
  which differs only for non-reflexive elements.
- **yake:** lone surrogates raise in the UTF-32 encode step, where yake
  returns `[]`; the grapheme approximation lacks Hangul and Prepend rules.

Checker bug found while grading: `run_upstream.py` ran 48 of 57 tests once
the class was swapped (finding 6). Fixed before the final grade.

### r1-unseen — the snowball translator on all 36 languages

`gen/translate.py` retargeted by changing `LANGS` and the class-name rule.
`gen/translate_all.py` and `gen/compile_each.py` are harness wrappers.
Rejected at translation: arabic, esperanto, finnish, greek, hindi,
lithuanian, polish, tamil, turkish. Failed to compile: czech, danish, dutch,
porter. Checked against Snowball's vocabularies (snowball-data, not vendored,
38 MB) with `harness/snowball_unseen.py`. All 23 that compiled were exact
(armenian 33,000/33,000 … nepali 7,000/7,000); speedups 10.6× (sesotho) to
56.8× (dutch_porter).
