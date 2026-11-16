# remex-proptest — property-testing remex with four parallel agents

[remex](https://github.com/oaustegard/remex) compresses embedding vectors: it
rotates each vector, stores a few bits per coordinate, and searches the
compressed codes. It had about 600 tests. This experiment applies the method in Inanna
Malick's [Property Testing with Agent Swarms](https://recursion.wtf/posts/agents-and-property-tests/)
to it. Agents write *property tests* (Hypothesis generates random inputs and
operation sequences, and each result is checked against a simple, independent
reference). The agents then reduce every disagreement to a short deterministic
test, and plant deliberate bugs (*mutants*) to check that their tests can catch
them. The method's prompt is `proptest-praxis-SKILL.md` here.

Four Sonnet subagents each took one area of remex 2.2.0 (commit b8d527a) and
reported findings. The orchestrating session reproduced each finding and wrote
the fixes. Run 2026-11-16, following the smaller `bm25-proptest/` pilot.

Terms used below: `d` is the vector dimension and `bits` the bits stored per
coordinate. A *container* (`CompressedVectors`, or the bit-packed
`PackedVectors`) holds the codes, and `.npz` is its file format. A *warm cache*
is decoded data a container keeps after its first search. *Scalar mode*
(`normalize=False`) and *centred mode* (`mean=`) are two encoding options;
`renorm=False` turns off a length correction. `rht` is the default rotation and
IVF an optional index over the codes.

## Answer

Does the method find real bugs in a library that already has a careful test
suite? Yes. The four harnesses produced deterministic regression tests that fail
on remex 2.2.0 for 68 cases <!-- @evidence[[results/summary.json#totals.regression_failures_on_old]] -->.
Those cases cover the 21 distinct defects listed under Findings. The worst was
silent data corruption. The two container classes wrote `.npz` files in
different bit layouts, so whenever `d * bits` was not a multiple of 8, a file
saved by one and loaded by the other returned wrong codes with no error. All of
remex's existing tests passed on 2.2.0, so none of them made that cross-class
read at such a width. Others included a segfault after pickling the default
quantizer, and a search guard skipped on a warm cache.

The fixes are in remex 2.2.1 (0319e2d). With them, 10 regression cases
still fail <!-- @evidence[[results/summary.json#totals.regression_failures_on_new]] -->.
Four of the 10 are the two design questions in finding 7, both of which would
change stored codes. Two are float corner cases, and four test behaviour no
document specifies (finding 7). Further ambiguous cases the agents marked
`xfail` are not counted as failures. Planted bugs were caught at 72 of 73.
<!-- @evidence[[results/summary.json#totals.mutants_caught]] -->

What would change this: the harnesses share some inputs with remex (codebook
tables, the dense rotation matrix), so defects there are invisible to them.

## Findings

1. **`.npz` cross-class reads returned wrong codes silently** whenever
   `d * bits % 8 != 0`, because `CompressedVectors.save` packs one bit stream and
   `PackedVectors.save` pads each row to a byte. One repro is d=7, 3 bits, n=5:
   26 of 35 codes wrong. Files now record a `layout` key. For old files where
   the two sizes coincide, the reader still cannot tell them apart.
   (containers F1)
2. **`Quantizer.search` skipped the bits and mean checks on a warm cache.** A
   reader with the wrong bit width or no mean got plausible scores instead of an
   error, with `renorm=False` or in scalar mode. Three of the four agents found
   it independently. (search F1, containers F2, stateful F2)
3. **A pickled or deep-copied default `Quantizer` segfaulted.** `RHTOperator`
   cached raw buffer addresses, which pickling copied verbatim. (stateful F4)
4. **A rejected first `add([])` left a `RemexIndex` with no model unusable**: it
   set `dim=0`, and every later `add` failed. (stateful F1)
5. **Vectors shorter than 1e-8 were coded off their direction.** An absolute
   floor shrank them into the central cells. At d=64 and 4 bits, decode quality
   fell to 1-bit level by a norm of 7e-12. (encode F2)
6. **Fourteen smaller defects, all fixed:**
   - `RemexIndex.load` bypassed `open_npz`;
   - IVF built indexes from codes its quantizer refuses;
   - `save_arrow` extra metadata could relabel the encoding;
   - damaged `.npz` members leaked raw zip errors;
   - seeds of 2**63 or more could not be saved;
   - `PackedVectors.load` failed on legacy files;
   - `_score_vectors` failed on uncentred packed codes;
   - the IVF flat scan was not byte-identical for centred codes;
   - `candidate_indices` returned a writable view;
   - degenerate `scale` codebooks were accepted;
   - `assign_codes` mishandled dtypes;
   - a spurious warning fired on inf input;
   - `coordinate_sigma` accepted `d <= 0`;
   - `corpus_mean` of an empty corpus returned NaN.
7. **Two design questions, not changed.**
   - The 8-bit Lloyd-Max table stops at 300 iterations, with MSE 20% above its
     fixed point (4.94e-5 against 4.12e-5).
   - `rht` at d=2 decodes identically for every seed, and at d=8 27% of seeds
     leave a coordinate unmixed.

   Fixing either changes stored codes. (encode F6, F1)

   Also left open: two float corner cases. One is a float32 query offset that
   can tie two rows at ‖mu‖/‖x−mu‖ ≥ ~1e5 (search F4). The other is overflow
   near float64 max under `rht` (encode F7). Four cases test behaviour no
   document specifies:
   - two search cases marked ambiguous;
   - `seed=None` giving an `.R` that differs from the rotation encode applies
     (encode F9);
   - normalized `encode` accepting NaN silently (encode F11).
8. **Planted bugs were caught at 72 of 73.** The one survivor was in the
   stateful area: a GPU norms cache that ignores precision. It survived 153
   examples in the final run, and an earlier run of the same harness caught it
   after 55. Warm GPU searches at mixed precisions are drawn too rarely to catch
   it reliably.
   (`results/summary.json#mutants_caught_of`)

## Method

Each agent got the `proptest-praxis` prompt and one area of the library:

| area | what it generated | reference |
|---|---|---|
| `search/` | every search path (`search`, ADC, two-stage, batch, IVF at full probe, GPU numpy backend) over modes, bit widths, precisions, ties and edge shapes | brute-force float64 `q · decode(x)` |
| `containers/` | sequences of subset, pack, precision change, save and load in every format | numpy model, an independent bit packer, fresh-container differential |
| `stateful/` | `RemexIndex` add/search/save histories, cache-warming sequences, GPU and IVF builds | from-scratch rebuild and a float64 brute force reading raw arrays |
| `encode/` | chunked, permuted and repeated encodes, codebooks, rotations, scalar mode, dtype and non-finite inputs | `np.searchsorted`, an independent Lloyd iteration, a float64 rotation |

Each area holds the Hypothesis harness (`test_*_props.py`), deterministic
regressions (`test_*_regressions.py`), and its mutant runner and results.
Regression tests marked `xfail(strict)` are the agents' ambiguous cases. The
`search/` and `stateful/` areas hold the agents' `REPORT.md`. The tooling
refused to let the other two agents write theirs, so their findings are
summarized here.

`python3 record.py <remex clone>` reruns all four regression files against
2.2.0 and the fixed commit, tallies the mutant results each area recorded, and
writes `results/summary.json`. The property harnesses take 2–5 minutes each and
are run with `pytest test_*_props.py` in their area.

The fixes, and 35 regression cases distilled from these files, are in remex
`tests/test_property_findings.py`. All 35 fail on 2.2.0. On 2.2.1 the full
remex suite gives 651 passed and 1 skipped, as recorded in the remex commit
message; `record.py` does not rerun it.

Process notes from the session; none of these are recorded in the files here.
Two of the four agents hit the account's spend limit mid-run. Three hit an
auto-mode denial for a `sleep` or polling loop. All four were resumed and
finished. The `containers/` and `encode/` agents could not write their
`REPORT.md`, because the tool refused report files from subagents.

## Log

### Round 1 — 2026-11-16

Oskar: "Fix the found bugs, yes. Then run the test skill against remex."
Four Sonnet subagents ran in parallel, one per area, told not to modify remex
and to stop after one mutation-calibration pass. I reproduced each confirmed
finding before fixing it, made the fixes in a separate worktree, and checked
them against the agents' regressions and the full remex suite. Per area, the
regression results on 2.2.0 → 2.2.1:

| area | failed on 2.2.0 | failed on 2.2.1 | left open |
|---|---|---|---|
| search | 12 | 3 | float32 query offset can tie rows at ‖mu‖/‖x−mu‖ ≥ ~1e5; two ambiguous |
| containers | 22 | 0 | 14 ambiguous cases, xfail |
| stateful | 15 | 0 | 4 ambiguous cases, xfail |
| encode | 19 | 7 | 8-bit codebook, rht mixing (3 tests), float64-max overflow under rht, `seed=None` `.R`, NaN input |

(`results/summary.json#regressions`)

One remex test changed: `test_ivf_empty_cell_still_runs_the_guards` now builds
the index inside `pytest.raises`, because the guard fires at build time.

### Round 2 — 2026-11-16, count corrected

The Answer said 19 distinct defects, and Findings item 6 listed twelve smaller
ones. The 2.2.1 changelog fixes 19, including `coordinate_sigma` refusing
`d <= 0` and `corpus_mean` refusing an empty corpus, which item 6 had left out.
Adding the two design questions gives 21, the figure Oskar used when he
announced the release. remex 2.2.1 was released the same day (tag `v2.2.1` at
0319e2d, published to PyPI).
