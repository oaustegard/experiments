# Port difflib.SequenceMatcher's matching core to Mojo

Reference: CPython 3.13 `difflib.SequenceMatcher`
(`/usr/lib/python3.13/difflib.py`).

## Deliverable

A package `mdifflib/` in this directory exposing `SequenceMatcher`, a
drop-in replacement for `difflib.SequenceMatcher`:

- Same constructor (`isjunk=None, a='', b='', autojunk=True`) and the same
  public methods: `set_seqs`, `set_seq1`, `set_seq2`, `find_longest_match`
  (with its default arguments), `get_matching_blocks`, `get_opcodes`,
  `get_grouped_opcodes`, `ratio`, `quick_ratio`, `real_quick_ratio`.
- The public attributes `a`, `b`, `bjunk`, `bpopular` and `b2j` keep their
  meaning (`b2j` may be built on first access).
- Identical results to the reference for any sequences of hashable elements
  (strings, lists of lines, tuples of ints, mixed types), with or without an
  `isjunk` predicate, with `autojunk` on and off.
- The b-side index, `find_longest_match` and the `get_matching_blocks`
  search run in Mojo. Subclassing `difflib.SequenceMatcher` for the API
  surface is fine, as is keeping cheap, non-hot methods (like `get_opcodes`,
  which only walks the matching blocks) in Python. Map elements to integer
  ids in Python; Mojo should see integers only.
- `build.sh` builds everything from source.

Besides `check.sh` (a differential fuzzer), the grade runs CPython's own
`test_difflib` with `difflib.SequenceMatcher` replaced by yours, so
`unified_diff`, `ndiff`, `HtmlDiff`, `get_close_matches` and the module
doctests all go through your class.

## Performance

Measure with `python3 /home/user/experiments/mojo-port-haiku/bench/bench.py --target difflib --pkg .`. It times character
diffs of ~3,000-character texts, `unified_diff` over line lists, and
`get_close_matches` (many short sequences, which punishes per-call
overhead). Aim to beat Python on all three; report what you get.
