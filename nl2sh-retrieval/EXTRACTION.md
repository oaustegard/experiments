# Parameter extraction — component writeup

*(Component of `nl2sh-retrieval`. Fold into `RESULTS.md` when the retrieval
side lands.)*

`extract_params.py` lifts literal parameter values out of a shell request and
binds each to a character span. The premise is `monad-bsky`'s one
unconditionally transferable finding — **extract, never generate; a model must
never retype an identifier** — which in a shell helper stops being a quality
property and becomes a safety one. A generator that reconstructs
`/etc/nginx.conf` as `/etc/nginx/nginx.conf`, or `*.log` as `*.log.1`, is worse
than no helper.

Sixteen kinds fire: `path`, `glob`, `filename`, `extension`, `literal` (quoted),
`var`, `url`, `hostname`, `ip`, `port`, `size`, `duration`, `date`, `time`,
`pid`, `perm`, `signal`, `user`, `process`, `git_ref`, `identifier`, `number`.
Everything is gated on *structural* evidence — a quote, a slash, a
dot-extension, a `$`, a digit plus a unit, a cue word plus a number. Nothing is
guessed from semantics.

## How it was scored, stated before the numbers

NL2Bash has no parameter annotations, so the answer key is **derived**, and the
derivation decides the result. Full detail is in `eval_extract.py`'s docstring;
in brief:

* **Precision** — an extracted value is correct if it occurs verbatim in the
  gold command. A proxy, wrong in both directions: it over-credits `txt`
  against `"*.txt"` by substring, and it counts a genuinely correct extraction
  the reference solution declined to use (request says "the current directory",
  command says `.`) as an error.
* **Recall** — the denominator is the command's *operand* tokens (stage utility
  dropped, `-flags` dropped, shell operators dropped) that occur in the request.
  Where a whole operand is absent, it is split on shell metacharacters and its
  pieces tested instead, so `s/1\.2\.3\.4/5.6.7.8/g` contributes `5.6.7.8`.
  Splitting is conditional on the whole token missing, so a path that *is*
  present contributes itself rather than four inflating fragments.

400 pairs, seed 20260819: 300 dev (patterns were iterated against these), 100
holdout (looked at once, at the end).

## Results

| | dev (300) | holdout (100) | all 400 |
|---|---|---|---|
| values extracted | 408 | 136 | 544 (1.36/request) |
| **precision** (verbatim in gold cmd) | **0.907** [0.875–0.931] | **0.971** [0.927–0.989] | **0.923** [0.897–0.942] |
| precision, case-insensitive | 0.917 | 0.971 | 0.930 |
| **recall**, all gold values | **0.828** [0.786–0.864] | **0.872** [0.796–0.922] | **0.838** [0.802–0.869] |
| recall, structurally marked | 0.984 (n=255) | 0.954 (n=87) | 0.977 (n=342) |
| recall, unmarked | 0.453 (n=106) | 0.545 (n=22) | 0.469 (n=128) |
| value is a whole command token | 0.532 | 0.529 | 0.531 |

Intervals are Wilson 95%. Holdout precision sits *above* dev and the intervals
overlap, so there is no evidence of overfitting to the dev split — but with 136
extractions the holdout cannot rule out a few points either way, and this repo's
`gh-mcp-regex-fit` entry records a hand-written rule set falling 0.984 → 0.239
across *phrasing families*. NL2Bash is one family. This number does not transfer
to a different request distribution without being re-measured.

**Read `recall_marked` = 0.977 as close to tautological**, not as a result: the
"marked" denominator is defined as *quoted, or containing a non-alphabetic
character*, which is very nearly the set the extractor is built to fire on. The
two informative recall numbers are the unconditional 0.838 and the unmarked
0.469.

### Per-kind precision (400 pairs)

| kind | n | precision | | kind | n | precision |
|---|---|---|---|---|---|---|
| literal | 180 | 0.917 | | duration | 23 | **0.739** |
| path | 95 | 0.937 | | size | 13 | 0.923 |
| number | 49 | 0.898 | | perm | 11 | 0.909 |
| extension | 45 | 0.889 | | user | 6 | 1.000 |
| filename | 32 | 0.969 | | hostname | 4 | 1.000 |
| glob | 29 | 1.000 | | ip | 2 | 1.000 |
| var | 28 | 0.964 | | signal | 2 | 0.500 |
| identifier | 24 | 1.000 | | port | 1 | 1.000 |

## The most common failure

**Unit mismatch between request and command — the extraction is right and the
metric is wrong.** `duration` is the worst kind at 0.739, and its errors are all
one shape: *"modified in the last 24 hours"* → `find . -mtime 0`. The request
says 24 hours, `find` counts days, and the reference command therefore contains
no `24`. Six of 42 false positives across both splits are this exact pattern,
the largest single cluster. It is not fixable by better regexes; it says the
extractor must hand the generator **value plus unit** and let the generator
convert, which is why `size` spans carry a `unit` field.

The next cluster is `literal` (15 of 42), and it splits three ways: unbalanced
quotes in the NL2Bash source itself (`Create intermediate directories "b and
"c"`, which no quote-pairing rule can recover), case differences the command
introduced (`'dateiname'` → `"Dateiname"`, recovered by the case-insensitive
variant), and correct extractions the reference solution simply did not use
(`Set variable OS to ... "Linux"` → `OS=$(uname -s)`).

**The dominant recall failure is different and is by design.** 128 of 470 gold
values carry no structural marker at all — `file`, `dir`, `name`, `directory`,
`CVS`, `junk`, `emacs` — bare English words that are simultaneously prose and
the argument. The extractor recovers 47% of them incidentally (they fall inside
a quoted or path span) and cannot in principle recover the rest without
guessing, which is the property it exists to provide. That 27% of copyable
content is unmarked is the honest ceiling on this layer.

## Two findings for the layer above

**Only 37.4% of requests contain every operand their command needs** (138 of the
369 sampled pairs that have operands at all). Extraction is not a path to a
complete command on the majority of requests — the rest require a value the
request never states. This is the same measurement `gh-mcp-regex-fit`'s
`context_probe.py` made on GitHub MCP routing, where templated queries scored
56–66% and hand-authored ones 14.9%; NL2Bash sits between, as scraped forum
prose should.

**Only 53% of correct extractions are used as a whole command token.** The rest
are *composed* into a larger argument: `20` → `+20M`, `1080` → `-D1080`, `.log`
→ `'*.log'`, `/path/to/target/directory` → `anotherhost:/path/to/target/dir…`.
So the extractor supplies a slot filler, never the argument. A design that
splices extracted spans directly into an argv position would be wrong roughly
half the time even when the extraction is perfect.

## Reproduce

```bash
git clone --depth 1 https://github.com/TellinaTool/nl2bash.git
python3 test_extract.py                      # no data, no network
python3 eval_extract.py --data nl2bash/data/bash
python3 extract_params.py --text "Find *.mov under /mnt/raid older than 30 days"
```

Writes `results_extract.json` (both splits, per-kind breakdown, ranked failure
lists). The NL2Bash clone is not vendored — public corpus, own licence.
