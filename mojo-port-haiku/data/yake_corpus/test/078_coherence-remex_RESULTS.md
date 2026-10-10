# Coherence on remex

**Status: done.** [`daniloc/coherence`](https://github.com/daniloc/coherence) adopted into
[`oaustegard/remex`](https://github.com/oaustegard/remex) as a live trial, 2026-08-24.
One mechanism in it is worth having; the rest is a large apparatus around that mechanism.
The trial shipped as [remex#80](https://github.com/oaustegard/remex/pull/80).

## Where this came from

Danilo Campos, [Bluesky, 2026-08-22](https://bsky.app/profile/daniloc.xyz/post/3mtofpi2pxc2y),
13 posts. His argument: models write fine code at the file level, and the failure is
complexity across files and subsystems, swamped by a volume no human reads —
*"entire galleries and chambers within the cave system of the code that no human has laid
eyes upon."* He cites Conant and Ashby (a good regulator must be a model of the system) and
predicts the next gains come from modelling the project rather than from inference.
Post 12 of the thread points at his tool. Oskar asked for a read of the thread, then for an
implementation against one of our own repos.

## How the harness works

MIT TypeScript, created 2026-06-17, 217 commits, 24 stars, 1 fork, one author. On npm as
`@danilocampos/coherence` since 2026-08-18, now 0.37.1, 1,686 downloads in the week to
2026-08-22.

Cloned, built and run rather than read:

| check | result |
|---|---|
| `npm install` + `tsc` on Node 22 | clean |
| `npm test` | 936 tests, 936 pass, 3m04s |
| `coherence verify` on its own repo | 83 claims, 83 green, 0 red, 0 skipped |
| dogfood ledger in `.coherence/` | 643 decisions, 48 defects, 22 experiments |

You write `*.spec.md` files next to code. Each carries a `## works when` block of claim
lines and a `## why` block of authored rationale. The claim grammar is a nine-form registry
in `src/phrasebook.ts`, first match wins: `typechecks`, `exists`, `imports`, `responds`,
`passes test`, `boundary`, `lives in`, `parity`, `conforms to`. A line matching no form is
skipped rather than failed, so a typo'd verb is a silent no-op. `coherence verify` derives a
symbol graph through vendored tree-sitter wasm grammars (TypeScript, Python, Ruby), grades
every claim, and exits nonzero on red.

### The meta-oracle

`src/oracle-domain.ts`, `analyzeOracle`. For a claim of the form
`boundary "<invariant>" at <chokepoint> via test "<name>"` it locates the named test,
parses **that test's own source** with the TypeScript compiler API (or the Python arm for
`.py`), and classifies how the test iterates its domain:

- **LIVE**: an imported registry, a call result, member access on an import, or the anchor
  symbol itself. Passes.
- **LITERAL**: every loop iterates an array literal or a same-file `const`. Fails the claim.
- **NO-ITERATION**: no domain iteration at all. Fails.
- **NOT-FOUND**: no test with that exact name. Fails, and deliberately does not fall
  through to the runner.

A test that loops a hand-written list passes while proving nothing about completeness.
The classification catches that. It is static AST work, so it runs under `--fast`.

### Its own repo uses `via guard` everywhere

66 of 66 boundary claims in coherence's own specs use `via guard`, the escape hatch that
skips the meta-oracle entirely. Zero use `via test`. Zero `parity` claims, zero
`conforms to`. So the flagship mechanism is unit-tested in that repo and never exercised
through its own enforcement path — while its Known Limits section calls `via guard`
"a laundering channel for hand-lists dressed as guards. Review every `via guard` by hand."

That is what made a trial on someone else's code the only way to assess it.

## The trial

`remex` is a Python vector-quantization library: 53 source files, 11 test files, 267 tests
before this work. Adoption was a config file and two spec files.

### It ran on Python out of the box

`"language": "python"` resolves to a shipped grammar. No adapter to write. The first
`verify` on two structural claims came back green with an accurate nag:

```
○ adoption — step 4: 2 claim(s), none backed by an oracle.
  Every claim here can only fail structurally (`via test` / `via guard` /
  `passes test` appear nowhere) — nothing red-flags a behavioral break
```

### It flagged a real sampling oracle

`remex/rotation.py` holds `ROTATION_CODES = {"haar": 0, "rht": 1, "none": 2}`, the single
source of truth for which rotations exist and what byte each is written as.
`tests/test_rotation_rht.py` checks the on-disk round-trip with
`@pytest.mark.parametrize("rotation", ["haar", "rht"])` — a hand-written list, two of three
members. Pointing a boundary claim at it:

```
✗ boundary "every declared rotation survives a round-trip through the on-disk formats"
    at rotation_from_code via test "test_pq_and_npz_round_trip_rotation"
    — [oracle] iterates a LITERAL domain (inline ["haar", "rht"] literal)
      — a sampling oracle, not totality.
```

The tool was right about the oracle. The gap it implies is smaller than it looks: `"none"`
does round-trip, and a scalar-mode serialization test covers that path, because scalar mode
happens to use `rotation="none"`. Coverage by coincidence rather than by enumeration.

### Perturbation results

Three perturbations of `remex`, each reverted:

| perturbation | pre-existing suite (267 tests) | the live-domain oracle |
|---|---|---|
| `"hadamard2": 3` into `ROTATION_CODES`, no construction behind it | 267 passed | red on `test_every_declared_rotation_round_trips[hadamard2]` |
| `"hadamard2"` into `Quantizer.ROTATIONS` only | 267 passed | red on `test_constructible_and_persistable_rotations_agree` |
| `5` into `SUPPORTED_BITS`, no packing branch | `test_bits_validation` also fails | red on `test_every_supported_width_round_trips[5]` |
| `header[17] = 0` for `"none"` in `save_pq` | `TestSerialization::test_pq_round_trip` fails | red on `test_every_declared_rotation_round_trips[none]` |

In rows one and two an entire 267-test suite stays green while a member sits in a registry
with nothing behind it. Rows three and four are co-detected, and
`lib.spec.md` records row three as the weaker refutation it is.

### The refactor the tool demanded

The tool flagged a second literal domain: `tests/test_packed_vectors.py` parametrizes
`[1, 2, 3, 4, 8]`. There was no registry to loop, because the supported bit widths were
spelled five times as the negative guard `bits in (5, 6, 7)` across `packing.py`, `core.py`
and `pq_format.py`, and twice more as that literal in the tests. Seven sites held equal by
memory, which is tier 3 on coherence's own enforcement ladder.

Closing that claim meant editing `remex`, not the spec: extract `SUPPORTED_BITS`, point four
guards at it, write an oracle that loops it. That is the honest shape of adoption. The tool
names a convention and demands a refactor; it cannot promote anything on its own.

Final state on the branch: 7 claims, 7 green, 3 invariants each anchored and each carrying a
recorded refutation, 288 Python tests passing.

### A false failure in the parity arm

The parity meta-oracle false-failed a correct oracle. This is a complete, total check:

```python
persistable = set(ROTATION_CODES)
for name in sorted(persistable):
    assert rotation_from_code(ROTATION_CODES[name]) == name
```

The analyzer refused it:

```
[parity] iterates `sorted(persistable)` — never the declared domain `ROTATION_CODES`
```

The analyzer requires the loop to name the declared domain literally, and one alias defeats
it. Coherence's README claims the analyzer "is conservative and never false-fails" — that holds for the
boundary arm, not this one. Rewriting the loop as `for name in ROTATION_CODES:` passed.

The `redundancy` advisory found two README benchmark tables spelling the same bit-level
domain and already disagreeing, which is a genuine hit. It did not find the five-site
`bits in (5, 6, 7)` duplication in the Python source, which the claim machinery had just
flagged from the other direction.

## Judgment

**Take the mechanism, not the apparatus.** The meta-oracle is the contribution: an
oracle-about-oracles that distinguishes a totality check from a sample by reading the test's
own AST. It works on Python, it works on a codebase its author has never seen, and the
perturbation table above is what a passing `via test` claim buys that a passing test does
not.

Around it sit 40+ CLI subcommands, a 217 KB README (~54k tokens) that instructs agents to
read the full reference, a 62 KB `AGENTS.md`, and invented vocabulary: doctrine, gyroscope,
regulate, premise leases, mass, atlas. A harness whose pitch is spending less inference asks
for a 54k-token read before an agent can use it.

Positioned against our own tooling: `xr`, `tree-sitting` and `exploring-codebases` attack
cross-file opacity from retrieval; coherence attacks it from enforcement in CI. Those are
complements. The transplantable piece for `verifying-claims` or `gating` is the LIVE /
LITERAL / NO-ITERATION classification, which needs none of the ledger machinery to be useful.

## Files

- `artifacts/coherence.config.json` — the Python adoption config, 11 lines
- `artifacts/root.spec.md`, `artifacts/lib.spec.md` — the two spec files
- `artifacts/test_rotation_totality.py`, `artifacts/test_packing_totality.py` — the oracles
- `recheck.py` — checks this writeup's numbers against the artifacts
- `ERRORS.md` — what was wrong on the way here
