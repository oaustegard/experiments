# remex product description

A written description of what a programmer experiences when they use remex: what each call accepts, what it does, what it refuses, and exactly what happens when a call is interrupted or the thing it acts on changes underneath it.

This is a trial of the `describing-products` skill (oaustegard/claude-skills) on a library rather than an app. Two documents are drafted (the pilot and one foundation), with a verification pass and a bug triage behind them; the rest of the structure is the plan. How the trial went is in [RESULTS.md](RESULTS.md).

## Purpose

remex is, from the caller's side, a state chart. The caller moves through it with calls: build a `Quantizer`, `encode` a matrix into a `CompressedVectors`, `save` it, `load` it in another process, `decode` or `search` it, possibly at a lower `precision`. Most of the rules that govern those moves are spread across guard methods in `remex/core.py`, docstrings, the README, `CLAUDE.md` and twenty test files. There is no single place that says, in plain language, "when you call X with this container, this is what happens, and this is what happens if the process dies halfway through".

This project is that place. It describes the remex 2.0.0 Python API in its default configuration: `pip install remex`, NumPy and SciPy present, a C compiler available so the native kernel builds, no environment variables set.

The documents are for people who need to use or change remex: callers deciding whether a file is safe to rely on, maintainers deciding whether a behavior is intended. They are written from the caller's side. They describe what a call does and what a caller can observe, not how it is implemented.

### What this is not

- Not API reference. Signatures and argument lists are in the docstrings and the README.
- Not organized by module. `core.py`, `packing.py` and `rotation.py` are not described separately. A behavior is described once, where the caller meets it.
- Not a design document. Where a mechanism changes what a caller would expect, it appears in a block quote labeled `Technical note:` and nowhere else.

## Conventions

- Describe what the caller observes. "Decoding with a quantizer built from a different seed returns vectors with no relation to the originals, and no error", not "`_check_rotation` compares rotation names but not seeds".
- Technical detail goes in block quotes, prefixed with `Technical note:`.
- Sentence case for headings.
- The [glossary](glossary.md) is the source of truth for terms like container, encoding parameters, scalar mode, centred, precision and refused.
- Every document ends with the remex commit it was verified against and a list of open questions.
- Surprising behavior is stated plainly, with the reason if the code or a comment gives one.

## The work to be done

Each document describes one feature: a call or a closely related pair of calls (`save` and `load`), a container type, or a concern that runs through every call.

### Document template

1. **Summary.** One paragraph, abstract, then where the feature lives and how it is reached.
2. **The simple case.** The common path in prose.
3. **The call, phase by phase.** The five phases of a call: *the call is made* (arguments checked, what is captured); *refused at once* (raises before any effect); *the first effect* (the moment state changes: a file is opened or truncated, a cache is built, an object is mutated); *while it runs* (what is partially done, what is visible to other threads or processes); *it returns* (what exists afterwards, what was recorded, the failure path). A Mermaid `stateDiagram-v2` of the states.
4. **Variants.** A table of the container and quantizer properties that change the outcome, the same rows in every document: *Container mode*; *Centred*; *Rotation*; *Bit width and precision*; *Environment variables*. Columns: *Set when the call begins* and *Changed while it runs*.
5. **Cancel and interrupt.** The same seven rows in every document, in this order:
   - An exception or KeyboardInterrupt inside the call
   - Another thread uses the same object during the call
   - A later call reuses what this one cached
   - Resources fail: MemoryError, disk full, no C compiler
   - The process dies: SIGKILL, power loss, OOM killer
   - The target changes underneath: file replaced or deleted, container fields reassigned
   - Input in an unexpected form: dtype, layout, NaN or inf, wrong dimension, a path object or file handle
6. **Interactions with other systems.** One bold-led paragraph per concern, in this order: **The encoding contract.** **Matryoshka precision.** **Determinism across machines.** **Memory and caches.** **Thread safety.** **Other formats.** **Version compatibility.**
7. **Edge cases.**
8. **Open questions and verification.**

Item 5 matters most: the same questions asked of every call is how gaps show up.

### Method

For each document:

1. Read the methods in `remex/core.py` (and `packing.py`, `rotation.py`, `st.py` where they apply) and the guard methods every decode and search path passes through.
2. Read the matching tests in `tests/`. `test_coverage_gaps.py`, `test_scalar_mode.py`, `test_centered_mode.py` and `test_v2_fixes.py` read as specifications of edge cases.
3. Draft the document.
4. Run anything ambiguous against the installed package. For a library, "the running product" is a Python process; most claims are a five-line script.
5. Record the commit verified against.

### Verification

The `verification/` directory holds one checklist per cluster of documents, each item a single observable claim with setup, steps, expected result and priority. Because every item is a short script, a pass is a script: `verification/run_pass.py` runs the items that can be automated and prints a result per ID. A document moves from `drafted` to `verified` when every P1 and P2 item has passed or been filed.

[`bug-triage.md`](bug-triage.md) holds every behavior the documents flagged as a likely defect, deduplicated, with reproduction, cause, severity and the decision needed.

### Order of work

1. **Pilot: saving and loading `.npz` files.** Small, self-contained, and every interrupt row has a real answer.
2. **Foundation: the encoding contract.** What makes two quantizers compatible and which mismatches are refused. Everything else links to it.
3. **The hardest area: search.** Four search paths and the IVF index, which must agree with each other on every variant.
4. **Everything else**, in parallel, then a consistency pass and a verification pass.

### Scope decisions

- **GPU backend (`remex/gpu.py`).** Out of scope: optional, and documented as a wrapper that must agree with `Quantizer.search`; a later `search/gpu.md` can cover it.
- **The Mojo port and the `.pq` CLI parity.** Out of scope: a separate product in a separate repo.
- **Benchmarks and recall figures.** Out of scope. These documents describe behavior, not quality; recall numbers live in the remex README and `bench/RESULTS.md`.
- **Interaction shape.** The unit of interaction is a call; its phases, the variant rows, the interrupt rows and the order of cross-cutting concerns are fixed as written above.
- **Numbered rules.** Prose, not numbered specifications. Heading anchors are enough for cross-references.

## Structure

```
README.md                        this file
RESULTS.md                       how the describing-products trial went (experiments-repo convention)
goal.md                          standing instructions for whoever drafts
AGENTS.md, CLAUDE.md             entry points for agents: read README.md, then goal.md
glossary.md                      shared vocabulary
bug-triage.md                    suspected defects, with repro steps and decisions needed

verification/
  README.md                      how to run a pass and record results
  persistence.md                 checklists for persistence/
  run_pass.py                    the automated part of a pass

foundations/
  the-encoding-contract.md       what determines codes, what a reader must match, which mismatches are refused
  containers.md                  CompressedVectors and PackedVectors: what a container holds and reports

encoding/
  quantizer-construction.md      building a Quantizer: argument checks, defaults, the native kernel
  encode.md                      encode: input handling, the anisotropy warning, scalar mode
  decode.md                      decode and precision=

search/
  search.md                      search: the cached float32 path
  search-adc.md                  search_adc: scoring from the codes
  search-twostage.md             search_twostage: coarse then fine
  search-batch.md                search_batch
  ivf.md                         IVFCoarseIndex

persistence/
  npz-files.md                   save and load to .npz (the pilot)
  arrow-files.md                 save_arrow and load_arrow, memory mapping
  pq-files.md                    save_pq, load_pq, save_params

integration/
  remex-index.md                 RemexIndex: add, search, save, load, MaxSim
  semantic-search-remex.md       the faiss-shaped one-call search

cross-cutting/
  determinism.md                 what is bit-identical across seeds, machines and versions
  memory.md                      caches, low_memory, drop_cache, resident_bytes
  warnings.md                    AnisotropyWarning, the native-kernel fallback, deprecations
```

## Coverage

Status is one of `not started`, `drafted`, or `verified`.

| Document | Status |
| --- | --- |
| glossary.md | drafted |
| bug-triage.md | drafted |
| verification/ (1 checklist) | drafted |
| foundations/the-encoding-contract.md | drafted |
| foundations/containers.md | not started |
| encoding/quantizer-construction.md | not started |
| encoding/encode.md | not started |
| encoding/decode.md | not started |
| search/search.md | not started |
| search/search-adc.md | not started |
| search/search-twostage.md | not started |
| search/search-batch.md | not started |
| search/ivf.md | not started |
| persistence/npz-files.md | drafted |
| persistence/arrow-files.md | not started |
| persistence/pq-files.md | not started |
| integration/remex-index.md | not started |
| integration/semantic-search-remex.md | not started |
| cross-cutting/determinism.md | not started |
| cross-cutting/memory.md | not started |
| cross-cutting/warnings.md | not started |

## Reference

The source of truth is [oaustegard/remex](https://github.com/oaustegard/remex) at commit `b3a0ed9` (release 2.0.0).

- `remex/__init__.py`: the public surface
- `remex/core.py`: `Quantizer`, `CompressedVectors`, `PackedVectors`, and the guard methods (`_check_rotation`, `_check_bits`, `_check_mean`, `_check_mode`) every decode and search path passes through
- `remex/packing.py`: bit-packing and `SUPPORTED_BITS`
- `remex/rotation.py`: the three rotations and `LEGACY_ROTATION`
- `remex/pq_format.py`, `remex/st.py`, `remex/ivf.py`: the other formats, the sentence-transformers wrapper, the IVF index
- `tests/`: behavioral tests; `test_coverage_gaps.py`, `test_scalar_mode.py`, `test_centered_mode.py`, `test_v2_fixes.py` are closest to executable specifications
- `CLAUDE.md`: design decisions and invariants, with the reasons
