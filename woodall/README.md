# Woodall τ=3 counterexample search

Phase 1 of issue #163 (SAT/MIP dijoin-packing verifier + Edmonds-Giles seed
liftings). Builds and calibrates the verifier the issue specifies; runs a
first, deliberately modest, search pass; defers the heavy generator work
(null-arc resolutions, Williams' catalog, iso-dedup) with an explicit list
of what's left.

## What this is

Woodall's conjecture: in every digraph, min dicut size τ equals max
disjoint-dijoin-packing size ν. Lucchesi-Younger gives ν ≤ τ always; the
open direction is whether ν can be strictly less. The *capacitated* version
(Edmonds-Giles) is already known false — Schrijver 1980, Cornuéjols-Guenin
2002 gave explicit counterexamples, all using capacity-0 ("null") arcs. This
experiment builds the machinery to search for an uncapacitated (Woodall)
counterexample with τ=3, by looking for null-arc "resolutions" of the known
capacitated counterexamples and by direct structured/random search.

## Repro

```bash
cd experiments/woodall
python3 -m pip install python-sat   # bundled cadical/glucose/minisat, no system solver needed
PYTHONPATH=. python3 calibration/test_calibration.py
PYTHONPATH=. python3 generator/search.py    # ~25s per (n, p_edge) cell, see logs/
```

## Layout

- `verifier/digraph.py` — capacitated digraph, τ via brute-force closed-set
  enumeration (predecessor-closed vertex sets), dijoin check (dicut-meeting
  definition), and an independent contract-and-check-strong-connectivity
  cross-check of the same definition (issue's "equivalently" clause).
- `verifier/packing.py` — ν(D,u) via a CEGAR SAT loop (python-sat /
  cadical+glucose): colors active arcs 1..k, adds violated-dicut clauses
  lazily, verifies each proposed color class against the *exhaustive*
  dijoin check (not just the working clause set) before accepting.
- `calibration/schrijver_d1.py` — Schrijver's (D1,u1), transcribed by
  rendering Figure 6 of the Feofiloff survey to a 600dpi PNG and reading
  vertex positions / arrowheads directly off the image (not from any
  secondary source). See the file's own docstring for a transcription bug
  found and fixed via cross-checking the graph's derived 3-fold rotational
  symmetry and the paper's stated "4 critical cuts" against an initial
  read that gave only 3 and ν=2 instead of the correct ν=1.
- `calibration/test_calibration.py` — the calibration gates (see below).
- `generator/ring.py` — the ring-of-length-2i family generalizing D1,
  reconstructed from D1's own verified orbit structure (not transcribed
  from Figure 8 — see its docstring). Validated against the paper's stated
  parity claim (odd i counterexample, even i not) for i=2..5.
- `generator/search.py` — a first, explicitly light, structure-guided
  random search pass (see Results below).
- `logs/` — raw search logs + `search_summary.json`.

## Calibration gates (issue #163 §"Calibration gates")

| Gate | Status | Result |
|---|---|---|
| 1. Reproduce Fact 7.1 (Schrijver D1: ν=1, τ=2) | **PASS** | exact match, all 4 special joins verified two ways |
| 2. Reproduce Facts 8.1/8.2 (Cornuéjols-Guenin D2, D3) | **NOT DONE** | not transcribed this pass — see Deferred |
| 3. Odd/even ring parity (i=5,7 counterexample; i=4 not) | **PASS** (i=2..5 checked; i=7 needs a non-brute-force τ) | exact match to the paper's stated parity |
| 4. Random small-DAG sanity (source-sink-connected ⇒ ν=τ) | **PASS** | 25/25 random DAGs (n=7, filtered) satisfy EG |

Gate 2 is genuinely not done, not silently skipped — it's called out both
here and in the calibration script's own output line. D2/D3 (Cornuéjols-
Guenin, Figures 9-10) involve ~14+ vertices with vertex identities the
survey itself references by number (Williams' "vertices 14 and 8") —
transcribing those correctly needs the same careful crop-and-cross-check
process used for D1, which took the bulk of this session's time for a
single 12-vertex graph. Flagged as the top priority for the next pass.

## Results — search pass (Phase 1, light)

Ran `generator/search.py`: unstructured random DAGs, n ∈ {7,8,9,10},
p_edge ∈ {0.25, 0.4}, 2000 trials/cell, 25s wall-clock budget/cell (hit in
every cell — brute-force τ is the bottleneck at these n). Hard filters
(DAG, not source-sink-connected, ≥2 sources and ≥2 sinks, τ≥3) applied per
issue spec.

| n | p_edge | trials | passed filters (τ≥3 survivors) | candidates found |
|---|---|---|---|---|
| 7 | 0.25 | 2000 | 0 | 0 |
| 7 | 0.4 | 2000 | 0 | 0 |
| 8 | 0.25 | 2000 | 0 | 0 |
| 8 | 0.4 | 2000 | 3 | 0 |
| 9 | 0.25 | 2000 | 0 | 0 |
| 9 | 0.4 | 2000 | 3 | 0 |
| 10 | 0.25 | 2000 | 1 | 0 |
| 10 | 0.4 | 2000 | 17 | 0 |

**No counterexample found — expected, and the actual finding is the filter
hit-rate, not the null result.** Across 16,000 trials only 24 instances
survived all four filters (0.15%). The dominant eliminator by far is
τ≥3 (>90% of trials at every cell): unstructured random DAGs almost never
have min-dicut ≥3, because that requires *every* predecessor-closed vertex
set to have ≥3 outgoing arcs, which sparse random generation rarely
produces. This confirms the issue's own framing: unstructured random search
is not a serious generator for this problem. The real search space is in
the structured families (null-arc resolutions of D1/D2/D3, Williams'
minimal-counterexample catalog) — deferred below, not attempted this pass.

## Deferred work (honest list, not silently dropped)

1. **Calibration gate 2** (Cornuéjols-Guenin D2, D3) — transcribe Figures
   9-10 with the same rigor as D1.
2. **Generator priority 1**: null-arc resolutions of D1/D2/D3 (subdivisions,
   anti-parallel path pairs, local widgets) — the actual "known-open lifting
   question" the issue centers on. Not started.
3. **Generator priority 2**: Williams' thesis catalog (Waterloo 2004) — not
   fetched or encoded.
4. **Iso-dedup**: no nauty/pynauty in this environment this pass; not
   installed or wired in.
5. **Literature gate**: no search performed this pass for prior
   computational verification (Feofiloff's pages, post-2020 citations,
   Abdi-Cornuéjols line). Should run before any serious compute commitment
   — a virgin-space claim from this repo currently rests only on the
   generic "no known result" framing in issue #163 itself, not on a fresh
   check.
6. **n > ~20**: `tau()`/`closed_sets()` is brute-force 2^n; the ring
   generator already can't run past i≈5 (n=20) in reasonable time. A
   max-flow-based dicut algorithm (issue's own suggested approach for the
   uncapacitated τ computation) would be needed to scale the generator to
   the n≈16-20 range the issue targets.

## What's solid

The verifier's correctness rests on more than "it compiled": Fact 7.1
reproduced exactly (not approximately), all four of the paper's own
"special join" fractional-packing witnesses independently verified under
two different dijoin definitions, and the ring generalization's odd/even
parity matches the paper's stated claim across the range that's
brute-force-tractable. The one transcription bug that did occur was caught
*by* the calibration process, not despite it — exactly the workflow issue
#163 asks for.
