# Anchors registry

An **anchor** is the part of a check that your implementation did not produce:
a published constant, a closed form, a conservation law, a theoretical bound.
Without one, the strongest thing a check can say is "this run agrees with the
last run," which is true of code that has been quietly wrong since the first
commit.

This file exists for one reason: **anchors have ranges, and range gaps are
invisible from inside a green run.** Max (1960)'s quantizer table stops at 5
bits. `remex-vs-higgs-ablation` swept to 8. For two runs nothing checked the
rates past the end of the table, a 16%-high value lived there, and — because
that value was the threshold a downstream check compared against — the error
made the suite *more permissive* exactly where it was weakest. The gap was
discoverable at any point by writing the covered range next to the constant.
Nobody did, because there was nowhere to write it.

Every row here therefore carries a **covered range** and a **gap** column. The
gap column is the one to read.

Conventions:

- **Write the source next to the number, always.** An unattributed constant
  becomes a golden value within one release, and then nobody can tell whether
  it is authority or history.
- **Never derive an anchor from the code under test**, including "I ran it once
  and it looked right."
- A **manufactured** anchor (synthetic data with a known answer, a degenerate
  case, an independent draw) is legitimate and is marked as such. It is weaker
  than published, stronger than nothing, and it must not share code with the
  subject.
- Prefer **brackets** to point checks: better than a baseline *and* not better
  than a bound. A one-sided check passes for a result that collapsed as readily
  as for one that is right.

---

## Published constants

| anchor | value | source | covered range | **gap — what it cannot see** | used in |
|---|---|---|---|---|---|
| BEIR SciFact split shape | 5,183 docs / 1,109 claims / 300 test queries / 339 test qrels pairs | Thakur et al., *BEIR*, arXiv:2104.08663 (NeurIPS 2021 D&B), table 2 + the published `BeIR/scifact` split | **counts only** — the four cardinalities of the test split | **Pins the split, not the strings.** It certifies that a rebuild from the upstream AllenAI release picked the right claims (dev, not test), the right relevance field (`cited_doc_ids`, not `evidence`) and deduped the one doubly-cited pair — but it cannot see wrong *document text*. **Gap measured, not hypothetical:** a reconstruction passing all four counts still differs from `BeIR/scifact` on **1,055 of 5,183 documents (20.4%)**. See `ttt-embed-quantized/crosscheck_allenai.py`. | `ttt-embed-quantized/crosscheck_allenai.py`; originally `encode_scifact.py::verify_shape` (PR #35, closed) |
| SciFact nDCG@10, small retrieval encoder | ≈ 0.60–0.72 | issue #33's pre-registered band; consistent with BEIR's published SciFact leaderboard for dense retrievers of this class | dense retrievers of roughly this size; **binary-gain nDCG@10 only** | **Wide, and demonstrated insufficient.** Two encodes differing on a fifth of the corpus scored **0.7152** (BEIR text) and **0.7067** (AllenAI rebuild) — both comfortably inside the band, which therefore certified a corpus it could not see was wrong. A band this loose also passes a pooling or prefix error costing a few points. Never the only check on document text. | `ttt-embed-quantized/encode.py::ndcg_at_k`, `RESULTS.md` |
| Optimal fixed-rate scalar quantizer MSE, N(0,1) | b=1 0.3634, b=2 0.1175, b=3 0.03454, b=4 0.009497, b=5 0.002499 | Max, *Quantizing for minimum distortion*, IRE Trans. IT-6 (1960), table 1 | **b = 1…5 only** | **b = 6, 8 are unanchored by this table.** The sweep uses both. A defect above b=5 is structurally unreachable here and reports the same green as everywhere else. Covered from below by Panter–Dite. | `remex-vs-higgs-ablation/grids.py::MAX_1960_MSE`, gated in `gate.py` |
| Panter–Dite high-rate asymptote | D → 2.7207·2⁻²ᵇ (= √3π/2) | Panter & Dite (1951); Gersho & Gray, *Vector Quantization and Signal Compression*, ch. 6 | high rate; a **hard upper bound at every finite rate**, approached from below | **Asymptotic, so it bounds but does not pin.** The true value sits ~2.6% below it at b=6 and ~0.6% at b=8, so an error smaller than that margin is invisible. Tight enough to reject the historical +16%. | `gate.py::PANTER_DITE`, b=6 and b=8 brackets |
| E8 normalised second moment | 0.0716821 | Conway & Sloane, *Sphere Packings, Lattices and Groups*, table 2.3 | the E8 lattice, exactly | Says nothing about a *trained* codebook — only that the lattice plumbing (nearest-point decode, Voronoi sampling) is right. | `calibrate.py::e8_nsm`, `gate.py` |
| QuIP# E8P codebook family | ball-shaped subset of E8, scale-optimised here | Tseng et al., *QuIP#*, arXiv:2402.04396 (ICML 2024) | 2 bits/coordinate at m=8 | **One configuration of nine.** It is the only *practical* floor in the gate — the others are statistical. m=4, m=5, m=2 have no published-family comparison. | `gate.py` G4-equivalent |
| HIGGS practical envelope | grid dimension p ∈ [1,5], grid size n ∈ [9, 4096] | Malinovskii et al., *HIGGS*, arXiv:2411.17525 (NAACL 2025), §4.3 | the configurations HIGGS itself reports | Used to show this experiment's grids (to m=8, K=2¹⁶) meet or exceed the published envelope, i.e. the vector arm is not a weakened stand-in. Does not certify quality, only scope. | `RESULTS.md`, deviation 4 |
| Rate-distortion bound, Gaussian | D ≥ 2⁻²ᵇ per unit-variance dimension | Shannon; standard | all rates | One-sided. Used as the far side of a bracket — a result beating its own bound is a bug, not a triumph. | `gate.py` grid brackets |
| Lloyd–Max fixed point is unique for a log-concave density | — | Fleischer (1964) | log-concave densities, incl. Gaussian | Justifies treating a converged fixed point as the global optimum, which is why the b=5 disagreement with Max's table is attributed to the 1960 table's last digit rather than to the code. | `gate.py` b=5 tolerance note |
| EIA state average retail price, Maryland **commercial** | 16.4 ¢/kWh (Aug 2026); 15.86 ¢ (Dec 2025 report); industrial 13.59 ¢ | EIA state electricity data, **via search-result summary — the primary table was egress-blocked** | Maryland statewide, commercial *class average*, monthly | **Not a tariff and not Montgomery County.** A class average blends every rate schedule and load factor in the state; it cannot tell you what Pepco's GS/GT schedule charges a 14 kW demand-metered 100%-load-factor customer, which is the only rate that matters here. EIA also published a **wrong** MD residential figure in Mar 2026 (35 ¢, corrected to 22.2 ¢), so the series itself has a demonstrated defect rate. Anchors the *order of magnitude only* — which is enough, because `luna-onprem-tco`'s negative control shows **free electricity does not change the conclusion**. | `luna-onprem-tco/params.json:electricity`, `recheck.py` phase 5 |
| DeepSeek V4 Pro 1.6T decode throughput on Blackwell | 976 tok/s/GPU (8×B200) and 6,644 tok/s/GPU (GB200 NVL72), both @105 tok/s/user | SemiAnalysis InferenceX, **via search-result summary — inferencex.semianalysis.com was egress-blocked** | **decode only**, one model, two operating points on the interactivity curve | **Says nothing about prefill**, which is the token class the whole cost question is about — so `luna-onprem-tco` models prefill from FLOPs×MFU and the anchor cannot check it. Also single-vendor and single-serving-stack: an SGLang or vLLM version change moves these by tens of percent. Back-solves to ~7% MFU, which is *consistent with* GEMV-bound decode but would also be consistent with a badly-tuned deployment. | `luna-onprem-tco/params.json:hardware`, `model.py::decode_tok_s` |
| B200 tensor-core peak | 9 PFLOPS dense FP4 (18 sparse); 4.5 PFLOPS dense FP8; 2.25 dense BF16 | NVIDIA Blackwell datasheet, via secondary summary | peak, per GPU, marketing conditions | **Peak, not achievable.** Everything downstream rides on an *authored* 30% MFU multiplier; the anchor pins the numerator and leaves the only contested factor unanchored. Swept 20–35% in `model.py --sweep-mfu`; the experiment's headline finding holds across the whole band, which is the only reason the gap is tolerable. | `luna-onprem-tco/params.json:hardware`, `prefill` |
| Artificial Analysis Intelligence Index, Luna vs V4 Pro | Luna max 52 / xhigh 50 / high 47 / medium 39; DeepSeek V4 Pro 0813 (reasoning, max) 53 | Artificial Analysis, via secondary summary | one composite index, one snapshot, reasoning-effort-dependent | **A composite hides the axes that matter for a given deployment.** 52 vs 53 licenses the claim "capability-matched" for sizing purposes and nothing finer; a workload dominated by one benchmark inside the composite could see a large real gap. Effort setting is load-bearing — the same two models compare as 39 vs 53 at different settings, which is how a comparison table can be built to say either thing. | `luna-onprem-tco/RESULTS.md` capability match |

## Closed forms and conservation laws

| anchor | statement | why it is strong | used in |
|---|---|---|---|
| Scalar quantizer distortion by direct integration | exact integral against the normal density | Independent of the sampled KD-tree instrument, so agreement between the two exonerates the *instrument* before anything blames the subject. Converts "one of these two is broken" into "this one is broken." | `gate.py` instrument check |
| Product-grid decomposition | nearest-neighbour assignment on an m-fold product grid decomposes per coordinate | Lets a sampled m-dimensional measurement and a 1-D closed form measure the same quantity. The degenerate case that makes a complex path checkable. | `gate.py` instrument check |
| Orthogonality | ‖Rx‖ = ‖x‖ and R⁻¹Rx = x | Cheap, and rules out a large family of indexing and sign errors in one line. | `gate.py` rotation checks |
| Quantizer idempotence | Q(Q(x)) = Q(x) | The reconstruction is already a codepoint, so re-encoding cannot move it. Fails the instant decision boundaries disagree with the levels they came from — which mutation testing showed nothing else was checking. | `gate.py` codec checks |
| Codebook membership | every output value is a codepoint | Definitional. Catches an encoder that returns something it never could have chosen. | `gate.py` codec checks |
| Constant-modulus 1-bit code | ‖x̂‖/‖x‖ = √(2/π) = 0.79788 with **zero** spread | Turns the mechanism story behind the 1-bit MIPS reversal into an assertion. It was prose for one whole run. | `gate.py` |

## Manufactured anchors

| anchor | how it is made | why it does not share code with the subject | used in |
|---|---|---|---|
| E[max\|coord\|] of a uniform random unit vector | draw Gaussians, normalise, take the max | Touches neither rotation implementation. A Haar rotation maps any fixed unit vector to exactly this distribution, so it is the ideal both arms are measured against — rather than measuring them against each other, which is what let two identity rotations pass. | `gate.py` incoherence brackets |
| Analytic paired standard error | two codebooks scored on one shared sample; the per-sample difference carries its own se | The common sampling fluctuation cancels. Validated against the observed spread over independent streams — a tolerance is only as good as the noise model it came from. **See the caveat below.** | `gate.py::paired_gain` |

> **Caveat on the paired se, recorded because it is the sharpest known weakness
> of the current gate.** The se *shrinks* as the two codebooks converge, so the
> margin certifies "the gain is real", not "the gain is worth having." Measured:
> a product grid perturbed by N(0, 1e-3) gains +0.0001 dB against a 3-se margin
> of 1.2e-06, and is accepted; real grids gain 0.35–1.41 dB. A statistical
> margin is not a practical floor. The only practical floor in that gate is the
> QuIP# E8P row above, and it constrains one configuration of nine.

## Internal settled results (weakest tier — use as controls, not as authority)

These came from our own earlier measurements. They are *not* anchors in the
strict sense: they share an author and a codebase with the work that cites
them, so they cannot detect an assumption both inherited. They belong in a gate
as **replication controls** — "if this comes out differently, the harness is
broken" — never as ground truth.

| result | measured | used as |
|---|---|---|
| TurboQuant `prod` (Lloyd-Max + 1-bit QJL residual) is strictly dominated for retrieval at every bit width | 2026-04-02 | replication control in `remex-vs-higgs-ablation`, gated to stay dominated |
| Fitting and evaluating on the same corpus flatters learned methods; the advantage reverses under an honest protocol | `recall-per-byte`, `rotation-decorrelation` | why every codebook here is data-oblivious |

See `METHODS.md` → "Negative results — do not re-derive" for the full list.

---

## Not yet swept

This registry was seeded from `remex-vs-higgs-ablation` and is **complete for
that directory only**. Other experiments in this repo use published constants
that are not yet catalogued here — `ms13-campaign`, `discrepancy` (Komlós),
`lattice-representation-hypothesis`, `te-bridges` and others reference external
values in code without a recorded range.

That is itself the finding this file is meant to surface: an uncatalogued
anchor has an unrecorded range, and an unrecorded range is a gap nobody can see
until it bites. Adding a row costs a minute. Add one when you use a constant,
not when you audit.
