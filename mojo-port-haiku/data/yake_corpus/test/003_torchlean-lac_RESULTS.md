# LAC in TorchLean

TorchLean can carry the llm-as-computer executor's spec and proof layers but not
its graph IR. The parabolic-addressing theorem that the whole executor rests on
is four tactic lines and is proved here; the hard-argmax read cannot be lowered
to `NN.IR.Graph`, so autograd and CROWN bounds are unavailable to it.

LAC's measured float32 capacity ceiling, exact reads through address 4096 and
failure from 4097, is now a theorem over TorchLean's `FP32` arithmetic
(`LAC/Capacity.lean`, second half of this file).

Ran 2026-09-05 on CCotw against `lean-dojo/TorchLean` @ `12f5c651` (Lean 4.33.0,
Mathlib v4.33.0).

## Proved lemmas

`LAC/Check.lean` (Mathlib only, no TorchLean import) states LAC's score function
and five facts about it. All five elaborate; `lake env lean` exits 0.

| theorem | statement |
|---|---|
| `score_eq` | `score j i = i*i - (j-i)*(j-i)` |
| `score_self` | `score i i = i*i` |
| `score_lt_self` | `j ≠ i → score j i < score i i` |
| `score_le_self` | `score j i ≤ score i i` |
| `score_gap` | for `i j : ℤ`, `j ≠ i → score j i + 1 ≤ score i i` |

`score j i = 2*j*i - j*j` is the dot product of the key `(2j, -j²)` written at
address `j` against the query `(i, 1)` formed to read address `i`. Exactness of
LAC's addressing is a ring identity, not a numerical property: `ring` closes
`score_eq`, and `linarith` plus `mul_self_pos` closes the strict inequality.

`score_gap` is the quantitative one. Adjacent *integer* addresses differ by at
least 1 in score, which is the input a softmax-vs-argmax agreement bound needs.

`LAC/Core.lean` restates the same core inside TorchLean's spec layer and adds the
memory encoding: `paraKey`/`paraQuery` as `Spec.Tensor ℝ [2]`, `Mem C` as
`Spec.Tensor ℝ [C, 3]`, and `rowScores` scoring every row against a query.
`lake env lean LACScratch/Core.lean` exits 0 against a fully built `NN`.

`#print axioms` on `score_lt_self` and on `Spec.Tensor.argmax` reports
`[propext, Classical.choice, Quot.sound]` — the three standard Lean axioms and
nothing else.

## Spec-layer primitives

`Spec.Tensor` is `inductive Tensor (α : Type) : Shape → Type` with no scalar
class attached at the type, so LAC's tensors are ordinary TorchLean tensors.
`Spec.Tensor.argmax` takes a nonemptiness proof, is total, and breaks ties toward
the smaller flattened index, which is already LAC's left-to-right recency
convention. A LAC read is `keys · query`, then `argmax`, then index, and all
three are spec primitives. It never touches TorchLean's `Attention` layer.

`NN.Spec.Dynamics.System` supplies `DynamicalSystem`, `iterate`, `trajectory`,
`iterate_add`, `IsFixedPoint`, and `FixedPointCertificate`. HALT is a fixed
point.

## Graph IR limits

`NN/IR/Graph.lean`'s `OpKind` has no argmax, no gather, and no one-hot. Its ops
are input, const, permute, transpose, detach, randUniform, bernoulliMask, add,
sub, mul_elem, abs, sqrt, inv, maxElem, minElem, maxPool, avgPool, broadcastTo,
reduceSum, reduceMean, sum, matmul, linear, conv, batchNormEval, relu, tanh,
sigmoid, exp, log, sin, cos, softmax, hardMaskedSoftmax, layernorm, reshape,
flatten, concat, mseLoss. Hard-argmax LAC therefore cannot be lowered to the
canonical graph, and the verifier bridge, autograd, and IBP/CROWN checkers all
sit behind that lowering. Adding an argmax op means re-discharging the
semantic-preservation proof and supplying a `CrownTransferSound` instance, which
`TRUST_BOUNDARIES.md` lists as a caller obligation.

`DynamicalSystem` is over `SpecTensor s = Tensor SpecScalar s` with
`abbrev SpecScalar := ℝ` (`NN/Spec/Core/Scalar.lean:33`), and its state shape is
fixed. LAC's append-only growing context has to be re-encoded as bounded capacity
plus a write pointer. The real executor has a capacity anyway, so this is
faithful, but it is a re-encoding rather than a transcription.

`class Context` extends `Div`, `Pow α α`, `MathFunctions`, `Numbers`, and
`Coe Nat α`, so ℤ cannot instantiate it without stub transcendentals, and ℚ is
deliberately not installed globally (`open scoped NN.Spec.RationalAlgebraic`
opts in). Doing LAC over ℝ costs nothing, since integers embed exactly and the
addressing proof is ring-level, and it leaves finite precision to `NN/Floats`.

## `Spec.Tensor.argmax` has no characterization lemma

Nothing in `NN/` proves that `Spec.Tensor.argmax` returns a maximal index.
`grep` for theorems mentioning argmax returns three hits, all in
`NN/MLTheory/Proofs/Verification/Robustness/LipschitzCertified.lean` and all
about `argmaxClassifier`, a different function. So a LAC read-exactness theorem, saying that
argmax over `rowScores` selects the row holding the queried address, cannot cite
an existing lemma; it has to unfold the `loop` in
`NN/Spec/Core/TensorReductionShape/Reductions.lean:61` and prove the invariant.
The lemma is missing from TorchLean generally; LAC is one caller among others.

## The binary32 capacity ceiling

LAC's measured capacity ceiling is `j² > 2^24` in fp32 (memory 5bd74ba1,
2026-07-22: zero misaddresses through 4096, then every read fails). That is a
statement about the 24-bit binary32 significand. `LAC/Capacity.lean` proves it
against TorchLean's `FP32` model: `NF binaryRadix fexp32 rnd32`, the Flocq-style
FLT(-149, 24) format with round-to-nearest-even, whose `*` and `+` round after
every primitive. The bridge theorem `toReal_roundDyadicToIEEE32_eq_fp32Round`
(`NN/Floats/IEEEExec/Bridge/FP32/RoundDyadic.lean`) identifies that rounding with
the executable bit-level kernel on the finite range, so these are statements
about IEEE-754 binary32 words.

The read is modelled the way the executor computes it. Address `j` stores the
float32 row `(fl(2j), fl(-j²))`; a read of `i` forms the float32 query `(fl(i), 1)`
and scores every row as `fl(fl(k₀ · i) + k₁)`, TorchLean's `FP32` multiply then
add:

```lean
noncomputable def score32 (j i : ℤ) : ℝ := ((key j).1 * query i + (key j).2).val
```

| theorem | statement |
|---|---|
| `r32_dyadic` | `\|m\| ≤ 2^24`, `0 ≤ k` → `fl(m · 2^k) = m · 2^k` |
| `r32_int` | `\|n\| ≤ 2^24` → `fl(n) = n` |
| `score32_exact` | `0 ≤ j ≤ i ≤ 4096` → `score32 j i = 2ji - j²` (every intermediate exact) |
| `read_exact_below_ceiling` | `0 ≤ j ≤ i ≤ 4096`, `j ≠ i` → `score32 j i < score32 i i` |
| `r32_odd_sq`, `r32_neg_odd_sq`, `r32_two_odd_sq` | odd `s`, `2^24 < s² < 2^25` → `fl(s²) = s² - 1`, `fl(-s²) = -(s² - 1)`, `fl(2s²) = 2s² - 2` |
| `read_ties_above_ceiling` | `4097 ≤ i ≤ 5792` → `score32 (i-1) i = score32 i i` |
| `capacity_ceiling` | the two read theorems as one conjunction |
| `generic_even_of_ge` | a binary32 value `≥ 2^24` is an even integer |
| `not_both_representable_above_ceiling` | `4097 ≤ i` → `i²` and `i² - 1` are not both binary32 values |
| `kernel_tie_4097`, `kernel_distinct_4096`, `kernel_tie_11585`, `kernel_separates_11587` | bit-level instances on `IEEE32Exec.roundDyadicToIEEE32`, closed by `decide` in the kernel |

`#print axioms` on the three main theorems reports `[propext, Classical.choice,
Quot.sound]`; the kernel instances need only `[propext, Quot.sound]`. No `sorry`,
no native evaluation. `lake env lean LACScratch/Capacity.lean` takes about 50 s
against the built `NN.Floats` subtree (2257 jobs).

The proof below the ceiling is that every quantity is a dyadic `m · 2^k` with
`|m| ≤ 2^24`: `2j`, `i`, `j²`, the product `2ji = (ji) · 2` and the score
`2ji - j² ∈ [0, i²]`. Rounding fixes all of them, so the float32 score is the
exact integer score and `score_lt_self` finishes it. The proof above the ceiling
is a parity computation in the first binade, where the grid spacing is 2. For odd
`s` with `s²` in `(2^24, 2^25)` the tie at `s²` resolves to the even significand,
which is `s² - 1` because `(s² - 1)/2 = 2t(t+1)`. With `i` odd both `fl(2i²)`
and `fl(-i²)` lose one unit and the address-`i` score lands on `i² - 1`, while the
even neighbour `i - 1` computes `i² - 1` exactly. With `i` even the roles swap:
address `i` is exact at `i²`, and the odd neighbour's `fl(-(i-1)²)` gains one unit
that the exact product `2(i-1)i` cancels, and that score is `i²` too. Either
way the two scores coincide, and a leftmost-tie argmax (LAC's recency
convention and `Spec.Tensor.argmax`'s) returns `i - 1`.

`generic_even_of_ge` is the all-binade statement: any binary32 value of
magnitude `≥ 2^24` has canonical exponent `≥ 1`, hence is an even integer, so
the exact winner `i²` and runner-up `i² - 1` are not both on the grid. That
bound is pipeline-independent; it does not say which way each rounds.

### Numerics above the first binade

`capacity_numerics.py` runs three models in numpy float32 for `i < 12000`:

| model | first failure | universal-failure run | reads that succeed afterwards |
|---|---|---|---|
| A: `fl(i²) = fl(i² - 1)`, one rounding of the exact scores | 4097 | 4097–11586 | 11587, 11597, 11603, … |
| B: the `score32` pipeline, keys and products rounded | 4097 | 4097–5793 | 5794, 5802, 5810, … (every 8) |
| C: numpy float32 `K @ q` (BLAS) | 4097 | 4097–5793 | identical to B |

The theorem covers 4097–5792, the whole first binade (`i² < 2^25`). Model B
keeps failing at 5793 and then succeeds sporadically from 5794, so the
"every address above 4096 fails" reading of the July measurement holds through
the first binade and not beyond it. With grid spacing 4 or more, rounding the
stored `-j²` and the product `2j·i` separately can land the winner above its
neighbour again. Model A ties through 11586 and first separates at 11587, where
`i² ≡ 9 (mod 32)` puts `i² - 1` on a midpoint that resolves downward while `i²`
rounds up; `kernel_tie_11585` and `kernel_separates_11587` are those two bit
patterns checked in the kernel. A theorem for the second binade and up would be
a per-residue-class case analysis and is not attempted.

## Not done

- The read-exactness theorem over `Mem C`, blocked on the argmax lemma above.
- The softmax-vs-argmax agreement bound that `score_gap` was proved for.
- The capacity tie above the first binade (`i ≥ 5793`), where
  `capacity_numerics.py` reports it is no longer universal.
- Any of the 55-opcode ISA. Only addressing is formalized.

## Reproducing

Needs a built TorchLean (`lake exe cache get` then `lake build NN`, ~7.6 GB of
Mathlib oleans and roughly 40 minutes on four cores; the full
`lake build NN NNCI NNExamples NNTests` is 4352 jobs and completed clean).
`Capacity.lean` only needs the float layer:
`lake build NN.Floats.IEEEExec NN.Floats.FP32` (2257 jobs, about 25 minutes
from the Mathlib cache).

```bash
lake env lean LAC/Check.lean   # Mathlib only
cp LAC/Core.lean <torchlean>/LACScratch/Core.lean
cp LAC/Capacity.lean <torchlean>/LACScratch/Capacity.lean
cd <torchlean> && lake env lean LACScratch/Core.lean && lake env lean LACScratch/Capacity.lean
python3 capacity_numerics.py 12000
```

`recheck.py` checks this file against the three Lean sources and the numeric
boundary claims without a toolchain.
