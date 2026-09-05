# LAC in TorchLean

TorchLean can carry the llm-as-computer executor's spec and proof layers but not
its graph IR. The parabolic-addressing theorem that the whole executor rests on
is four tactic lines and is proved here; the hard-argmax read cannot be lowered
to `NN.IR.Graph`, so autograd and CROWN bounds are unavailable to it.

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

LAC's measured capacity ceiling is `j² > 2^24` in fp32. That is a statement about
the 24-bit binary32 significand, and TorchLean already carries the machinery:
`NN/Floats/IEEEExec/Rounding/RoundDyadicToIEEE32Bounds.lean`,
`NN/Floats/IEEEExec/Exec32/Dyadic.lean`,
`NN/Floats/IEEEExec/Bridge/LeanFloat32.lean` (`2^23 ≤ mant < 2^24`), and
`NN/Floats/IEEEExec/Bridge/FP32/Ulp`. Turning that measured ceiling into a
machine-checked theorem about binary32 is the one thing TorchLean offers LAC
that LAC does not already have, and it is not attempted here.

## Not done

- The read-exactness theorem over `Mem C`, blocked on the argmax lemma above.
- The softmax-vs-argmax agreement bound that `score_gap` was proved for.
- The binary32 capacity theorem.
- Any of the 55-opcode ISA. Only addressing is formalized.

## Reproducing

Needs a built TorchLean (`lake exe cache get` then `lake build NN`, ~7.6 GB of
Mathlib oleans and roughly 40 minutes on four cores; the full
`lake build NN NNCI NNExamples NNTests` is 4352 jobs and completed clean).

```bash
lake env lean LAC/Check.lean   # Mathlib only
cp LAC/Core.lean <torchlean>/LACScratch/Core.lean
cd <torchlean> && lake env lean LACScratch/Core.lean
```

`recheck.py` checks this file against the two Lean sources without a toolchain.
