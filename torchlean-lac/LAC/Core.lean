/-
LAC core in TorchLean — scratch.

Goal: state and prove the one theorem the whole llm-as-computer executor rests on,
in TorchLean's spec layer, and connect the memory step to `NN.Spec.Dynamics`.

`score j i = ⟨(2j, -j²), (i, 1)⟩ = -(j - i)² + i²`, a downward parabola centred at `j = i`,
so a hard argmax over stored addresses returns the queried address exactly.
-/
module

import NN

open Spec

namespace LAC

/-! ## Parabolic addressing -/

/-- Key written for memory address `j`: `(2j, -j²)`. -/
def paraKey (j : ℝ) : Spec.Tensor ℝ [2] :=
  Spec.Tensor.dim (fun i : Fin 2 =>
    Spec.Tensor.scalar (if i = 0 then 2 * j else -(j * j)))

/-- Query formed to read memory address `i`: `(i, 1)`. -/
def paraQuery (i : ℝ) : Spec.Tensor ℝ [2] :=
  Spec.Tensor.dim (fun k : Fin 2 =>
    Spec.Tensor.scalar (if k = 0 then i else 1))

/-- The attention score of the key at address `j` under a query for address `i`. -/
def score (j i : ℝ) : ℝ := 2 * j * i - j * j

/-- The parabola identity: the score is `i²` minus the squared address error. -/
theorem score_eq (j i : ℝ) : score j i = i * i - (j - i) * (j - i) := by
  unfold score; ring

/-- Self-score at the queried address. -/
theorem score_self (i : ℝ) : score i i = i * i := by
  unfold score; ring

/-- **Exact addressing.** Any address other than the queried one scores strictly lower.
No tolerance, no temperature: this is why LAC's reads are exact rather than approximate. -/
theorem score_lt_self {i j : ℝ} (h : j ≠ i) : score j i < score i i := by
  rw [score_eq, score_self]
  have hne : j - i ≠ 0 := sub_ne_zero.mpr h
  have hpos : 0 < (j - i) * (j - i) := mul_self_pos.mpr hne
  linarith

/-- Monotone form: scores fall off quadratically with address distance. -/
theorem score_le_self (i j : ℝ) : score j i ≤ score i i := by
  rw [score_eq, score_self]
  nlinarith [mul_self_nonneg (j - i)]

/-! ## Memory as a dynamical system

LAC's context is append-only and grows one row per write. `NN.Spec.Dynamics.DynamicalSystem`
requires a FIXED state shape, so the faithful encoding is a bounded-capacity memory with a
write pointer: state = (capacity × 3) of (key₀, key₁, value) plus a scalar pointer.
That is a `SpecTensor`, and `Dynamics.iterate` then runs the machine.
-/

/-- Memory of capacity `C`: rows of `(2j, -j², value)`. -/
abbrev Mem (C : Nat) := Spec.Tensor ℝ [C, 3]

/-- Scores of every row against a query for address `i`. -/
def rowScores {C : Nat} (m : Mem C) (i : ℝ) : Spec.Tensor ℝ [C] :=
  match m with
  | Spec.Tensor.dim rows =>
      Spec.Tensor.dim (fun r : Fin C =>
        match rows r with
        | Spec.Tensor.dim cells =>
            match cells 0, cells 1 with
            | Spec.Tensor.scalar k0, Spec.Tensor.scalar k1 =>
                Spec.Tensor.scalar (k0 * i + k1))

end LAC
