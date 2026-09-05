/-
LAC's binary32 addressing capacity, as theorems over TorchLean's `FP32` model.

LAC (llm-as-computer) reads memory address `i` by scoring every stored key `(2j, -j²)`
against the query `(i, 1)` and taking a hard argmax. The exact score is
`2·j·i - j² = i² - (i - j)²`, so the queried address wins by a margin of exactly 1 over
its neighbours. Measured on real float32 hardware the read is exact through `i = 4096`
and fails at `i = 4097`, i.e. once `i² > 2^24`.

This file proves that cliff against TorchLean's rounded-real binary32 semantics
(`FP32 = NF binaryRadix fexp32 rnd32`: Flocq-style FLT(-149, 24), round-to-nearest-even),
using the same `FP32` `*` and `+` the rest of TorchLean's error analysis uses:

* `read_exact_below_ceiling`: for `0 ≤ j ≤ i ≤ 4096`, `j ≠ i`, the fp32 score of address
  `j` is strictly below the fp32 score of address `i`. Every intermediate is exact.
* `read_ties_above_ceiling`: for `4097 ≤ i ≤ 5792` (the first binade above `2^24`), the
  fp32 scores of addresses `i - 1` and `i` are equal, so a leftmost-tie argmax returns
  `i - 1` and the read fails.
* `not_both_representable_above_ceiling`: for every `i ≥ 4097`, `i²` and `i² - 1` are not
  both binary32 values, in any binade. Above `2^24` the grid has spacing ≥ 2, so the exact
  winner/runner-up pair can never both survive rounding.

`toReal_roundDyadicToIEEE32_eq_fp32Round` (NN/Floats/IEEEExec/Bridge/FP32/RoundDyadic.lean)
identifies `fp32Round` with the executable bit-level kernel on the finite range, so these are
statements about IEEE-754 binary32, not only about the real-valued abstraction.
-/
module

import NN.Floats.FP32
import NN.Floats.IEEEExec.Bridge.FP32.Core

open TorchLean.Floats

namespace LAC

/-- binary32 round-to-nearest-even on `ℝ`: the rounding `FP32` arithmetic applies after every
primitive operation. Definitionally `TorchLean.Floats.IEEE754.fp32Round`. -/
noncomputable abbrev r32 (x : ℝ) : ℝ :=
  neuralRound (β := binaryRadix) (fexp := fexp32) rnd32 x

/-! ## Rounding facts about the binary32 grid -/

/-- TorchLean's radix power in base 2 is the real power of two. -/
lemma bpow_two (e : ℤ) : neuralBpow binaryRadix e = (2 : ℝ) ^ e := by
  simp [neuralBpow, binaryRadix, NeuralRadix.toReal]

/-- The binary32 exponent function: 24 bits of precision, gradual underflow to `2^-149`. -/
lemma fexp32_eq (e : ℤ) : fexp32 e = max (e - 24) (-149) := rfl

/-- Rounding at a known canonical exponent `c` is nearest-even on the grid of spacing `2^c`. -/
lemma r32_eq_of_cexp (x : ℝ) (c : ℤ) (hc : neuralCexp binaryRadix fexp32 x = c) :
    r32 x = (neuralNearestEven (x / (2 : ℝ) ^ c) : ℝ) * (2 : ℝ) ^ c := by
  simp only [r32, neuralRound, neuralToReal, neuralScaledMantissa, rnd32, hc, bpow_two,
    zpow_neg, div_eq_mul_inv]

/-- A real whose scaled mantissa is already an integer is fixed by rounding. -/
lemma r32_of_scaledMantissa_int (x : ℝ) (n : ℤ)
    (h : neuralScaledMantissa binaryRadix fexp32 x = n) : r32 x = x := by
  have hid : rnd32 (n : ℝ) = n := NeuralValidRnd.id (rnd := rnd32) n
  have hne : neuralBpow binaryRadix (neuralCexp binaryRadix fexp32 x) ≠ 0 :=
    neuralBpow.ne_zero _ _
  have h' : x * (neuralBpow binaryRadix (neuralCexp binaryRadix fexp32 x))⁻¹ = n := by
    simpa [neuralScaledMantissa, neuralBpow.neg_exp] using h
  simp only [r32, neuralRound, neuralToReal, h, hid]
  rw [← h']
  field_simp

/-- `m · 2^k` with `|m| < 2^24` and `0 ≤ k` is a binary32 value: rounding fixes it. This is the
"24-bit significand" fact in the form the score computation needs. -/
theorem r32_dyadic_lt (m : ℤ) (k : ℕ) (hm : |m| < 2 ^ 24) :
    r32 ((m : ℝ) * 2 ^ k) = (m : ℝ) * 2 ^ k := by
  by_cases hm0 : m = 0
  · subst hm0
    simp only [Int.cast_zero, zero_mul]
    exact r32_of_scaledMantissa_int 0 0 (by simp [neuralScaledMantissa])
  · have hm0' : (m : ℝ) ≠ 0 := by exact_mod_cast hm0
    have hx0 : (m : ℝ) * 2 ^ k ≠ 0 := mul_ne_zero hm0' (by positivity)
    have hmR : |(m : ℝ)| < 2 ^ 24 := by exact_mod_cast hm
    have hupper : |(m : ℝ) * 2 ^ k| < neuralBpow binaryRadix (24 + k) := by
      have hb : neuralBpow binaryRadix (24 + k) = (2 : ℝ) ^ 24 * 2 ^ k := by
        rw [bpow_two, zpow_add₀ two_ne_zero, zpow_natCast]
        norm_num
      rw [hb, abs_mul, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ k)]
      exact mul_lt_mul_of_pos_right hmR (by positivity)
    have hmag := neuralMagnitude_le_of_abs_lt_bpow binaryRadix _ (24 + k) hx0 hupper
    have hc : neuralCexp binaryRadix fexp32 ((m : ℝ) * 2 ^ k) ≤ k := by
      unfold neuralCexp; rw [fexp32_eq]; omega
    generalize hcdef : neuralCexp binaryRadix fexp32 ((m : ℝ) * 2 ^ k) = c at hc
    obtain ⟨d, hd⟩ : ∃ d : ℕ, (k : ℤ) - c = d := Int.eq_ofNat_of_zero_le (by omega)
    apply r32_of_scaledMantissa_int _ (m * 2 ^ d)
    unfold neuralScaledMantissa
    rw [hcdef, bpow_two]
    push_cast
    have h2 : (2 : ℝ) ^ k * (2 : ℝ) ^ (-c) = 2 ^ d := by
      rw [← zpow_natCast, ← zpow_add₀ two_ne_zero, ← zpow_natCast (2 : ℝ) d,
        show (k : ℤ) + -c = d by omega]
    rw [mul_assoc, h2]

/-- Same as `r32_dyadic_lt` with `|m| ≤ 2^24`: the boundary case `±2^24 = ±2^23 · 2`. -/
theorem r32_dyadic (m : ℤ) (k : ℕ) (hm : |m| ≤ 2 ^ 24) :
    r32 ((m : ℝ) * 2 ^ k) = (m : ℝ) * 2 ^ k := by
  rcases lt_or_eq_of_le hm with hlt | heq
  · exact r32_dyadic_lt m k hlt
  · rcases abs_eq (by norm_num : (0 : ℤ) ≤ 2 ^ 24) |>.mp heq with h | h
    · subst h
      have := r32_dyadic_lt (2 ^ 23) (k + 1) (by norm_num)
      rwa [show ((2 ^ 23 : ℤ) : ℝ) * 2 ^ (k + 1) = ((2 ^ 24 : ℤ) : ℝ) * 2 ^ k by push_cast; ring]
        at this
    · subst h
      have := r32_dyadic_lt (-2 ^ 23) (k + 1) (by norm_num)
      rwa [show ((-2 ^ 23 : ℤ) : ℝ) * 2 ^ (k + 1) = ((-2 ^ 24 : ℤ) : ℝ) * 2 ^ k by push_cast; ring]
        at this

/-- Every integer of magnitude at most `2^24` is a binary32 value. -/
theorem r32_int (n : ℤ) (hn : |n| ≤ 2 ^ 24) : r32 n = n := by
  simpa using r32_dyadic n 0 hn

/-- Nearest-even on a half-integer: the tie goes to the even neighbour. -/
lemma nearestEven_add_half (m : ℤ) :
    neuralNearestEven ((m : ℝ) + 1 / 2) = if Even m then m else m + 1 := by
  have hfl : ⌊(m : ℝ) + 1 / 2⌋ = m := by
    rw [Int.floor_eq_iff]
    constructor <;> linarith
  have h1 : ¬ ((m : ℝ) + 1 / 2 - (m : ℝ) < 1 / 2) := by norm_num
  have h2 : ¬ ((m : ℝ) + 1 / 2 - (m : ℝ) > 1 / 2) := by norm_num
  simp only [neuralNearestEven, hfl]
  rw [if_neg h1, if_neg h2]

/-- The canonical exponent of a real in the binade `[2^(e-1), 2^e)`, for `e ≥ 25` (above the
subnormal range), is `e - 24`: the grid spacing there is `2^(e-24)`. -/
lemma cexp_of_binade (x : ℝ) (e : ℤ) (hx : x ≠ 0)
    (hlo : (2 : ℝ) ^ (e - 1) ≤ |x|) (hhi : |x| < (2 : ℝ) ^ e) (he : 25 ≤ e) :
    neuralCexp binaryRadix fexp32 x = e - 24 := by
  unfold neuralCexp
  rw [neuralMagnitude_eq_of_bpow_bounds binaryRadix x e hx (by rwa [bpow_two]) (by rwa [bpow_two]),
    fexp32_eq]
  omega

/-! ## The first binade above `2^24`

For an odd `s` with `2^24 < s² < 2^25` the grid spacing is 2, `s²` is odd, and the tie resolves
to the even significand, which is `s² - 1` because `(s² - 1)/2 = 2t(t+1)` is even. -/

/-- For odd `s` with `s²` in `(2^24, 2^25)`: `fl(s²) = s² - 1`. -/
theorem r32_odd_sq (s : ℤ) (hs : Odd s) (hlo : 2 ^ 24 < s * s) (hhi : s * s < 2 ^ 25) :
    r32 ((s : ℝ) * s) = (s : ℝ) * s - 1 := by
  obtain ⟨t, rfl⟩ := hs
  set m : ℤ := 2 * t * (t + 1) with hm
  have hsq : (((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ)) = 2 * (m : ℝ) + 1 := by
    rw [hm]; push_cast; ring
  have hpos : (0 : ℝ) < ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by
    have : (2 ^ 24 : ℝ) < ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by exact_mod_cast hlo
    linarith
  have hloR : (2 ^ 24 : ℝ) < ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by exact_mod_cast hlo
  have hhiR : ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) < (2 ^ 25 : ℝ) := by exact_mod_cast hhi
  have hc : neuralCexp binaryRadix fexp32 (((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ)) = 1 := by
    refine (cexp_of_binade _ 25 hpos.ne' ?_ ?_ le_rfl).trans (by norm_num)
    · rw [abs_of_pos hpos]; push_cast at hloR ⊢; nlinarith
    · rw [abs_of_pos hpos]; push_cast at hhiR ⊢; nlinarith
  rw [r32_eq_of_cexp _ 1 hc, hsq]
  have hhalf : (2 * (m : ℝ) + 1) / (2 : ℝ) ^ (1 : ℤ) = (m : ℝ) + 1 / 2 := by
    rw [zpow_one]; ring
  have heven : Even m := ⟨t * (t + 1), by rw [hm]; ring⟩
  rw [hhalf, nearestEven_add_half, if_pos heven, zpow_one]
  ring

/-- For odd `s` with `s²` in `(2^24, 2^25)`: `fl(-s²) = -(s² - 1)`. -/
theorem r32_neg_odd_sq (s : ℤ) (hs : Odd s) (hlo : 2 ^ 24 < s * s) (hhi : s * s < 2 ^ 25) :
    r32 (-((s : ℝ) * s)) = -((s : ℝ) * s - 1) := by
  obtain ⟨t, rfl⟩ := hs
  set m : ℤ := 2 * t * (t + 1) with hm
  have hsq : (((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ)) = 2 * (m : ℝ) + 1 := by
    rw [hm]; push_cast; ring
  have hpos : (0 : ℝ) < ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by
    have : (2 ^ 24 : ℝ) < ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by exact_mod_cast hlo
    linarith
  have hloR : (2 ^ 24 : ℝ) < ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by exact_mod_cast hlo
  have hhiR : ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) < (2 ^ 25 : ℝ) := by exact_mod_cast hhi
  have hc : neuralCexp binaryRadix fexp32 (-(((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ))) = 1 := by
    refine (cexp_of_binade _ 25 (neg_ne_zero.mpr hpos.ne') ?_ ?_ le_rfl).trans (by norm_num)
    · rw [abs_neg, abs_of_pos hpos]; push_cast at hloR ⊢; nlinarith
    · rw [abs_neg, abs_of_pos hpos]; push_cast at hhiR ⊢; nlinarith
  rw [r32_eq_of_cexp _ 1 hc, hsq]
  have hhalf : (-(2 * (m : ℝ) + 1)) / (2 : ℝ) ^ (1 : ℤ) = ((-m - 1 : ℤ) : ℝ) + 1 / 2 := by
    rw [zpow_one]; push_cast; ring
  have hodd : ¬ Even (-m - 1) := by
    rw [Int.not_even_iff_odd]
    exact ⟨-t * (t + 1) - 1, by rw [hm]; ring⟩
  rw [hhalf, nearestEven_add_half, if_neg hodd, zpow_one]
  push_cast
  ring

/-- For odd `s` with `s²` in `(2^24, 2^25)`: `fl(2s²) = 2s² - 2` (spacing 4, same tie). -/
theorem r32_two_odd_sq (s : ℤ) (hs : Odd s) (hlo : 2 ^ 24 < s * s) (hhi : s * s < 2 ^ 25) :
    r32 (2 * (s : ℝ) * s) = 2 * (s : ℝ) * s - 2 := by
  obtain ⟨t, rfl⟩ := hs
  set m : ℤ := 2 * t * (t + 1) with hm
  have hsq : 2 * ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) = 4 * (m : ℝ) + 2 := by
    rw [hm]; push_cast; ring
  have hpos : (0 : ℝ) < 2 * ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by
    have : (2 ^ 24 : ℝ) < ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by exact_mod_cast hlo
    linarith
  have hloR : (2 ^ 24 : ℝ) < ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) := by exact_mod_cast hlo
  have hhiR : ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ) < (2 ^ 25 : ℝ) := by exact_mod_cast hhi
  have hc : neuralCexp binaryRadix fexp32 (2 * ((2 * t + 1 : ℤ) : ℝ) * ((2 * t + 1 : ℤ) : ℝ)) = 2 := by
    refine (cexp_of_binade _ 26 hpos.ne' ?_ ?_ (by norm_num)).trans (by norm_num)
    · rw [abs_of_pos hpos]; push_cast at hloR ⊢; nlinarith
    · rw [abs_of_pos hpos]; push_cast at hhiR ⊢; nlinarith
  rw [r32_eq_of_cexp _ 2 hc, hsq]
  have hhalf : (4 * (m : ℝ) + 2) / (2 : ℝ) ^ (2 : ℤ) = (m : ℝ) + 1 / 2 := by
    norm_num; ring
  have heven : Even m := ⟨t * (t + 1), by rw [hm]; ring⟩
  rw [hhalf, nearestEven_add_half, if_pos heven]
  norm_num; ring

/-! ## The LAC read in `FP32`

The executor stores the key row `(2j, -j²)` for address `j` and reads address `i` with the
query `(i, 1)`. Each stored entry is a rounded float32; the score is the float32 dot product
`fl(fl(k₀ · i) + k₁)` (the second query component is exactly 1). These are TorchLean's
`FP32` operations: `*` and `+` round after every step. -/

/-- The float32 key row written at address `j`. -/
noncomputable def key (j : ℤ) : FP32 × FP32 :=
  (NF.ofReal (2 * (j : ℝ)), NF.ofReal (-((j : ℝ) * j)))

/-- The float32 query for address `i`. -/
noncomputable def query (i : ℤ) : FP32 := NF.ofReal (i : ℝ)

/-- The float32 score of stored address `j` under a read of address `i`. -/
noncomputable def score32 (j i : ℤ) : ℝ := ((key j).1 * query i + (key j).2).val

/-- The score unfolds to three roundings around exact real arithmetic. -/
theorem score32_eq (j i : ℤ) :
    score32 j i = r32 (r32 (r32 (2 * (j : ℝ)) * r32 (i : ℝ)) + r32 (-((j : ℝ) * j))) := rfl

/-! ## Below the ceiling: every intermediate is exact -/

/-- For `0 ≤ j ≤ i ≤ 4096` the float32 score equals the exact integer score `2ji - j²`. -/
theorem score32_exact (j i : ℤ) (hj : 0 ≤ j) (hji : j ≤ i) (hi : i ≤ 4096) :
    score32 j i = 2 * (j : ℝ) * i - (j : ℝ) * j := by
  rw [score32_eq]
  have hiR : (i : ℝ) ≤ 4096 := by exact_mod_cast hi
  have hjR : (0 : ℝ) ≤ j := by exact_mod_cast hj
  have hjiR : (j : ℝ) ≤ i := by exact_mod_cast hji
  have h2j : r32 (2 * (j : ℝ)) = 2 * j := by
    have := r32_int (2 * j) (by rw [abs_le]; omega)
    simpa using this
  have hi' : r32 (i : ℝ) = i := r32_int i (by rw [abs_le]; omega)
  have hjj : r32 (-((j : ℝ) * j)) = -((j : ℝ) * j) := by
    have := r32_int (-(j * j)) (by rw [abs_le]; constructor <;> nlinarith)
    simpa using this
  have hprod : r32 (2 * (j : ℝ) * i) = 2 * (j : ℝ) * i := by
    have := r32_dyadic (j * i) 1 (by rw [abs_le]; constructor <;> nlinarith)
    rwa [show ((j * i : ℤ) : ℝ) * 2 ^ 1 = 2 * (j : ℝ) * i by push_cast; ring] at this
  have hsum : r32 (2 * (j : ℝ) * i + -((j : ℝ) * j)) = 2 * (j : ℝ) * i - (j : ℝ) * j := by
    have := r32_int (2 * j * i - j * j) (by rw [abs_le]; constructor <;> nlinarith)
    rw [show ((2 * j * i - j * j : ℤ) : ℝ) = 2 * (j : ℝ) * i + -((j : ℝ) * j) by push_cast; ring]
      at this
    rw [this]; ring
  rw [h2j, hi', hprod, hjj, hsum]

/-- **Exact addressing below the ceiling.** For `0 ≤ j ≤ i ≤ 4096` and `j ≠ i`, the float32
score of address `j` is strictly below that of the queried address `i`: a hard argmax over the
float32 scores returns `i`. -/
theorem read_exact_below_ceiling (j i : ℤ) (hj : 0 ≤ j) (hji : j ≤ i) (hi : i ≤ 4096)
    (hne : j ≠ i) : score32 j i < score32 i i := by
  rw [score32_exact j i hj hji hi, score32_exact i i (by omega) le_rfl hi]
  have hneR : (j : ℝ) ≠ i := by exact_mod_cast hne
  have hpos : 0 < ((i : ℝ) - j) * ((i : ℝ) - j) := mul_self_pos.mpr (sub_ne_zero.mpr hneR.symm)
  nlinarith

/-! ## Above the ceiling: the neighbour ties -/

/-- **The read fails in the first binade above `2^24`.** For `4097 ≤ i ≤ 5792` the float32 scores
of addresses `i - 1` and `i` coincide. A leftmost-tie argmax (LAC's recency convention, and
`Spec.Tensor.argmax`'s) therefore returns `i - 1`, not `i`. -/
theorem read_ties_above_ceiling (i : ℤ) (hlo : 4097 ≤ i) (hhi : i ≤ 5792) :
    score32 (i - 1) i = score32 i i := by
  rw [score32_eq, score32_eq]
  push_cast
  have hiR : r32 (i : ℝ) = i := r32_int i (by rw [abs_le]; omega)
  rw [hiR]
  rcases Int.even_or_odd i with ⟨t, ht⟩ | ⟨t, ht⟩
  · -- `i = 2t` even: score i = 4t² exactly; address `2t - 1` is odd and its `-j²` rounds up by one,
    -- which the exact product `2j·i` then cancels.
    have ht' : (i : ℝ) = 2 * t := by rw [ht]; push_cast; ring
    have htlo : 2049 ≤ t := by omega
    have hthi : t ≤ 2896 := by omega
    rw [ht']
    -- score at `i`
    have ha : r32 (2 * (2 * (t : ℝ))) = 2 * (2 * (t : ℝ)) := by
      have := r32_int (4 * t) (by rw [abs_le]; omega)
      rwa [show ((4 * t : ℤ) : ℝ) = 2 * (2 * (t : ℝ)) by push_cast; ring] at this
    have hc : r32 (2 * (2 * (t : ℝ)) * (2 * t)) = 2 * (2 * (t : ℝ)) * (2 * t) := by
      have := r32_dyadic (t * t) 3 (by rw [abs_le]; constructor <;> nlinarith)
      rwa [show ((t * t : ℤ) : ℝ) * 2 ^ 3 = 2 * (2 * (t : ℝ)) * (2 * t) by push_cast; ring] at this
    have hd : r32 (-(2 * (t : ℝ) * (2 * t))) = -(2 * (t : ℝ) * (2 * t)) := by
      have := r32_dyadic (-(t * t)) 2 (by rw [abs_le]; constructor <;> nlinarith)
      rwa [show ((-(t * t) : ℤ) : ℝ) * 2 ^ 2 = -(2 * (t : ℝ) * (2 * t)) by push_cast; ring] at this
    have hsi : r32 (2 * (2 * (t : ℝ)) * (2 * t) + -(2 * (t : ℝ) * (2 * t))) = 4 * (t : ℝ) * t := by
      have := r32_dyadic (t * t) 2 (by rw [abs_le]; constructor <;> nlinarith)
      rw [show 2 * (2 * (t : ℝ)) * (2 * t) + -(2 * (t : ℝ) * (2 * t)) = ((t * t : ℤ) : ℝ) * 2 ^ 2 by
        push_cast; ring, this]
      push_cast; ring
    -- score at `i - 1 = 2t - 1`
    have hodd : Odd (2 * t - 1) := ⟨t - 1, by ring⟩
    have hjlo : 2 ^ 24 < (2 * t - 1) * (2 * t - 1) := by nlinarith
    have hjhi : (2 * t - 1) * (2 * t - 1) < 2 ^ 25 := by nlinarith
    have ha' : r32 (2 * (2 * (t : ℝ) - 1)) = 2 * (2 * (t : ℝ) - 1) := by
      have := r32_int (2 * (2 * t - 1)) (by rw [abs_le]; omega)
      rwa [show ((2 * (2 * t - 1) : ℤ) : ℝ) = 2 * (2 * (t : ℝ) - 1) by push_cast; ring] at this
    have hc' : r32 (2 * (2 * (t : ℝ) - 1) * (2 * t)) = 2 * (2 * (t : ℝ) - 1) * (2 * t) := by
      have := r32_dyadic (2 * t * t - t) 2 (by rw [abs_le]; constructor <;> nlinarith)
      rwa [show ((2 * t * t - t : ℤ) : ℝ) * 2 ^ 2 = 2 * (2 * (t : ℝ) - 1) * (2 * t) by
        push_cast; ring] at this
    have hd' : r32 (-((2 * (t : ℝ) - 1) * (2 * t - 1))) = -((2 * (t : ℝ) - 1) * (2 * t - 1) - 1) := by
      have := r32_neg_odd_sq (2 * t - 1) hodd hjlo hjhi
      push_cast at this
      exact this
    have hsj : r32 (2 * (2 * (t : ℝ) - 1) * (2 * t) + -((2 * (t : ℝ) - 1) * (2 * t - 1) - 1)) =
        4 * (t : ℝ) * t := by
      have := r32_dyadic (t * t) 2 (by rw [abs_le]; constructor <;> nlinarith)
      rw [show 2 * (2 * (t : ℝ) - 1) * (2 * t) + -((2 * (t : ℝ) - 1) * (2 * t - 1) - 1) =
        ((t * t : ℤ) : ℝ) * 2 ^ 2 by push_cast; ring, this]
      push_cast; ring
    rw [ha, hc, hd, hsi, ha', hc', hd', hsj]
  · -- `i = 2t + 1` odd: both `2i²` and `-i²` round toward `i² - 1`; address `2t` is exact.
    have ht' : (i : ℝ) = 2 * t + 1 := by rw [ht]; push_cast; ring
    have htlo : 2048 ≤ t := by omega
    have hthi : t ≤ 2895 := by omega
    rw [ht']
    have hodd : Odd i := ⟨t, ht⟩
    have hilo : 2 ^ 24 < i * i := by nlinarith
    have hihi : i * i < 2 ^ 25 := by nlinarith
    -- score at `i`
    have ha : r32 (2 * (2 * (t : ℝ) + 1)) = 2 * (2 * (t : ℝ) + 1) := by
      have := r32_int (2 * (2 * t + 1)) (by rw [abs_le]; omega)
      rwa [show ((2 * (2 * t + 1) : ℤ) : ℝ) = 2 * (2 * (t : ℝ) + 1) by push_cast; ring] at this
    have hc : r32 (2 * (2 * (t : ℝ) + 1) * (2 * t + 1)) = 2 * (2 * (t : ℝ) + 1) * (2 * t + 1) - 2 := by
      have := r32_two_odd_sq i hodd hilo hihi
      rw [ht'] at this
      exact this
    have hd : r32 (-((2 * (t : ℝ) + 1) * (2 * t + 1))) = -((2 * (t : ℝ) + 1) * (2 * t + 1) - 1) := by
      have := r32_neg_odd_sq i hodd hilo hihi
      rw [ht'] at this
      exact this
    have hsi : r32 (2 * (2 * (t : ℝ) + 1) * (2 * t + 1) - 2 + -((2 * (t : ℝ) + 1) * (2 * t + 1) - 1)) =
        4 * (t : ℝ) * t + 4 * t := by
      have := r32_dyadic (2 * t * (t + 1)) 1 (by rw [abs_le]; constructor <;> nlinarith)
      rw [show 2 * (2 * (t : ℝ) + 1) * (2 * t + 1) - 2 + -((2 * (t : ℝ) + 1) * (2 * t + 1) - 1) =
        ((2 * t * (t + 1) : ℤ) : ℝ) * 2 ^ 1 by push_cast; ring, this]
      push_cast; ring
    -- score at `i - 1 = 2t`
    have ha' : r32 (2 * (2 * (t : ℝ) + 1 - 1)) = 2 * (2 * (t : ℝ) + 1 - 1) := by
      have := r32_int (4 * t) (by rw [abs_le]; omega)
      rwa [show ((4 * t : ℤ) : ℝ) = 2 * (2 * (t : ℝ) + 1 - 1) by push_cast; ring] at this
    have hc' : r32 (2 * (2 * (t : ℝ) + 1 - 1) * (2 * t + 1)) =
        2 * (2 * (t : ℝ) + 1 - 1) * (2 * t + 1) := by
      have := r32_dyadic (2 * t * t + t) 2 (by rw [abs_le]; constructor <;> nlinarith)
      rwa [show ((2 * t * t + t : ℤ) : ℝ) * 2 ^ 2 = 2 * (2 * (t : ℝ) + 1 - 1) * (2 * t + 1) by
        push_cast; ring] at this
    have hd' : r32 (-((2 * (t : ℝ) + 1 - 1) * (2 * t + 1 - 1))) =
        -((2 * (t : ℝ) + 1 - 1) * (2 * t + 1 - 1)) := by
      have := r32_dyadic (-(t * t)) 2 (by rw [abs_le]; constructor <;> nlinarith)
      rwa [show ((-(t * t) : ℤ) : ℝ) * 2 ^ 2 = -((2 * (t : ℝ) + 1 - 1) * (2 * t + 1 - 1)) by
        push_cast; ring] at this
    have hsj : r32 (2 * (2 * (t : ℝ) + 1 - 1) * (2 * t + 1) + -((2 * (t : ℝ) + 1 - 1) * (2 * t + 1 - 1))) =
        4 * (t : ℝ) * t + 4 * t := by
      have := r32_dyadic (2 * t * (t + 1)) 1 (by rw [abs_le]; constructor <;> nlinarith)
      rw [show 2 * (2 * (t : ℝ) + 1 - 1) * (2 * t + 1) + -((2 * (t : ℝ) + 1 - 1) * (2 * t + 1 - 1)) =
        ((2 * t * (t + 1) : ℤ) : ℝ) * 2 ^ 1 by push_cast; ring, this]
      push_cast; ring
    rw [ha, hc, hd, hsi, ha', hc', hd', hsj]

/-- **The capacity ceiling is `i² ≤ 2^24`, i.e. `i ≤ 4096`, and it is sharp.** -/
theorem capacity_ceiling :
    (∀ j i : ℤ, 0 ≤ j → j ≤ i → i ≤ 4096 → j ≠ i → score32 j i < score32 i i) ∧
    (∀ i : ℤ, 4097 ≤ i → i ≤ 5792 → score32 (i - 1) i = score32 i i) :=
  ⟨read_exact_below_ceiling, read_ties_above_ceiling⟩

/-! ## Every binade above `2^24`: the winner/runner-up pair cannot both be represented -/

/-- A binary32 value of magnitude at least `2^24` is an even integer: its canonical exponent is
at least 1. -/
theorem generic_even_of_ge (x : ℝ) (hx : (2 : ℝ) ^ 24 ≤ x)
    (hg : neuralGenericFormat binaryRadix fexp32 x) : ∃ n : ℤ, x = 2 * n := by
  have hx0 : x ≠ 0 := by positivity
  have hmag : 25 ≤ neuralMagnitude binaryRadix x := by
    have hs := (neuralMagnitude_spec binaryRadix x hx0).2
    rw [bpow_two, abs_of_pos (by positivity)] at hs
    have h24 : (2 : ℝ) ^ (24 : ℤ) < (2 : ℝ) ^ neuralMagnitude binaryRadix x := by
      calc (2 : ℝ) ^ (24 : ℤ) = (2 : ℝ) ^ (24 : ℕ) := by norm_cast
        _ ≤ x := hx
        _ < _ := hs
    have := (zpow_lt_zpow_iff_right₀ (by norm_num : (1 : ℝ) < 2)).mp h24
    omega
  have hc : 1 ≤ neuralCexp binaryRadix fexp32 x := by
    unfold neuralCexp; rw [fexp32_eq]; omega
  set n : ℤ := ⌊neuralScaledMantissa binaryRadix fexp32 x⌋ with hn
  set c : ℤ := neuralCexp binaryRadix fexp32 x with hcdef
  have hg' : x = (n : ℝ) * neuralBpow binaryRadix c := hg
  obtain ⟨d, hd⟩ : ∃ d : ℕ, c - 1 = d := Int.eq_ofNat_of_zero_le (by omega)
  refine ⟨n * 2 ^ d, ?_⟩
  rw [hg', bpow_two, show c = (d : ℤ) + 1 by omega, zpow_add₀ two_ne_zero, zpow_one, zpow_natCast]
  push_cast
  ring

/-- **Above the ceiling no rounding can separate the winner from its neighbour.** For every
`i ≥ 4097`, in any binade, the exact winning score `i²` and the runner-up score `i² - 1` are not
both binary32 values: the grid above `2^24` has spacing at least 2. -/
theorem not_both_representable_above_ceiling (i : ℤ) (hi : 4097 ≤ i) :
    ¬ (neuralGenericFormat binaryRadix fexp32 ((i : ℝ) * i) ∧
       neuralGenericFormat binaryRadix fexp32 ((i : ℝ) * i - 1)) := by
  rintro ⟨h1, h2⟩
  have hiR : (4097 : ℝ) ≤ i := by exact_mod_cast hi
  obtain ⟨a, ha⟩ := generic_even_of_ge _ (by norm_num; nlinarith) h1
  obtain ⟨b, hb⟩ := generic_even_of_ge _ (by norm_num; nlinarith) h2
  have h : ((1 : ℤ) : ℝ) = 2 * ((a : ℝ) - b) := by push_cast; linarith
  have h' : (1 : ℤ) = 2 * (a - b) := by exact_mod_cast h
  omega

/-! ## The executable kernel agrees

`IEEE32Exec.roundDyadicToIEEE32` is TorchLean's bit-level binary32 rounder, and
`toReal_roundDyadicToIEEE32_eq_fp32Round` proves it agrees with `fp32Round` on the finite range.
The kernel is executable, so the boundary instances reduce by `decide` inside Lean's kernel
(no native evaluator, no additional axioms). -/

section Kernel

open TorchLean.Floats.IEEE754 IEEE32Exec

/-- Bit-level: `4097²` and `4097² - 1` round to the same binary32 word. -/
theorem kernel_tie_4097 :
    roundDyadicToIEEE32 ⟨false, 4097 * 4097, 0⟩ = roundDyadicToIEEE32 ⟨false, 4097 * 4097 - 1, 0⟩ := by
  decide

/-- Bit-level: `4096²` and `4096² - 1` are distinct binary32 words. -/
theorem kernel_distinct_4096 :
    roundDyadicToIEEE32 ⟨false, 4096 * 4096, 0⟩ ≠ roundDyadicToIEEE32 ⟨false, 4096 * 4096 - 1, 0⟩ := by
  decide

/-- Bit-level: the exact-then-round tie persists through `i = 11585` (`i² < 2^27`). -/
theorem kernel_tie_11585 :
    roundDyadicToIEEE32 ⟨false, 11585 * 11585, 0⟩ =
      roundDyadicToIEEE32 ⟨false, 11585 * 11585 - 1, 0⟩ := by
  decide

/-- Bit-level: `i = 11587` is the first address where nearest-even separates `i²` from `i² - 1`
again (`i² ≡ 9 mod 32` puts `i² - 1` on a midpoint that resolves downward while `i²` rounds up).
Above `2^27` the exact-then-round tie is therefore sporadic, not universal; see RESULTS.md for
what the full key-rounding pipeline does there. -/
theorem kernel_separates_11587 :
    roundDyadicToIEEE32 ⟨false, 11587 * 11587, 0⟩ ≠
      roundDyadicToIEEE32 ⟨false, 11587 * 11587 - 1, 0⟩ := by
  decide

end Kernel

end LAC

#print axioms LAC.read_exact_below_ceiling
#print axioms LAC.read_ties_above_ceiling
#print axioms LAC.not_both_representable_above_ceiling
#print axioms LAC.kernel_tie_4097
#print axioms LAC.kernel_separates_11587
