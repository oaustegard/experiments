import Mathlib.Analysis.SpecialFunctions.Pow.Real

def score (j i : ℝ) : ℝ := 2 * j * i - j * j

theorem score_eq (j i : ℝ) : score j i = i * i - (j - i) * (j - i) := by
  unfold score; ring

theorem score_self (i : ℝ) : score i i = i * i := by
  unfold score; ring

theorem score_lt_self {i j : ℝ} (h : j ≠ i) : score j i < score i i := by
  rw [score_eq, score_self]
  have hne : j - i ≠ 0 := sub_ne_zero.mpr h
  have hpos : 0 < (j - i) * (j - i) := mul_self_pos.mpr hne
  linarith

theorem score_le_self (i j : ℝ) : score j i ≤ score i i := by
  rw [score_eq, score_self]
  nlinarith [mul_self_nonneg (j - i)]

/-- Adjacent integer addresses are separated by at least 1 in score. -/
theorem score_gap (i j : ℤ) (h : j ≠ i) :
    score (j : ℝ) (i : ℝ) + 1 ≤ score (i : ℝ) (i : ℝ) := by
  rw [score_eq, score_self]
  have hne : ((j : ℝ) - (i : ℝ)) ≠ 0 := by
    simpa [sub_eq_zero] using (fun hh => h (by exact_mod_cast hh))
  have h1 : (1 : ℝ) ≤ ((j : ℝ) - (i : ℝ)) * ((j : ℝ) - (i : ℝ)) := by
    have : ((j - i : ℤ) : ℝ) = (j : ℝ) - (i : ℝ) := by push_cast; ring
    have hz : (j - i : ℤ) ≠ 0 := sub_ne_zero.mpr h
    have : (1 : ℤ) ≤ (j - i) * (j - i) := by
      rcases lt_or_gt_of_ne hz with hlt | hgt
      · nlinarith
      · nlinarith
    calc (1:ℝ) = ((1:ℤ) : ℝ) := by norm_num
    _ ≤ (((j - i) * (j - i) : ℤ) : ℝ) := by exact_mod_cast this
    _ = ((j : ℝ) - (i : ℝ)) * ((j : ℝ) - (i : ℝ)) := by push_cast; ring
  linarith
