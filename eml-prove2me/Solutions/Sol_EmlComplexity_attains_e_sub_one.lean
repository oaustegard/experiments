import Definitions.Def_EmlComplexity
import Mathlib.Analysis.Complex.ExponentialBounds

open EmlComplexity in
theorem solution : Attains (Real.exp 1 - 1) 2 := by
  have h₁ := Real.exp_one_gt_d9
  have h₂ := Real.exp_one_lt_d9
  have h₃ := Real.log_two_gt_d9
  have h₄ := Real.log_two_lt_d9
  have h₅ : (7 : ℝ) < Real.exp (Real.exp 1) := by
    have := Real.quadratic_le_exp_of_nonneg (Real.exp_pos 1).le
    nlinarith
  have : (0:ℝ) < Real.exp 1 := by first | positivity | linarith | (ring_nf; linarith) | nlinarith [Real.exp_pos 1]
  refine ⟨(.node .one (.node .one .one)), ?_, rfl, ?_⟩
  · simp (disch := assumption) only [Tree.valid_node, Tree.valid_one, Tree.eval_node, Tree.eval_one,
      Real.log_one, sub_zero, sub_sub_cancel, Real.log_exp, Real.exp_log, true_and, and_true, and_self,
      zero_lt_one, Real.exp_pos, *]
    all_goals first | assumption | positivity | linarith | trace_state
  · simp (disch := assumption) only [Tree.eval_node, Tree.eval_one, Real.log_one, sub_zero, sub_sub_cancel, Real.log_exp, Real.exp_log]
    all_goals first | ring1 | (norm_num; done) | (ring_nf; done) | (ring_nf; norm_num; done) | (rw [Real.exp_neg, Real.exp_log two_pos]; norm_num; done) | (ring_nf; rw [Real.exp_neg, Real.exp_log two_pos]; norm_num; done) | trace_state
