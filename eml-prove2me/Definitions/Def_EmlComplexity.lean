import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# EML trees and the EML complexity of a real constant

The single binary operator `eml(x, y) = exp(x) - ln(y)` together with the
constant `1` generates all elementary functions (Odrzywolek, "All elementary
functions from a single operator", arXiv:2603.21852, 2026). A *closed EML tree*
is a full binary tree whose leaves are the constant `1` and whose internal
nodes are `eml`. Its *size* is the number of `eml` nodes. On the real branch
every logarithm must be taken of a positive real; a tree satisfying that at
every node is *valid*. The *EML complexity* of a real `c` is the least size of a
valid closed tree evaluating exactly to `c`.
-/

namespace EmlComplexity

/-- A closed EML tree: the leaf is the constant `1`, `node a b` denotes `eml(a, b)`. -/
inductive Tree
  | one : Tree
  | node : Tree → Tree → Tree
  deriving DecidableEq, Repr

namespace Tree

/-- Number of `eml` nodes. -/
def size : Tree → ℕ
  | one => 0
  | node a b => a.size + b.size + 1

/-- Real-branch evaluation. `Real.log` is total in Mathlib, so a tree that takes
the logarithm of a non-positive real still evaluates; `valid` rules those out. -/
noncomputable def eval : Tree → ℝ
  | one => 1
  | node a b => Real.exp a.eval - Real.log b.eval

/-- Every logarithm in the tree is taken of a positive real. -/
def valid : Tree → Prop
  | one => True
  | node a b => a.valid ∧ b.valid ∧ 0 < b.eval

@[simp] theorem size_one : size one = 0 := rfl
@[simp] theorem size_node (a b : Tree) : size (node a b) = a.size + b.size + 1 := rfl
@[simp] theorem eval_one : eval one = 1 := rfl
@[simp] theorem eval_node (a b : Tree) : eval (node a b) = Real.exp a.eval - Real.log b.eval := rfl
@[simp] theorem valid_one : valid one := trivial
@[simp] theorem valid_node (a b : Tree) : valid (node a b) ↔ a.valid ∧ b.valid ∧ 0 < b.eval := Iff.rfl

end Tree

/-- `c` is the value of some valid closed EML tree with exactly `n` nodes. -/
def Attains (c : ℝ) (n : ℕ) : Prop := ∃ t : Tree, t.valid ∧ t.size = n ∧ t.eval = c

/-- The EML complexity of `c` is exactly `n`: attained at `n` and at no smaller size. -/
def Complexity (c : ℝ) (n : ℕ) : Prop := Attains c n ∧ ∀ m, m < n → ¬ Attains c m

end EmlComplexity
