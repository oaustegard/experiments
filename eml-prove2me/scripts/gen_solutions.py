"""Generate solutions: mirror simp only [eval_node, eval_one, log_one, sub_zero, log_exp, exp_log] to list the
positivity side goals, prove them up front, then discharge by assumption."""
import json, sys, pathlib
sys.path.insert(0, "/home/user/eml-sr/benchmarks")
from eml_complexity import parse_witness
W = "/home/user/claude-workspace/.spokes/prove2me_workspace"
items = json.load(open(f"{W}/eml_items.json"))

class E:  # reduced Lean expression: kind in {one, exp, log, sub, num}
    def __init__(s, kind, *args): s.kind, s.args = kind, args
    def lean(s):
        if s.kind == "one": return "1"
        if s.kind == "exp": return f"Real.exp {s.args[0].lean_atom()}"
        if s.kind == "log": return f"Real.log {s.args[0].lean_atom()}"
        if s.kind == "sub": return f"{s.args[0].lean()} - {s.args[1].lean_atom()}"
    def lean_atom(s):
        t = s.lean(); return t if s.kind == "one" else f"({t})"

facts = []   # reduced expressions that must be > 0 (validity and exp_log side goals)
def reduce(t):
    if t is None: return E("one")
    A, B = reduce(t[0]), reduce(t[1])
    facts.append(B)                      # validity: 0 < eval b
    if A.kind == "log":                  # exp (log X) = X needs 0 < X
        facts.append(A.args[0]); expA = A.args[0]
    else: expA = E("exp", A)
    if B.kind == "one": return expA      # log 1 = 0, sub_zero
    logB = B.args[0] if B.kind == "exp" else E("log", B)
    if logB.kind == "sub" and logB.args[0].lean() == expA.lean():   # sub_sub_cancel: a - (a - b) = b
        return logB.args[1]
    return E("sub", expA, logB)

for name, it in items.items():
    facts.clear()
    tree = parse_witness(it["witness"]) if it["witness"] != "1" else None
    red = reduce(tree)
    seen, hyps = set(), []
    for f in facts:
        s = f.lean()
        if s in seen or s == "1": continue
        seen.add(s); hyps.append(s)
    haves = "\n".join(f"  have : (0:ℝ) < {s} := by first | positivity | linarith | (ring_nf; linarith) | nlinarith [Real.exp_pos 1]" for s in hyps)
    slug = it["theorem_name"].replace(".", "_")
    sol = f"""import Definitions.Def_EmlComplexity
import Mathlib.Analysis.Complex.ExponentialBounds

open EmlComplexity in
theorem solution : Attains ({it['expr']}) {it['size']} := by
  have h₁ := Real.exp_one_gt_d9
  have h₂ := Real.exp_one_lt_d9
  have h₃ := Real.log_two_gt_d9
  have h₄ := Real.log_two_lt_d9
  have h₅ : (7 : ℝ) < Real.exp (Real.exp 1) := by
    have := Real.quadratic_le_exp_of_nonneg (Real.exp_pos 1).le
    nlinarith
{haves}
  refine ⟨{it['tree']}, ?_, rfl, ?_⟩
  · simp (disch := assumption) only [Tree.valid_node, Tree.valid_one, Tree.eval_node, Tree.eval_one,
      Real.log_one, sub_zero, sub_sub_cancel, Real.log_exp, Real.exp_log, true_and, and_true, and_self,
      zero_lt_one, Real.exp_pos, *]
    all_goals first | assumption | positivity | linarith | trace_state
  · simp (disch := assumption) only [Tree.eval_node, Tree.eval_one, Real.log_one, sub_zero, sub_sub_cancel, Real.log_exp, Real.exp_log]
    all_goals first | ring1 | (norm_num; done) | (ring_nf; done) | (ring_nf; norm_num; done) | (rw [Real.exp_neg, Real.exp_log two_pos]; norm_num; done) | (ring_nf; rw [Real.exp_neg, Real.exp_log two_pos]; norm_num; done) | trace_state
"""
    pathlib.Path(f"{W}/Solutions/Sol_{slug}.lean").write_text(sol)
    print(f"{name:>5} {it['size']:2d} facts {len(hyps):2d}  final: {red.lean()[:110]}")
