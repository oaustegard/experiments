# EML complexity of constants as a Prove2Me mission

The witness half of `oaustegard/eml-sr` `benchmarks/eml_complexity.md` (real branch), formalised in Lean 4 and verified on [Prove2Me](https://prove2.me) on 2026-09-05, plus the lower bounds posted as open theorems and a mission proposal built on top. Environment: Lean v4.33.1, Mathlib `0df444a`.

## What is on the platform

Definition `EmlComplexity` (id `4356d2c2-a5af-4cb4-b610-0a2b890f11c8`): the inductive `Tree` (`one | node`), `size`, real-branch `eval` with Mathlib's total `Real.log`, the `valid` predicate (every logarithm of a positive real), `Attains c n` and `Complexity c n`.

| theorem | id | status |
|---|---|---|
| `EmlComplexity.attains_e` | f3535c20-7272-4d81-8dfd-2db0e6069425 | ACCEPTED |
| `EmlComplexity.attains_e_sub_one` | 21d4632c-20ec-4359-9f7f-35a0f892180f | ACCEPTED |
| `EmlComplexity.attains_zero` | eb5c6937-e5e3-4ca5-8466-ae63ade36d41 | ACCEPTED |
| `EmlComplexity.attains_e_sub_two` | b5e196a4-c270-43ac-bfef-948524c86db5 | ACCEPTED |
| `EmlComplexity.attains_neg_one` | 61d4901a-2e6e-4784-a918-71014171d527 | ACCEPTED |
| `EmlComplexity.attains_two` | f916205d-b1cc-473a-b26f-f06abaae8875 | ACCEPTED |
| `EmlComplexity.attains_log_two` | 7f2bdfd8-d181-432c-9305-14f642a9aa26 | ACCEPTED |
| `EmlComplexity.attains_three` | 9c72d39e-4bc0-467a-b57d-3b2b07c00a79 | ACCEPTED |
| `EmlComplexity.attains_half` | 53b8878c-024c-4eee-8a16-bf358c59d2a2 | ACCEPTED |
| `EmlComplexity.attains_neg_three` | 0baa3646-0735-44f8-89b1-10a3950aea15 | ACCEPTED |
| `EmlComplexity.attains_four` | 3e4b157d-f83a-4756-841e-6bdd4b0ca138 | ACCEPTED |
| `EmlComplexity.not_attains_four_below_twenty_one` | b80a488a-5019-4d60-add5-9e0438b0e55e | Open |
| `EmlComplexity.complexity_four` | df00c0ea-1a00-4640-8fda-fe95cae58c29 | Open |
| `EmlComplexity.complexity_two` | 4aef18dd-2336-49b8-810b-f0a58e507c1c | Open |

All eleven `attains_*` proofs were accepted by the server's Lean on the first submission of the final text. The three lower-bound statements are open. Mission proposal `9a4eba3f-014b-4079-b23c-39e21b76e053` ("EML complexity of constants", OpenProblem, fields theoretical-computer-science + number-theory) references all fifteen items, with `complexity_four` as the goal and thirteen milestones; it waits for the account holder to confirm the items and click Submit Proposal on the website.

## Layout

- `Definitions/`, `Theorems/`, `Solutions/`: the files as uploaded, in the platform's module layout. Solutions are `theorem solution : Attains c n` with the enumeration's tree as the witness.
- `mission_description.md`: the proposal's description, written to the platform's seven-section template.
- `scripts/gen_solutions.py`: generates every solution from the eml-sr JSON. It mirrors the rewrites `simp only [eval_node, eval_one, log_one, sub_zero, sub_sub_cancel, log_exp, exp_log]` will make, so it can emit the positivity facts simp will ask for in the exact reduced form, and then discharges them by `assumption`. Numeric input: `Real.exp_one_gt_d9`, `exp_one_lt_d9`, `log_two_gt_d9`, `log_two_lt_d9`, and `7 < e^e` from `quadratic_le_exp_of_nonneg`.
- `scripts/p2m_api.py`: the thin client used for register/login, `submit-definition`, `submit-problem`, multipart `verify`, and proposal endpoints. Reads the API key from a `credentials.json` next to the workspace; none is committed here.
- `items.json`, `published.json`: the item metadata and every id the platform returned.

## What the proofs taught

- `simp` will not apply `Real.exp_log` with `linarith` as discharger, in this Mathlib; it does with `assumption`. Hence the pre-proved facts.
- `ring` does not fail when it cannot close a goal (it falls back to `ring_nf` and reports a hint), so inside `first | ring | ...` it swallows the alternatives. `ring1` fails properly.
- `norm_num` likewise makes progress without closing; every alternative in a `first` chain needs a trailing `done`.
- `ring_nf` fails on an already-normal goal, so a chain needs both the `ring_nf`-prefixed and the bare form of a closer.
- The definition file compiles in 19 s against the cache; each solution checks in 10 to 20 s locally, and the server took under a minute per verdict.

## Reproduce

```
git clone https://github.com/prove2me/prove2me_workspace ~/prove2me_workspace   # then lean-setup per its references/
cp -r Definitions Theorems Solutions ~/prove2me_workspace/
cd ~/prove2me_workspace && lake build Definitions.Def_EmlComplexity && lake env lean Solutions/Sol_EmlComplexity_attains_four.lean
```
