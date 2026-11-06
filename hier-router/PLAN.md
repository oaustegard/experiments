# PLAN: hierarchical routing for the spiky ball

Registered 2026-11-06, before any training run.

## Question

The spiky ball (`adapter-library`) puts one router in front of its LoRA spikes. The router reads one
description per spike, and every description sits in the same 1,024-token sequence as the input, so the
number of spikes a router can offer is bounded: at about 18 tokens per description, options alone fill the
sequence near 56 spikes. Accuracy may also fall well before that. Oskar proposed progressive disclosure:
a router whose options can be leaves (decision spikes) or other routers, with no fixed depth.

1. As the library grows from 4 to 62 leaves, how does flat routing accuracy fall, and does routing down a
   tree hold up better?
2. What does each cost per decision, in router calls and tokens read?
3. When an input belongs to a leaf the library lacks, does the tree still get it to the right branch, and
   does the router there say `none`, so the gap shows where a new leaf should grow?
4. When a new leaf is added by description only, does the tree or the flat router find it more often?
5. End to end on Typed Decisions, with each family answered by its own sub-spike: does the routing
   difference survive into decision accuracy?
6. (Oskar, 2026-11-06) A full fine-tune that keeps the token table frozen stores only the 7 layers:
   3.97 MB at the shipped int8 (about 2 MB at 4 bits), against 0.49 MB for a LoRA spike. Does a
   whole-body tree router route better than a LoRA one?

## Prior art (3 web searches, 2026-11-06)

Components exist; I did not find this composition in 3 searches.

- **Hierarchical classification** is old. Top-down (Pachinko) classifiers propagate early errors, and in
  the 2009 LSHTC challenge flat methods ranked highest and top-down lowest; top-down wins on cost when
  classes number in the thousands. Mitigations include flattening the hierarchy and keeping several
  branches alive. The expectation going in is that flat wins on accuracy while it fits.
- **Tool routing** does category-then-tool: AnyTool (2402.04253) uses category agents over 16K APIs;
  Hi-RAG/MCPBench retrieves services, then tools. These use LLMs or retrievers per level.
- **LoRA libraries:** LoRAuter (2601.21795) routes 1,500 adapters through task representations; S'MoRE
  (2504.06426) is a per-token tree of low-rank experts inside one model.

What differs here: one 17M-parameter description-reading decision model routes at every node, so a
router is just another option; depth varies per branch; the per-node `none` is a gap detector that says
where in the tree a missing leaf belongs; and it runs in a browser.

## Fixture (`run/tree.py`)

69 leaves in a tree of depth 1 to 4:

- 54 Amazon MASSIVE en-US intents (CC BY 4.0) under their 18 scenarios, the voice router above them.
  iot is split once more (lights, plugs). Six intents with under 10 test utterances are dropped;
  single-intent scenarios (news, weather, cooking after the drop) hang their intent directly under voice.
- 12 Typed Decisions families (adapter-library round 4) under their 4 workflows, under a business router.
- 3 everyday sets (transactions, emails, reviews) directly under the root, beside the voice and business
  routers. The root therefore mixes leaves and routers.

7 leaves are held out of all router training (`HELD_OUT`): six intents and `sec_privilege`. Caps: 150
train and 30 test cases per leaf (7,531 train, 1,846 test). Leaf counts are drawn from the 62 trained
leaves: 4, 8, 16, 32 (3 seeded draws each) and 62. A library of N leaves induces a tree: empty routers
are removed and a router with one child is replaced by that child.

## Arms (`run/train_eval.py`, one L4 run)

All routers are zero-start adapters on frozen stock Ettin-17m, trained 3 epochs, best epoch by
validation KL, sequences capped at 2,048 tokens so the flat router is not cut off.

- **flat**: LoRA. Three questions per training case, each with the gold leaf among a random subset of 3
  to 61 others plus `none`; 15% drop the gold leaf and answer `none`. Scored two ways: `flat_long`, all
  N leaves in one sequence; `flat_shard`, chunks of 16, compared through log(p/p_none).
- **tree**: one shared LoRA for every router node. Per case, one question per router on its path
  (non-gold siblings dropped with p 0.25), plus one question at an off-path router answering `none`.
  Scored `greedy` (argmax per node) and `beam2/3/5` (product of probabilities along the path).
- **tree_body**: tree's data, every layer trained, token table frozen, lr 8e-5.

End to end: adapter-library's 12 family sub-spikes (retrained: round 4's per-case probabilities were not
committed) answer the five Typed Decisions questions for the family the route picked; a route that leaves
the case's workflow counts every decision wrong. Oracle routing gives the ceiling.

## Metrics

Routing accuracy on test cases of library leaves (micro, and macro over leaves), by source; misroutes
that stay in the right top-level branch; router calls and tokens read per decision; for absent leaves,
whether greedy reaches the correct parent, how often it says `none` there, and AUROC of the parent's
P(none) against trained siblings' cases (flat: AUROC of its P(none) against all trained cases); for added
leaves, zero-shot routing to them and the change for the others; Typed Decisions end-to-end accuracy.

## Predictions

- **P1** Flat routing falls with N: `flat_long` at 62 is at least 0.05 below its mean at 8.
- **P2** At 62 leaves `tree_beam3` routes at least as well as `flat_long` (my credence 0.5; prior art
  says flat wins while it fits).
- **P3** Beam helps: `tree_beam3` ≥ `tree_greedy` + 0.01 at 62.
- **P4** Tree reads fewer tokens: `tree_greedy` tokens at 62 ≤ 0.5 × `flat_long`'s.
- **P5** At least 80% of tree misroutes at 62 stay in the right top-level branch.
- **P6** An absent leaf's cases reach its parent under greedy ≥ 0.7 of the time, and the parent's
  P(none) separates them from trained siblings with AUROC ≥ 0.65.
- **P7** Added by description, the tree sends more of a new leaf's cases to it than flat does.
- **P8** End to end on Typed Decisions at 62, `tree_beam3` is within 0.01 of `flat_long` or better.
- **P9** `tree_body` beats `tree` by less than 0.01 at 62 (whole-body bytes do not buy routing).

## Decision rule

If P2 and P8 hold, the tree becomes the spiky ball's routing design, since it also removes the
sequence-length ceiling. If flat wins by more than 0.02 at 62, keep flat up to the ceiling and use the
tree only past it, with the beam width that closes most of the gap.
