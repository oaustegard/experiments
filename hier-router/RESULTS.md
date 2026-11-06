# RESULTS: routing a library of LoRA adapters through a tree of routers

`adapter-library` (nicknamed the "spiky ball") is a decision system small enough to run in a browser: one
frozen 17M-parameter text encoder, Ettin-17m, plus many small LoRA adapters, each trained for one kind of
decision (classifying a bank transaction, checking an invoice, and so on). A **router**, itself a LoRA
adapter on the same encoder, reads the case together with a one-line description of every adapter and
outputs a probability for each adapter and for `none`. Every description shares one input sequence with
the case. At about 18 tokens a description, the 1,024-token sequence that library's adapters were trained
with is full of descriptions near 56 adapters, before the case gets a single token.

This experiment replaces the single router with a tree: a router's options can be adapters or further
routers, to any depth. The question is whether, on a library of 62 adapters, routing down that tree picks
the right adapter as often as one flat router does; what each costs per decision; and what each does with
an adapter that is added from its description alone, or that is missing from the library.

Terms used throughout. A **leaf** is one decision adapter. A **router node** asks one question: which of
its children (leaves or routers, by description) fits this case, or `none`. **Greedy** takes the most
probable child at each node. **Beam B** keeps the B most probable root-to-node paths (product of the
routers' probabilities) and returns the best finished one. **Flat long** puts every leaf in one sequence;
**flat shard** scores the leaves in chunks of 16, one call each, scores every option as log(p / p(none))
within its chunk, and picks the highest positive score across chunks (`none` if none is positive).
**Typed Decisions** is the business-workflow benchmark from `adapter-library`; **end to end** is the
accuracy of its final decisions once a case has been routed. The arms are listed in Method.

## Answer

At 62 leaves the tree and the flat router tie, and the tree reads fewer tokens. The best tree
(beam 2 or 3) sent 0.839 of 1,666 test cases to the right leaf, against 0.843 for flat long and 0.851 for flat shard. With one training seed the standard error is about 0.009
per arm. Greedy tree routing scored 0.830 reading 610 tokens per decision against flat's 1,198, but
over 2.9 router calls in sequence where flat makes one.

The tie depends on a repair chosen from test errors. The tree used one adapter for every router
node, and it never learned one node, the invoice router; a separate adapter trained 40 epochs on that node
alone was spliced in. In the 3-epoch runs PLAN.md registered, with no repair, flat led by 0.016 to 0.031
(0.801 and 0.816 against 0.785).

For a leaf added by description only, neither design is reliably better: the tree led at 9 epochs (0.617
against 0.511) and trailed at 3 (0.561 to 0.606 against 0.606 to 0.622), on 180 cases.

On the registered runs the plan's decision rule keeps flat; on the later runs it picks neither. The tree
is workable if each node is checked for convergence. Several seeds, a validation split held out by case,
and a tree that converges without repair would change this answer.

## Findings

1. **At 62 leaves the tree and flat routers tie.** Tree beam 2 0.839, <!-- @evidence[[results/summary.json#configs.N62-d0.tree_e9_node40_beam2.acc]] -->
   tree greedy 0.830, <!-- @evidence[[results/summary.json#configs.N62-d0.tree_e9_node40_greedy.acc]] -->
   flat long 0.843, <!-- @evidence[[results/summary.json#configs.N62-d0.flat_e9_long.acc]] -->
   flat shard 0.851, <!-- @evidence[[results/summary.json#configs.N62-d0.flat_e9_shard.acc]] -->
   on n = 1,666 test cases, one seed, standard error about 0.009 per arm (about 0.013 for an unpaired
   difference). The tree-vs-flat-long gap (0.004) is well inside that; the widest gap, greedy against flat
   shard (0.021), is under two standard errors of a difference. No paired test was run. On smaller libraries
   the mean over three random draws favours tree beam 2 (0.904 against flat 0.896 at 8 leaves, 0.917 against
   0.903 at 16, 0.883 against 0.867 long and 0.879 shard at 32), but one draw reverses the order at 8 and 16
   leaves, and at 32 one draw puts flat shard ahead; at 8 leaves the draws alone span 0.816 to 0.952. Both routers fall by about the same amount from 8 to 62 leaves (flat 0.896 to 0.843, tree 0.904 to
   0.839); the confusable voice-assistant intents are what make larger libraries harder, at any depth.
   (`tree_e9_node40_*`, `flat_e9_*`, `configs.N16-d*`, `configs.N32-d*`)
2. **The tree reads fewer tokens but makes more calls in sequence.** At 62 leaves, per decision: tree greedy
   610 tokens over 2.9 calls, <!-- @evidence[[results/summary.json#configs.N62-d0.tree_e9_node40_greedy.calls]] -->
   tree beam 2 957 tokens over 4.9 calls, flat long 1,198 tokens in 1 call, flat shard 1,459 tokens over 4 calls. <!-- @evidence[[results/summary.json#configs.N62-d0.flat_e9_shard.tokens]] -->
   Token counts include re-reading the case at each call. In a browser the tree's calls cannot overlap, so
   latency may track calls more than tokens; no device timing was taken. (`results/summary.json` tokens, calls)
3. **The flat router has a sequence ceiling; the tree's routers stay small.** Flat's sequences averaged
   1,198 tokens at 62 leaves, and it was trained with the cap raised to 2,048. The tree's largest router
   question, the voice router, has 18 options. At about 18 tokens a description, one flat sequence would pass
   Ettin's 8,192 positions somewhere above 400 leaves; sharding avoids that but already costs 22% more tokens
   than one sequence at 62. Nothing above 62 leaves was run, so the tree's behaviour there is untested; it
   keeps each question short only as long as the tree keeps branching. (`results/summary.json` tokens)
4. **One shared tree router can leave a node untrained without the overall number showing it.** Trained on
   9 epochs, the shared router routed Typed Decisions cases at 0.804 <!-- @evidence[[results/summary.json#configs.N62-d0.tree_e9_beam3.by_source.typed_decisions]] -->
   against flat's 0.971, because at the invoice router (which picks between quantity, pricing and process
   problems, mostly by comparing figures in the case) it sent 55 of 60 pricing and process cases to
   `inv_quantity`. The flat router collapsed the same way on 3 epochs (Typed Decisions 0.801) <!-- @evidence[[results/summary.json#configs.N62-d0.flat_long.by_source.typed_decisions]] -->
   and learned the split on 9. The per-family counts come from evaluation files that are not committed.
   (`tree_e9_*`, Round 3)
5. **A separate adapter for that node fixed it, after a long plateau.** A LoRA trained on the invoice
   router's question alone (855 examples) held validation KL at the value for a uniform guess over three
   options (ln 3 = 1.10) for its first 19 epochs, then fell to 0.03 at epoch 26 of 40. A 9-epoch version never
   left the plateau. Spliced into the tree, it raised Typed Decisions routing from 0.804 to 0.918, <!-- @evidence[[results/summary.json#configs.N62-d0.tree_e9_node40_beam2.by_source.typed_decisions]] -->
   and end-to-end accuracy from 0.594 to 0.644, <!-- @evidence[[results/summary.json#configs.N62-d0.tree_e9_node40_beam2.e2e_td]] -->
   against 0.645 for flat long, 0.650 for flat shard and 0.654 with every case routed correctly. <!-- @evidence[[results/summary.json#scaling.oracle.62.e2e_td]] -->
   This node was chosen from test-set errors, and its validation set shares cases with its training set, so
   the result shows the repair works, not that the method finds such nodes. (`node_invoice_e40`, Round 4b)
6. **Training schedule moved accuracy more than the choice of architecture.** The same shared tree router
   scored 0.785 on 3 epochs and 0.818 on 9; flat went from 0.801 to 0.843. Each change of schedule moved
   accuracy further than switching between flat and tree on a fixed schedule. (`tree_beam3`, `tree_e9_beam3`, `flat_long`, `flat_e9_long`)
7. **Which design finds a newly added leaf depends on the schedule.** With the 7 held-out leaves'
   descriptions added and no training examples, 180 test cases: on 9 epochs the tree sent 0.617 of them to <!-- @evidence[[results/summary.json#configs.added.tree_e9_node40_greedy.zero_shot_held_out]] -->
   the new leaf, flat long 0.511; <!-- @evidence[[results/summary.json#configs.added.flat_e9_long.zero_shot_held_out]] -->
   on 3 epochs the tree sent 0.561 (greedy) to 0.606 (beam 5), flat long 0.606 and flat shard 0.622. <!-- @evidence[[results/summary.json#configs.added.flat_shard.zero_shot_held_out]] -->
   The standard error at this size is about 0.037. Longer training made flat more accurate on its known leaves
   and less willing to pick a new one. Each new leaf was placed under its true parent in the tree by hand,
   which is information flat does not get. Adding the descriptions cost the 62 existing leaves 0.009 under
   tree beam 2 (0.839 to 0.830) and 0.020 under flat long (0.843 to 0.824). (`configs.added`, Round 2)
8. **When a leaf is missing, the tree still takes its cases to the right parent, but rarely says `none`
   there.** Under greedy, 0.856 of a held-out leaf's cases reached its parent router. <!-- @evidence[[results/summary.json#configs.absent.tree_e9_node40_greedy.held_out_reached_parent]] -->
   That router's P(none) separates them from its trained children's cases at AUROC 0.750, <!-- @evidence[[results/summary.json#configs.absent.tree_e9_node40_greedy.gap_auroc_at_parent]] -->
   but `none` won there for only 0.083. <!-- @evidence[[results/summary.json#configs.absent.tree_e9_node40_greedy.held_out_none_at_parent]] -->
   Flat long's P(none) separates them from all trained cases at AUROC 0.813 <!-- @evidence[[results/summary.json#configs.absent.flat_e9_long.gap_auroc]] -->
   and wins for 0.433, but says nothing about where the leaf belongs. The two AUROCs use different negatives
   (siblings, against every trained case), so they are not directly comparable. Beam search never returns
   `none` below the root. (`configs.absent`, Round 2)
9. **Training the whole encoder for the router bought nothing at 62 leaves.** Training all 7 layers with the token table
   frozen (3.97 MB at int8, against 0.49 MB for a LoRA adapter) scored 0.779 at 62 leaves with beam 3, <!-- @evidence[[results/summary.json#configs.N62-d0.tree_body_beam3.acc]] -->
   against 0.785 for the LoRA tree router on the same 3-epoch schedule. It was ahead at 8 and 16 leaves
   (0.904 against 0.870, 0.890 against 0.861, about 0.03 each) and level at 32. It was run on 3 epochs only. (`tree_body_*`, Round 1)
10. **The top level of the tree is easy on this fixture.** Every tree misroute stayed in the right
    top-level branch, as did 99.4% of flat long's, <!-- @evidence[[results/summary.json#configs.N62-d0.flat_e9_long.misroute_same_top]] -->
    and the 3 everyday leaves routed at 1.0. The hard choices are between siblings. (`misroute_same_top`)

### Predictions and decision rule

PLAN.md registered P1 to P9 for 3-epoch arms with no node repair. The table scores them on those arms,
and again on the final arms (9 epochs, invoice adapter spliced in), which were added after the first results.

| | prediction | as registered (3 epochs) | final arms |
|---|---|---|---|
| P1 | flat long at 62 at least 0.05 below its mean at 8 | held: 0.876 → 0.801 | held: 0.896 → 0.843 |
| P2 | tree beam 3 ≥ flat long at 62 | failed: 0.785 vs 0.801 | failed: 0.839 vs 0.843 (a tie in noise) |
| P3 | tree beam 3 ≥ greedy + 0.01 at 62 | held: +0.013 | failed: +0.008 |
| P4 | tree greedy tokens ≤ 0.5 × flat long's at 62 | failed narrowly: 609 vs 1,198 (0.51) | failed narrowly: 610 vs 1,198 (0.51) |
| P5 | ≥ 80% of tree misroutes stay in the top-level branch | held: 100% | held: 100% |
| P6 | absent leaf reaches parent ≥ 0.7, parent AUROC ≥ 0.65 | held: 0.778, 0.708 | held: 0.856, 0.750 |
| P7 | tree zero-shot to added leaves > flat's | failed: 0.561–0.606 vs 0.606 | held: 0.617 vs 0.511 |
| P8 | end-to-end tree beam 3 within 0.01 of flat long | held: 0.587 vs 0.587 (both collapsed) | held: 0.644 vs 0.645 |
| P9 | whole-body tree beats LoRA tree by < 0.01 | held: it did not beat it (0.005 behind) | not run |

The rule: if P2 and P8 hold, the tree becomes the routing design; if flat wins by more than 0.02 at 62,
keep flat up to the ceiling and use the tree past it. As registered, P2 failed and flat led by 0.016 (long)
or 0.031 (shard), so on flat's best scoring the second branch fires: keep flat. On the final arms P2 failed by
0.004 and flat's best scoring led by 0.012, under the 0.02 margin, so neither branch fires. The tree is a
workable design here, with each node checked for convergence, rather than a clear win over flat.

## Method

**Base model and router.** Every router is a zero-start LoRA adapter (rank 8 on Wqkv/Wo/Wi, an MLP head over
the options) on frozen stock Ettin-17m, a 17M-parameter encoder with 8,192 positions, trained with
`adapter-library`'s recipe (AdamW, warmup then linear decay), sequences capped at 2,048. The input is the
case followed by the option descriptions and `none`; the output is a probability over those options. The
best epoch is chosen by validation KL against the one-hot gold option.

**Fixture** (`run/tree.py`): 69 leaves in a tree of depth 1 to 4.
- 54 intents from Amazon MASSIVE en-US (CC BY 4.0), a voice-assistant intent dataset: 60 intents less 6
  with fewer than 10 test utterances. They sit under their scenarios under a voice router; iot is split once
  more into lights and plugs, and single-intent scenarios hang their intent directly under voice.
- The 12 Typed Decisions families under their 4 workflows under a business router.
- 3 everyday sets (bank transactions, emails, reviews) directly under the root, which therefore offers 2
  routers and 3 leaves.

7 leaves (6 intents and `sec_privilege`) are held out of all router training. At most 150 train and 30 test
cases per leaf: 7,531 train and 1,846 test, of which 1,666 belong to the 62 trained leaves and 180 to the
held-out ones; 306 of the 1,666 are Typed Decisions cases.

**Libraries.** 4, 8, 16 and 32 leaves (three seeded draws each, results averaged) and all 62 trained leaves
(one library). A library induces a tree: empty routers go, and a router left with one child is replaced by
that child. Accuracy is micro over the library leaves' test cases (macro agreed within 0.01).

**Training questions.** Flat: three questions per training case, the gold leaf among 3 to 61 random others
plus `none`; 15% drop the gold leaf and answer `none`. Tree: one question per router on the case's path,
non-gold siblings dropped with probability 0.25, plus one question at an off-path router answered `none`.

**Arms.**

| arm | what it is | epochs | training tokens |
|---|---|---|---|
| `flat_*` | one flat LoRA router, scored long and shard | 3 | 43M |
| `tree_*` | one shared LoRA router for every node, scored greedy and beam 2/3/5 | 3 | 14M |
| `tree_body_*` | as `tree`, but all 7 layers trained with the token table frozen | 3 | 14M |
| `flat_e9_*` | `flat` on a 9-epoch schedule (best: last epoch) | 9 | 130M |
| `tree_e9_*` | `tree` on a 9-epoch schedule (best: epoch 4) | 9 | 42M |
| `tree_e9_node_*` | `tree_e9` with its own LoRA at the invoice router | 9 + 9 | +3.3M |
| `tree_e9_node40_*` | as above, the invoice LoRA trained 40 epochs (best: epoch 26) | 9 + 40 | +14.7M |

The invoice adapter trains on 285 invoice cases, each with 3 orderings of the options.

**Added and absent leaves.** *Absent*: the 62-leaf library, scored on the held-out leaves' 180 cases.
*Added*: the held-out leaves' descriptions are inserted at their true place in the tree (and into the flat
option list), with no training examples. Greedy returns `none` wherever a router picks it; beam returns
`none` only at the root, so beam arms never abstain.

**End to end.** `adapter-library`'s 12 Typed Decisions family adapters, retrained because their
probabilities were never committed, answer each case's 5 questions for the family the route picked. A route
that leaves the case's workflow counts all 5 wrong. Routing every case to its own leaf gives the ceiling
(0.654).

**Caveats.** One training seed for every router. `flat_e9` was still improving at its last epoch, so flat
may be undertrained relative to the tree, which favours the tree. Validation questions are a random 5% of training questions,
and each case produces several questions, so validation shares cases with training in every arm; best-epoch
choices are made on near-training data. The invoice repair and the 9-epoch arms were chosen after the
3-epoch results. Per-family and per-scenario counts quoted in the Log come from `out/eval*.json`, which are
not committed. Epochs are counted from 1 above; the Log and `results/summary.json` count from 0.

**Cost.** One Colab L4 session. Router training: 466 s (flat), 204 s (tree), 171 s (tree_body), 568 s
(tree_e9), 1,412 s (flat_e9), 55 s and 151 s (invoice adapters), about 50 minutes in all, plus the 12 family
adapters and evaluation (round 1's whole job was about 25 minutes). No browser timing was taken: token
counts stand in for cost.

## Log

### Round 1 (2026-11-06): flat vs shared tree router, 3 epochs

Asked: does routing down a tree hold up better than flat as the library grows (PLAN.md P1–P9)? Ran: flat LoRA,
tree LoRA and tree whole-body routers, 3 epochs each, one Colab L4 job (~25 min, including the 12 sub-spikes).
Flat read 43M training tokens and the tree routers 14M; all stopped on their last or second-to-last epoch.

| leaves | flat long | flat shard | tree greedy | tree beam 3 | body greedy | body beam 3 |
|---|---|---|---|---|---|---|
| 4 | 0.916 | 0.916 | 0.940 | 0.979 | 0.958 | 0.973 |
| 8 | 0.876 | 0.876 | 0.847 | 0.870 | 0.872 | 0.904 |
| 16 | 0.880 | 0.880 | 0.839 | 0.861 | 0.878 | 0.890 |
| 32 | 0.852 | 0.866 | 0.819 | 0.834 | 0.826 | 0.833 |
| 62 | 0.801 | 0.816 | 0.771 | 0.785 | 0.773 | 0.779 |

Tokens per decision at 62: flat long 1,198, shard 1,459, tree greedy 609, beam 3 1,067. On MASSIVE at 62 the
tree picked the scenario about as well as flat (0.832–0.855 vs 0.842–0.863) and lost inside it (intent given the
right scenario 0.899 vs 0.935). End to end on Typed Decisions every method scored 0.580–0.587 (oracle 0.654): all routed
those families at 0.80. Reading at the time: flat wins by 0.016–0.03, with the tree undertrained (a third of
flat's training tokens).

### Round 2: tree router on a 9-epoch schedule

Asked: is round 1's gap the tree's training budget? Ran: the tree LoRA for 9 epochs (42M tokens, about flat's);
validation KL was best at epoch 3 and rose after it. Result at 62: greedy 0.809, beam 2 and 3 0.818, above round
1's flat. Unseen leaves: reached the parent 0.856, parent AUROC 0.750; zero-shot to added leaves 0.617 (beam 3 0.622),
against round 1 flat's 0.606.

### Round 3: flat router on the same schedule

Asked: round 2 changed the schedule, not only the tokens, so does flat gain the same? Ran: flat for 9 epochs
(130M tokens, 24 min on the L4, best epoch 8 of 9). Result at 62: one sequence 0.843, shards 0.851; at 32
0.867/0.879; at 16 0.903. Zero-shot to added leaves fell to 0.511, P(none) AUROC 0.813. By source at 62: MASSIVE flat shard
0.811 vs tree beam 3 0.808; Typed Decisions 0.974 vs 0.804. The tree's 0.804 came from one node: 28 of 30
`inv_pricing` and 27 of 30 `inv_process` test cases went to `inv_quantity`, while the other nine families were 0.90–1.0.
Flat had collapsed the same way in round 1 and learned the split only on this longer run.

### Round 4: the invoice node gets its own adapter (9 epochs)

Asked: since a router is just another spike, can the one hard node have its own? Ran: a LoRA on the invoice node's
question alone, 855 examples (3 option orders per case), 9 epochs (486 steps, 55 s). Validation KL stayed at
1.08–1.13, against ln 3 = 1.099 for a uniform guess: nothing learned. Spliced in, Typed Decisions routing 0.797.

### Round 4b: the same node adapter for 40 epochs

Ran: 40 epochs (151 s). Validation KL sat at 1.06–1.12 through epoch 18, fell to 0.57 at epoch 19 and 0.03 at
epoch 25 (best). Spliced into round 2's tree at 62 leaves: greedy 0.830, beam 2 0.839, Typed Decisions routing
0.918, end to end 0.644 against flat's 0.645–0.650 and the oracle's 0.654. At 16 and 32 leaves: beam 2 0.917 and 0.883.
