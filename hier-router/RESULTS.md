# RESULTS: routing a spiky ball through a tree of routers

The spiky ball (`adapter-library`) is one frozen Ettin-17m with many small LoRA adapters ("spikes"),
each good at one kind of decision, and a router spike that reads a one-line description of every spike
and picks one. All those descriptions share one input sequence with the case itself. At about 18 tokens
a description, the 1,024-token sequence the decider was trained with (browser-decider's cap) is full of
options near 56 spikes, before the case gets a single token. Oskar proposed progressive disclosure instead: a router whose options can be spikes or further
routers, to any depth. This experiment builds that tree on 69 leaves and asks how well it routes, what it
costs and what it does with leaves it has never seen, against one flat router that sees every leaf at once.

Terms: a **leaf** is a decision spike; a **router node** is a router question over its children's
descriptions plus `none`; **greedy** takes the top child at each node; **beam B** keeps the B most
probable root-to-node paths (product of the routers' probabilities) and returns the best finished one.
**Shared tree router** is one LoRA adapter answering at every node. **Flat long** puts every leaf in one
sequence; **shard** scores the flat router's options in chunks of 16 and compares them through each chunk's
own `none`. **Typed Decisions** is the business-workflow benchmark adapter-library uses; **oracle** routing
sends every case to its own leaf, the ceiling for end-to-end accuracy.

## Answer

A tree of routers routes about as well as one flat router over the same leaves, reads fewer tokens per
decision and has no sequence ceiling, once each hard node is trained to convergence. At 62 leaves the
tree with beam 2 routes 0.839 of test cases to the right leaf, at 0.8× flat's tokens, <!-- @evidence[[results/summary.json#scaling.tree_e9_node40_beam2.62.acc]] -->
against 0.843 for the flat router with every option in one sequence <!-- @evidence[[results/summary.json#scaling.flat_e9_long.62.acc]] -->
and 0.851 with the options in shards. <!-- @evidence[[results/summary.json#scaling.flat_e9_shard.62.acc]] -->
Greedy, 0.013 behind flat, reads half the tokens: 610 per decision against 1,198. <!-- @evidence[[results/summary.json#scaling.tree_e9_node40_greedy.62.tokens]] -->
At 16 and 32 leaves the tree was ahead by 0.004 to 0.016.

The tree adds new leaves better: from a description alone, 0.617 of a new leaf's cases reached it,
against 0.511 for flat (0.606 on a shorter schedule). A missing leaf's cases reach the right parent 0.856
of the time, where P(none) flags them less sharply (AUROC 0.75) than flat's global P(none) does (0.81).

One shared adapter never learned the invoice node, where the family is read from the numbers; that
node's own adapter learned it after 19 epochs without progress. Per-node convergence checks would move
this result most.

## Findings

1. **The tree routes within 0.004–0.013 of flat at 0.5–0.8× the tokens.** At 62 leaves: tree greedy 0.830 <!-- @evidence[[results/summary.json#configs.N62-d0.tree_e9_node40_greedy.acc]] -->
   at 610 tokens and 2.9 router calls per decision, tree beam 2 0.839 at 957 tokens, flat in one sequence
   0.843 at 1,198 tokens, flat in 4 shards 0.851 at 1,459.
   At 16 leaves beam 2 scored 0.917 against 0.903 for both flat scorings, and at 32 leaves 0.883 against 0.867
   and 0.879. Both routers fall by about the same amount from 8 to 62 leaves (flat 0.896 to 0.843, tree beam
   0.904 to 0.839): the confusable MASSIVE intents are what make larger libraries harder, at every depth. (Rounds 3–4b)
2. **The tree's sequences stay short as the library grows.** The flat router needed sequences of
   1,200 tokens at 62 leaves and was trained with the cap raised to 2,048; the tree's longest router question
   has 18 options. Past about 400 leaves the flat router outgrows Ettin's 8K positions and has to shard,
   and sharding already costs it 22% more tokens than one sequence at 62. (Round 1, `results/summary.json` tokens)
3. **A shared tree router can leave a node untrained without showing it.** The tree router routed Typed
   Decisions families at 0.804 in rounds 1–3 because it sent 55 of 60 invoice pricing and process cases to
   `inv_quantity`; the other 10 families were 0.90–1.0. The flat router had the same collapse after 3 epochs
   (Typed Decisions 0.801) and learned the split only on a 9-epoch run, still improving at its last epoch. (Rounds 1–3)
4. **A node's own adapter fixes it, after a long plateau.** A LoRA trained on the invoice node's question
   alone (855 examples) held validation KL at ln 3, the value for a uniform guess over 3 families, for 19 epochs,
   then dropped to 0.03 by epoch 25. With 9 epochs it never left the plateau. Spliced into the tree, it raised
   Typed Decisions routing from the shared router's 0.804 to 0.918, and end-to-end decision accuracy from
   0.594 to 0.644. <!-- @evidence[[results/summary.json#configs.N62-d0.tree_e9_node40_beam2.e2e_td]] -->
   The flat router gets 0.645 (one sequence) and 0.650 (shards); the oracle gets 0.654. (Rounds 4, 4b)
5. **The tree adds a leaf by description better than flat.** With the 7 held-out leaves' descriptions added
   and no examples: 0.617 of their cases reached them under the tree, <!-- @evidence[[results/summary.json#configs.added.tree_e9_node40_greedy.zero_shot_held_out]] -->
   0.511 under flat in one sequence. Flat on the 3-epoch schedule sent 0.606, so the margin depends on how long flat trains: longer training made
   it more accurate on its known leaves and less willing to pick a new one.
   Routing for the 62 existing leaves fell 0.009 under tree beam 2 (0.839 to 0.830) and 0.019 under flat (0.843 to 0.824).
   A new description only competes with its siblings in the tree, and with every leaf in the flat router. (Round 2, `added`)
6. **When a leaf is missing, the tree still puts its cases in the right place.** Under greedy, 0.856 <!-- @evidence[[results/summary.json#configs.absent.tree_e9_node40_greedy.held_out_reached_parent]] -->
   of a held-out leaf's cases reached its parent.
   The parent's P(none) separates them from trained siblings' cases at AUROC 0.75, but as a hard decision `none` fired on only 0.083
   of them. Flat's single P(none) separates held-out cases from all trained cases better (AUROC 0.813) and fires
   on 0.433, but says nothing about where the new leaf belongs. (Round 2, `absent`)
7. **Training schedule mattered more than architecture in this run.** The same shared tree router trained on a
   3-epoch schedule scored 0.785 (beam 3), and 0.818 on a 9-epoch schedule (best epoch 3 of 9). Each change of schedule
   moved accuracy further than switching between flat and tree. The 9-epoch run's best epoch came early
   but at a higher learning rate, because the linear decay is stretched over 9 epochs. (Rounds 1–2)
8. **A whole-body tree router bought nothing at 62 leaves.** Training all 7 layers with the token table frozen (3.97 MB at
   int8, against 0.49 MB for LoRA) scored 0.779 at 62 leaves with beam 3, <!-- @evidence[[results/summary.json#configs.N62-d0.tree_body_beam3.acc]] -->
   against 0.785 for the LoRA tree router on the same 3-epoch schedule.
   It was ahead at 8 and 16 leaves (0.904 vs 0.870, 0.890 vs 0.861) and level at 32. It was trained only on the
   3-epoch schedule. (Round 1)
9. **Top-level routing is trivial on this fixture.** Every misroute by every method stayed in the right
   top-level branch, and the 3 everyday leaves routed at 1.0. The hard decisions are between siblings. (All rounds)

## Method

**Fixture** (`run/tree.py`): 69 leaves in a tree of depth 1 to 4. 54 Amazon MASSIVE en-US intents (CC BY
4.0) under their scenarios under a voice router, with iot split into lights and plugs; six intents with fewer
than 10 test utterances dropped; single-intent scenarios hang their intent directly under voice. The 12 Typed
Decisions families from adapter-library round 4 under their 4 workflows under a business router. The 3 everyday
sets directly under the root, which therefore offers 2 routers and 3 leaves. 7 leaves (6 intents and
`sec_privilege`) are held out of all router training. At most 150 train and 30 test cases per leaf: 7,531 train,
1,846 test.

**Libraries**: 4, 8, 16 and 32 leaves (3 seeded draws each) and all 62 trained leaves. A library induces a tree:
empty routers go, and a router left with one child is replaced by that child. Accuracy is micro over the library
leaves' test cases (macro over leaves agreed within 0.01).

**Routers**: zero-start adapters on frozen stock Ettin-17m with adapter-library's LoRA recipe (rank 8 on
Wqkv/Wo/Wi, MLP head, AdamW, warmup then linear decay), best epoch by validation KL, sequences capped at 2,048.
Flat: three questions per training case, the gold leaf among 3 to 61 random others plus `none`, 15% with the gold
leaf removed and answered `none`. Tree: one question per router on the case's path, non-gold siblings dropped with
p 0.25, plus one question at an off-path router answered `none`. Final arms: flat 9 epochs (130M training tokens);
tree 9 epochs (42M, best epoch 3), with the invoice node answered by its own LoRA (3 option orders per case,
40 epochs, best 25).

**End to end**: adapter-library's 12 family sub-spikes, retrained (their probabilities were never committed),
answer each Typed Decisions case's 5 questions for the family the route picked; a route out of the workflow
counts all 5 wrong.

**Not measured**: wall-clock in a browser. Token counts stand in for cost; the GPU bridge was not used, so no
device timings are reported. The invoice node's validation set holds other option orders of training cases, so
its curve shows when the split was learned, not held-out accuracy.

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
right scenario 0.899 vs 0.935). End to end on Typed Decisions every method scored 0.587 (oracle 0.654): all routed
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
`inv_pricing` and 27 of 30 `inv_process` test cases went to `inv_quantity`, while the other 10 families were 0.90–1.0.
Flat had collapsed the same way in round 1 and learned the split only on this longer run.

### Round 4: the invoice node gets its own adapter (9 epochs)

Asked: since a router is just another spike, can the one hard node have its own? Ran: a LoRA on the invoice node's
question alone, 855 examples (3 option orders per case), 9 epochs (486 steps, 55 s). Validation KL stayed at
1.08–1.13, against ln 3 = 1.099 for a uniform guess: nothing learned. Spliced in, Typed Decisions routing 0.797.

### Round 4b: the same node adapter for 40 epochs

Ran: 40 epochs (151 s). Validation KL sat at 1.06–1.12 through epoch 18, fell to 0.57 at epoch 19 and 0.03 at
epoch 25 (best). Spliced into round 2's tree at 62 leaves: greedy 0.830, beam 2 0.839, Typed Decisions routing
0.918, end to end 0.644 against flat's 0.645–0.650 and the oracle's 0.654. At 16 and 32 leaves: beam 2 0.917 and 0.883.

Predictions (PLAN.md), against the final arms unless noted:

| | prediction | result |
|---|---|---|
| P1 | flat at 62 at least 0.05 below its mean at 8 | held: 0.896 → 0.843 (round 1 0.876 → 0.801) |
| P2 | tree beam 3 ≥ flat long at 62 | failed: 0.839 (beam 2 and 3 alike) vs 0.843 (held in round 2 against round 1's flat) |
| P3 | beam 3 ≥ greedy + 0.01 at 62 | failed: +0.008 for the final tree (round 1: +0.014) |
| P4 | greedy tokens ≤ 0.5 × flat long at 62 | failed narrowly: 610 vs 1,198 (0.51) |
| P5 | ≥ 80% of tree misroutes stay in the top-level branch | held: 100% (also for flat) |
| P6 | absent leaf reaches parent ≥ 0.7, parent AUROC ≥ 0.65 | held: 0.856, 0.750 |
| P7 | tree zero-shot to added leaves > flat's | held: 0.617 vs 0.511 |
| P8 | end-to-end tree beam 3 within 0.01 of flat long | held: 0.644 vs 0.645 (failed before round 4b: 0.594) |
| P9 | whole-body tree beats LoRA tree by < 0.01 | held: it was 0.006 behind |

Decision rule: P2 failed and flat's best scoring led by 0.012, under the 0.02 margin that would have kept flat as
the default. So the tree is a workable design here, with each node checked for convergence (round 4b), rather than a
clear win over flat.
