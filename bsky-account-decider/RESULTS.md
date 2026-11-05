# RESULTS: a Bluesky account classifier in the browser, taught by Sonnet

This is the second half of `browser-decider`. That experiment found that a 10 MB Ettin-17m decision model
matches a hosted model on questions it was trained on and fails on questions it was not. Here the
recipe is applied to a question set nobody had labels for. Claude Sonnet labels Bluesky accounts, and
Ettin-17m learns to give Sonnet's answers. The page is `web/index.html`, live at
https://austegard.com/experiments/bsky-account-decider/web/: type a handle, and it reads the account's
latest posts from the public API and answers in the tab.

## Answer

Can a 10 MB model learn AI-themed account questions from Sonnet's labels, and how many labels does it
need? Yes. Trained on 1,542 accounts, it picks Sonnet's answer on 300 held-out accounts
92.1% of the time over all 5,058 decisions, mean of three runs. <!-- @evidence[[results/summary.json#arms.all.overall.mean]] -->
Per question it ranges from 81% (how much is about AI) to 94% (is this post about AI), 3 to 11 points
below Sonnet's agreement with itself. On the five account questions it averages 0.872 against 0.841 for a classifier on frozen embeddings. <!-- @evidence[[results/summary.json#arms.all.account_mean.mean]] -->
With 100 labelled accounts it averages 0.784, and 300 bring it within a point of the frozen classifier. <!-- @evidence[[results/summary.json#arms.n100.account_mean.mean]] -->
Bots are its weak spot: of 18 test accounts Sonnet called bots, it finds 5 to 8. Labels cost
$14.58 for 1,842 accounts plus a 150-account repeat pass. More bot examples would change the bot
result.

## Findings

1. **The fine-tuned model is ahead of both baselines on four of six questions.** Account type 0.917 <!-- @evidence[[results/summary.json#arms.all.account_type.mean]] -->
   (frozen vectors 0.867, majority 0.850), how much about AI 0.813 (0.783, 0.643), own writing 0.854
   (0.757, 0.683), post about AI 0.942 (0.927, 0.839). On AI role the frozen vectors lead (0.897
   against 0.881) and on attitude they tie (0.900 against 0.897). Structure-heavy questions gain most.
   (`results/summary.json`, `results/baselines.json`)
2. **Sonnet is a consistent teacher.** A second labelling pass over 150 accounts in fresh batches
   agrees on the top answer 97.3% for account type, 92.0% for share, 95.3% role, 94.7% attitude, <!-- @evidence[[results/labels.json#self_agreement.account_type.argmax]] -->
   88.0% own writing, 98.0% for the per-post question.
   (`results/labels.json`)
3. **More labels keep helping.** Mean over the five account questions: 0.784 at 100 accounts, 0.835 at <!-- @evidence[[results/summary.json#arms.n300.account_mean.mean]] -->
   300, 0.854 at 700, 0.872 at 1,542.
   Prediction P4 (300 accounts give 90% of the full gain over the majority baseline) missed: they give 71%.
4. **Bot recall is low.** 85 of 1,842 accounts are bots by Sonnet's labels, 18 in test. The three seeds <!-- @evidence[[results/summary.json#bot_detection.test_bots]] -->
   find 5, 7 and 8 of them with 0, 1 and 2 false alarms.
5. **The labels are cheap.** `claude -p --model sonnet` with tools off (`--tools ""`), five accounts a
   call, $0.04 a call, 15 minutes for 1,842 accounts at 6 parallel calls. Leaving the tools on
   costs about 4x as much per call: the tool definitions ride along on every request.
6. **The JS state builder matches Python exactly** on 25,589 strings (`run/check_state.mjs`), after
   matching Python's float rounding (exact ties to even) and its idea of whitespace (U+FEFF is not
   whitespace to Python).

## Method

**Accounts** (`run/collect.py`, 2026-11-05, public AppView and Jetstream): 495 members of Oskar's AI
list, 411 accounts they follow, 399 Discover-feed authors, 537 authors of English firehose posts in 180
s; at least 5 feed items each. **State** (`run/state.py`, mirrored in `web/state.js`): handle and name,
bio, counts, an activity summary, and the latest 12 items cut to 140 characters, with trailing items
dropped until it fits 830 Ettin tokens. **Questions** (`run/questions.json`): five account questions
and a per-item "about AI" yes/no. **Labels** (`run/label.py`): Sonnet 5.5, 5 accounts per call, a
probability for every option; pass 2 relabels a fixed 150 in different batches. **Split**: those 150
plus 150 more held out by account (all their items); training keeps 6 items per account.
**Model**: stock Ettin-17m with browser-decider's cross-encoder head and trainer (`--data`), 10 epochs,
seeds 0 to 2, plus 100, 300 and 700 training accounts; Colab A100, about 7 minutes a full run.
**Baselines** (`run/baseline_probe.py`): each question's majority label; logistic regression (C=3) on
the frozen 10 MB embedder's vectors. **Page build** (`run/pack_page.py`): browser-decider's packer,
int8 matmuls and a 4-bit token table.

## Log

### 2026-11-05: replies read with the post they answer

The page scored each reply on its own text, so a reply under an AI post ("agreed, it can't do that")
came out as not about AI. The page now appends the parent post to a reply's state,
`[reply] <text> (in reply to @handle: <parent, 140 chars>)`, from the `reply.parent` the author feed
already returns. The model was not retrained: training states are unchanged because they carry no parent.

Measured with `run/reply_context_eval.py` on 388 fresh replies from 45 AI-list members and 45 Discover
authors, labelled by Sonnet with the parent visible under the rule that a reply in a conversation about
AI is about AI ($0.33). 107 of the 388 are about AI by that rule. <!-- @evidence[[results/reply_context.json#sonnet_positive]] -->

| model input | accuracy | recall | false positives |
|---|---|---|---|
| reply only (before) | 0.884 | 0.617 | 1.4% |
| reply, then parent (shipped) | 0.912 | 0.776 | 3.6% |
| parent, then reply | 0.910 | 0.794 | 4.6% |

The shipped format gains 16 points of recall for 2 points of false positives. <!-- @evidence[[results/reply_context.json#variants.reply_then_parent.all]] -->
On AI-list accounts the false-positive rate rises from 3.6% to 9.0%: replies on an AI post that change
the subject now get partial credit from the parent. A retrain on parent-aware labels would be the next
step; the account-level questions still read replies without their parents, because the parent text
would push items out of the 830-token budget the model was trained on.

### 2026-11-05: labels, arms, page

Registered `PLAN.md` before reading labels. Predictions against results:

| prediction | result |
|---|---|
| P1 oracle self-agreement ≥ 0.85 on type and post, 0.65 to 0.80 on role and attitude | 0.973 and 0.980; role 0.953 and attitude 0.947, higher than predicted |
| P2 within 0.10 of self-agreement on every question | five of six; how much about AI is 0.107 short |
| P3 account type ≥ 0.85, 0.05 over majority | 0.917, +0.067 |
| P4 300 accounts give 90% of the gain | 71% |
| P5 post question at least matches the frozen-vector classifier | 0.942 against 0.927 |

The decision rule (P2 on type, share and the post question) is met for type and post and missed by
0.007 on share; the page shipped with that noted.

What broke along the way: installing sentence-transformers upgraded transformers to 5.22, which nests
the RoPE bases; the packer now reads both layouts, and the training environment stayed on 4.57.1.

### 2026-11-05: replies read with their whole thread

The parent alone missed replies deep in a thread: a "thanks, did the team have an answer?" six posts
below a post about an embedding paper scored 23% with its parent. Each reply now carries every
ancestor back to the thread's root (one `getPostThread` per reply; the page's fetch for 20 items with
11 replies took 1.3 s), appended nearest first, `(in reply to @parent: … ← @…: … ← @root: …)`, each cut
to 140 characters, with threads longer than 8 ancestors keeping the nearest 7 and the root.

Measured on 376 new replies (45 AI-list members and 45 Discover authors, a different sample from the
first round), labelled by Sonnet seeing the whole thread ($0.47); 130 are about AI. <!-- @evidence[[results/reply_thread.json#sonnet_positive]] -->

| model input | accuracy | recall | false positives |
|---|---|---|---|
| reply only | 0.838 | 0.538 | 0.4% |
| reply, then parent (previous round's page) | 0.904 | 0.785 | 3.3% |
| reply, then parent and root | 0.928 | 0.869 | 4.1% |
| reply, then the whole chain (shipped) | 0.923 | 0.877 | 5.3% |
| the whole chain, then the reply | 0.912 | 0.869 | 6.5% |

On the 99 replies at least three posts deep, the whole chain finds 97.8% of the AI ones against 86.7%
for the parent alone, with false positives at 7.4% against 0%. Parent and root alone is the more
accurate input overall (0.970 on that subset, 1.9% false positives); the whole chain shipped because
it is what a reader of the thread sees, and the two differ by 2 replies of 376 overall. On the two
threads that prompted this round, the whole chain scores 74% and 44% (2% and 25% on the reply alone).
The second stays under the line: its thread is about EmbeddingGemma on GCP A100s, which the 17M model
does not read as AI strongly enough. A retrain on thread-aware labels is the fix for both that and
the false positives.

