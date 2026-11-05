# PLAN: a Bluesky account classifier in the browser, labelled by Sonnet

Registered 2026-11-05, while the labels were being written and before any were read.

## Question

Oskar, after `browser-decider`: add a Bluesky account profile classifier that reads an account's latest
posts and answers a set of AI-themed choice questions (bot or human among them), plus a yes/no per post,
with a Sonnet subagent as the oracle. Can the 10 MB Ettin-17m decider learn this from Sonnet's labels, and
how many labelled accounts does it need?

## Fixture

1,842 accounts collected 2026-11-05 from the public AppView (`run/collect.py`): 495 members of Oskar's
"Artificial (and real) Intelligence" list, 411 accounts those members follow, 399 authors in the Discover
feed, 537 authors of English posts on the Jetstream firehose. The state (`run/state.py`) is the profile,
counts, an activity summary and the latest 12 feed items cut to 140 characters, fitted to 830 Ettin
tokens. Five account questions (`run/questions.json`): `account_type` (human, bot, organization,
aggregator), `ai_share` (score 0 to 3), `ai_role` (6 options), `ai_stance` (5), `original_content`
(score 0 to 2); and per item, the noul "this post is about AI".

Labels: Claude Sonnet 5.5 through `claude -p` (`run/label.py`), 5 accounts per call, a probability for
every option. A second pass over a fixed 150 accounts, in different batches, measures the oracle's
agreement with itself. Test set: those 150 plus 150 more random accounts, held out by account.

## Arms

1. **E-all**: stock Ettin-17m, the browser-decider recipe (cross-encoder, `[MASK]` per option, 10 epochs),
   trained on all training accounts, account questions and item questions together.
2. **Data curve**: the same on 100 and 300 training accounts.
3. **Baselines**: per question, the training-set majority label; for the item question, the List Scout
   logistic regression on Ettin-17m embeddings if it can be run on the same items.
4. **Browser**: the E-all build on the WGSL engine, an "enter a handle" page, timed on the Mac.

## Predictions

- P1. Oracle self-agreement (argmax, pass 1 vs pass 2) is at least 0.85 on `account_type` and the item
  question, and lower (0.65 to 0.80) on `ai_role` and `ai_stance`.
- P2. E-all agrees with the oracle within 0.10 of the oracle's self-agreement on every question.
- P3. `account_type`: E-all reaches 0.85 or more, above the majority baseline by at least 0.05; bots are
  rare (under 10% of accounts), so bot recall is reported on its own.
- P4. The data curve flattens: 300 accounts reach at least 90% of E-all's mean accuracy gain over the
  majority baseline.
- P5. Item question: E-all at least matches the List Scout regression's accuracy against the oracle.

## Decision rule

Ship the page if E-all meets P2 on `account_type`, `ai_share` and the item question.
