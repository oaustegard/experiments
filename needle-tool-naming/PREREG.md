# Pre-registration — does Cactus Needle 2 route on tool *names* or tool *descriptions*?

Written and committed **before any variant was run**. Predictions are numeric
where a number was available to predict against.

## Motivation

`needle-bsky` found two categories that no schema rewrite and no fine-tune could
move: `profile` (0.250 in all four arms) and `identity` (0.333 in the tuned
arms). Re-reading those failures against the tuned schemas:

* `get_profile`'s description already reads *"…display name, bio, **follower
  count**, following count, post count."* The query "how many followers does
  pfrazee.com have" returns `get_followers` at confidence 0.80.
* `resolve_identity`'s description already reads *"Look up the **DID** and
  hosting **PDS** server behind a handle."* The query "which pds hosts
  pfrazee.com" misroutes in all four arms.
* The one `identity` query correct in every tuned arm — "resolve did:plc:… to a
  handle" — is the one whose verb is the tool's **name**.

An oracle five-tool catalogue (the correct tool guaranteed present) moves
`profile` only 0.250 → 0.500 and leaves `identity` at 0.333, so this is
decode-side, not retrieval-side.

This is the *opposite* failure from the one the Cactus attention-only paper
(arXiv:2607.18363) predicts. That paper localizes the residual SAN deficit to
low-context tokens, "the positions with the least context to route from". Here
the discriminating fact is in context, verbatim, and is ignored anyway.

## Hypothesis

**H1.** Needle 2's routing is driven substantially more by lexical overlap
between the query and the tool **name** than by the content of the tool
**description**.

## Design

A 2×2 factorial over *what carries information*, holding the catalogue at 18
tools, the arity at `tuned-min` (required arguments only), and the query set at
`needle-bsky/evalset.jsonl` (62 queries, 54 routable, 8 off-topic — written for
a different purpose, before this hypothesis existed, and not modified here).

|                              | descriptions intact | descriptions stripped |
|------------------------------|---------------------|-----------------------|
| **names intact (canonical)** | `canon`             | `names-only`          |
| **names opaque (`tool_NN`)** | `desc-only`         | `neither`             |

Two further conditions vary name *quality* with descriptions intact:

* `separated` — every name rewritten as `<verb>_<distinguishing object>`, where
  the object is the noun phrase a user of *this* tool would say. Where two tools
  genuinely share a noun ("followers"), the head noun disambiguates
  (`follower_count` vs `follower_accounts`).
* `adversarial` — the `separated` names **cyclically rotated within confusable
  groups**, so the distinguishing term sits on a neighbour. Groups are fixed in
  advance in `names.py` and are not chosen from any result.

Stripping a description replaces it with the constant string `"A Bluesky API
operation."`, identical for all 18, and replaces every argument description with
`"value"`. Argument *names* (`handle`, `actor`, `q`) are kept in every arm: a
developer cannot avoid having them, so they are part of the floor, not part of
the manipulation.

Every arm is run twice: flat 18 declared, and with an oracle five-tool catalogue
(same seed and subset rule as `needle-bsky/oracle.py`), so a name effect on the
contrastive retrieval head is separated from a name effect on the decode.

## Predictions

Routable top-1 accuracy (54 queries), flat 18 declared, `tuned-min` arity.
`canon` is the same configuration `needle-bsky` measured at **0.611**.

| # | prediction |
|---|---|
| P1 | `canon` reproduces 0.611 exactly (harness check; the engine is deterministic) |
| P2 | `neither` ≤ 0.15 (chance over 18 tools is 0.056; argument names leak some signal) |
| P3 | **`desc-only` ≤ 0.35** — if descriptions carried the routing this would stay near 0.611 |
| P4 | **`names-only` ≥ 0.50** — losing every description costs less than 12 points |
| P5 | **`names-only` − `desc-only` ≥ 0.15** — this contrast *is* H1 |
| P6 | `separated` ≥ 0.69 |
| P7 | `adversarial` ≤ 0.41 |
| P8 | in `separated`, `profile` ≥ 0.75 and `identity` ≥ 0.75 |

**H1 is falsified if `desc-only` ≥ `names-only`.**

## Second question this answers

`needle-bsky` concluded "keep every agent at five tools" — the flat 18 arm cost
11 points against an oracle five and 3.6× the latency. If P6 holds, the flat-18
number under good names approaches the oracle-five ceiling (0.815 measured), and
declaring all 18 becomes defensible without a pre-filter. `separated` flat-18
against 0.815 is the number that decides it.

## What would make this uninteresting

If `neither` is not near chance, the argument names alone carry the routing and
neither factor is doing the work the design attributes to it. That is reported
as the result rather than worked around.
