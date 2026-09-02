# bts-coordinates — pre-registration

**Written before any experimental data was collected.** Required by
`thesis-discipline-check` (ops), which exists because two prior campaigns
locked a conclusion early and then spent six arms elaborating it.

## Question

Large Discovery Models (arXiv:2608.15669, §4.2) fit their surrogate over a
**growing** feature set: when the proposer finds a new mechanism, that
mechanism enters the value model as a new coordinate and the model is refit.
Their ablation freezes the feature set at iteration 1 and converges to a
strictly worse plateau.

Our own singular-learning-theory objection (memory `a8b97f70`, 2026-05-20)
says that if novelty appears as a *new dimension* of representation space,
no search over a fixed flat embedding can find it. We published that
objection. We never tested a mechanism against it, because we did not have
one. LDM supplies one.

**So: does an LLM-named, growing coordinate system plus an acquisition-ranked
surrogate surface a cross-field prior-art target from a fixed candidate pool
in fewer expensive reads than flat dense similarity — and does the *growth*
of the coordinate set carry the difference, or only the naming?**

## Thesis date and prior

Thesis formed **2026-09-01**, after one reading of the LDM paper, on zero
data of our own. My prior before running anything: **B < C < A** in reads to
hit (B best). That prior is exactly the thing this pre-registration exists to
protect against. Recorded so the writeup can be scored against it.

## Arms

All three rank the **same fixed candidate pool** per test case. They differ
only in how the ranking is produced.

| arm | mechanism | what it tests |
|---|---|---|
| **A — flat dense** | cosine between query text and candidate text, one shot, bge-small-en-v1.5 | the control our published BTS claim is about |
| **B — growing coordinates** | LLM names features → candidates scored on them → ridge/GP surrogate on observed verdicts → UCB ranks the unread → top-b get read → **on a plateau, the proposer names new features and the surrogate is refit on the widened matrix** | the LDM transplant |
| **C — frozen coordinates** | identical to B, except no features are added after round 1 | LDM's own no-discovery ablation, isolating growth from naming |

"Read" = an LLM verifier judging, from title+abstract, whether the candidate
actually contains the described result. It is the expensive step and the unit
of budget. Cost is reported in reads, not wall-clock.

## Pre-registered null

If the growing-coordinate mechanism contributes nothing:

- **B and C report the same reads-to-hit**, within seed noise. The plateau
  branch in B either never fires, or fires and buys nothing.
- If *naming* helps but *growth* does not, B ≈ C and both beat A. This is a
  distinct outcome from the above and is the one I expect to be hardest to
  tell apart at small n. It is reported as its own verdict, not folded into
  a win.
- If the SLT objection is simply wrong for this corpus, **A hits inside the
  first few reads** and there is nothing for B or C to improve. This is a
  live possibility and would be a correction to our own published post, so it
  is reported loudly rather than buried.

A metric that cannot separate these three readings is not diagnostic. The
separation is: reads-to-hit for A alone answers the third; B−C answers the
first two.

## Confounds, named before they bite

1. **Doerr's abstract is elided by the publisher** (S2 returns
   `abstract: null`; confirmed again today). The target would enter the dense
   arm with title-only text while every distractor carries title+abstract —
   a length and information asymmetry that would hand arm A a loss it did not
   earn. **Primary condition is therefore title-only for all documents in all
   arms.** Title+abstract runs as a secondary condition where abstracts exist.
2. **Pool construction must not use the answer.** For cases whose target is
   on arXiv, the pool is the union of arXiv keyword searches over term
   hypotheses that were extracted *blind* (July 2026, no access to the
   answer). For P1 the target is not on arXiv and can never appear in such a
   pool, so P1's pool is assembled the same way and the target is then
   **injected explicitly**. P1 therefore tests **ranking, not retrieval**, and
   every P1 number carries that label. Conflating the two is the failure this
   note exists to prevent.
3. **Verifier leakage.** The verifier sees only title+abstract and the query.
   It is never told which paper is the target, and the same verifier prompt
   runs in all three arms.
4. **Small n.** Test cases are single instances, not samples. No arm
   difference under ~2x in reads will be called a result.

## Kill criteria

- If arm A hits the P1 target within the first 10 reads, the premise is dead:
  report that flat embedding already solves the case and stop.
- If the pool for P2 does not contain MSW25, the corpus builder is broken;
  fix it before running arms, do not score it.
- If B and C are within one read of each other on every case, report the
  growing-coordinate mechanism as not transferring, and say so as the headline.

## Scheduled adversary

Before writing any positive conclusion, a subagent is given the results and
the raw per-candidate scores with the instruction to break the finding —
specifically to check whether an arm B win is explained by the feature-naming
LLM having leaked the answer into a feature name. That check runs **before**
the writeup, not after.

## Scope

Cases run: **P1** (Doerr 2004, cross-field, the hard one), **P2** (MSW25,
same-field control, the gate that killed the July project), **N3** (synthetic
pseudo-problem, false-alarm floor). P3/P4/P5 and N1/N2 are not run; the
budget goes to the contrast that answers the question rather than to
breadth. Stated as a scope cut, not an omission.

## Provenance

Test-case text is verbatim from `claude-workspace` PR #180
(`experiments/prior-art-probe/testcases.md`, issue #179). The blind term
hypotheses reused for pool construction are from that PR's
`p2_signature_framing_a.md` / `_b.md`, generated by subagents with no access
to the answers.

---

## Amendment 1 (2026-09-01, after corpus construction, before any arm was run)

Two changes, both forced by data collected before scoring:

**A. Retrieval depth was the P2 miss, not the queries.** The first P2 pool
(15 blind July queries × 40 results) did not contain MSW25, tripping kill
criterion 2. Diagnosis: the blind query `two-terminal series-parallel network
reduction` *does* retrieve the target, at **rank 77**. Depth 40 cut it off.
Fixed to depth 100 for every case, decided once and applied uniformly. No
query text was changed, and no answer vocabulary was added. Recorded because
"tune the corpus until the target appears" is the exact leakage this document
exists to prevent — the fix is a depth constant, not a query edit.

**B. A fourth arm, because the naming step may already be the whole result.**
The blind P1 extraction (subagent, **0 tool uses**, given only the
discrepancy-stripped flow text) emitted `linear discrepancy totally unimodular
matrices` as its highest-confidence query. That string is the verbatim title
of the P1 target. It also derived "network matrix, hence totally unimodular"
unprompted in its structural description.

If a single naming step hands over the target's exact term of art, then any
downstream ranking difference between B and C is measuring the wrong thing.
So:

| arm | mechanism |
|---|---|
| **N — naming only** | one blind extraction, then rank candidates by dense similarity to the *extracted signature* rather than the raw query. No surrogate, no rounds, no growth. |

Pre-registered reading: if **N ≈ B ≈ C and all beat A**, the verdict is that
the naming step carries the result and LDM's growing-coordinate machinery adds
nothing on this task. That is a negative result for the transplant and a
positive one for the cheaper mechanism, and it will be reported that way.

**C. Confound added to the list.** The extraction agent may be *recognising*
this problem rather than deriving the bridge — Morell–Skutella, Swamy and
Doerr are all public and old. The campaign writeup postdates the model's
June 2026 cutoff, but the underlying mathematics does not. This experiment
therefore cannot distinguish "the model derived the bridge" from "the model
already held the bridge and we finally asked". The honest claim available is
the second one, which is also what memory `3d35c1e0` recorded as this
pipeline's real operating envelope in May.
