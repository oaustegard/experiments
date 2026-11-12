### E-01 — Out-of-stack recipes cap at LOW
**Type.** avoid
**When.** The method is a training recipe or architecture for a system the reader does not run, linked to his stack only by a general idea.
**Guidance.** Name a component he could run within weeks; if none, cap at LOW, or NONE when the only overlap is a principle.
**Why.** Plain reviews inflated these to MEDIUM or HIGH on a cheap-looking snippet or ablation.
**Seen in.** 2610.06805, 2610.20331, 2610.14136, 2610.20533, 2610.31642

### E-02 — Check reliability before rating above LOW
**Type.** act
**When.** A single-run delta against a control or base, or judge or model output with no reported agreement or calibration.
**Guidance.** Compare the margin with the paper's own spread and control; if they cover it, or agreement is missing, rate LOW.
**Why.** Plain reviews treated small deltas and unvalidated labels as MEDIUM evidence.
**Seen in.** 2610.31642, 2610.18615, 2610.11959, 2610.32285

### E-03 — Shared vocabulary is not usefulness
**Type.** avoid
**When.** The subject uses a term from the reader's stack (memory, retrieval, routing, agents) but the mechanism does not run on his components.
**Guidance.** Judge only whether the mechanism runs on his components at his scale; a shared keyword does not raise the rating.
**Why.** Keyword overlap drove MEDIUM ratings where gains needed other domains or budgets.
**Seen in.** 2610.39302, 2610.24087, 2610.08077, 2610.10528

### E-04 — Artifacts count only if they run
**Type.** avoid
**When.** Code, weights or a snippet exist, but the headline depends on an unreleased or flagship model or on data outside his hardware.
**Guidance.** Count an artifact only if it runs on his hardware and reproduces the headline result; otherwise it stays LOW.
**Why.** Release status alone lifted papers to MEDIUM, though the released piece was smaller or lacked the gain.
**Seen in.** 2610.11959, 2610.24087, 2610.39302, 2610.20331

### E-05 — Infrastructure-bound methods stay LOW
**Type.** hold
**When.** Adoption needs a modified serving engine, from-scratch multi-GPU training, large RL budgets, or unreleased data, and nothing runs on his hardware.
**Guidance.** Keep LOW; do not raise it for a transferable idea unless he can test that idea with models he already runs.
**Why.** Plain reviews already land here, so raising it overstates the paper.
**Seen in.** 2610.18861, 2610.26825, 2610.30136, 2610.08077
