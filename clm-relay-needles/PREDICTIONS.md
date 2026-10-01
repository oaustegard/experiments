# Predictions (written 2026-10-01 before the full run, after a 3-chunk pilot where all three conditions scored 1.0)

1. `jev` recall stays at or above 0.95 at every size: the handover rule is easy for a per-line
   classifier, and its kept lines are byte-exact by construction.
2. `state` recall falls with stream length and is below `jev` at 96 chunks, through some mix of
   missed lines, retyping errors (`corrupted` > 0) and cuts at the cap.
3. `both` recall is about `jev`'s at roughly `state`'s cost. The pilot already shows the subagent
   copying lines the filter kept despite being told not to, so `both` carries more tokens than
   `jev` and is the condition most likely to hit the cap.
4. Cost: `state` and `both` about $0.07 per chunk with Haiku 4.5, most of it output (thinking)
   tokens; `jev` under a cent per episode.
5. The labeled variant (the paper's format) is near 1.0 for both `jev` and `state`, as the paper
   reports for CLM on Needle Retention.
