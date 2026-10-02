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

# Phase 2: Custody Register (written before the full run, after a 3-chunk pilot where all three conditions scored 1.0)

40 assets, 16 handovers per chunk among 120 filler lines. 32 chunks = 512 handovers, whose lines
(~15k tokens) fit the 24k cap; 96 chunks = 1,536 handovers (~46k tokens), which do not. Final
answer by one Haiku 4.5 call without tools in every condition.

6. At 32 chunks `jev` and `jev_fifo` get holders right (≥ 0.9), since every line fits and the
   answerer only needs the last mention of each asset, but miss many counts (exact < 0.7):
   counting up to ~21 mentions per asset across 512 lines in one pass is where a tool-less
   reader slips.
7. At 32 chunks `state` matches on holders (≥ 0.9) and beats Jev on exact counts, because it
   increments as it goes; single misses still accumulate, so not 1.0.
8. At 96 chunks `jev` (evicting by score) loses both holders and counts (≤ 0.5 each): its
   scores do not encode recency, so the lines it drops are arbitrary in time.
9. At 96 chunks `jev_fifo` keeps holders high (≥ 0.8; most assets move within the last ~50
   chunks) and counts collapse toward 0, since the early handovers are gone.
10. At 96 chunks `state` has the best holders and counts of the three; exact counts fall below
    its 32-chunk level. Relay cost about $0.12 per chunk.

# Phase 3: combined register + recent lines (written before the run, after a 3-chunk pilot at 1.0)

`both`: per chunk, Jev flags likely handovers; the subagent sees the flags and updates the
register (8k reserved); flagged lines also go to a most-recent verbatim store (16k reserved).
The answer call reads the register, then the recent lines.

11. Holders 40/40 at both sizes, as for `state`.
12. Fewer missed handovers than `state` at 96 chunks (≤ 3 counts wrong against 7): the flags act
    as a second check on the subagent's reading, and Jev's per-line misses are unlikely to fall
    on the same lines as Haiku's.
13. Cheaper per chunk than `state` (pilot $0.075 vs $0.12), because a flagged list lets the
    subagent skip reasoning over all 136 lines.
14. The recent-lines store does not hurt the one-pass answer: with the register labelled
    authoritative, the answerer copies it.
