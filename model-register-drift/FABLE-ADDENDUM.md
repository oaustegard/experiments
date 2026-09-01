# Fable 5.1 addendum (2026-09-01)

Two bare API samples from `claude-fable-5-1` (no system prompt; prompt reconstructed from
RESULTS.md, not the verbatim issue-244 prompt; `max_tokens` 6000 / 2500). Both over the 900-word
ceiling (984 / 907 prose words). Blind labels S11 = fable-5-1-a, S12 = fable-5-1-b
(`blind_key_fable.json`). Each judged against all ten existing samples plus each other, both
positions, Opus 5 via API with the round-2 two-question prompt, 22 verdicts
(`verdicts_fable.json`). Bradley-Terry over round-2 verdicts + these.

STAGING by model mean: opus-5 +1.90, opus-4-8 +1.26, haiku-4-5 +0.41, opus-4-6 +0.23,
sonnet-5 -0.44, sonnet-4-6 -1.40, fable-5-1 -2.06. fable-5-1-a [-2.73,-0.70], -b [-4.19,-1.50].
AI by model mean: haiku-4-5 +2.61 ... opus-5 -2.00, fable-5-1 -2.52. fable-5-1-b 0/12.
Lint: fable-5-1-a 3.05/1k, -b 9.92/1k; em-dash 0.15 and 0.0 per 150w.

Residual register in both samples: bold-lead enumerated causes, bolded maxim takeaways, an
"X not Y" closer, and the literal phrase "the one/part that stung" in both. Invented numbers
converge across the two independent samples (db.r6g.2xlarge, ~4k rps, 5-min TTL, pgbouncer,
99.6%/99.7% buffer hit ratio).

Caveats: judge ran via API rather than subagent; Fable samples have n=12 vs n=8/4 for the rest;
two samples; prompt not verbatim.

sample                  words   hits   per1k   staging   flat    struct  dash/150  short%   1-sent
----------------------  ------  -----  ------  --------  ------  ------  --------  -------  -----
fable-5-1-a             984     3      3.05    1.02      2.03    9       0.15      17.2     6    
opus-5-b                802     4      4.99    2.49      2.49    7       1.12      6.1      2    
opus-5-a                794     4      5.04    1.26      3.78    6       1.32      4.2      1    
sonnet-4-6-a            748     4      5.35    4.01      1.34    13      1.4       26.2     1    
opus-4-6-a              852     5      5.87    3.52      2.35    10      1.94      17.5     3    
opus-4-8-a              826     5      6.05    2.42      2.42    4       1.45      4.0      0    
sonnet-5-a              868     7      8.06    6.91      0.0     7       2.94      20.8     1    
sonnet-4-6-b            817     7      8.57    4.9       1.22    10      0.73      14.1     3    
haiku-4-5-b             862     8      9.28    4.64      3.48    10      0.52      22.7     5    
fable-5-1-b             907     9      9.92    5.51      4.41    7       0.0       5.7      2    
PROVENANCE              157     2      12.74   0.0       6.37    1       0.0       11.1     0    
haiku-4-5-a             919     12     13.06   5.44      7.62    10      0.49      14.5     7    
opus-4-8-b              819     13     15.87   6.11      4.88    6       1.65      6.8      3    

categories by sample
  fable-5-1-a: header 5, editorializing 2, cadence 2, density 2, significance 1
  opus-5-b: header 5, em-dash 2, triad 1, editorializing 1, cadence 1, density 1
  opus-5-a: header 3, density 3, triad 2, em-dash 1, participle 1
  sonnet-4-6-a: header 5, cadence 3, density 3, typography 2, negation-first 1, significance 1, triad 1, em-dash 1
  opus-4-6-a: header 4, typography 3, editorializing 2, density 2, negation-first 1, locator 1, staging 1, cadence 1
  opus-4-8-a: header 3, triad 2, em-dash 2, flat-certainty 1, density 1
  sonnet-5-a: negation-first 3, header 3, density 3, significance 2, em-dash 1, reversal 1, typography 1
  sonnet-4-6-b: header 5, density 3, em-dash 2, flat-certainty 2, negation-first 1, triad 1, rtfm 1, typography 1
  haiku-4-5-b: header 5, cadence 3, negation-first 2, list-shape 2, density 2, significance 1, rhetorical-q 1, participle 1
  fable-5-1-b: header 6, triad 4, negation-first 3, significance 2, density 1
  PROVENANCE: spec-ese 1, juridical 1, header 1
  haiku-4-5-a: header 4, participle 3, typography 3, negation-first 2, significance 2, triad 2, list-shape 2, cadence 2
  opus-4-8-b: em-dash 4, flat-certainty 4, triad 3, header 3, density 2, negation-first 1, participle 1, reuse 1

