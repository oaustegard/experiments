# Plan: does a 1-bit index hold up for Bluesky Answer Search?

Question (Oskar, 2026-11-04): test the accuracy of a 1-bit index for austegard.com/bsky/answer-search,
whose 2-bit codes are 19.2 MB of a 37.7 MB page. A 1-bit index would be 9.6 MB (page about 28 MB).

Fixture: the 100,000-post snapshot built 2026-11-04T04:08Z (data/snapshot.bin, gitignored; header in
results/fixture.json). Ettin-17m fp32 document vectors, the page's query prompt.

Arms: fp32 exact (reference) · 2-bit remex (what ships) · 1-bit remex (native) · 2-bit codes read at
precision 1 (one file serving both). All centered on the shipped mean, seed 42, scored with remex 2.1.0
search_adc, the formula the page's GPU scan implements.

Queries: 60 hand-written keyword queries in the page's style (queries.json, written before any run) and
200 post-snippet queries (the first 6 words of 200 seeded random posts with at least 8 words).

Metrics, against fp32: recall@10 and recall@40 of the post ranking (the page searches 40 posts); and the
page's own output: overlap of the 10 quoted sentences (best sentence per post, top 10, sentences embedded
fp32 in every arm), which is what a reader sees. Paired bootstrap CIs over queries.

Predictions (before running):
- P1: 2-bit recall@40 >= 0.85 on keyword queries; 1-bit native 0.65-0.80.
- P2: the quoted-sentence overlap is higher than recall@40, because sentence re-ranking forgives swaps
  near the cutoff: 2-bit >= 8.5/10, 1-bit 6-8/10.
- P3: 2-bit read at precision 1 is worse than native 1-bit (its top bit is not an optimal 1-bit code).
Decision rule: ship 1-bit only if its sentence overlap is within 1.0/10 of 2-bit on keyword queries.
