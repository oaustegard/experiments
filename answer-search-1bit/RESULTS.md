# Does a 1-bit index hold up for Bluesky Answer Search?

[Bluesky Answer Search](https://austegard.com/bsky/answer-search/) quotes sentences from the last 100,000
posts of a Bluesky list. Ettin-17m, a 17-million-parameter text embedder, turns the query into a vector; a
WebGPU scan over the posts' compressed vectors finds the 40 closest; Ettin then scores every sentence of
those 40 posts, and the best sentence of each, top 10, is what a reader sees. The compressed vectors are
remex codes: each dimension of a randomly rotated vector stored in 2 bits plus a per-post scale and bias.
They are 19.2 MB of the 37.7 MB page. The question: would 1-bit codes, half the size, change what the reader
sees?

## Answer

Would a 1-bit index change what the reader sees? Very little on this corpus. Against exact search with
uncompressed (fp32) vectors, the 1-bit index finds 84% of the same 40 posts on 60 keyword queries, against 92% for the
2-bit index that ships. But of the 10 sentences the page quotes, the same sentences appear as in the fp32
result in 9.33 of 10 at 1 bit <!-- @evidence[[results/summary.json#keyword.1bit.picks.0]] -->
and 9.82 at 2 bits <!-- @evidence[[results/summary.json#keyword.2bit.picks.0]] -->:
1-bit minus 2-bit is -0.48 sentences <!-- @evidence[[results/summary.json#keyword.1bit-2bit.picks.0]] --> (95% CI -0.67 to -0.32).
On 200 queries made from post snippets the post recall falls further (0.87 to 0.74) while the quoted
sentences still differ by under half a sentence (9.65 vs 9.18). That passes the rule set beforehand for
keyword queries (within 1.0 sentence of 2-bit) and would take the page from 37.7 MB to about 28 MB. It
measures agreement with fp32, not quality. Shipping needs a 1-bit path in the page's GPU scan (`tail.js`). Because each 1-bit code is exactly the top bit
of the 2-bit code, one file could serve both: load the top bits first and search, then upgrade when the
low bits arrive. Real user queries could change the gap.

## Findings

1. **What the reader sees changes by about half a sentence in ten.** Keyword queries: 9.82 / 10 quoted
   sentences match fp32 at 2 bits, 9.33 at 1 bit, a difference of -0.48 [-0.67, -0.32] <!-- @evidence[[results/summary.json#keyword.1bit-2bit.picks.0]] -->
   Post-snippet queries: 9.18 at 1 bit <!-- @evidence[[results/summary.json#snippet.1bit.picks.0]] -->
   against 9.65 at 2 bits <!-- @evidence[[results/summary.json#snippet.2bit.picks.0]] --> (rerun, `results/summary.json`)
2. **The post ranking loses more than the answer does.** A likely reason, not tested here: the sentence
   re-ranking draws its 10 sentences mostly from the best posts, while index errors mostly swap posts near
   the 40th place. Recall@40 against fp32 on keyword queries is
   0.84 at 1 bit <!-- @evidence[[results/summary.json#keyword.1bit.r40.0]] -->
   and 0.92 at 2 bits <!-- @evidence[[results/summary.json#keyword.2bit.r40.0]] -->;
   on snippets 0.74 at 1 bit <!-- @evidence[[results/summary.json#snippet.1bit.r40.0]] -->
   and 0.87 at 2 bits <!-- @evidence[[results/summary.json#snippet.2bit.r40.0]] -->. Recall@10 drops the same
   way, 0.91 to 0.82 on keywords (rerun, `results/keyword.json`, `results/snippet.json`).
3. **remex's 2-bit code contains the 1-bit code exactly.** For all 15,360,000 coordinates of 20,000 posts, the
   1-bit code equals the top bit of the 2-bit code (fraction 1.0 <!-- @evidence[[results/nesting.json#top_bit_equal_fraction]] -->),
   and scoring the 2-bit codes at `precision=1` (remex's option to read only the top bits) gives the
   same result as a 1-bit index on every query (difference 0.0
   <!-- @evidence[[results/summary.json#keyword.2bit@1-1bit.picks.0]] -->), so a page can ship the top bits
   and the low bits as separate sections and search after the first (rerun, `results/summary.json`).
4. **Size:** 96 + 8 bytes per post at 1 bit against 192 + 8 at 2 bits, so 9.6 MB less for 100,000 posts
   (`results/summary.json` bytes_per_post).

## Method

Fixture: the 100,000-post snapshot built 2026-11-04T04:08Z (`results/fixture.json`; the file itself is
gitignored in `data/`). Document vectors: Ettin-17m fp32 from the site's ONNX model via the builder's
`Ettin` class; queries use the page's query prompt.

Arms: fp32 exact (the reference), 2-bit remex (what ships), 1-bit remex, and the 2-bit codes read at
precision 1. All use remex 2.1.0 `Quantizer(768, seed=42, mean=<the shipped mean>)` and `search_adc`, the
score the page's GPU scan implements.

Queries: 60 keyword queries in the page's style, written before any run (`queries.json`), and 200
post-snippet queries (the first six words of 200 random posts with at least eight words, `random.Random(42)`).
The keyword queries were written by the experimenter (Claude), so they are a stand-in for real queries.
"Quoted sentences matching fp32" counts sentences, by post and text, that appear in both top-10 lists.

Metrics against fp32: recall@10 and recall@40 of the post ranking, and the overlap of the page's 10 quoted
sentences (best sentence per post, top 10; sentence vectors fp32 in every arm, so only the retrieved posts
differ). Paired bootstrap over queries, 5,000 resamples, 95% intervals. Code: `run.py embed`, `run.py eval`.

Costs: embedding 100,000 posts took 373 s on 4 vCPUs (shared with a browser test); the evaluation about
5 minutes.

## Log

### 2026-11-04: first run

Asked: test the accuracy of a 1-bit index (Oskar). Registered in `PLAN.md` before running.

| queries | arm | recall@10 | recall@40 | quoted sentences matching fp32 (of 10) |
|---|---|---|---|---|
| keyword (60) | 2-bit | 0.907 | 0.925 | 9.82 |
| keyword (60) | 1-bit | 0.820 | 0.839 | 9.33 |
| snippet (200) | 2-bit | 0.873 | 0.869 | 9.65 |
| snippet (200) | 1-bit | 0.750 | 0.744 | 9.18 |

Predictions against measurements: P1 (2-bit recall@40 >= 0.85; 1-bit 0.65-0.80) half right: 2-bit 0.92,
but 1-bit did better than predicted at 0.84. P2 (quoted-sentence overlap above recall@40; 2-bit >= 8.5,
1-bit 6-8) half right: the overlap is higher than recall and 2-bit is 9.8, but 1-bit is 9.3, above the
predicted range. P3 (2-bit read at precision 1 worse than native 1-bit) wrong: they are the same code.
The misses on P1 and P2 were pessimism about 1-bit; P3 was wrong about how the codes relate.

The first evaluation crashed on the post-snippet half: a post with no quotable sentence got no cache entry
when it was the only new post in a batch. Fixed and rerun; every number here is from the rerun, and its
keyword half matches the crashed run's keyword half exactly. The bit-level check in Finding 3 was added
after a fresh reader asked whether "contains" had been compared directly.
