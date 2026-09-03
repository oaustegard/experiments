# Pass 4 — a static table compatible with a remax/remex-quantized SPECTER2 index

Question (Oskar, 2026-09-03): could we craft a static model whose query embeddings
work against a remex/remax-quantized SPECTER2 vector set?

Data, already on disk (do not re-fetch):
- `/home/user/oaustegard/remax/bench/.cache/SPECTER2/embeddings.npy` — (10000, 768) float32
  SPECTER2 vectors, UNNORMALIZED (norm mean 21.7). Ground truth in the remax bench is
  raw inner product, and centering is the biggest single lever there (see
  `/home/user/oaustegard/remax/bench/sketch_matryoshka.py` and
  `bench/results/SKETCH_MATRYOSHKA.md`).
- `/home/user/oaustegard/remax/bench/.cache/SPECTER2/texts.json` — 10000 strings,
  `"<title> [SEP] <abstract>"`, index-aligned with the vectors.
- SPECTER2 tokenizer: `/home/user/models/specter2_base/tokenizer.json` (BERT WordPiece,
  vocab 31,090). Drop `[SEP]` handling: just tokenize the whole string, max 512.
- remex 0.6.0 (pip), remax at `/home/user/oaustegard/remax/src`.

No SPECTER2 model forward passes are needed: queries are held-out PAPERS from the same
set, exactly as the remax bench does it. Reuse `fit_table.py` from pass 3 (parametrize
tokenizer path, vectors, texts).

Protocol (mirror sketch_matryoshka.py where it applies: seed 99, 100 queries):
1. Split: rng(99) picks 100 query papers; the remaining 9,900 are the corpus. Ground
   truth = top-10 corpus papers by raw inner product with the TEACHER query vector.
   Metrics R@10 and R@100 (the bench's), plus top-1 hit rate.
2. Fit the static table by ridge regression on the 9,900 corpus rows (bag-of-tokens X,
   teacher vectors Y, RAW not normalized, so the table learns the teacher's mean and
   scale). Choose alpha by 5-fold CV on the corpus. Also fit a second table with
   Model2Vec-style targets for comparison ONLY IF cheap: skip if it needs a model
   download; the token-level distill already failed on bekko and is not the question.
3. Student query vectors = mean of table rows for the query paper's tokens (no
   normalization; report both raw and L2-normalized variants if they differ in rank).
   Report cosine(student, teacher) on the 100 queries and on a 1,000-row held-out
   subset (refit on 8,900 for that number only, or use the CV folds).
4. Cells, each scored on the 100 queries against the 9,900 corpus:
   - teacher query / teacher float index (reference, should be R@10 = 1.0 by construction)
   - teacher query / teacher remax 1-bit index (centered SimHash, d=768 single signature,
     `StackedSignBitQuantizer(d=768, k=1, seed=99).fit(corpus)`) — the bench's ~0.62 R@10
   - teacher query / teacher remex 2-bit and 4-bit index (query stays float, ADC-style:
     decode codes, dot with query — same as run_code_quant.py)
   - student query / teacher FLOAT index
   - student query / teacher remax 1-bit index (encode the student query with the SAME
     fitted quantizer — its centering vector and rotation come from the teacher corpus)
   - student query / teacher remex 2-bit and 4-bit index
   - student query / student index (self-consistent, float) — for reference only
5. One more variant that targets the 1-bit case directly: after the ridge fit, report
   sign-agreement rate between student and teacher query vectors AFTER the quantizer's
   centering+rotation (fraction of the 768 bits that match). That number explains
   whatever the remax cell shows.

Output: `specter2_results.json` (rows: cell, index_codec, bits, R@10, R@100, top1) and a
new section appended to `RESULTS.md`. Same prose rules as before. State clearly which
cell answers Oskar's question (student query vs teacher quantized index) and give
the number without spin. Do not commit or push.
