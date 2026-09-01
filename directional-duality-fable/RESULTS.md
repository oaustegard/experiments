# Directional-duality probe, paired: claude-fable-5 vs claude-fable-5-1

Muninn, 2026-09-01. Rebuild of the 2026-07-22 self-probe on claude-fable-5
(memory 84edd20f; never filed). Same design: walk a single measurable variable
out of training density (digit length of addition, run length in a count) and
compare against a decorated frame at matched difficulty. This time both models
run in the same harness on the same seeded operands, so the 5 vs 5.1 comparison
is within-run.

## Protocol

- `probe.py`. Raw Messages API, no system prompt, no temperature (deprecated on
  fable-5), `max_tokens=16000` so thinking cannot starve the answer.
- Addition, plain frame: 16, 40, 60, 90 digits (seed 42), 150, 250 (seed 4242,
  refills seed 777). Decorated frame (Neptune fog bank, from the July 23
  companion) at 40 digits. Count: N copies of `hare`, N in {80, 200, 500, 1000}.
- 3 trials per cell. Score: last integer in the text, exact match. Empty text and
  `stop_reason: refusal` are harness events, never model errors.
- 88 calls total.

## Results (correct / valid trials; thinking tokens per trial)

| axis | n | fable-5 | fable-5-1 |
|---|---|---|---|
| add plain | 16 | 3/3 (338–620) | 3/3 (0, 0, 327) |
| add plain | 40 | 3/3 (400–466) | 3/3 (324–469) |
| add plain | 60 | 3/3 (625–1025) | 3/3 (418–729) |
| add plain | 90 | 3/3 (1037–1415) | 3/3 (752–1234) |
| add plain | 150 | 2/2, 3 refused (2354–2767) | 4/4, 1 refused (1825–1888) |
| add plain | 250 | 4/4, 1 refused (3366–5759) | 2/2 + 1 parse-ambiguous¹, 2 refused (2574–2809) |
| add decor | 40 | 3/3 | 3/3 |
| count | 80 | 3/3 | 3/3 |
| count | 200 | 3/3 | 1/3 — misses 220, 220 |
| count | 500 | 0/3 — 400, 365, 400 | 2/3 — miss 446 |
| count | 1000 | 0/3 — 700, 700, 700 | 3/3 |

¹ Response was a correct digit string that the last-integer regex split; re-fetch
of the same prompt returned the correct sum as one token. Classed `other`, not a
miss.

## Findings

1. **The July addition cliff on fable-5 (40–90 digits) did not reproduce.**
   fable-5 is 3/3 at 90 and 4/4 at 250 today with 1–6k thinking tokens. The July
   run reported 90d 0/2 after "fixing" the harness; its budget after the fix was
   not recorded. Second time this probe's headline cliff has moved when the
   budget changed. No addition cliff exists through 250 digits for either model
   at this budget, and the operand-length axis is now bounded by the refusal
   classifier, not by arithmetic.

2. **The count axis still degrades for fable-5 and the misses are round-number
   estimates** (400, 365, 400 at 500; 700 ×3 at 1000). Same texture as July's
   350/470. fable-5-1 counts 1000 correctly 3/3 but misses 200 twice at exactly
   220 — an overcount that repeats, so it is a systematic chunking error, not a
   ballpark guess. 5.1's count behaviour is non-monotone in N on this sample; the
   cliff is gone but a bump appeared.

3. **Decorated frame flat**, both models, as in July and the Haiku/Flash-Lite
   companion. Unchanged.

4. **Harness: long random digit strings trip the API refusal classifier.** 7 of
   22 calls with 150- or 250-digit operands returned `stop_reason: refusal` with
   zero output tokens, on both models, dependent on the specific operands. This
   is a second cause of empty text alongside the July max_tokens starvation, and
   it caps how far the digit axis can be walked through the public API.

5. **Thinking spend**: 5.1 used fewer thinking tokens than 5 at every addition
   length from 60 up (250d: 2.6–2.8k vs 3.4–5.8k) and reached the same answers.

## Caveats

n=3 per cell; a probe, not a result. Bare API, no product harness. The July
harness is reconstructed from its memory record, not reused, so cross-run
comparison with July is weaker than the within-run 5 vs 5.1 comparison.
