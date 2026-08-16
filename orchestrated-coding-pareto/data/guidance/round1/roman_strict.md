# Review: `roman_strict.py` (orch-haiku, round 1)

**Verdict:** the structure of this solution is sound. There is exactly **one** root cause, it is
a single missing check, and it is not in the parsing logic. Do not rewrite the module — make the
one targeted change described below and leave everything else alone.

Test result being explained: `test_invalid_raises[MMMM]` — `DID NOT RAISE ValueError`.

---

## Root cause 1 — the output range bound `1..3999` is never enforced

**Spec rule violated:** *"Parse a strictly classical Roman numeral into an integer in **1..3999**"*
(spec line 9). `"MMMM"` is 4000, which is outside that range, so it must raise `ValueError`.

**Why the existing validation misses it.** The module's only canonicity gate is the round-trip on
lines 53–56:

```python
canonical = _to_roman(total)
if canonical != s:
```

That check is genuinely correct as a *canonicity* test, but it is **not** a *range* test, because
`_to_roman` (lines 61–77) is unbounded. Its greedy loop takes `count = n // 1000` with no cap, so
`_to_roman(4000)` happily returns `"MMMM"`. The chain for the failing input is therefore:

1. line 20 — every char is in `IVXLCDM`, passes.
2. lines 39–51 — no descending-value pair, so four additive `M`s sum to `total = 4000`.
3. line 54 — `_to_roman(4000) == "MMMM" == s`, so the round-trip *agrees* and the function returns
   4000 instead of raising.

The round-trip is self-consistent above 3999, so it can never catch this class on its own. The same
hole passes `"MMMMM"` (5000) and `"MMMMCMXCIX"` (4999); `MMMM` is just the smallest witness the
hidden suite happened to use. Note the spec's list of consequences (lines 17–21) never mentions
overflow — the bound comes from the signature sentence on line 9, which is easy to read past. That
is the actual reasoning error here: treating the bulleted "consequences" as the complete set of
rejection rules.

**What a correct approach does.** Enforce the numeric bound explicitly as its own condition, after
`total` is computed and before returning — reject when `total` is not within `1 <= total <= 3999`,
raising `ValueError`. It may sit on either side of the round-trip comparison; both paths raise
`ValueError`, so ordering does not affect observable behavior. Equivalently you may bound
`_to_roman` (make it raise or refuse for `n > 3999`), but the explicit bound on `total` in
`from_roman` is the clearer expression of the spec sentence.

Two traps while making this change:

- **Do not** fix it as "reject more than three consecutive `M`s" or by counting `M`s. That is a
  symptom fix aimed at the one failing test string; the spec constrains the *value*, and a
  character-count rule re-derives the bound indirectly and will read as coincidence to the next
  reader.
- **Do not** clamp, saturate, or return a truncated value. The required behavior is to raise
  `ValueError`, not to produce a nearest in-range integer.

---

## Verified as *not* defects — do not "fix" these

A one-shot fix is easiest to get wrong by changing more than necessary. The following were checked
against the spec and are correct as written:

- **`valid_subtractive` (lines 27–31)** correctly encodes I→{V,X}, X→{L,C}, C→{D,M}, so `"IC"`,
  `"IL"`, `"XM"`, `"XD"`, `"VX"`, `"IM"` all raise at line 44. Leave it.
- **The character whitelist (lines 19–21)** rejects lowercase, mixed case, whitespace and any
  non-Roman character in one pass, before any `values[...]` lookup — so the parse loop cannot
  `KeyError`. Leave it.
- **Empty string (lines 15–16)** raises. Leave it.
- **Non-canonical repetition and wrong ordering** (`"IIII"`, `"VV"`, `"IXIX"`, `"IVI"`, `"CMCM"`)
  are all caught by the round-trip, which is a *sound* canonicity test for in-range values: if
  `_to_roman(total) == s`, then `s` is by definition the greedy encoding of `total`. The redundancy
  between the subtractive-pair check and the round-trip is harmless.
- **`_to_roman` being a second module-level name** does not violate "exactly one public function" —
  the leading underscore makes it private. No need to inline or nest it.

Everything else in the file already passes; 34 of 35 hidden tests are green.
