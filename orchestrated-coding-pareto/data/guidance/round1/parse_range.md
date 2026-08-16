# Review: `parse_range` (round 1)

All 7 failures trace back to how a token is split into `A` and `B`. The comma
handling, the empty-token check, and `sorted(set(...))` are all correct — do not
change those. The token-level parse needs to be replaced.

---

## Root cause 1 — the range separator is discovered by brute force, so a `-` sign is mistaken for the operator

**Spec rule:** "Each token is either a single integer or a range `A-B` (inclusive
on both ends)" — the token has a *grammar*; `B`'s leading `-` is only a sign when
a separator hyphen already appeared.

**Offending code:** lines 21–34.

```python
for i in range(1, len(token)):
    left = token[:i]; right = token[i:]
```

This tries every character boundary, keeps every boundary where both halves
happen to parse as `int`, and then takes `valid_splits[0]` — the *leftmost* one.
No boundary is required to sit on a hyphen at all.

For `"1-3"` the very first boundary is `left="1"`, `right="-3"`. Both parse. So
the token is read as the range `1 → -3`, `left_val > right_val`, and line 36
raises. That single mechanism is what fails `test_simple_range`,
`test_mixed_sorted_dedup`, `test_overlap_dedup`, `test_whitespace`, and
`test_single_element_range` — every one of those five reports
`ValueError: Range A > B` at line 36 for input that is plainly well-formed. The
`A > B` check itself is right; it is being fed garbage.

The same mechanism produces the *wrong answer* rather than an exception in
`test_negative_to_positive`: for `"-2-1"` the leftmost parseable boundary is
`left="-2"`, `right="-1"`, giving `[-2, -1]` where the spec requires the range
from `-2` to `1` → `[-2, -1, 0, 1]`. The trailing `1` was silently absorbed as a
sign.

**What a correct approach does:** decide the split *structurally* instead of by
trial. Match the whole stripped token against an anchored grammar in which the
separator is an explicit hyphen sitting between two signed integers, e.g. an
anchored pattern equivalent to

```
^(-?\d+)\s*-\s*(-?\d+)$
```

Applied to `"1-3"` this can only yield `A="1"`, `B="3"` — there is no boundary
freedom, because `(-?\d+)` cannot stop mid-number and still let the rest anchor.
Applied to `"-2-1"` it yields `A="-2"`, `B="1"`. If you prefer not to use `re`,
the equivalent hand-rolled rule is: skip one optional leading `-`, consume
digits, and the *next* `-` (after optional whitespace) is the separator — the
first hyphen that is not in sign position.

Order the two cases: try the range form first; only if it does not match, try the
single-integer form; if neither matches, raise `ValueError`.

## Root cause 2 — a negative lower bound written as `A--B` has no representation in this parser

**Spec rule:** "`-3--1` is the range from -3 to -1 (i.e. `[-3, -2, -1]`)."

**Offending code:** the same loop, and the fallback at lines 40–43.

For `"-3--1"` *no* boundary yields two parseable halves: the candidates are
`"-3"/"--1"`, `"-3-"/"-1"`, `"-3--"/"1"` — each has one invalid side. So
`valid_splits` is empty, control falls through to `int("-3--1")`, and line 43
raises `Invalid token`. That is exactly the traceback in the output
(`invalid literal for int() with base 10: '-3--1'` → `ValueError: Invalid token`)
and is why `test_negative_range` fails.

Note this is not merely "the same bug again": root cause 1 is *choosing the wrong
split*, this is *finding no split at all*. Both disappear under a grammar-based
parse, because there the separator hyphen and `B`'s sign hyphen are distinct
symbols in the pattern: in `^(-?\d+)\s*-\s*(-?\d+)$`, `"-3--1"` binds `A="-3"`,
separator `-`, `B="-1"`. Verify your fix against all four sign combinations —
`"1-3"`, `"-3--1"`, `"-2-1"`, `"5-5"` — before you consider it done.

Also confirm your grammar still *rejects* what it must: `"1-2-3"` and `"--3"`
must raise, and they do under the anchored pattern (nothing can consume `"1-2"`
as a single integer, and `"--3"` has two signs). These currently pass by
accident; do not lose them.

## Root cause 3 — whitespace is handled by deleting spaces, which both under- and over-accepts

**Spec rules:** "Whitespace is allowed around tokens and around the range hyphen"
*and* "any token that is not a valid integer or integer range" raises.

**Offending code:** line 18, `token = token.replace(' ', '')`.

Two defects, neither currently caught by a visible test but both live risks on a
one-shot fix:

- It deletes spaces *everywhere*, so `"1 2"` becomes `"12"` and is accepted as
  the integer 12. The spec requires `ValueError` — `"1 2"` is not a valid
  integer or range. Same for `"1 - 3 - 5"`-style malformations that collapse into
  something parseable.
- It only removes the literal space character. Tabs and newlines are whitespace
  too; `token.strip()` on line 16 clears them at the edges, but an interior tab
  around the hyphen would not be handled, while an interior *space* would — an
  inconsistency with no basis in the spec.

**What a correct approach does:** do not mutate the token's interior. Strip the
token's ends once, and let the grammar itself allow optional whitespace only
where the spec permits it — around the separator hyphen (`\s*` on both sides of
it in the pattern above). Interior whitespace anywhere else then fails to match
and correctly raises.

---

## Things that are already right — leave them alone

- `s.strip()` + early `return []` for empty/whitespace-only input.
- Splitting on `,` and rejecting any token that is empty after `.strip()` — this
  correctly covers `"1,,3"`, `",1"`, `"1,"`.
- `if left_val > right_val: raise ValueError` — keep it, applied to a correctly
  parsed `A`/`B`.
- `result.extend(range(left_val, right_val + 1))` (inclusive upper bound).
- `return sorted(set(result))`.

## One caution on the single-integer path

`int(token)` is looser than the spec's notion of "a valid integer": it accepts
`"+5"`, `"1_000"`, surrounding whitespace, and non-ASCII digits. If you validate
the single-token form with an anchored `^-?\d+$` before calling `int`, you get
the strict behavior for free and stay consistent with the range path. Whichever
you choose, apply the *same* notion of "integer" to `A`, `B`, and the standalone
token.
