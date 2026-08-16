# Task: roman_strict

Write a Python module defining exactly one public function:

```python
def from_roman(s: str) -> int
```

Parse a **strictly classical** Roman numeral into an integer in 1..3999.

Rules:
- Accepted symbols: I=1, V=5, X=10, L=50, C=100, D=500, M=1000 (uppercase only).
- Only the canonical (strict) form of each number is accepted — the form produced by the
  standard greedy encoding using the values
  1000=M, 900=CM, 500=D, 400=CD, 100=C, 90=XC, 50=L, 40=XL, 10=X, 9=IX, 5=V, 4=IV, 1=I.
- Consequences (all of these raise `ValueError`):
  - non-canonical repetition: `"IIII"` (use `"IV"`), `"XXXX"`, `"CCCC"`, `"VV"`, `"LL"`, `"DD"`
  - invalid subtractive pairs: `"IC"`, `"IL"`, `"XM"`, `"XD"`, `"VX"`, `"IM"`
  - wrong ordering: `"IXIX"`, `"XCXC"`, `"IVI"`, `"CMCM"`
  - lowercase or mixed case: `"iv"`, `"Xii"`
  - empty string, whitespace anywhere, or any non-Roman character
- Valid examples: `"I"` -> 1, `"IV"` -> 4, `"MCMXCIV"` -> 1994, `"MMMCMXCIX"` -> 3999.

No I/O. Standard library only. Only `from_roman` is tested.
