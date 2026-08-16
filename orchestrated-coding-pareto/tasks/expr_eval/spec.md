# Task: expr_eval

Write a Python module defining exactly one public function:

```python
def evaluate(expr: str) -> float
```

Evaluate an arithmetic expression string. Do **not** use `eval`, `exec`, `ast`,
`compile`, or any expression-evaluation library — implement tokenizer and parser
yourself.

Grammar and semantics:
- Operands: integer and decimal literals (`3`, `2.5`, `.5`, `10.`). All arithmetic is
  done in Python floats; the result is a float.
- Binary operators: `+`, `-`, `*`, `/`, `%`, `**`.
- Precedence (low to high): `+ -` < `* /  %` < unary minus < `**` ... with the two
  standard Python quirks, which you must reproduce exactly:
  - `**` is **right**-associative: `2**3**2` = `2**(3**2)` = 512.
  - Unary minus binds **looser** than `**` on its left: `-2**2` = `-(2**2)` = -4.0,
    but a unary minus **immediately after** `**` binds tighter: `2**-1` = 0.5,
    `2**-2**2` = `2**(-(2**2))` = 0.0625.
- Unary minus (and unary plus) may be stacked: `--3` = 3, `+-+3` = -3.
- `/` is true division; `%` is Python's modulo (sign follows the divisor:
  `-7 % 3` = 2). Division or modulo by zero raises `ZeroDivisionError`.
- Parentheses group arbitrarily deep. Whitespace (spaces/tabs) is allowed anywhere
  between tokens and is insignificant.
- Errors — raise `ValueError` for every malformed input: empty/whitespace-only string,
  unbalanced parentheses, dangling operators (`"1+"`, `"*3"`), two operands in a row
  (`"1 2"`), invalid characters, malformed numbers (`"1.2.3"`), empty parentheses
  (`"()"`).
- `0**0` is 1.0 (Python semantics). `(-2)**0.5` may raise or return complex-ish
  results — it is **not tested**; don't special-case it.

Examples: `evaluate("1+2*3")` -> 7.0; `evaluate("-2**2")` -> -4.0;
`evaluate("2**-1")` -> 0.5; `evaluate("(1+2)*3")` -> 9.0; `evaluate("-7 % 3")` -> 2.0.

No I/O. Standard library only (`re` allowed). Only `evaluate` is tested.
