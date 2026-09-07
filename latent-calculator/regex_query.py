"""Deterministic query extraction: the null model for the asking side.

The operands are literals the user typed, so they are lifted from the prompt
by span-bound regex rather than re-read out of the residual stream
(`nl2sh-retrieval/extract_params.py`, `monad-bsky` repair.py: a model must
never retype an identifier it was given).  Operator from cue words; operand
order from the one template that reverses it ("Subtract {b} from {a}.").
Returns (op, a, b) or None when the prompt carries no two integers.
"""
import re

OPS = ["add", "sub", "mul", "cmp"]
_CUES = [
    ("cmp", r"\blarger\b|\bcompare\b|\bgreater\b|\bless\b|\bsmaller\b|\bbigger\b"),
    ("mul", r"\*|\btimes\b|\bmultipl|\bproduct\b"),
    ("sub", r"(?<![\w-])-\s|\bminus\b|\bsubtract\b|\bdifference\b|\bless\s+\d"),
    ("add", r"\+|\bplus\b|\badd\b|\bsum\b|\btogether\b|\btotal\b"),
]
_INT = re.compile(r"(?<![\w])-?\d+(?![\w]|\.\d)")


def extract(prompt):
    nums = [m.group(0) for m in _INT.finditer(prompt)]
    if len(nums) < 2:
        return None
    a, b = nums[0], nums[1]
    low = prompt.lower()
    op = None
    for name, pat in _CUES:
        if re.search(pat, low):
            op = name
            break
    if op is None:
        return None
    if op == "sub" and re.search(r"\bsubtract\b.*\bfrom\b", low):
        a, b = b, a
    return op, int(a), int(b)


def calculate(op, a, b):
    if op == "add":
        return str(a + b)
    if op == "sub":
        return str(a - b)
    if op == "mul":
        return str(a * b)
    return "greater" if a > b else "less" if a < b else "equal"


def results_for(rows):
    """Result strings for dataset rows; a blank string where extraction fails."""
    out = []
    for r in rows:
        q = extract(r["prompt"])
        out.append(calculate(*q) if q else "")
    return out
