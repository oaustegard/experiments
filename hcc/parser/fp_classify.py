"""Spellings of the classification builtins that take no f/l suffix (see hcc/builtins/fpclass.py)."""
from __future__ import annotations

# Plain names such as __builtin_isinf (ends in f) and __builtin_isnormal (ends in l) look
# suffixed, so they are listed explicitly; any other spelling's last letter is the suffix.
FP_CLASS_STEMS = frozenset({"isnan", "isinf", "isfinite", "finite", "isnormal", "isinf_sign"})
