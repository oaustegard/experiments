"""Unicode normalisation for cross-surface string matching.

Extracted from `te-bridges/scripts/te_common.py`.

The trap this exists for: NFKD does not decompose precomposed *stroke*
letters. `unicodedata.normalize("NFKD", s).encode("ascii", "ignore")`
turns "Odrzywołek" into "Odrzywoek" — the ł is dropped entirely rather
than folded to l, because it has no canonical decomposition. The same
holds for ø, đ, ð, þ, æ, œ, ß and Turkish dotless ı.

Silently dropping a letter breaks substring matching in a way that looks
like a data problem rather than a normalisation bug, so fold those
explicitly *before* handing the string to NFKD.
"""
from __future__ import annotations

import unicodedata

#: Letters NFKD will not decompose, and what to fold them to.
STROKED_LETTER_FOLD = str.maketrans({
    "ł": "l", "Ł": "L",    # Polish
    "ø": "o", "Ø": "O",    # Nordic
    "đ": "d", "Đ": "D",    # Croatian / Vietnamese
    "ð": "d", "Ð": "D",    # Icelandic / Faroese
    "þ": "th", "Þ": "Th",  # Icelandic / Old English
    "æ": "ae", "Æ": "AE",  # Latin / Nordic
    "œ": "oe", "Œ": "OE",  # French
    "ß": "ss",             # German sharp s
    "ı": "i",              # Turkish dotless i
})


def ascii_fold(s: str) -> str:
    """Strip diacritics, fold stroked letters, lowercase.

    Use for substring matching across mixed-Unicode and ASCII-normalised
    text surfaces — e.g. matching an author name from structured metadata
    against the same name as it appears in an abstract.
    """
    s = (s or "").translate(STROKED_LETTER_FOLD)
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
