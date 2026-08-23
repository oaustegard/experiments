"""Directive conditions.

The design leans on one property of Baguettotron's tokenizer: capitalising a
directive keyword costs a variable number of extra tokens. "do not" -> "DO NOT"
costs nothing; "never" -> "NE|VER" costs one; "under no circumstances" costs
four. Holding the sentence frame fixed and flipping case within a keyword gives a
CAPS contrast whose token-count cost is known and, for two of the keywords, zero.
"""

# (name, lowercase directive, capitalised-keyword directive, keyword lowercase)
# {W} is the forbidden word.
KEYWORD_FRAMES = [
    ("do_not",  "Do not mention the word {W}.",
                "DO NOT mention the word {W}.", "Do not"),
    ("on_no_account", "On no account mention the word {W}.",
                "ON NO ACCOUNT mention the word {W}.", "On no account"),
    ("never",   "Never mention the word {W}.",
                "NEVER mention the word {W}.", "Never"),
    ("avoid",   "Avoid mentioning the word {W}.",
                "AVOID mentioning the word {W}.", "Avoid"),
    ("forbidden", "It is forbidden to mention the word {W}.",
                "It is FORBIDDEN to mention the word {W}.", "forbidden"),
    ("unc",     "Under no circumstances mention the word {W}.",
                "UNDER NO CIRCUMSTANCES mention the word {W}.", "Under no circumstances"),
]

# Length-matched set requested by issue #45 Q2: same semantics, four surface forms.
MATCHED = [
    ("m_lower", "You must never mention the word {W}.",         "never"),
    ("m_caps",  "You must NEVER mention the word {W}.",         "NEVER"),
    ("m_pad",   "You must not ever mention the word {W}.",      "not ever"),
    ("m_bold",  "You must **never** mention the word {W}.",     "**never**"),
    ("m_caps_all", "YOU MUST NEVER MENTION THE WORD {W}.",      "YOU MUST NEVER MENTION THE WORD"),
]

# Instrument validity checks.
CONTROLS = [
    ("none",      None,                                   None),
    ("mention",   "The word {W} may be relevant here.",    "The word"),
    ("positive_lower", "Always mention the word {W}.",     "Always"),
    ("positive_caps",  "ALWAYS mention the word {W}.",     "ALWAYS"),
]

# Directive after the question rather than before it.
POSITION = [("post_lower", "Never mention the word {W}.", "Never"),
            ("post_caps",  "NEVER mention the word {W}.", "NEVER")]


def caps_fraction(text, frac):
    """Capitalise the first `frac` of the text's word tokens, left to right."""
    words = text.split(" ")
    k = round(len(words) * frac)
    return " ".join([w.upper() for w in words[:k]] + words[k:])


DOSE_BASE = "You must never mention the word {W} anywhere in your answer."
DOSE_LEVELS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
