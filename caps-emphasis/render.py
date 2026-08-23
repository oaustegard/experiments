"""Render a directive in one surface form, keeping everything else fixed."""

MARK = "\x00"


def render(frame, keyword, word, mode):
    """mode: title | caps | lower | bold | italic | alt | acronym

    `title` is sentence case, the grammatical baseline. `lower` drops the
    sentence-initial capital as well, and exists only as a grammaticality
    control: without it, CAPS-vs-lowercase compares emphasis against a
    well-formedness violation.
    """
    s = frame.format(K=MARK, W=word)
    if mode == "caps":
        k = keyword.upper()
    elif mode == "bold":
        k = "**" + keyword + "**"
    elif mode == "italic":
        k = "*" + keyword + "*"
    elif mode == "alt":
        k = "".join(c.upper() if i % 2 == 0 else c.lower()
                    for i, c in enumerate(keyword))
    elif mode == "acronym":
        # a capitalised non-word of the same length, to test whether an
        # all-caps span acts as an acronym cue rather than an emphasis cue
        k = "".join("QXVZKJWPFB"[i % 10] if c.isalpha() else c
                    for i, c in enumerate(keyword))
    else:                                    # title, lower
        k = keyword
    at_start = s.startswith(MARK)
    s = s.replace(MARK, k)
    if mode != "lower":
        s = s[0].upper() + s[1:]
        if at_start:
            # sentence-casing rewrote the keyword's own first letter, so the
            # span text the caller must search for is the cased form, not `k`
            k = k[0].upper() + k[1:]
    return s, k
