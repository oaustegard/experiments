"""Greedy paragraph-aware text wrapping."""


def _wrap_words(words, width):
    """Greedily wrap a flat list of words into lines of at most `width`
    characters, hard-breaking any word longer than `width`."""
    lines = []
    current = ""
    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            start = 0
            while len(word) - start > width:
                lines.append(word[start:start + width])
                start += width
            current = word[start:]
        else:
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= width:
                current += " " + word
            else:
                lines.append(current)
                current = word
    if current:
        lines.append(current)
    return lines


def wrap_text(text: str, width: int) -> str:
    """Greedily wrap `text` to `width` columns, preserving paragraph breaks.

    Paragraphs are separated by one or more blank (empty or whitespace-only)
    lines in the input, and are separated by exactly one blank line in the
    output. Within a paragraph all whitespace collapses to single spaces
    between words. Words longer than `width` are hard-broken into
    `width`-sized chunks. Raises ValueError if `width < 1`.
    """
    if width < 1:
        raise ValueError("width must be >= 1")

    if text is None or text.strip() == "":
        return ""

    lines = text.split("\n")
    paragraphs = []
    current_lines = []
    for line in lines:
        if line.strip() == "":
            if current_lines:
                paragraphs.append(current_lines)
                current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        paragraphs.append(current_lines)

    wrapped_paragraphs = []
    for para_lines in paragraphs:
        words = " ".join(para_lines).split()
        wrapped_lines = _wrap_words(words, width)
        wrapped_paragraphs.append("\n".join(wrapped_lines))

    return "\n\n".join(wrapped_paragraphs)
