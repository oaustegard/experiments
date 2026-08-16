"""Greedy, paragraph-aware text wrapping.

Public API: ``wrap_text(text, width)``.
Implemented without the ``textwrap`` module.
"""

__all__ = ["wrap_text"]


def _split_paragraphs(text):
    """Split ``text`` into paragraphs on blank (empty/whitespace-only) lines.

    Returns a list of word-lists; empty paragraphs are dropped, so runs of
    blank lines collapse into a single break and leading/trailing blank
    lines produce nothing.
    """
    paragraphs = []
    current = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                paragraphs.append(current)
                current = []
        else:
            current.extend(line.split())
    if current:
        paragraphs.append(current)
    return paragraphs


def _wrap_paragraph(words, width):
    """Greedily wrap a list of words into lines of at most ``width`` chars."""
    lines = []
    line = ""
    for word in words:
        while len(word) > width:
            # Overlong word: finish the current line, then hard-break.
            if line:
                lines.append(line)
                line = ""
            lines.append(word[:width])
            word = word[width:]
        if not word:
            continue
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= width:
            line += " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def wrap_text(text: str, width: int) -> str:
    """Wrap ``text`` to ``width`` columns, preserving paragraph breaks.

    Paragraphs are separated by blank lines in the input and by exactly one
    empty line in the output. Within a paragraph all whitespace collapses and
    words are packed greedily. Words longer than ``width`` are hard-broken
    into ``width``-sized chunks.

    Raises:
        ValueError: if ``width`` is less than 1.
    """
    if width < 1:
        raise ValueError("width must be at least 1")

    out = []
    for words in _split_paragraphs(text):
        lines = _wrap_paragraph(words, width)
        if lines:
            out.append("\n".join(lines))
    return "\n\n".join(out)
