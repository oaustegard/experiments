def _wrap_paragraph(words, width):
    lines = []
    cur = ""
    for w in words:
        if len(w) > width:
            if cur:
                lines.append(cur)
                cur = ""
            while len(w) > width:
                lines.append(w[:width])
                w = w[width:]
            cur = w  # final chunk starts the new current line
        elif not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def wrap_text(text: str, width: int) -> str:
    if width < 1:
        raise ValueError("width must be >= 1")
    paragraphs = []
    current = []
    for line in text.split("\n"):
        if line == "":
            if current:
                paragraphs.append(current)
                current = []
        else:
            current.extend(line.split())
    if current:
        paragraphs.append(current)
    blocks = ["\n".join(_wrap_paragraph(words, width)) for words in paragraphs]
    return "\n".join(blocks)
