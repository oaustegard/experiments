def wrap_text(text: str, width: int) -> str:
    if width < 1:
        raise ValueError("width must be at least 1")

    # Split into paragraphs by blank lines
    paragraphs = []
    current_paragraph = []

    for line in text.split('\n'):
        if line.strip():
            current_paragraph.append(line)
        else:
            if current_paragraph:
                paragraphs.append('\n'.join(current_paragraph))
                current_paragraph = []

    if current_paragraph:
        paragraphs.append('\n'.join(current_paragraph))

    if not paragraphs:
        return ""

    wrapped_paragraphs = []

    for para in paragraphs:
        words = para.split()

        lines = []
        current_line = ""

        for word in words:
            if not current_line:
                if len(word) <= width:
                    current_line = word
                else:
                    while len(word) > width:
                        lines.append(word[:width])
                        word = word[width:]
                    if word:
                        current_line = word
            else:
                if len(current_line) + 1 + len(word) <= width:
                    current_line += " " + word
                else:
                    lines.append(current_line)

                    if len(word) <= width:
                        current_line = word
                    else:
                        while len(word) > width:
                            lines.append(word[:width])
                            word = word[width:]
                        current_line = word if word else ""

        if current_line:
            lines.append(current_line)

        wrapped_paragraphs.append('\n'.join(lines))

    return '\n\n'.join(wrapped_paragraphs)
