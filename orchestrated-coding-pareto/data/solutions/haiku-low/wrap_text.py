def wrap_text(text: str, width: int) -> str:
    if width < 1:
        raise ValueError("width must be at least 1")

    # Split into paragraphs by blank lines
    paragraphs = []
    current_para = []

    for line in text.split('\n'):
        if line.strip():  # non-blank line
            current_para.append(line)
        else:  # blank line
            if current_para:
                paragraphs.append(current_para)
                current_para = []

    # Don't forget the last paragraph if it exists
    if current_para:
        paragraphs.append(current_para)

    # Process each paragraph
    wrapped_paragraphs = []

    for para_lines in paragraphs:
        # Join lines in the paragraph and split into words
        para_text = ' '.join(para_lines)
        words = para_text.split()

        # Wrap words
        wrapped_lines = []
        current_line = ""

        for word in words:
            # Check if word fits on current line
            if not current_line:
                # Starting a new line
                if len(word) <= width:
                    current_line = word
                else:
                    # Word is too long, hard-break it
                    while len(word) > width:
                        wrapped_lines.append(word[:width])
                        word = word[width:]
                    current_line = word
            else:
                # Current line has content
                if len(current_line) + 1 + len(word) <= width:
                    # Word fits
                    current_line += " " + word
                else:
                    # Word doesn't fit
                    wrapped_lines.append(current_line)

                    # Handle overlong word
                    if len(word) <= width:
                        current_line = word
                    else:
                        # Word is too long, hard-break it
                        while len(word) > width:
                            wrapped_lines.append(word[:width])
                            word = word[width:]
                        current_line = word

        # Add the last line of the paragraph
        if current_line:
            wrapped_lines.append(current_line)

        if wrapped_lines:
            wrapped_paragraphs.append('\n'.join(wrapped_lines))

    # Join paragraphs with empty lines between them
    result = '\n\n'.join(wrapped_paragraphs)

    return result
