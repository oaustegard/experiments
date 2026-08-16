import re


def parse_character_class(pattern, start_idx):
    """Parse a [...] character class in a glob pattern."""
    i = start_idx + 1  # Skip '['

    if i >= len(pattern):
        raise ValueError("Unterminated character class")

    is_negated = False
    if pattern[i] == "!":
        is_negated = True
        i += 1

    class_chars = ""

    # Handle ] as first character after [ or [!
    if i < len(pattern) and pattern[i] == "]":
        class_chars += "]"
        i += 1

    # Collect characters until closing ]
    while i < len(pattern) and pattern[i] != "]":
        class_chars += pattern[i]
        i += 1

    if i >= len(pattern):
        raise ValueError("Unterminated character class")

    # i is now at the closing ]
    return (is_negated, class_chars, i + 1)


def segment_to_regex(pattern):
    """Convert a segment glob pattern to regex."""
    regex = "^"
    i = 0

    while i < len(pattern):
        if pattern[i] == "*":
            regex += "[^/]*"
            i += 1
        elif pattern[i] == "?":
            regex += "[^/]"
            i += 1
        elif pattern[i] == "[":
            # Parse character class
            is_negated, class_chars, next_i = parse_character_class(pattern, i)

            # Escape special chars in class_chars for use in regex
            escaped_class = ""
            for ch in class_chars:
                if ch == "\\":
                    escaped_class += "\\\\"
                elif ch == "]":
                    escaped_class += "\\]"
                else:
                    escaped_class += ch

            if is_negated:
                regex += f"[^{escaped_class}]"
            else:
                regex += f"[{escaped_class}]"

            i = next_i
        else:
            # Literal character - escape regex special chars
            if pattern[i] in ".^$+(){}|[]":
                regex += "\\" + pattern[i]
            else:
                regex += pattern[i]
            i += 1

    regex += "$"
    return regex


def segment_match(pattern, segment):
    """Match a glob pattern segment against a path segment."""
    regex = segment_to_regex(pattern)
    return bool(re.match(regex, segment))


def match_segments(pattern_segs, path_segs, pi, pathi):
    """Recursively match pattern segments against path segments."""
    if pi == len(pattern_segs):
        return pathi == len(path_segs)

    if pattern_segs[pi] == "**":
        if pi == len(pattern_segs) - 1:
            # Last segment
            if len(pattern_segs) == 1:
                # Pattern is just "**" - matches any path
                return True
            else:
                # Trailing **, must match at least one path segment
                return pathi < len(path_segs)
        else:
            # Not last segment
            # Try matching 0 segments (skip this **)
            if match_segments(pattern_segs, path_segs, pi + 1, pathi):
                return True
            # Try matching 1 or more segments
            if pathi < len(path_segs):
                return match_segments(pattern_segs, path_segs, pi, pathi + 1)
            return False
    else:
        # Regular segment
        if pathi >= len(path_segs):
            return False
        if segment_match(pattern_segs[pi], path_segs[pathi]):
            return match_segments(pattern_segs, path_segs, pi + 1, pathi + 1)
        return False


def glob_match(pattern: str, path: str) -> bool:
    """Match a path against a glob pattern.

    Semantics: git/.gitignore-style matching of slash-separated paths.
    - The path is a `/`-separated string with no leading or trailing slash.
    - `?` matches exactly one character within a segment (never `/`).
    - `*` matches zero or more characters within a segment (never `/`).
    - `**` as a whole segment matches zero or more whole path segments.
    - Character classes `[abc]`, `[a-z]`, `[!...]` are supported.
    - No escape character; other characters match literally.
    - Matching is anchored: the whole pattern must match the whole path.
    """
    if pattern == "":
        return path == ""

    pattern_segs = pattern.split("/")
    path_segs = path.split("/") if path else []

    return match_segments(pattern_segs, path_segs, 0, 0)
