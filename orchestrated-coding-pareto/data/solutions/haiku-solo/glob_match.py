import re


def glob_match(pattern: str, path: str) -> bool:
    if not pattern:
        return not path

    if pattern == "**":
        return True

    pat_segs = pattern.split('/')
    path_segs = path.split('/') if path else []

    def segment_match(pat_seg, path_seg):
        regex = _glob_segment_to_regex(pat_seg)
        return re.fullmatch(regex, path_seg) is not None

    def match_segs(pat_idx, path_idx):
        if pat_idx == len(pat_segs):
            return path_idx == len(path_segs)

        pat_seg = pat_segs[pat_idx]

        if pat_seg == "**":
            if pat_idx == len(pat_segs) - 1:
                return path_idx < len(path_segs)
            else:
                for i in range(path_idx, len(path_segs) + 1):
                    if match_segs(pat_idx + 1, i):
                        return True
                return False
        else:
            if path_idx >= len(path_segs):
                return False
            if segment_match(pat_seg, path_segs[path_idx]):
                return match_segs(pat_idx + 1, path_idx + 1)
            return False

    return match_segs(0, 0)


def _glob_segment_to_regex(seg):
    result = []
    i = 0
    while i < len(seg):
        if seg[i] == '*':
            result.append('.*')
            i += 1
        elif seg[i] == '?':
            result.append('.')
            i += 1
        elif seg[i] == '[':
            char_class, end_idx = _parse_char_class(seg, i)
            result.append(char_class)
            i = end_idx
        else:
            result.append(re.escape(seg[i]))
            i += 1

    return ''.join(result)


def _parse_char_class(seg, start_idx):
    i = start_idx + 1
    if i >= len(seg):
        raise ValueError()

    negated = False
    if seg[i] == '!':
        negated = True
        i += 1
        if i >= len(seg):
            raise ValueError()

    class_chars = [seg[i]]
    i += 1

    while i < len(seg) and seg[i] != ']':
        class_chars.append(seg[i])
        i += 1

    if i >= len(seg):
        raise ValueError()

    i += 1

    class_str = ''.join(class_chars)
    if class_str and class_str[0] == '^':
        class_str = '\\' + class_str
    class_str = class_str.replace('\\', '\\\\').replace(']', '\\]')

    if negated:
        return '[^' + class_str + ']', i
    else:
        return '[' + class_str + ']', i
