"""#include search-path resolution and cached file tokenization.

IncludeMixin is mixed into Preprocessor. Search order (CONTRACT section 6):
  "file":  dir of current file, -iquote dirs, then the <file> order
  <file>:  -I dirs, -isystem dirs, gcc's include dir, /usr/local/include,
           /usr/include/x86_64-linux-gnu, /usr/include, -idirafter dirs
#include_next resumes the search after the directory the current file came from.
"""
from __future__ import annotations

import glob
import os

from hcc.lexer import tokenize_file
from hcc.tokens import Token, TokenKind


def default_system_dirs() -> list[str]:
    dirs = []
    gcc_dirs = glob.glob("/usr/lib/gcc/x86_64-linux-gnu/*/include")

    def ver(p):
        v = p.split("/")[-2]
        return [int(x) if x.isdigit() else 0 for x in v.split(".")]
    if gcc_dirs:
        dirs.append(max(gcc_dirs, key=ver))
    dirs += ["/usr/local/include", "/usr/include/x86_64-linux-gnu", "/usr/include"]
    return [d for d in dirs if os.path.isdir(d)]


def hcc_include_dir() -> str:
    """Directory of hcc's own override headers (hcc/include), searched first."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")


class IncludeMixin:

    def init_search_path(self) -> None:
        o = self.opts
        sysdirs = [] if o.nostdinc else default_system_dirs()
        self.search_path: list[str] = list(o.include_dirs) + list(o.system_dirs)
        if os.path.isdir(hcc_include_dir()) and not o.nostdinc:
            self.search_path.append(hcc_include_dir())
        self.search_path += sysdirs + list(o.after_dirs)
        self.file_dir_index: dict[str, int] = {}
        self.token_cache: dict[str, list[Token]] = {}
        self.include_guards: dict[str, str] = {}
        self.pragma_once: set[str] = set()

    def find_include(self, name: str, quoted: bool, cur_file: str, next_: bool = False):
        """Return (path, dir_index) or (None, -1)."""
        if os.path.isabs(name):
            return (name, -1) if os.path.isfile(name) else (None, -1)
        start = 0
        if next_:
            start = self.file_dir_index.get(cur_file, -1) + 1
        elif quoted:
            for d in [os.path.dirname(cur_file) or "."] + list(self.opts.quote_dirs):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    return os.path.normpath(p), -1
        for i in range(start, len(self.search_path)):
            p = os.path.join(self.search_path[i], name)
            if os.path.isfile(p):
                return os.path.normpath(p), i
        return None, -1

    def file_tokens(self, path: str) -> list[Token]:
        toks = self.token_cache.get(path)
        if toks is None:
            toks = tokenize_file(path, self.line_comments, self.u8_chars)
            self.token_cache[path] = toks
            guard = detect_include_guard(toks)
            if guard:
                self.include_guards[path] = guard
        return toks

    def should_skip_include(self, path: str) -> bool:
        if path in self.pragma_once:
            return True
        guard = self.include_guards.get(path)
        return guard is not None and guard in self.macros

    def include_file(self, path: str, dir_index: int) -> None:
        """Push the tokens of `path` so they are read next."""
        if self.should_skip_include(path):
            return
        if dir_index >= 0:
            self.file_dir_index[path] = dir_index
        self.include_depth += 1
        self.line_marks.pop(path, None)      # #line marks are per inclusion
        self.push_tokens(self.file_tokens(path))


def detect_include_guard(toks: list[Token]) -> str | None:
    """Recognize `#ifndef X / #define X ... #endif <EOF>`; return X."""
    if len(toks) < 6:
        return None
    t = toks
    if not (t[0].text == "#" and t[0].bol and t[1].text == "ifndef" and t[2].kind == TokenKind.IDENT
            and t[3].text == "#" and t[3].bol and t[4].text == "define" and t[5].text == t[2].text):
        return None
    name = t[2].text
    depth = 0
    i = 0
    n = len(t) - 1      # skip EOF
    while i < n:
        if t[i].text == "#" and t[i].bol and i + 1 < n:
            d = t[i + 1].text
            if d in ("if", "ifdef", "ifndef"):
                depth += 1
            elif d == "endif":
                depth -= 1
                if depth == 0:
                    return name if i + 2 == n else None
            elif d in ("elif", "else") and depth == 1:
                return None
        i += 1
    return None
