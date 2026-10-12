"""GNU #assert / #unassert extension and the `#pred(answer)` test in #if.

Assertions are stored per Preprocessor in `pp.assertions`: predicate name ->
list of canonical answers. An answer is the token texts of the parenthesized
answer joined by single spaces (internal whitespace collapsed, external
whitespace dropped). Answers are not macro-expanded.

`replace_assertion_tests(pp, line)` rewrites `# pred` / `# pred ( answer )`
in an #if line into 0/1 number tokens; it runs before macro expansion.
"""
from __future__ import annotations

from hcc import diag
from hcc.tokens import Token, TokenKind


def _parse_pred(pp, where: Token, line: list[Token]) -> tuple[str, str | None, int]:
    """Parse `pred [ ( answer ) ]` from line; return (pred, answer-or-None, end index)."""
    if not line or line[0].kind not in (TokenKind.IDENT, TokenKind.KEYWORD):
        diag.error(where, "predicate must be an identifier")
    pred = line[0].text
    if len(line) == 1 or line[1].text != "(":
        return pred, None, 1
    depth = 0
    for j in range(1, len(line)):
        if line[j].text == "(":
            depth += 1
        elif line[j].text == ")":
            depth -= 1
            if depth == 0:
                body = line[2:j]
                if not body:
                    diag.error(where, "assertion without an answer")
                return pred, " ".join(t.text for t in body), j + 1
    diag.error(where, "missing ')' in assertion")


def dir_assert(pp, hash_tok: Token, line: list[Token]) -> None:
    pred, ans, _ = _parse_pred(pp, hash_tok, line)
    if ans is None:
        diag.error(hash_tok, "missing '(' in assertion")
    answers = pp.assertions.setdefault(pred, [])
    if ans not in answers:
        answers.append(ans)


def dir_unassert(pp, hash_tok: Token, line: list[Token]) -> None:
    pred, ans, _ = _parse_pred(pp, hash_tok, line)
    if ans is None:
        pp.assertions.pop(pred, None)
        return
    answers = pp.assertions.get(pred, [])
    if ans in answers:
        answers.remove(ans)
    if not answers:
        pp.assertions.pop(pred, None)


def replace_assertion_tests(pp, line: list[Token]) -> list[Token]:
    """Replace `#pred` / `#pred(answer)` in an #if line with 0 or 1."""
    out: list[Token] = []
    i = 0
    n = len(line)
    while i < n:
        t = line[i]
        if not (t.kind == TokenKind.PUNCT and t.text == "#" and i + 1 < n
                and line[i + 1].kind in (TokenKind.IDENT, TokenKind.KEYWORD)):
            out.append(t)
            i += 1
            continue
        pred, ans, end = _parse_pred(pp, t, line[i + 1:])
        answers = pp.assertions.get(pred, [])
        if ans is None:
            v = bool(answers)
        else:
            v = ans in answers
        out.append(Token(TokenKind.NUMBER, str(int(v)), t.loc, False, True))
        i += 1 + end
    return out
