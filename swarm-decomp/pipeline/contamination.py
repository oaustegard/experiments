#!/usr/bin/env python3
"""How much of each matched function is the upstream source, verbatim?

    contamination.py RESULTS.json UPSTREAM_DIR [--out runs/contamination.json]

UPSTREAM_DIR/<proj>/*.c is the re-downloaded original source (fetched only after
every worker has stopped). For every function that PASSes in RESULTS.json, the
reconstruction src/<proj>/<tu>/<base>.c is compared with the upstream definition.

Matching bytes fixes control flow, operations and the names in the symbol table.
It does not fix what the binary never contains: local variable and parameter names,
struct/field/typedef names, comments, formatting. Those are the signal:

  free_ident_recall   share of the upstream function's free identifiers (not a symbol
                      of any target object, not a C keyword, not a libc/public-header
                      name) that the reconstruction also uses
  verbatim_lines      share of the reconstruction's informative lines (>= 20 chars after
                      whitespace normalisation, not a lone brace) found verbatim in the
                      upstream function
  comment_lines       upstream comment lines reproduced verbatim
  token_ratio         difflib ratio of the two token sequences, identifiers included

A function is flagged COPIED when verbatim_lines >= 0.6 over >= 3 informative lines,
or free_ident_recall >= 0.8 over >= 5 free identifiers with token_ratio >= 0.9, or it
reproduces any upstream comment line of >= 20 chars. The obscure target (neat*) is the
control: its distributions show what independent reconstruction looks like.
Only text comparison runs here; no target code is executed.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests" / "lib"))
import objcmp  # noqa: E402

KEYWORDS = set("""auto break case char const continue default do double else enum extern float for goto if inline int
long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while
_Bool _Complex _Noreturn bool true false NULL""".split())
TOKEN = re.compile(r"[A-Za-z_]\w*|0[xX][0-9a-fA-F]+|\d+|\S")
IDENT = re.compile(r"[A-Za-z_]\w*")


def strip_comments(text: str) -> tuple[str, list[str]]:
    """Text with comments and string/char literals blanked (newlines kept), plus the comment lines."""
    out, comments, i, n = [], [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            seg = text[i:j]
            comments += [" ".join(l.strip(" */").split()) for l in seg.splitlines()]
            out.append("".join("\n" if ch == "\n" else " " for ch in seg))
            i = j
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            comments.append(" ".join(text[i + 2:j].split()))
            out.append(" " * (j - i))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(n, j + 1)
            out.append(c + "".join("\n" if ch == "\n" else "_" for ch in text[i + 1:j - 1]) + c)
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out), [c for c in comments if c]


def functions(text: str) -> dict[str, tuple[str, list[str]]]:
    """{name: (definition text, comment lines inside it)} for top-level function definitions."""
    clean, _ = strip_comments(text)
    res: dict[str, tuple[str, list[str]]] = {}
    depth, start, i = 0, 0, 0
    while i < len(clean):
        ch = clean[i]
        if ch == "{":
            if depth == 0:
                header = clean[start:i]
                # drop preprocessor lines (a #define with parentheses would read as a function header)
                lines = header.split("\n")
                keep = [k for k, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")]
                hdr = "\n".join(lines[keep[0]:]) if keep else ""
                hdr_off = start + sum(len(l) + 1 for l in lines[:keep[0]]) if keep else i
                m = re.search(r"([A-Za-z_]\w*)\s*\([^;{}]*\)\s*$", hdr.strip())
                if m and "=" not in hdr and not re.search(r"\b(struct|union|enum)\s+\w*\s*$", hdr.strip()):
                    name = m.group(1)
                    j, d = i, 0
                    while j < len(clean):
                        if clean[j] == "{":
                            d += 1
                        elif clean[j] == "}":
                            d -= 1
                            if d == 0:
                                break
                        j += 1
                    body_raw = text[hdr_off:j + 1]
                    _, cm = strip_comments(body_raw)
                    res.setdefault(name, (body_raw, cm))
                    depth, i, start = 0, j + 1, j + 1
                    continue
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                start = i + 1
        elif ch == ";" and depth == 0:
            start = i + 1
        i += 1
    return res


def norm_lines(text: str) -> list[str]:
    clean, _ = strip_comments(text)
    return [" ".join(l.split()) for l in clean.splitlines()]


def informative(l: str) -> bool:
    return len(l) >= 20 and l not in ("{", "}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("upstream")
    ap.add_argument("--out", default="runs/contamination.json")
    a = ap.parse_args()
    results = json.loads(Path(a.results).read_text())
    up_root = Path(a.upstream)
    # names fixed by the binaries: every symbol of every target object, plus the public headers' names
    symnames: set[str] = set()
    for sd in (ROOT / "tests" / "suites").iterdir():
        obj = objcmp.Obj(str(sd / "target.o"))
        symnames |= {re.sub(r"\..*$", "", s.name) for s in obj.syms if s.name}
    for h in (ROOT / "tests" / "include").rglob("*.h"):
        symnames |= set(IDENT.findall(strip_comments(h.read_text(errors="replace"))[0]))
    up_funcs: dict[str, dict[str, tuple[str, list[str]]]] = {}
    for pdir in up_root.iterdir():
        fs: dict[str, tuple[str, list[str]]] = {}
        for f in sorted(pdir.glob("*.c")):
            for k, v in functions(f.read_text(errors="replace")).items():
                fs.setdefault(f"{f.stem}:{k}", v)
        up_funcs[pdir.name] = fs
    rows = []
    for tid, r in sorted(results.items()):
        if r.get("status") != "PASS":
            continue
        suite, func = tid.split("/", 1)
        proj, tu = suite.split("-", 1)
        base = re.sub(r"\.(part|isra|constprop|cold)\.\d+.*$", "", func)
        mine_file = ROOT / "src" / proj / tu / f"{base}.c"
        mine = functions(mine_file.read_text(errors="replace")).get(base) if mine_file.exists() else None
        up = (up_funcs.get(proj, {}).get(f"{tu}:{base}"))
        if not mine or not up:
            rows.append({"test": tid, "error": "no definition " + ("upstream" if mine else "in reconstruction")})
            continue
        m_text, _ = mine
        u_text, u_comments = up
        m_lines = [l for l in norm_lines(m_text) if informative(l)]
        u_lines = set(norm_lines(u_text))
        verb = sum(1 for l in m_lines if l in u_lines)
        m_tok = TOKEN.findall(strip_comments(m_text)[0])
        u_tok = TOKEN.findall(strip_comments(u_text)[0])
        free = lambda toks: {t for t in toks if IDENT.fullmatch(t) and t not in KEYWORDS and t not in symnames}
        uf, mf = free(u_tok), free(m_tok)
        m_comment_text = " ".join(strip_comments(m_text)[1])
        ccopy = [c for c in u_comments if len(c) >= 20 and c in m_comment_text]
        row = {"test": tid, "proj": proj, "informative": len(m_lines),
               "verbatim_lines": round(verb / len(m_lines), 3) if m_lines else None,
               "free_upstream": len(uf), "free_ident_recall": round(len(uf & mf) / len(uf), 3) if uf else None,
               "token_ratio": round(difflib.SequenceMatcher(None, u_tok, m_tok, autojunk=False).ratio(), 3),
               "comment_lines": len(ccopy)}
        row["copied"] = bool((row["verbatim_lines"] or 0) >= 0.6 and len(m_lines) >= 3
                             or (row["free_ident_recall"] or 0) >= 0.8 and len(uf) >= 5 and row["token_ratio"] >= 0.9
                             or ccopy)
        rows.append(row)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for target in ("neat", "lua"):
        rs = [r for r in rows if r["test"].startswith(target) and "error" not in r]
        if not rs:
            continue
        def mean(k):
            v = [r[k] for r in rs if r.get(k) is not None]
            return sum(v) / len(v) if v else float("nan")
        print(f"{target}: {len(rs)} matched functions compared; copied {sum(r['copied'] for r in rs)}; "
              f"mean verbatim_lines {mean('verbatim_lines'):.3f}, free_ident_recall {mean('free_ident_recall'):.3f}, "
              f"token_ratio {mean('token_ratio'):.3f}; functions with an upstream comment {sum(r['comment_lines'] > 0 for r in rs)}")
    errs = [r for r in rows if "error" in r]
    if errs:
        print(f"{len(errs)} not compared (no definition found), e.g. {errs[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
