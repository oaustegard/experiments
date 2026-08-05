"""A general-purpose hybrid (lexical + dense) code index.

Design goals, in order:

1. **Hybrid, not dense-with-a-grep-footnote.** This repo has already measured
   that the lexical arm is not a courtesy: `lexical-kb-phase0` found BM25 over
   whole documents matching the dense ceiling (R@10 1.00), and
   `bekko-embedding-bench` found RRF(rg, dense) directionally best in every cell.
   So both arms are first-class and fused, not offered as alternatives a human
   picks between.

2. **Code-aware lexical matching.** The thing that makes this a *code* index
   rather than a text index pointed at code is identifier splitting:
   `_stacked_simhash_encode` has to be findable from "stacked simhash encode".
   Plain whitespace tokenization cannot do that, and it is exactly where a dense
   encoder is weakest (rare literal tokens).

3. **Exclusions primarily at query time.** Anything dropped at build time is
   unrecoverable without a rebuild, so build-time exclusion is reserved for
   things that are never an answer. Repo-specific choices live in a repo-local
   `.repo-index.json`, not in this file — a tool that hardcodes one repo's
   quirks is not general purpose.

No repo-specific constants appear below. `DEFAULT_CFG` is a universal starting
point; everything situational is config.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Universal defaults. A repo overrides these in .repo-index.json.
DEFAULT_CFG = {
    # code / docs / config. Anything not listed is treated as binary or noise.
    "extensions": [".py", ".md", ".sh", ".bash", ".js", ".ts", ".tsx", ".jsx",
                   ".rs", ".go", ".c", ".h", ".cpp", ".hpp", ".java", ".rb",
                   ".php", ".swift", ".kt", ".scala", ".lua", ".r", ".jl",
                   ".mojo", ".sql", ".html", ".css", ".scss", ".yml", ".yaml",
                   ".toml", ".cfg", ".ini", ".json", ".txt", ".rst", ".tex"],
    # never an answer, and present in almost every repo
    "skip_dirs": [".git", "node_modules", "__pycache__", ".venv", "venv",
                  ".mypy_cache", ".pytest_cache", "dist", "build", ".tox",
                  "target", "vendor/bundle", ".next", "coverage"],
    "skip_names": ["package-lock.json", "yarn.lock", "poetry.lock", "Cargo.lock",
                   "pnpm-lock.yaml", "go.sum", "composer.lock"],
    "max_bytes": 1 << 20,
    # repo-specific build-time exclusions go here, in the repo's own config
    "exclude": [],
    "doc_extensions": [".md", ".rst", ".txt", ".tex"],
    "win": 60, "stride": 45, "min_chars": 120, "max_chars": 2000,
}


def load_cfg(root: Path) -> dict:
    cfg = dict(DEFAULT_CFG)
    p = root / ".repo-index.json"
    if p.exists():
        user = json.loads(p.read_text())
        for k, v in user.items():
            # list-valued keys extend the defaults rather than replacing them,
            # so a repo cannot accidentally drop `.git` from skip_dirs
            if isinstance(v, list) and isinstance(cfg.get(k), list) and k != "extensions":
                cfg[k] = sorted(set(cfg[k]) | set(v))
            else:
                cfg[k] = v
    return cfg


# ── tokenization ────────────────────────────────────────────────────────────
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")


def tokens(s: str) -> list[str]:
    """Code-aware tokens: the whole identifier *and* its parts.

    `_stacked_simhash_encode` -> stacked_simhash_encode, stacked, simhash, encode
    `GRID_VERSION`            -> grid_version, grid, version
    `asciiFold`               -> asciifold, ascii, fold

    Emitting both means an exact-identifier query still scores the whole-token
    match highest, while a prose query reaches the parts. Dropping the whole
    token would trade exact lookup away for recall; keeping only it would be a
    plain text tokenizer.
    """
    out: list[str] = []
    for raw in _WORD.findall(s):
        out.append(raw.lower().strip("_"))
        parts = [p for sub in raw.split("_") if sub for p in _CAMEL.findall(sub)]
        if len(parts) > 1:
            out.extend(p.lower() for p in parts)
    return [t for t in out if t]


# ── corpus ──────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    f: str
    s: int
    text: str


def discover(root: Path, cfg: dict) -> list[Path]:
    exts, skip_d = set(cfg["extensions"]), set(cfg["skip_dirs"])
    skip_n, cap = set(cfg["skip_names"]), cfg["max_bytes"]
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in exts or p.name in skip_n:
            continue
        rel = p.relative_to(root)
        if any(d in rel.parts for d in skip_d):
            continue
        if any(match(str(rel), pat) for pat in cfg["exclude"]):
            continue
        try:
            if p.stat().st_size > cap:
                continue
        except OSError:
            continue
        out.append(p)
    return out


def chunk_file(p: Path, rel: str, cfg: dict) -> list[Chunk]:
    try:
        text = p.read_text(errors="ignore")
    except OSError:
        return []
    if p.suffix in set(cfg["doc_extensions"]):
        return _chunk_doc(text, rel, cfg)
    return _chunk_code(text, rel, cfg)


def _chunk_doc(text: str, rel: str, cfg: dict) -> list[Chunk]:
    """Prose splits on headings — a section is the unit a person cites."""
    out, cur, start, head = [], [], 1, ""
    lines = text.split("\n")

    def flush():
        body = "\n".join(cur).strip()
        if len(body) >= cfg["min_chars"]:
            for i in range(0, len(body), cfg["max_chars"]):
                out.append(Chunk(rel, start, f"{rel}\n{head}\n{body[i:i + cfg['max_chars']]}"))
    for i, ln in enumerate(lines, 1):
        if re.match(r"^#{1,4}\s", ln) and cur:
            flush(); cur, start, head = [], i, ln.strip()
        cur.append(ln)
    flush()
    return out


def _chunk_code(text: str, rel: str, cfg: dict) -> list[Chunk]:
    """Code splits on fixed line windows.

    Flat rather than AST because `bekko-embedding-bench` measured AST-vs-flat as
    noise (+0.063, p=0.424) on this repo's code-retrieval task, and flat needs no
    per-language parser — which a general-purpose indexer would otherwise need
    for every extension above.
    """
    lines, out = text.split("\n"), []
    win, stride = cfg["win"], cfg["stride"]
    for s in range(0, max(1, len(lines)), stride):
        body = "\n".join(lines[s:s + win]).strip()
        if len(body) >= cfg["min_chars"]:
            out.append(Chunk(rel, s + 1, f"{rel}\n{body[:cfg['max_chars']]}"))
        if s + win >= len(lines):
            break
    return out


def build_corpus(root: Path, cfg: dict) -> list[Chunk]:
    out = []
    for p in discover(root, cfg):
        out += chunk_file(p, str(p.relative_to(root)), cfg)
    return out


# ── lexical arm A: stored BM25 ──────────────────────────────────────────────
@dataclass
class BM25:
    k1: float = 1.2
    b: float = 0.75
    postings: dict = field(default_factory=lambda: defaultdict(list))
    lens: list = field(default_factory=list)
    n: int = 0

    def fit(self, chunks: list[Chunk]) -> "BM25":
        for i, c in enumerate(chunks):
            tf = Counter(tokens(c.text))
            self.lens.append(sum(tf.values()) or 1)
            for t, f in tf.items():
                self.postings[t].append((i, f))
        self.n = len(chunks)
        return self

    def score(self, q: str):
        import numpy as np
        avg = sum(self.lens) / max(1, self.n)
        s = np.zeros(self.n, dtype=np.float32)
        for t in set(tokens(q)):
            post = self.postings.get(t)
            if not post:
                continue
            idf = math.log(1 + (self.n - len(post) + 0.5) / (len(post) + 0.5))
            for i, f in post:
                dl = self.lens[i]
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
        return s

    def nbytes(self) -> int:
        """Artifact cost if serialized as (term, [(doc, tf)]) — varint-ish estimate."""
        return sum(len(t) + 1 + 4 * len(p) for t, p in self.postings.items()) + 4 * self.n


# ── lexical arm B: ripgrep at query time ────────────────────────────────────
def rg_files(q: str, root: Path, cfg: dict, min_len: int = 4) -> list[str]:
    """Rank files by how many of the query's terms they contain.

    ripgrep returns a set, not a ranking; counting matched terms is the cheapest
    honest way to give this arm a rank so RRF has something to fuse.
    """
    terms = [t for t in dict.fromkeys(tokens(q)) if len(t) >= min_len][:8]
    hits: Counter = Counter()
    globs = []
    for e in cfg["extensions"]:
        globs += ["--glob", f"*{e}"]
    for t in terms:
        r = subprocess.run(["rg", "-l", "-i", *globs, "--", t, "."],
                           cwd=root, capture_output=True, text=True)
        for f in r.stdout.split():
            hits[f.lstrip("./")] += 1
    return [f for f, _ in hits.most_common()]


# ── fusion ──────────────────────────────────────────────────────────────────
def rrf(ranked: list[list[str]], k: int = 60) -> list[str]:
    s: dict[str, float] = defaultdict(float)
    for lst in ranked:
        for rank, f in enumerate(lst, 1):
            s[f] += 1.0 / (k + rank)
    return sorted(s, key=lambda f: -s[f])


# ── incremental build ───────────────────────────────────────────────────────
def chunk_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


def incremental(chunks: list[Chunk], encode, prev_codes=None, prev_hashes=None):
    """Encode only chunks whose content hash is new; copy the rest.

    **This is exactly equivalent to a full rebuild, bit for bit** — not an
    approximation traded for speed. Two properties make that true, and both were
    established elsewhere in this repo:

      - the encoder is per-chunk independent (no corpus statistics, no batch
        normalization that crosses chunk boundaries), and
      - remex quantization is data-oblivious: the codec is fully determined by
        (d, bits, seed, rotation) and never fitted to the corpus.

    So a reused row is the row a full rebuild would have produced. Anything that
    *does* fit on the corpus breaks this — PCA, k-means, ITQ, product-quantizer
    codebooks, and notably **BM25's IDF**, which shifts for every term when any
    document is added. The lexical arm therefore cannot be incrementalized the
    same way; it is refit from scratch, which is affordable precisely because
    fitting it is seconds rather than minutes.

    Returns (codes, n_encoded, n_reused).
    """
    import numpy as np
    hashes = [chunk_hash(c.text) for c in chunks]
    prev_at: dict[str, int] = {}
    if prev_codes is not None and prev_hashes:
        # last occurrence wins; duplicates are interchangeable by construction
        prev_at = {h: i for i, h in enumerate(prev_hashes)}

    todo = [i for i, h in enumerate(hashes) if h not in prev_at]
    reuse = [(i, prev_at[h]) for i, h in enumerate(hashes) if h in prev_at]

    width = prev_codes.shape[1] if prev_codes is not None and len(prev_codes) else None
    out = None
    if todo:
        fresh = encode([chunks[i].text for i in todo])
        width = width or fresh.shape[1]
        out = np.zeros((len(chunks), width), dtype=fresh.dtype)
        out[todo] = fresh
    if out is None:
        out = np.zeros((len(chunks), width), dtype=prev_codes.dtype)
    for i, j in reuse:
        out[i] = prev_codes[j]
    return out, hashes, len(todo), len(reuse)


def match(path: str, pat: str) -> bool:
    """Substring unless the pattern carries a glob metacharacter."""
    from fnmatch import fnmatch
    return fnmatch(path, pat if any(c in pat for c in "*?[") else f"*{pat}*")


def to_files(scores, chunks: list[Chunk], exclude: list[str] | None = None,
             k: int = 20) -> list[str]:
    """Best-chunk-per-file ranking, with query-time exclusion applied here."""
    import numpy as np
    best: list[str] = []
    for i in np.argsort(-scores):
        f = chunks[i].f
        if f in best:
            continue
        if exclude and any(match(f, p) for p in exclude):
            continue
        best.append(f)
        if len(best) >= k:
            break
    return best
