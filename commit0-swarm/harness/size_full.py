"""Size every non-lite Commit0 library: what the stub removed and how many tests it has.

    python harness/size_full.py      -> data/full_sizes.json, table on stdout

Bare-clones each mirror (git only, no repo code runs), then from the base and
reference commits counts lines the stub removed under src_dir, test functions at
the reference, and compiled-extension sources (.pyx/.c/.rs) under src_dir,
which this harness cannot build.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, LITE, MIRRORS, PARQUET, git  # noqa: E402


def rows():
    import pandas as pd
    d = pd.read_parquet(PARQUET)
    for _, r in d.iterrows():
        name = r["repo"].split("/")[1]
        if name not in LITE:
            yield {"name": name, "base": r["base_commit"], "ref": r["reference_commit"],
                   "src": r["src_dir"].rstrip("/"), "test_dir": dict(r["test"])["test_dir"]}


def size(t: dict) -> dict:
    m = MIRRORS / f"{t['name']}.git"
    if not m.exists():
        git("clone", "-q", "--bare", f"https://github.com/commit-0/{t['name']}", str(m))
    num = git("--git-dir", str(m), "diff", "--numstat", t["base"], t["ref"], "--", t["src"], check=False)
    added = sum(int(a) for a, _, _ in (l.split("\t") for l in num.splitlines()) if a.isdigit())
    files = git("--git-dir", str(m), "ls-tree", "-r", "--name-only", t["ref"], check=False).split()
    compiled = [f for f in files if f.startswith(t["src"]) and f.endswith((".pyx", ".pxd", ".c", ".cpp", ".rs"))]
    tests = git("--git-dir", str(m), "grep", "-c", "def test_", t["ref"], "--", t["test_dir"].rstrip("/") or ".",
                check=False)
    n_tests = sum(int(l.rsplit(":", 1)[1]) for l in tests.splitlines() if l.rsplit(":", 1)[-1].isdigit())
    return {**t, "removed_lines": added, "test_functions": n_tests, "compiled_sources": len(compiled)}


def main():
    with ThreadPoolExecutor(2) as ex:   # the git proxy caps concurrent clones per repo, not per session
        out = [r for r in ex.map(lambda t: _safe(t), rows())]
    out.sort(key=lambda r: -r.get("removed_lines", 0))
    (DATA / "full_sizes.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in out:
        print(f"{r['name']:24s} removed {r.get('removed_lines', '?'):>7}  tests {r.get('test_functions', '?'):>6}  "
              f"compiled {r.get('compiled_sources', '?'):>4}  {r.get('error', '')}")


def _safe(t):
    try:
        return size(t)
    except Exception as e:  # noqa: BLE001
        return {**t, "error": f"{type(e).__name__}: {e}"[:200]}


if __name__ == "__main__":
    main()
