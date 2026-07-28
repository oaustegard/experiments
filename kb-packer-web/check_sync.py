#!/usr/bin/env python3
"""Prove kb-packer.html is in sync with its vendored runtime and with canonical creating-kb.

Two independent edges can drift; this checks both and exits nonzero on either.

  Edge 1  vendor/ <-> canonical creating-kb (oaustegard/claude-skills)
          The three files with an upstream — bundle_SKILL.md, search.js,
          search.py — must match creating-kb/scripts/ byte-for-byte. This is the
          vendor-freshness edge: the failure where creating-kb changes and the
          vendored snapshot is left stale. (lexkb-web.mjs has NO upstream there —
          it is the browser port of build_lexkb.js and lives only in vendor/, so
          it is hash-pinned by Edge 2 but not upstream-diffed.)

  Edge 2  kb-packer.html <-> vendor/
          The runtime hash baked into the HTML <meta> must equal the hash
          recomputed from current vendor/ (same export-strip the generator
          applies). This is the regeneration edge: the failure where vendor/
          changes but nobody re-ran build_packer.py.

Usage:
    python3 check_sync.py            # both edges (Edge 1 needs network + GH_TOKEN)
    python3 check_sync.py --offline  # Edge 2 only (no network)

Exit 0 if every checked edge is clean, 1 otherwise.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = HERE / "vendor"
HTML = HERE / "kb-packer.html"

# vendored-file -> canonical path in creating-kb (None = no upstream, hash-pinned only)
UPSTREAM = {
    "bundle_SKILL.md": "creating-kb/scripts/bundle_SKILL.md",
    "search.js": "creating-kb/scripts/search.js",
    "search.py": "creating-kb/scripts/search.py",
    "lexkb-web.mjs": None,
}
CANON_REPO = "oaustegard/claude-skills"


def runtime_hash() -> str:
    """Recompute the hash the generator embeds: export-stripped core + 3 runtime files."""
    core = (V / "lexkb-web.mjs").read_text(encoding="utf-8")
    core = re.sub(r"^export ", "", core, flags=re.M)  # mirror build_packer.py
    search_js = (V / "search.js").read_text(encoding="utf-8")
    search_py = (V / "search.py").read_text(encoding="utf-8")
    bundle_skill = (V / "bundle_SKILL.md").read_text(encoding="utf-8")
    concat = "\n--KBPACKER-RUNTIME--\n".join([core, search_js, search_py, bundle_skill])
    return hashlib.sha256(concat.encode("utf-8")).hexdigest()


def embedded_hash() -> str | None:
    html = HTML.read_text(encoding="utf-8")
    m = re.search(r'<meta name="kb-packer-runtime-hash" content="([0-9a-f]{64})">', html)
    return m.group(1) if m else None


def fetch_canonical(path: str) -> str:
    tok = os.environ.get("GH_TOKEN")
    url = f"https://api.github.com/repos/{CANON_REPO}/contents/{path}?ref=main"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "kb-packer-sync-check")
    req.add_header("Accept", "application/vnd.github.v3.raw")
    if tok:
        req.add_header("Authorization", f"token {tok}")
    with urllib.request.urlopen(req) as r:
        return r.read().decode("utf-8")


def check_edge2() -> list[str]:
    fails = []
    emb = embedded_hash()
    cur = runtime_hash()
    if emb is None:
        fails.append("Edge 2: no kb-packer-runtime-hash meta in kb-packer.html — regenerate with build_packer.py")
    elif emb != cur:
        fails.append(
            "Edge 2: kb-packer.html is stale vs vendor/.\n"
            f"          embedded {emb}\n          current  {cur}\n"
            "          -> run: python3 build_packer.py")
    pin = (V / "RUNTIME_HASH")
    if pin.exists() and pin.read_text(encoding="utf-8").strip() != cur:
        fails.append("Edge 2: vendor/RUNTIME_HASH disagrees with recomputed vendor hash — regenerate.")
    return fails


def check_edge1() -> list[str]:
    fails = []
    for fn, up in UPSTREAM.items():
        if up is None:
            continue
        local = (V / fn).read_text(encoding="utf-8")
        try:
            canon = fetch_canonical(up)
        except Exception as e:  # noqa: BLE001
            fails.append(f"Edge 1: could not fetch canonical {up}: {e}")
            continue
        if local != canon:
            fails.append(
                f"Edge 1: vendor/{fn} differs from canonical {CANON_REPO}:{up} "
                f"(vendor {len(local)}B vs upstream {len(canon)}B) -> re-vendor + rebuild.")
    return fails


def main() -> int:
    offline = "--offline" in sys.argv
    fails = check_edge2()
    if not offline:
        fails += check_edge1()
    else:
        print("(--offline: skipping Edge 1 upstream diff)")
    if fails:
        print("SYNC CHECK FAILED:\n" + "\n".join("  - " + f for f in fails))
        return 1
    print("sync OK: kb-packer.html matches vendor/, vendor matches canonical creating-kb.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
