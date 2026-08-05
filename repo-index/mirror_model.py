#!/usr/bin/env python3
"""Mirror the pinned encoder into this repo's releases, so builds do not depend
on Hugging Face availability or rate limits.

bekko-embedding-v1-a8m is MIT-licensed, so redistribution is permitted.

    python3 repo-index/mirror_model.py                 # verify a local dir, print paths
    python3 repo-index/mirror_model.py DIR             # ... a specific one
    python3 repo-index/mirror_model.py --fetch DIR     # download from HF into DIR, verify

Paths go to stdout one per line (feed them to `gh release upload`); hashes and
progress go to stderr.

Publishing is `.github/workflows/repo-index-mirror.yml`, dispatched by hand.
A Claude Code session cannot create releases — the agent proxy refuses with
*"Creating, editing, or deleting releases is not permitted for this session
type"* — but an Actions run can.

**The sha256 pin is the whole point.** Mirroring a file that Hugging Face has
changed would republish a *different embedding space* under a name the index
trusts, which is exactly the silent failure `ask.py` refuses to allow. So a
mismatch is a hard error here, never an automatic update of the pin: if the
upstream file legitimately changed, that is a decision to make deliberately and
together with a full `--build`. (The first version of this script only *printed*
the hashes and left the comparison to whoever was reading the terminal, which is
not a check.)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ask import FILES, HF, MODEL_SHA256  # noqa: E402  (single source of truth)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def fetch(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        print(f"fetching {remote} from huggingface.co ...", file=sys.stderr)
        urllib.request.urlretrieve(f"{HF}/{remote}", dest / local)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=str(Path.home() / ".cache/repo-index"),
                    help="directory holding the encoder files")
    ap.add_argument("--fetch", metavar="DIR",
                    help="download from Hugging Face into DIR first")
    a = ap.parse_args()

    src = Path(a.fetch) if a.fetch else Path(a.src)
    if a.fetch:
        fetch(src)

    paths = []
    for local in FILES.values():
        p = src / local
        if not p.exists():
            raise SystemExit(f"missing {p}; run ask.py once to populate the cache, "
                             f"or pass --fetch DIR")
        print(f"{p}  sha256 {sha256(p)}", file=sys.stderr)
        paths.append(p)

    got = sha256(src / "model.onnx")
    if got != MODEL_SHA256:
        raise SystemExit(
            f"model.onnx sha256 does not match the pin in ask.py\n"
            f"  pinned {MODEL_SHA256}\n"
            f"  got    {got}\n"
            "Refusing to mirror. Publishing this would put a different embedding "
            "space behind a name the index trusts. If upstream genuinely changed, "
            "update MODEL_SHA256 and re-run `ask.py --build` in the same change.")
    print(f"model.onnx matches the pin ({MODEL_SHA256[:16]}...)", file=sys.stderr)

    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
