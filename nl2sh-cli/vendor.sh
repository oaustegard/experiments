#!/bin/bash
# Re-vendor the measured prompt builders and re-pin their hashes.
# Run after changing prompts.py or extract_params.py in the experiment dirs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
for f in nl2sh-instantiate/prompts.py nl2sh-retrieval/extract_params.py; do
  b=$(basename "$f"); sha=$(sha256sum "$SRC/$f" | cut -c1-16)
  { echo "# VENDORED from oaustegard/experiments $f"
    echo "# sha256[:16] = $sha"
    echo "# Do not edit here. Edit the original and re-vendor with vendor.sh, which"
    echo "# re-pins this hash. The pin exists because every number in"
    echo "# nl2sh-instantiate/RESULTS.md was produced by these exact bytes: a silent"
    echo "# drift here would keep working and quietly stop matching the measurements."
    echo
    cat "$SRC/$f"; } > "$HERE/nl2sh/vendor/$b"
  echo "vendored $b @ $sha"
done
python3 - <<'PY'
import pathlib
p = pathlib.Path(__file__).parent / "nl2sh/vendor/prompts.py" if False else pathlib.Path("nl2sh/vendor/prompts.py")
s = p.read_text()
old = '''RETRIEVAL = Path(__file__).resolve().parent.parent / "nl2sh-retrieval"
sys.path.insert(0, str(RETRIEVAL))'''
new = '''sys.path.insert(0, str(Path(__file__).resolve().parent))'''
if old in s:
    p.write_text(s.replace(old, new, 1)); print("repointed extract_params import")
PY
