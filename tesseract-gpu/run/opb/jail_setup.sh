#!/bin/sh
# One-time setup of the jailed scorer's dependencies for run/opb/score.py (run as root).
#
#   sudo sh run/opb/jail_setup.sh            # or as root; JAIL_DIR defaults to /tmp/jail/opb-tg
#
# Wheels are installed outside the jail, from the registry, into JAIL_DIR/deps (root-owned, read-only
# to nobody). fuzzysearch 0.8.1 publishes only an sdist, so it is fetched from PyPI with its sha256
# checked, then built into a wheel INSIDE the jail (--no-build-isolation, no index), and that wheel is
# unpacked into deps. Its setup.py never runs outside the jail. JAIL_DIR/work is the directory the jail
# runs in; nobody owns it and nothing is written there.
set -eu

[ "$(id -u)" -eq 0 ] || { echo "jail_setup.sh: run as root" >&2; exit 2; }
D=${JAIL_DIR:-/tmp/jail/opb-tg}
JAIL=/home/user/claude-workspace/scripts/jail.sh
mkdir -p "$D/deps" "$D/fz/wheelhouse"

# 1. Wheels only, outside the jail.
python3 -I -m pip install --quiet --only-binary=:all: --no-deps --target "$D/deps" "omni-parse-bench==1.7.1"
python3 -I -m pip install --quiet --only-binary=:all: --target "$D/deps" \
  "beautifulsoup4>=4.12" "mini-racer>=0.12" "omni-bench-core[hub]>=0.14,<0.15" "pylatexenc>=2.10" \
  "rapidfuzz>=3.0" "httpx>=0.27" "markdown-it-py>=3.0" "pillow>=10.0" "openai>=1.0" "pypdfium2>=4.30"

# 2. fuzzysearch: fetch the sdist (no build), check its digest, extract it.
python3 -I - "$D/fz" <<'PY'
import hashlib, json, subprocess, sys, tarfile
from pathlib import Path
root = Path(sys.argv[1])
meta = json.loads(subprocess.run(["curl", "-sSf", "https://pypi.org/pypi/fuzzysearch/0.8.1/json"],
                                 capture_output=True, check=True).stdout)
sd = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
blob = subprocess.run(["curl", "-sSfL", sd["url"]], capture_output=True, check=True).stdout
if hashlib.sha256(blob).hexdigest() != sd["digests"]["sha256"]:
    sys.exit("fuzzysearch sdist digest mismatch")
(root / "sdist.tar.gz").write_bytes(blob)
src = root / "src"
src.mkdir(exist_ok=True)
with tarfile.open(root / "sdist.tar.gz") as t:
    t.extractall(src, filter="data")
print("fuzzysearch sdist extracted, sha256 ok")
PY
SRC=$(ls -d "$D"/fz/src/fuzzysearch-0.8.1)

# 3. Build the wheel inside the jail.
chown -R nobody:nogroup "$D/fz"
sh "$JAIL" -C "$SRC" -- python3 -m pip wheel --quiet --no-deps --no-build-isolation --no-index \
  -w "$D/fz/wheelhouse" .

# 4. Unpack it into deps (a wheel is a zip).
python3 -I - "$D" <<'PY'
import glob, sys, zipfile
d = sys.argv[1]
(whl,) = glob.glob(f"{d}/fz/wheelhouse/fuzzysearch-0.8.1-*.whl")
zipfile.ZipFile(whl).extractall(f"{d}/deps")
print("unpacked", whl)
PY

# 5. The directory the jail runs in, writable by nobody only.
mkdir -p "$D/work"
chown -R root:root "$D/deps"
chmod -R go+rX "$D/deps"
chown nobody:nogroup "$D/work"
echo "jail deps ready in $D/deps"
