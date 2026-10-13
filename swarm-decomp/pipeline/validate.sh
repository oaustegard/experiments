#!/bin/sh
# Runner self-check: the ORIGINAL sources, fed through the runner as if they were the
# reconstruction, must match every function. Runs inside claude-workspace scripts/jail.sh.
#   sh pipeline/validate.sh O2
set -eu
opt=${1:-O2}
cc_opt=${2:-$opt}
here=$(cd "$(dirname "$0")/.." && pwd)
V=/tmp/jail/validate
python3 "$here/pipeline/stage_validation.py" "$opt" "$cc_opt" >/dev/null
chown -R nobody:nogroup "$V"
sh /home/user/claude-workspace/scripts/jail.sh -C "$V" -e DECOMP_EXTRA_I="$V/inc" -- \
    python3 tests/run.py --jobs 4 --json out.json
cp "$V/out.json" "$here/build/validate-$opt-$cc_opt.json"
