#!/bin/bash
# run.sh KERNEL N: run one native kernel binary.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec "$here/out/$1" "$2"
