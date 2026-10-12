#!/bin/bash
# Bootstrap a federated sibling container for the hcc swarm.
#
#   bash federation/setup.sh NAME
#
# Run from the checkout of oaustegard/experiments at branch hcc/main (this
# file's repo). Idempotent: rerun after a container restart. It
#   - installs what the suites need beyond the base image (tcl8.6-dev for the
#     sqlite/redis Tcl harnesses, cvise for the reducer, PyPy 3.11 via uv,
#     tests/bin/hcc-auto and hcc-pycheck on PATH, csmith built into tests/tools),
#   - puts the checkout on a local branch fed-NAME (the swarm's main here),
#   - adds the transport remote `fed` and fetches hcc/main + hcc/state.
# Suite drivers download their own pinned sources on first use.
set -u
NAME=${1:?usage: setup.sh NAME}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
warn() { echo "setup: WARN: $*" >&2; }

export DEBIAN_FRONTEND=noninteractive
need=()
dpkg -s tcl8.6-dev >/dev/null 2>&1 || need+=(tcl8.6-dev)
command -v cvise >/dev/null 2>&1 || need+=(cvise)
for t in cmake m4 make; do command -v $t >/dev/null 2>&1 || need+=($t); done
if [ ${#need[@]} -gt 0 ]; then
  (apt-get update -qq && apt-get install -y -qq "${need[@]}") >/tmp/fed-apt.log 2>&1 || warn "apt-get install ${need[*]} failed (see /tmp/fed-apt.log)"
fi

if ! ls /opt/uvpy/pypy-3.11*/bin/pypy3 >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    UV_PYTHON_INSTALL_DIR=/opt/uvpy uv python install pypy3.11 >/tmp/fed-pypy.log 2>&1 || warn "PyPy install failed; hcc-auto falls back to CPython"
  else
    warn "uv missing; no PyPy (hcc-auto falls back to CPython)"
  fi
fi
# Driver caches key on the compiler command, so hcc-auto lives at a tree-independent path.
install -m 755 tests/bin/hcc-auto /usr/local/bin/hcc-auto || warn "could not install hcc-auto"
# The merge gate byte-compiles hcc/ under PyPy 3.11 with this (hcc is pinned to 3.11 syntax).
install -m 755 tests/bin/hcc-pycheck /usr/local/bin/hcc-pycheck || warn "could not install hcc-pycheck"

if [ ! -x tests/tools/csmith/bin/csmith ]; then
  sh tests/csmith/build.sh >/tmp/fed-csmith.log 2>&1 || warn "csmith build failed (see /tmp/fed-csmith.log)"
fi

git config user.name 'Oskar Austegard'
git config user.email 'oskar@austegard.com'
git remote get-url fed >/dev/null 2>&1 || git remote add fed https://github.com/oaustegard/experiments
git fetch -q --no-tags fed '+refs/heads/hcc/main:refs/fed/main' '+refs/heads/hcc/state:refs/fed/state' || warn "fetch of hcc/main / hcc/state failed"
cur=$(git symbolic-ref --short -q HEAD || echo DETACHED)
if [ "$cur" != "fed-$NAME" ]; then
  if git rev-parse -q --verify "refs/heads/fed-$NAME" >/dev/null; then
    git checkout -q "fed-$NAME"
  else
    git checkout -q -b "fed-$NAME" refs/fed/main 2>/dev/null || git checkout -q -b "fed-$NAME"
  fi
fi
mkdir -p runs
echo "setup: ok  branch=$(git symbolic-ref --short HEAD) head=$(git rev-parse --short HEAD)"
echo "setup: claude=$(command -v claude || echo MISSING) pypy=$(ls /opt/uvpy/pypy-3.11*/bin/pypy3 2>/dev/null | head -1) hcc-auto=$(command -v hcc-auto) hcc-pycheck=$(command -v hcc-pycheck) csmith=$(ls tests/tools/csmith/bin/csmith 2>/dev/null) tclsh=$(command -v tclsh8.6)"
