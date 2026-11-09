#!/bin/sh
# One-time setup for a lane session: OpenEvolve's example problems at the pinned
# commit, and OpenEvolve 0.4.1 with its dependencies in the jail (wheels only,
# from PyPI, installed outside the jail because the jail has no network).
set -eu
OE_SHA=3e72b5bcce27cdd948d8cdff2fa00e67c4d6066c
if [ ! -d /tmp/src/openevolve/.git ]; then
  mkdir -p /tmp/src
  git clone -q https://github.com/algorithmicsuperintelligence/openevolve.git /tmp/src/openevolve
fi
git -C /tmp/src/openevolve checkout -q "$OE_SHA" 2>/dev/null || {
  git -C /tmp/src/openevolve fetch -q --depth 50 origin
  git -C /tmp/src/openevolve checkout -q "$OE_SHA"
}
if [ ! -d /tmp/jail/hsp/.deps/openevolve ]; then
  mkdir -p /tmp/jail/hsp
  python3 -I -m pip install -q --only-binary=:all: --target /tmp/jail/hsp/.deps \
    openevolve==0.4.1 openai==3.46.0 pyyaml==6.0.3 tqdm==4.70.1 flask==3.1.3 dacite==1.9.2 cloudpickle==3.1.2
fi
python3 -c "import yaml" 2>/dev/null || python3 -m pip install -q pyyaml
python3 -c "import matplotlib" 2>/dev/null || python3 -m pip install -q matplotlib
test -x /home/user/claude-workspace/scripts/jail.sh || test -f /home/user/claude-workspace/scripts/jail.sh
echo "lane setup ok: openevolve $(git -C /tmp/src/openevolve rev-parse --short HEAD), jail deps present"
