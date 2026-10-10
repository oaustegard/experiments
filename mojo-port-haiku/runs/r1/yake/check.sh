#!/bin/sh
cd "$(dirname "$0")"
sh build.sh || exit 1
python3 /home/user/experiments/mojo-port-haiku/oracle/yake/check.py --pkg . --split dev "$@"
