#!/bin/sh
# Inside the jail: start the socket bridge, then stock OpenEvolve.
# $1 = run directory (relative to the jail root), holding config.yaml and llm.sock
set -eu
run=$1
python3 jailside/bridge.py 18080 "$PWD/$run/llm.sock" &
sleep 1
exec python3 -m openevolve.cli "${HSP_PROBLEM_DIR:-problem}/initial_program.py" jailside/oe_evaluator.py \
  --config "$run/config.yaml" --output "$run/oe_out"
