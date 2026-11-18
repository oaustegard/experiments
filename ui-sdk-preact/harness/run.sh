#!/bin/sh
# Usage: run.sh <vite-config> <port> <playwright args...>
cfg=$1; port=$2; shift 2
cd /tmp/jail/ui-sdk/apps/playground
node ../../node_modules/vite/bin/vite.js --config "$cfg" --port "$port" --strictPort > /tmp/jail/runner/vite-$port.log 2>&1 &
vpid=$!
i=0; until node -e "fetch('http://localhost:$port/').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"; do i=$((i+1)); [ $i -gt 60 ] && { echo "vite did not start"; cat /tmp/jail/runner/vite-$port.log; kill $vpid; exit 3; }; sleep 1; done
cd /tmp/jail/ui-sdk/tests
PLAYGROUND_PORT=$port node ../node_modules/@playwright/test/cli.js test "$@"
rc=$?
kill $vpid
exit $rc
