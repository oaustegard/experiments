#!/bin/bash
# test/play.sh "map=E1M1&tics=700" -> compact summary of a headless run
NODE_PATH=$(npm root -g) node test/run_page.cjs test/play.html ${2:-1800} "$1" 2>&1 | grep -v "^console: {" > /tmp/play.out
python3 - <<'PY'
import json,re
s=open('/tmp/play.out').read(); i=s.find('{')
try: r=json.loads(s[i:])
except Exception: print(s[-3000:]); raise SystemExit
if 'error' in r: print(r['error']); raise SystemExit
print({k:r[k] for k in ['map','result','tics','wallMs','kills','totalKills','items','totalItems','health']})
print(r['stats'])
for e in r['events']: print(' ', e)
print('trace', r['trace'][::4])
PY
