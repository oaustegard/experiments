#!/usr/bin/env bash
# Mirror transformer-circuits.pub/2026/interference_effectiveness_helpfulness in full:
# main page, 21 interactive figures + their data/vendor assets, 34-page figure gallery,
# and the 23-feature feature_vis page. Result is browsable offline.
set -euo pipefail
B=https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness
OUT=${1:-mirror}
mkdir -p "$OUT" && cd "$OUT"
curl -sL -o index.html "$B/index.html"
python3 - <<'PY' > .assets
import re,html,json
h=open('index.html',encoding='utf-8',errors='replace').read()
blobs=re.findall(r'srcdoc="([^"]*)"',h)
refs=set()
for b in blobs:
    for m in re.findall(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']',html.unescape(b)):
        if not m.startswith(('http','data:','#')) and '<' not in m: refs.add(m)
print('\n'.join(sorted(refs)))
PY
while read -r p; do
  [ -z "$p" ] && continue; mkdir -p "$(dirname "$p")"
  curl -sf -o "$p" "$B/$p" || echo "miss $p" >&2
done < .assets
mkdir -p figure_gallery feature_vis
curl -sL -o figure_gallery/index.html "$B/figure_gallery/index.html"
curl -sL -o feature_vis/index.html   "$B/feature_vis/index.html"
for p in $(grep -oE "\./[A-Za-z0-9_-]+/index\.html" figure_gallery/index.html | sed 's|^\./||;s|/index.html$||' | sort -u); do
  mkdir -p "figure_gallery/$p"
  curl -sf -o "figure_gallery/$p/index.html" "$B/figure_gallery/$p/index.html" || echo "miss gallery/$p" >&2
done
# gallery pages reference the same shared/ and vendor/ trees relative to themselves
for d in figure_gallery/*/; do
  for t in shared vendor; do [ -d "$t" ] && cp -r "$t" "$d" 2>/dev/null || true; done
done
echo "mirrored to $(pwd)"
