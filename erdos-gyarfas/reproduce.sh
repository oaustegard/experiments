#!/usr/bin/env bash
# Reproduce the cubic censuses and diff them against Markstrom's own files.
#
#   ./reproduce.sh 24     # ~5 min on 4 cores   -> 4 graphs
#   ./reproduce.sh 26     # ~15 min on 4 cores  -> 23 graphs
#
# Enumerates every connected cubic graph of the given order and keeps those with
# no 4-cycle and no 8-cycle, then reports which of the survivors contain a
# 16-cycle. A survivor with no 16-cycle would be a counterexample to
# Erdos-Gyarfas; expect none.
#
# Needs: snarkhunter (see below), nauty (nauty-labelg), gcc, curl.
#
# snarkhunter is not packaged anywhere. Build it:
#   curl -sSLO https://caagt.ugent.be/cubic/snarkhunter-2.0b.zip && unzip -q snarkhunter-2.0b.zip
#   curl -sSLO https://pallini.di.uniroma1.it/nauty2_8_8.tar.gz && tar xzf nauty2_8_8.tar.gz
#   (cd nauty2_8_8 && ./configure >/dev/null && make nautyW1.a nautyL1.a >/dev/null)
#   cp nauty2_8_8/{nausparse.h,nauty.h,nautyW1.a,nautyL1.a} snarkhunter-2.0b/
#   (cd snarkhunter-2.0b && make)
#
# Do NOT reach for nauty's geng here. Asked for cubic graphs it enumerates
# everything and filters down to 3-regular: ~145 graphs/sec/core, generation-
# bound, with the cycle test idling under 1% CPU. snarkhunter constructs cubic
# graphs directly and runs this pipeline at ~385,000/sec. Order 24 is the
# difference between "never finishes" and five minutes.

set -euo pipefail
cd "$(dirname "$0")"

N=${1:-24}
SHUNTER=${SNARKHUNTER:-./snarkhunter}
SHARDS=${SHARDS:-$(nproc)}

[[ -x "$SHUNTER" ]] || { echo "snarkhunter not found at $SHUNTER — see header" >&2; exit 1; }
command -v nauty-labelg >/dev/null || { echo "nauty-labelg not found" >&2; exit 1; }

[[ -x src/filt ]] || { echo "building filt..."; gcc -O3 -march=native -o src/filt src/filt.c; }

# Do not restrict the girth. 3 is not a power of two, so triangles are legal in a
# counterexample -- and every extremal graph at orders 24 and 26 has girth 3.
# Asking snarkhunter for girth >= 5 shrinks the search ~60x and returns nothing,
# which looks like a clean result and is an artifact of excluding the answers.
echo "order $N: enumerating all connected cubic graphs across $SHARDS shards"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
for i in $(seq 0 $((SHARDS-1))); do
    ( "$SHUNTER" "$N" 3 s o g m "$i" "$SHARDS" 2>/dev/null \
        | src/filt 4 8 > "$tmp/shard_$i.g6" 2>"$tmp/shard_$i.err" ) &
done
wait

cat "$tmp"/shard_*.g6 | sort -u > "$tmp/survivors.g6"
scanned=$(awk -F'scanned=' '{print $2}' "$tmp"/shard_*.err | awk '{s+=$1} END{print s}')
found=$(grep -c . "$tmp/survivors.g6" || true)
echo "scanned $scanned graphs; $found have neither a 4-cycle nor an 8-cycle"

echo "16-cycle test on the survivors (absence would be a counterexample):"
src/filt 16 < "$tmp/survivors.g6" > "$tmp/no16.g6" 2>/dev/null || true
n16=$(grep -c . "$tmp/no16.g6" || true)
if [[ "$n16" -gt 0 ]]; then
    echo "  *** $n16 graph(s) with no 4-, 8- or 16-cycle. Verify independently before believing it. ***"
    cat "$tmp/no16.g6"
else
    echo "  all $found contain a 16-cycle — no counterexample at order $N"
fi

ref="data/order${N}_markstrom.g6"
if [[ -f "$ref" ]]; then
    echo "diffing against Markstrom's own file, up to isomorphism:"
    nauty-labelg -q < "$ref"                 | sort > "$tmp/theirs"
    nauty-labelg -q < "$tmp/survivors.g6"    | sort > "$tmp/mine"
    if diff -q "$tmp/theirs" "$tmp/mine" >/dev/null; then
        echo "  identical — $(grep -c . "$tmp/theirs") graphs, zero differences either way"
    else
        echo "  DIFFERS:"; diff "$tmp/theirs" "$tmp/mine" | head
    fi
else
    echo "no reference file for order $N (Markstrom published 24, 26, 28, 30)"
fi
