#!/bin/sh
# Live end-to-end capture: the router in front of the real Bluesky reads.
# Writes demo.txt. Needs BSKY_HANDLE / BSKY_APP_PASSWORD in the environment
# for the authenticated AppView reads.
set -e
cd "$(dirname "$0")"
: "${GATE:=0.6}"
{
  echo "# needle_bsky ask — default router (grouped, heuristic stage 1), gate ${GATE}"
  echo
  for q in \
    "grab pfrazee.com's timeline" \
    "open the account page for austegard.com" \
    "who follows austegard.com" \
    "where is austegard.com's repo hosted" \
    "anything broken in the network at the moment" \
    "dig up posts on cactus needle" \
    "set a timer for twenty minutes"
  do
    echo "\$ python3 -m needle_bsky ask \"$q\""
    python3 -m needle_bsky ask "$q" --threshold "$GATE" 2>&1 | head -6 || true
    echo
  done
  echo "# the same queries against --router flat, all 18 declared to one agent"
  echo
  for q in \
    "grab pfrazee.com's timeline" \
    "who follows austegard.com" \
    "where is austegard.com's repo hosted"
  do
    echo "\$ python3 -m needle_bsky ask \"$q\" --router flat"
    python3 -m needle_bsky ask "$q" --router flat --threshold "$GATE" 2>&1 | head -3 || true
    echo
  done
} > demo.txt
cat demo.txt
