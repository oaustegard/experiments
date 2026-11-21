#!/bin/sh
# Re-index ~/Projects write-ups. Run by cron; safe to run by hand. Skips if a run is already going.
cd /Users/austegard/Projects/ane-search || exit 1
LOCK=index/.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  # stale lock older than 30 minutes: take it over
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then rmdir "$LOCK"; mkdir "$LOCK" || exit 0; else exit 0; fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
OUT=$(/Users/austegard/Projects/ane-search/.venv/bin/python index.py 2>&1 | tail -1)
echo "$(date '+%Y-%m-%d %H:%M:%S') $OUT" >> index/reindex.log
tail -n 500 index/reindex.log > index/reindex.log.tmp && mv index/reindex.log.tmp index/reindex.log
