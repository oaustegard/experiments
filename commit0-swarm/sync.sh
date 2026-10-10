#!/bin/sh
# Commit and push run state while agents write into data/runs (stop hook wants a clean tree).
set -e
cd "$(dirname "$0")/.."
git add -A commit0-swarm
git diff --cached --quiet || git -c user.email=oskar@austegard.com -c user.name='Oskar Austegard' commit -q -m "commit0-swarm: ${1:-run state}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Ar1F6HvPNeMUeWp7uUXD4H"
git fetch -q origin main && git rebase -q --autostash origin/main && git push -q origin HEAD:main
git log --oneline -1; git status --short | wc -l
