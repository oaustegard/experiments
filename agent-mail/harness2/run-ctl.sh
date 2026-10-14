#!/bin/bash
# controls: headless default agent (no team-member system prompt); briefs = mail-team briefs, +charter in brief for ctl-briefcharter
for arm in ctl-nocharter ctl-briefcharter; do for r in 1 2 3; do for n in alice bob carol dave; do
  T=/tmp/ac2/$arm-$r
  ( cd $T && claude -p --model haiku --setting-sources project --settings /tmp/ac2/ctl-settings.json \
      --allowedTools Bash Read Edit Write Glob Grep --max-turns 80 --output-format json \
      "Read your brief at /tmp/ac2-briefs/$arm-$r-$n.md and do what it says. Final report: ONE line." < /dev/null > $T/result-$n.json 2> $T/err-$n.txt ) &
done; done; done
wait; echo ALL-DONE
