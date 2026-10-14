#!/bin/bash
# 3 teams x 4 headless team-member children, all concurrent; one JSON result per child
cd /tmp/ac2 || exit 1
for r in 1 2 3; do for n in alice bob carol dave; do
  T=/tmp/ac2/mail-team-$r
  ( cd $T && claude -p --agents /tmp/ac2/agents-def.json --agent team-member --model haiku \
      --setting-sources project --settings /home/user/claude-workspace/.claude/settings.local.json \
      --allowedTools Bash Read Edit Write Glob Grep --max-turns 80 --output-format json \
      "Your brief: /tmp/ac2-briefs/mail-team-$r-$n.md" < /dev/null > $T/result-$n.json 2> $T/err-$n.txt ) &
done; done
wait
echo ALL-DONE
