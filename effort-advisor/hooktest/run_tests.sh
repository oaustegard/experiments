#!/bin/zsh
# Runs every hook experiment (questions 1-7) headless and writes raw output to out/.
# Needs a logged-in CLI: `claude auth status` must say loggedIn true.
# Usage: ./run_tests.sh [all | q1 q2 q3 q4 q5 q7 ...]      (q6 is answered from the q1/q2 logs)
H=${0:A:h}; P=$H/proj; OUT=$H/out
mkdir -p $OUT; cd $P || exit 1
# Run like a terminal session: drop what a parent Claude Code session exports (CLAUDE_EFFORT etc.),
# so anything the hook sees in its env was put there by the session under test.
for v in ${(k)parameters[(R)*export*]}; do [[ $v == CLAUDE* ]] && unset $v; done
if ! claude auth status 2>/dev/null | grep -q '"loggedIn": true'; then echo "claude CLI is not logged in; run 'claude' and /login first"; exit 2; fi
# The workspace is untrusted, so permissions.allow in the project settings is ignored: allow tools on the command line.
COMMON=(--setting-sources project --strict-mcp-config --output-format json --allowedTools Bash Skill Agent Read Grep Glob)
run() {  # run <name> <claude args...> ; prompt is the last arg. Saves out/<name>.json, echoes the session id.
  local name=$1; shift
  claude -p $COMMON "$@" < /dev/null > $OUT/$name.json 2> $OUT/$name.err
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("session_id",""))' $OUT/$name.json 2>/dev/null
}
result() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("RESULT:", str(d.get("result"))[:600]); print("turns", d.get("num_turns"), "cost", d.get("total_cost_usd"), "ms", d.get("duration_ms"))' $OUT/$1.json; }
report() { { echo "### $1"; result $1; python3 $H/analyze.py $2; echo; } | tee -a $OUT/summary.txt; }
want() { [[ " $TESTS " == *" all "* || " $TESTS " == *" $1 "* ]]; }
TESTS="${*:-all}"
TWO='Run `echo one` with the Bash tool. Then, as a separate later tool call, run `echo two`. Then reply with the single word done.'

if want q1; then
  s=$(HOOKTEST_MODE=nonce HOOKTEST_NONCE=ZEBRA-4471 run q1_nonce --model haiku 'What is the advisor nonce? Reply with just the nonce, or NONE if you do not know one.')
  report q1_nonce $s
  s=$(HOOKTEST_MODE=none run q1_control --model haiku 'What is the advisor nonce? Reply with just the nonce, or NONE if you do not know one.')
  report q1_control $s
fi

if want q2; then
  # a: project effortLevel low, hook asks for the xhigh skill; then a second turn in the same session with no hook advice
  s=$(HOOKTEST_MODE=escalate HOOKTEST_SKILL=escalating-effort run q2a_xhigh_turn1 --model sonnet $TWO)
  report q2a_xhigh_turn1 $s
  s2=$(HOOKTEST_MODE=none run q2a_xhigh_turn2 --model sonnet --resume $s 'Run `echo three` with the Bash tool, then reply done.')
  report q2a_xhigh_turn2 $s2
  # control: same prompt, no skill
  s=$(HOOKTEST_MODE=none run q2_control --model sonnet $TWO); report q2_control $s
  # b: the ladder
  for lvl in medium high; do
    s=$(HOOKTEST_MODE=escalate HOOKTEST_SKILL=effort-$lvl run q2b_$lvl --model sonnet $TWO); report q2b_$lvl $s
  done
  # c: lowering. Session at high (--effort high), skill says low
  s=$(HOOKTEST_MODE=escalate HOOKTEST_SKILL=effort-low run q2c_lower --model sonnet --effort high $TWO); report q2c_lower $s
  s=$(HOOKTEST_MODE=none run q2c_high_control --model sonnet --effort high $TWO); report q2c_high_control $s
fi

THREE='Make three separate Bash tool calls, one after another, never in parallel: each runs `sleep 2; echo x`. After the third, reply with the advisor nonce if any message has told you one, otherwise reply NONE.'
if want q3; then
  for d in post pre; do
    s=$(HOOKTEST_MODE=async HOOKTEST_DELIVER=$d HOOKTEST_NONCE=HERON-$d-9024 run q3_async_$d --model haiku $THREE); report q3_async_$d $s
  done
  s=$(HOOKTEST_MODE=async HOOKTEST_DELIVER=both HOOKTEST_NONCE=CRANE-1 run q3_zero_tool --model haiku 'Do not use any tools. Reply with the advisor nonce if any message has told you one, otherwise reply NONE.'); report q3_zero_tool $s
  s=$(HOOKTEST_MODE=async HOOKTEST_DELIVER=both HOOKTEST_STOP=block HOOKTEST_NONCE=CRANE-2 run q3_zero_tool_stop --model haiku 'Do not use any tools. Reply with the advisor nonce if any message has told you one, otherwise reply NONE.'); report q3_zero_tool_stop $s
  grep -h '"via"' $H/logs/delivered.jsonl | tee -a $OUT/summary.txt
fi

FIND='find every place in /Users/austegard/Projects/claude-skills where a skill declares an effort frontmatter and summarise. Read-only: do not modify anything.'
if want q4; then
  for i in 1 2; do s=$(HOOKTEST_MODE=delegate run q4_delegate_$i --model sonnet --add-dir /Users/austegard/Projects/claude-skills $FIND); report q4_delegate_$i $s; done
  s=$(HOOKTEST_MODE=none run q4_control --model sonnet --add-dir /Users/austegard/Projects/claude-skills $FIND); report q4_control $s
fi

if want q5; then python3 $H/latency.py $OUT/q5_latency.json | tee -a $OUT/summary.txt; fi

if want q7; then
  for f in stripped bare; do
    touch $OUT/.marker; sleep 1
    n0=$(grep -c '"mode": "inner"' $H/logs/UserPromptSubmit.jsonl 2>/dev/null); n0=${n0:-0}
    s=$(HOOKTEST_MODE=recurse HOOKTEST_INNER_FLAGS=$f run q7_recurse_$f --model haiku 'Reply with the single word outer.')
    n1=$(grep -c '"mode": "inner"' $H/logs/UserPromptSubmit.jsonl 2>/dev/null); n1=${n1:-0}
    { echo "### q7_recurse_$f outer session $s"; echo "inner hook runs logged: $((n1-n0))"
      echo "session files written since the run started:"; find ~/.claude/projects -name '*.jsonl' -newer $OUT/.marker | sed "s|$HOME|~|"
      tail -1 $H/logs/inner_call.jsonl; echo; } | tee -a $OUT/summary.txt
  done
fi
echo "raw outputs in $OUT, hook logs in $H/logs"
