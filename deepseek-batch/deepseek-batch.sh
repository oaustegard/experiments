#!/usr/bin/env bash
# deepseek-batch.sh — overnight task runner for a local model behind an
# OpenAI-compatible endpoint. One task per git worktree, every task gated on a
# machine-checkable acceptance command. Nothing merges; you review branches.
#
#   ./deepseek-batch.sh            run the queue once
#   ./deepseek-batch.sh --dry-run  show what would run
#
# Layout (all under $BATCH_HOME, default ~/batch):
#   queue/*.task    pending, run in lexical order (prefix 010-, 020- to order)
#   running/        the task currently executing
#   done/           passed the gate
#   failed/         ran, failed the gate
#   logs/<id>/      transcript, diff, baseline + post test output
#   reports/        one markdown summary per run
#
# A .task file is sourced as bash. It is your own file; treat it as code.
#   REPO=/Users/you/src/thing        # git repo to work in
#   BASE=main                        # branch to fork from
#   PROMPT="..."                     # the instruction (or PROMPT_FILE=...)
#   FILES="src/a.py src/b.py"        # files added to the agent's context
#   TEST_CMD="pytest -x -q tests/test_a.py"   # the gate
#   FULL_TEST_CMD="pytest -x -q"     # collateral-damage check (default TEST_CMD)
#   BASELINE=green                   # green|red|skip — required state pre-run
#   MAX_DIFF_LINES=400               # reject runaway diffs
#   TIMEOUT=5400                     # seconds; kill the agent past this

set -uo pipefail

BATCH_HOME="${BATCH_HOME:-$HOME/batch}"
MODEL="${BATCH_MODEL:-deepseek-v4.1-flash}"
API_BASE="${BATCH_API_BASE:-http://127.0.0.1:8000/v1}"
API_KEY="${BATCH_API_KEY:-local}"
DEADLINE_HOUR="${BATCH_DEADLINE_HOUR:-8}"   # stop starting new tasks at 08:00
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$BATCH_HOME"/{queue,running,done,failed,logs,reports}
RUN_ID="$(date +%Y%m%d-%H%M%S)"
REPORT="$BATCH_HOME/reports/$RUN_ID.md"

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
say() { printf '%s\n' "$*" >>"$REPORT"; }

# --- preflight -------------------------------------------------------------
# A dead endpoint burns the whole night silently. Check before anything else.
if ! curl -sf --max-time 20 "$API_BASE/models" >/dev/null; then
  log "FATAL: no model server at $API_BASE"
  exit 1
fi
command -v mini >/dev/null || { log "FATAL: mini (mini-swe-agent) not on PATH"; exit 1; }

# Local endpoint via litellm's OpenAI-compatible path. Cost tracking is off
# because a local model has no price and litellm errors on an unknown one.
export MSWEA_COST_TRACKING="ignore_errors"
export OPENAI_API_KEY="$API_KEY"

# Written per task (see write_agent_config) so limits and trajectory path are
# per task. The limits are the unattended-safety half: a 2-bit model that
# starts emitting malformed actions spirals, and max_consecutive_format_errors
# stops it at three rather than at the wall clock.
write_agent_config() {
  local path="$1" traj="$2" secs="$3"
  cat >"$path" <<YAML
model:
  model_name: "hosted_vllm/$MODEL"
  model_kwargs:
    custom_llm_provider: "openai"
    api_base: "$API_BASE"
agent:
  cost_limit: 0
  step_limit: ${BATCH_STEP_LIMIT:-40}
  wall_time_limit_seconds: $secs
  max_consecutive_format_errors: 3
  output_path: "$traj"
YAML
}

# Measure prompt-processing speed once and record it. Prefill is what decides
# whether an agent loop is viable at all; if it has collapsed you want the
# number in the report rather than a mysteriously empty morning.
probe_prefill() {
  local pad start end ms
  pad="$(head -c 8000 /dev/urandom | base64 | head -c 8000)"
  start=$(date +%s%N)
  curl -sf --max-time 600 "$API_BASE/chat/completions" \
    -H "Authorization: Bearer $API_KEY" -H 'Content-Type: application/json' \
    -d "$(printf '{"model":"%s","max_tokens":1,"messages":[{"role":"user","content":%s}]}' \
          "$MODEL" "$(printf '%s' "ignore this: $pad" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')")" \
    >/dev/null || { echo "probe failed"; return; }
  end=$(date +%s%N); ms=$(( (end-start)/1000000 ))
  # ~2000 tokens of base64 padding; rough but comparable run to run.
  echo "~2000 tok prompt, 1 tok out: ${ms}ms  (=> ~$(( 2000000 / (ms>0?ms:1) )) tok/s prefill)"
}

say "# batch run $RUN_ID"
say ""
say "- endpoint: \`$API_BASE\` model \`$MODEL\`"
say "- prefill probe: $(probe_prefill)"
say ""

# --- per-task --------------------------------------------------------------
run_task() {
  local taskfile="$1"
  local id; id="$(basename "$taskfile" .task)"
  local dir="$BATCH_HOME/logs/$id"; mkdir -p "$dir"

  # defaults, then the task file overrides them
  REPO=""; BASE="main"; PROMPT=""; PROMPT_FILE=""; FILES=""
  TEST_CMD=""; FULL_TEST_CMD=""; BASELINE="skip"
  MAX_DIFF_LINES=400; TIMEOUT=5400
  # shellcheck disable=SC1090
  source "$taskfile"
  [[ -n "$PROMPT_FILE" ]] && PROMPT="$(cat "$PROMPT_FILE")"

  # The gate is not optional. A task with no acceptance command cannot be
  # verified overnight and must not be queued.
  if [[ -z "$REPO" || -z "$PROMPT" || -z "$TEST_CMD" ]]; then
    log "$id: REPO, PROMPT and TEST_CMD are all required — skipping"
    say "## $id — REJECTED (missing REPO/PROMPT/TEST_CMD)"; say ""
    mv "$taskfile" "$BATCH_HOME/failed/"; return
  fi
  : "${FULL_TEST_CMD:=$TEST_CMD}"

  local branch="batch/$id"
  local wt="$BATCH_HOME/wt/$id"

  # Isolate build caches per worktree. A shared CARGO_TARGET_DIR scored a
  # `todo!()` stub as 23 tests passed in ../harness-bench — a gate that cannot
  # fail is worse than no gate, because it ships the garbage with a green tick.
  # Extend this list for any toolchain whose cache lives outside the tree.
  export CARGO_TARGET_DIR="$wt/.batch-target"
  export GOCACHE="$wt/.batch-gocache"
  export PYTHONPYCACHEPREFIX="$wt/.batch-pycache"
  export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"

  if (( DRY_RUN )); then
    log "would run $id in $REPO -> $branch"; return
  fi

  log "$id: preparing worktree"
  git -C "$REPO" fetch --quiet origin "$BASE" 2>/dev/null
  git -C "$REPO" worktree remove --force "$wt" 2>/dev/null
  if ! git -C "$REPO" worktree add -B "$branch" "$wt" "origin/$BASE" >"$dir/setup.log" 2>&1 \
     && ! git -C "$REPO" worktree add -B "$branch" "$wt" "$BASE" >>"$dir/setup.log" 2>&1; then
    log "$id: worktree failed"; say "## $id — FAILED (worktree)"; say ""
    mv "$taskfile" "$BATCH_HOME/failed/"; return
  fi

  # Baseline. A fix task whose test is already green is a no-op; a refactor
  # task whose suite is already red cannot be judged afterwards. Either way
  # the run is worthless, so spend the 60s to find out now.
  local base_rc=0
  if [[ "$BASELINE" != "skip" ]]; then
    ( cd "$wt" && eval "$TEST_CMD" ) >"$dir/baseline.log" 2>&1; base_rc=$?
    if [[ "$BASELINE" == "green" && $base_rc -ne 0 ]] || \
       [[ "$BASELINE" == "red"   && $base_rc -eq 0 ]]; then
      log "$id: baseline is not $BASELINE (rc=$base_rc) — skipping"
      say "## $id — SKIPPED (baseline not $BASELINE, rc=$base_rc)"; say ""
      git -C "$REPO" worktree remove --force "$wt"
      mv "$taskfile" "$BATCH_HOME/failed/"; return
    fi
  fi

  # Hand the agent the test command rather than running the loop for it.
  # harness-bench measured both shapes at a fixed model: feeding the test
  # output back once scored 8/12, letting the agent run the suite itself
  # scored 11/12. The gate below is still ours — this is the agent's own
  # check, not the acceptance check.
  local rendered="$PROMPT

Verify your work by running: $TEST_CMD
Iterate until it passes. Do not edit the tests themselves."
  [[ -n "$FILES" ]] && rendered="$rendered
The relevant files are: $FILES"

  write_agent_config "$dir/mini.yaml" "$dir/trajectory.json" "$TIMEOUT"

  log "$id: running agent (timeout ${TIMEOUT}s)"
  local t0 t1 rc
  t0=$(date +%s)
  # timeout(1) is the outer bound; the config's wall_time_limit_seconds lets
  # the agent exit cleanly and write its trajectory before that fires.
  ( cd "$wt" && timeout $(( TIMEOUT + 120 )) mini \
      -y --exit-immediately \
      -c "$dir/mini.yaml" \
      -t "$rendered" ) >"$dir/agent.log" 2>&1
  rc=$?
  t1=$(date +%s)
  local mins=$(( (t1-t0)/60 ))
  [[ $rc -eq 124 ]] && log "$id: agent hit the timeout"

  # --- the gate ---
  local verdict="PASS" reason=""
  local post_rc full_rc added removed total
  ( cd "$wt" && eval "$TEST_CMD" )      >"$dir/post.log"  2>&1; post_rc=$?
  ( cd "$wt" && eval "$FULL_TEST_CMD" ) >"$dir/full.log"  2>&1; full_rc=$?
  git -C "$wt" add -A
  git -C "$wt" diff --cached >"$dir/diff.patch"
  read -r added removed < <(git -C "$wt" diff --cached --numstat |
    awk '{a+=$1; r+=$2} END {print a+0, r+0}')
  total=$(( added + removed ))

  if [[ $total -eq 0 ]];                 then verdict=FAIL; reason="no changes"
  elif [[ $post_rc -ne 0 ]];             then verdict=FAIL; reason="gate test failed (rc=$post_rc)"
  elif [[ $full_rc -ne 0 ]];             then verdict=FAIL; reason="full suite regressed (rc=$full_rc)"
  elif [[ $total -gt $MAX_DIFF_LINES ]]; then verdict=FAIL; reason="diff $total lines > MAX_DIFF_LINES=$MAX_DIFF_LINES"
  elif [[ $rc -eq 124 ]];                then verdict=FAIL; reason="agent timed out"
  fi

  if [[ "$verdict" == "PASS" ]]; then
    git -C "$wt" -c user.email="batch@local" -c user.name="batch" \
        commit -q -m "$id: $(printf '%s' "$PROMPT" | head -c 72)"
    git -C "$REPO" push -u origin "$branch" >>"$dir/setup.log" 2>&1 \
      || log "$id: push failed (branch kept locally)"
    mv "$taskfile" "$BATCH_HOME/done/"
  else
    # Keep the branch and the patch. A failed run is diagnostic material.
    mv "$taskfile" "$BATCH_HOME/failed/"
  fi
  git -C "$REPO" worktree remove --force "$wt" 2>/dev/null

  say "## $id — $verdict ${reason:+($reason)}"
  say ""
  say "- ${mins}m, +$added/-$removed lines, branch \`$branch\`"
  say "- gate \`$TEST_CMD\` rc=$post_rc; full rc=$full_rc"
  say "- logs: \`$dir\`"
  say ""
  log "$id: $verdict ${reason}"
}

# --- queue -----------------------------------------------------------------
shopt -s nullglob
for taskfile in "$BATCH_HOME"/queue/*.task; do
  if [[ $(date +%-H) -ge $DEADLINE_HOUR && $(date +%-H) -lt 20 ]]; then
    log "past deadline hour $DEADLINE_HOUR — stopping"
    say "_stopped at deadline; $(ls "$BATCH_HOME"/queue/*.task 2>/dev/null | wc -l) tasks left_"
    break
  fi
  mv "$taskfile" "$BATCH_HOME/running/" 2>/dev/null || continue
  run_task "$BATCH_HOME/running/$(basename "$taskfile")"
done

log "run $RUN_ID complete — report: $REPORT"
[[ $DRY_RUN -eq 0 ]] && cat "$REPORT"
