#!/bin/bash
# The #52 bake-off queue: same corpus, same eval, same prompts, only --model differs.
#
# One queue at a time per invocation, because 15 GB of RAM does not hold two
# multi-billion-parameter arms at once and because the point of the laptop bar is
# a batch=1 number that a co-tenant would spoil. Each step writes its result,
# commits it, and pushes before the next one starts — a reclaimed container then
# costs the step in flight, not the queue.
#
#   ./run_bakeoff.sh small     # 270M dtype control, 1B, and their bench rows
#   ./run_bakeoff.sh large     # the 4B arm's remaining conditions and its bench row
#   ./run_bakeoff.sh bench-only
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
TLDR=/home/user/corpora/tldr/pages
BRANCH=claude/issue-52-bakeoff
QUEUE="${1:-small}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

cd "$HERE" || exit 1
set -a; . /mnt/project/GitHub.env 2>/dev/null; set +a

keep() {  # keep <file> <message> — commit and push one artifact as it lands
  local f="$1" msg="$2"
  [ -s "$f" ] || { echo "!! $f missing or empty, not committing"; return 1; }
  ( cd "$REPO" \
    && git add "nl2sh-instantiate/$f" \
    && git -c commit.gpgsign=false -c user.email=muninn@austegard.com -c user.name=muninn \
         commit -q -m "nl2sh-instantiate: $f — $msg" \
    && { git pull --rebase -q origin "$BRANCH" && git push -q origin "$BRANCH"; } \
    && echo "++ pushed $f" ) || echo "!! keep failed for $f"
}

inflight() {  # another process is already producing this exact --out
  pgrep -f -- "--out $1" >/dev/null 2>&1
}

await_inflight() {  # let a run someone else launched finish rather than duplicating it
  local out="$1" waited=0
  while inflight "$out"; do
    [ "$waited" -eq 0 ] && echo "== $out is already running elsewhere, waiting for it"
    sleep 30; waited=$((waited + 30))
    [ "$waited" -gt 7200 ] && { echo "!! gave up waiting on in-flight $out"; return 1; }
  done
  [ "$waited" -gt 0 ] && echo "== in-flight $out finished after ${waited}s"
  return 0
}

gen() {  # gen <out> <model> <dtype> <condition> [extra args...]
  local out="$1" model="$2" dtype="$3" cond="$4"; shift 4
  await_inflight "$out"
  if [ -s "$out" ]; then echo "== skip $out (already present)"; return 0; fi
  echo "== $(date +%T) $out  [$model $dtype $cond $*]"
  python3 run_gen.py --model "$model" --dtype "$dtype" --condition "$cond" \
      --tldr "$TLDR" --out "$out" "$@" 2>&1 | grep -v 'max_new_tokens' | tail -4
  keep "$out" "$model $dtype, $cond, oracle sources"
}

bench() {  # bench <out> <model> <dtype>
  local out="$1" model="$2" dtype="$3"
  await_inflight "$out"
  if [ -s "$out" ]; then echo "== skip $out (already present)"; return 0; fi
  echo "== $(date +%T) $out  [bench $model $dtype]"
  python3 bench.py --model "$model" --dtype "$dtype" --out "$out" 2>&1 | tail -20
  keep "$out" "laptop bar for $model at $dtype, batch=1 on 4 CPU cores"
}

case "$QUEUE" in
small)
  # The dtype control. 4B cannot be float32 in 15 GB, so no cross-model row is
  # readable until bfloat16 is shown not to move the metrics on a model whose
  # float32 answer is already committed.
  gen results_it_instantiate_anchored_bf16.json unsloth/gemma-3-270m-it bfloat16 instantiate_anchored
  gen results_1b_generate.json            unsloth/gemma-3-1b-it bfloat16 generate
  gen results_1b_instantiate_anchored.json unsloth/gemma-3-1b-it bfloat16 instantiate_anchored
  gen results_1b_generate_anchored.json   unsloth/gemma-3-1b-it bfloat16 generate_anchored
  bench results_bench_270m_fp32.json unsloth/gemma-3-270m-it float32
  bench results_bench_270m_bf16.json unsloth/gemma-3-270m-it bfloat16
  bench results_bench_1b_bf16.json   unsloth/gemma-3-1b-it   bfloat16
  ;;
large)
  # 4B measured at ~0.6 min/row on two cores, so a full 179-row condition is
  # ~110 minutes. `generate` runs full because it is the row that has to be
  # comparable to stage 1; the second condition is capped and read on the rows
  # the two share, which is what bakeoff_table.py is for.
  gen results_4b_instantiate_anchored.json unsloth/gemma-3-4b-it bfloat16 instantiate_anchored --limit 100
  bench results_bench_4b_bf16.json unsloth/gemma-3-4b-it bfloat16
  ;;
bench-only)
  bench results_bench_270m_fp32.json unsloth/gemma-3-270m-it float32
  bench results_bench_270m_bf16.json unsloth/gemma-3-270m-it bfloat16
  bench results_bench_1b_bf16.json   unsloth/gemma-3-1b-it   bfloat16
  bench results_bench_4b_bf16.json   unsloth/gemma-3-4b-it   bfloat16
  ;;
*) echo "unknown queue: $QUEUE"; exit 2;;
esac

echo "QUEUE_${QUEUE}_DONE $(date +%T)"
touch "/tmp/bakeoff_${QUEUE}.done"
