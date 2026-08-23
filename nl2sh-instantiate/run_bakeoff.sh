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
    && { git diff --cached --quiet -- "nl2sh-instantiate/$f" \
         && { echo "== $f already committed"; exit 0; } || true; } \
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
  if [ -s "$out" ]; then echo "== skip run for $out (already present)"
    keep "$out" "$model $dtype, $cond, oracle sources"; return 0; fi
  echo "== $(date +%T) $out  [$model $dtype $cond $*]"
  python3 run_gen.py --model "$model" --dtype "$dtype" --condition "$cond" \
      --tldr "$TLDR" --out "$out" "$@" 2>&1 | grep -v 'max_new_tokens' | tail -4
  keep "$out" "$model $dtype, $cond, oracle sources"
}

bench() {  # bench <out> <model> <dtype>
  local out="$1" model="$2" dtype="$3"
  await_inflight "$out"
  if [ -s "$out" ]; then echo "== skip run for $out (already present)"
    keep "$out" "laptop bar for $model at $dtype, batch=1 on 4 CPU cores"; return 0; fi
  echo "== $(date +%T) $out  [bench $model $dtype]"
  python3 bench.py --model "$model" --dtype "$dtype" --out "$out" 2>&1 | tail -20
  keep "$out" "laptop bar for $model at $dtype, batch=1 on 4 CPU cores"
}

gguf() {  # gguf <out> <repo> <file> <condition> [extra args...]
  local out="$1" repo="$2" file="$3" cond="$4"; shift 4
  await_inflight "$out"
  if [ -s "$out" ]; then echo "== skip run for $out (already present)"
    keep "$out" "$repo $file, $cond, oracle sources, llama.cpp"; return 0; fi
  echo "== $(date +%T) $out  [$repo $file $cond $*]"
  python3 run_gen_gguf.py --model "$repo" --gguf-file "$file" --condition "$cond" \
      --tldr "$TLDR" --n-threads "$OMP_NUM_THREADS" --out "$out" "$@" 2>&1 | tail -4
  keep "$out" "$repo $file, $cond, oracle sources, llama.cpp"
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
gguf)
  # The quantised lane. Nemotron 3 Nano 4B is the row #52 most wants: a hybrid
  # Mamba2/attention model rather than a dense transformer, and the middle point
  # of its three-point vocabulary axis — 65k Pleias, 131k Nemotron, 262k Gemma.
  # Gemma 4 E2B is the other row #52 tabulates. Google's own q4_0 GGUF is gated
  # (307 to login) but unsloth's Q4_K_M mirror of the same weights is not —
  # api 200 gated:false, and a ranged 1-byte fetch of the weight file returns
  # 206, which is the only proof that counts. The mmproj vision towers in that
  # repo are not downloaded; this is a text task.
  NEM=nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
  NEMF=NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
  G4=unsloth/gemma-4-E2B-it-GGUF
  G4F=gemma-4-E2B-it-Q4_K_M.gguf
  # `generate` on both bases before either second condition, so a queue that
  # runs out of time still leaves the primary condition on every model rather
  # than two conditions on one.
  # Gemma 4 E2B does not reason, so it runs the shared 64-token budget over the
  # whole eval and its row sits beside every other row unqualified.
  gguf results_gemma4e2b_q4km_generate.json              "$G4"  "$G4F"  generate
  gguf results_gemma4e2b_q4km_instantiate_anchored.json  "$G4"  "$G4F"  instantiate_anchored
  # Nemotron does reason, and the shared budget is not neutral for it: at 64
  # tokens it scored 0.000 routing with every row truncated mid-thought, and a
  # /no_think system turn did not stop it — it reasons in untagged prose instead.
  # Given 200 tokens it answers (3 of 4 on a probe) and needs 107 new tokens on
  # average to reach a ~15-token command. So its row runs at 200 tokens and says
  # so, on a capped n: 65.8 s a row here means the full eval is 3.3 hours, and
  # the latency is itself the finding for a laptop bar.
  gguf results_nemotron4b_q4km_generate_budget200.json   "$NEM" "$NEMF" generate \
       --max-new-tokens 200 --limit 40
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
