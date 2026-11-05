#!/usr/bin/env bash
# Ettin arms, sequential. Each arm is skipped if its results file exists (resume-safe).
set -u
cd "$(dirname "$0")/.."
run() { local name=$1; shift; [ -f "results/$name.json" ] && { echo "skip $name"; return; }
        python3 run/train_enc.py "$name" "$@" 2>&1 | grep -v -i warning; }

run spec-e17c --init e17c
run spec-stock --init stock

# LOWO uses the init with the lower validation KL on the specialist arms
best=$(python3 - <<'EOF'
import json
v = {i: min(e["val_kl"] for e in json.load(open(f"results/spec-{i}.json"))["log"]) for i in ("e17c", "stock")}
print(min(v, key=v.get))
EOF
)
echo "lowo init: $best"
W="agent_trace_observability customer_service invoice_processing security_incidents"
for w in $W; do
  tr=$(echo $W | tr ' ' '\n' | grep -v "^$w$" | tr '\n' ' ')
  run "lowo-$best-$w" --init "$best" --train $tr --test "$w"
done
echo ALL_DONE
