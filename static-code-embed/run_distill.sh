#!/bin/sh
export OMP_NUM_THREADS=3
python3 scripts/distill_bekko.py > .distill.log 2>&1 || exit 1
python3 scripts/run_bench.py bekko-a25m=bekko:../bekko-embedding-bench/vecs_ast_a25m.f32 distill-bekko-a25m=models/distill-bekko-a25m > .bench_distill.log 2>&1
