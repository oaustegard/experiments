#!/bin/sh
# Four vocabulary-extension arms, then score them. Launched detached.
P=/home/user/models/potion-code-16M-v2; V=data/vocab_words.json
export OMP_NUM_THREADS=1
T=scripts/train_static.py
python3 $T --init $P --out models/potion-code-vocabsum --extend-vocab $V --init-mode sum --epochs 0 > .train_vocabsum0.log 2>&1
python3 $T --init $P --out models/potion-code-vocab --extend-vocab $V --init-mode mean --epochs 0 > .train_vocab0.log 2>&1
python3 $T --init $P --out models/potion-code-vocabsum-ft --extend-vocab $V --init-mode sum --epochs 8 > .train_vocabsum_ft.log 2>&1
python3 $T --init $P --out models/potion-code-vocab-ft --extend-vocab $V --init-mode mean --epochs 8 > .train_vocab_ft.log 2>&1
python3 scripts/run_bench.py potion-code-vocabsum=models/potion-code-vocabsum potion-code-vocab=models/potion-code-vocab potion-code-vocabsum-ft=models/potion-code-vocabsum-ft potion-code-vocab-ft=models/potion-code-vocab-ft > .bench_vocab.log 2>&1
