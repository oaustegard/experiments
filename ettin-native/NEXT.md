# ettin-native: restart commands (lock yielded 2026-10-21 ~02:00, coord #38)

Wait for coord to hand cpu.lock back. Every job goes through the lock wrapper:
`run/after_lock.sh 9999999999 "ettin <job> ETA" <cmd>` (starts only when cpu.lock is gone, deletes it after).

Done so far: screen (results/screen.json), arctic teacher targets (out/data/), distill e17-base (stopped after
ep1, 0.503), distill e17-cap (3 ep, SYM 0.802 / ASYM-arctic 0.832), contrast e17-cap-c and leafir-c
(results/contrast_*.json).

1. Pick the best Ettin checkpoint (contrast ep vs distill ep3) by SYM hit@10.
2. Export + compression ladder (CPU, ~4 min per eval variant):
   .venv/bin/python -I run/export.py out/<best ckpt dir> e17best fp32 t8-w8 t4-w8 k32000-t8-w8 k32000-t4-w8 k16000-t4-w8
   (pruning frequencies now include MS MARCO/NQ general English, GENERAL=True)
   .venv/bin/python -I run/eval_onnx.py e17best
   Same for the leaf-ir control's best checkpoint (name leafirbest), to compare at matched bytes.
3. Browser bench (wasm, 1 thread, uncontended, under the lock): http://localhost:8765/web/bench.html?m=/out/models/<name>/<variant>
   via window.runAll([...]) in the bench tab; also retime leaf-ir (wasm #29 asked).
4. Optional if the lock allows: continue e17-cap distill 2 more epochs (it was still rising 0.674/0.770/0.802):
   .venv/bin/python -I run/distill.py e17-cap5 --base out/students/e17-cap/epoch3 --epochs 2 --lr 3e-4
   (note: --base with a full ST dir would wrap it in a second pooling/Dense; needs a --init flag first).
5. Pareto plot + RESULTS.md: total download = ORT wasm 3.07 MB br + model br + tokenizer br; SYM panel.
