# ERRORS — eg2-layer-student

What broke, what it cost, and what changed because of it.

1. **EG2's published bf16 weights are not the teacher.** The index was built with
   `model_q4.onnx`. The bf16 safetensors, run in PyTorch, matched the fp32 ONNX
   exactly (cosine 0.9999998) but agreed with the q4 targets at cosine 0.990 and
   0.87 top-10 overlap on 200 real queries. A student initialised from them would
   have started 13 points below the teacher at k = 24. `run/model.py` now
   dequantizes the q4 ONNX's own weights into the HF modules; k = 24 then gives
   overlap 1.000.
2. **The q4 token table has the embedding scale folded in.** The first
   dequantized model gave cosine 0.63 to the targets. Every MatMul weight matched
   bf16 at cosine 0.997, and the table was 22.7x bf16's norm: the ONNX stores
   `embed_tokens * sqrt(512)`, and the HF module multiplies again. Divided out.
3. **transformers silently rewrites a cut model's last layer.**
   `EmbeddingGemma2TextConfig.__post_init__` forces the last layer to
   `full_attention` (a warning, then an in-place edit of the `layer_types` list it
   was passed). For every k except 6, 12, 18 and 24 that swaps the last layer's RoPE theta from
   1e4 to 1e6. Because the edit mutates the caller's list, a first fix that
   reassigned the same list object did nothing. Fixed by copying the list before
   building the config and reassigning after; checked by matching each cut
   layer's output to the full model's (max difference 0.0). The first L0 run was
   discarded.
4. **`onnx.save` appends to an existing external-data file.** Re-exporting into
   the same directory doubled every reported size (k = 2 q4: 21.8 MB became
   42.9 MB, with a model that still loaded). `run/export.py` deletes the old
   files first.
5. **`quantize_dynamic` names its external data `model.onnx.data`, not
   `model.onnx_data`.** The int8 size first counted the graph only (0.9 MB).
6. **Random batches padded to 69 tokens per row when the mean is 26.** A k = 4
   step took 5.0 s. Length-bucketed batches (sorted inside random 50-batch chunks)
   brought it to about 1.7 s.
7. **#102's learning rate is far too low for a cut model.** At lr 1e-4, k = 4 had
   0.10 real overlap after 300 steps; at 1e-3, 0.33; at 3e-3, 0.336. All runs
   use 3e-3.
8. **The 2-hour background cap stopped k = 8 at step 2,100 of 2,572**, as
   expected. It resumed from the step-2,100 checkpoint with optimizer state.
9. **`pkill -f` killed its own shell.** The repo guard caught the first attempt.
   The second bracketed the pattern (`[k]4_ep2`), but the same command line also
   held `k4_ep2` unbracketed in an `rm` path, so it matched the wrapper shell and
   killed the edit that followed. The k4 continuation was relaunched with a
   fresh batch order (`--seed 1`) instead of replaying epoch 1's.
