# Error log — minicpm-ocr-mlx

Every error made in this experiment, with how it was caught and which way it
pushed the result.

| # | Error | Caught by | Direction |
|---|---|---|---|
| 1 | Prior-art search ran after the port was built, not before. `nativ-community` had published MLX conversions of MiniCPM-OCR 1.0 (bf16 to mxfp4) the day before. | The repo convention, at write-up time | Changed the framing: this is a second port, distinguished by running on released mlx-vlm and by token-level parity, not a first port |
| 2 | First parity run used the bf16 checkpoint on MLX's CPU backend, which has no fast bf16 path; 11 CPU-minutes went by without one image finishing. | `ps` showing the job alive but silent | None on results; switched parity to an fp32 conversion |
| 3 | The mlx-cpu Linux wheel bundles the netlib reference BLAS (4.2 s for a 2048² fp32 matmul against numpy's 0.14 s). | Timing a single matmul | None on results; `LD_PRELOAD` of an OpenBLAS (OpenCV's bundled copy) cut it 10× |
| 4 | Sweep output went through `grep` into a file, which block-buffers, so finished results looked like a hung job; the job was killed with three of five models already done. | Their `sweep.json` files on disk | Lost about an hour of CPU; rerun with `python -u` and `grep --line-buffered` |
| 5 | Disk allowance ran out mid-conversion; three bf16 checkpoints were written truncated (1 MB, 36 KB, 869 MB) without an error reaching the summary line. | `du` showing sizes inconsistent with the reported bits per weight | Would have put wrong sizes in the table; reconverted after freeing space, and sizes in the README are file sizes read back afterwards |
| 6 | The first CLI prompt-format choice reused the 4.6 entry, which wraps images as `(<image>./</image>)` and leaves the parentheses as two prompt tokens. `parity.py` passes the transformers prompt directly, so it could not see this. | A separate check of `apply_chat_template` output against the reference ids | Would have shipped a CLI path with a prompt unlike the one the model was trained on; fixed to the list format, now id-identical on all four cases |
