# lateon-pool-remex — errors

1. **Encode OOM (exit 137, 13 GB) after all document chunks were done.** The merge read
   `z["T"][a:b]` inside the per-document loop. `NpzFile.__getitem__` loads and decompresses the whole
   array on every access, and each per-document slice is a view that keeps its own full copy alive:
   about 1,000 copies of a 130 MB chunk. Fix: read `idx`, `T`, `off` once per chunk. The chunk
   checkpoints survived, so the resume lost nothing but the merge.

2. **Rounds 1 and 2 scored compressed codes as `q · decode(x)`, not through remex's search.** remex's centered
   `decode` and its search estimator are different estimates of the same code (remex `CLAUDE.md`, design decision
   4). The search divides by the expected cosine of a code, and that ranks better at low bit widths. On SciFact,
   unpooled 1-bit LateOn scored 0.7175 by decode and 0.7486 through `RemexIndex` (4-seed means), so every margin
   against 1-bit codes was overstated by about 0.03. Found while checking remex 2.2.0's `RemexIndex` on the full
   corpus. Round 3 rescored everything through `RemexIndex`, and the post and video were corrected. Any experiment
   that scored remex codes by decode (`token-pooling-remex`, `mxbai-edge-remex-quant`) carries the same bias at
   1 bit.
