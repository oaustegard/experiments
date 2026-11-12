# lateon-pool-remex — errors

1. **Encode OOM (exit 137, 13 GB) after all document chunks were done.** The merge read
   `z["T"][a:b]` inside the per-document loop. `NpzFile.__getitem__` loads and decompresses the whole
   array on every access, and each per-document slice is a view that keeps its own full copy alive:
   about 1,000 copies of a 130 MB chunk. Fix: read `idx`, `T`, `off` once per chunk. The chunk
   checkpoints survived, so the resume lost nothing but the merge.
