"""The index as a remax_kb knowledge base: index/projects.kbi (codes, BM25 table, chunk map) + index/projects.kbc/ (passage text).
Codec: remex 4-bit Lloyd-Max over the first 256 dimensions of the leaf-ir vector (128 bytes per passage).
The .kbi also carries remex/rotation.f32 and remex/centroids.f32, so a reader without numpy (the Swift app) can score the codes."""
import os, json, struct, time, zipfile, numpy as np
from common import *
NAME, DIM, BITS = "projects", 256, 4
KBI, KBC = f"{IDX}/{NAME}.kbi", f"{IDX}/{NAME}.kbc"
SIDE_R, SIDE_C = "remex/rotation.f32", "remex/centroids.f32"

class Leaf:
    """remax_kb embedder protocol on top of the Core ML models."""
    model_id, task_adapter, pooling, full_dim, normalize_l2 = "MongoDB/mdbr-leaf-ir", "retrieval", "mean", 768, True
    prompts = {"query": QUERY_PREFIX, "document": ""}
    def __init__(self, emb, cache=None): self.emb, self.cache, self.n, self.seconds = emb, cache or {}, 0, 0.0
    def fingerprint(self): return {"model_id": self.model_id, "task_adapter": self.task_adapter, "pooling": self.pooling, "full_dim": self.full_dim}
    def encode(self, texts, *, prompt):
        if prompt == "query": return np.stack([self.emb.query(t) for t in texts])
        out = []
        for t in texts:
            v = self.cache.get(t)
            if v is None:
                t0 = time.perf_counter(); v = self.emb.doc_ids(self.emb.tok.encode(t or "empty").ids); self.seconds += time.perf_counter() - t0; self.n += 1
            out.append(np.asarray(v, np.float32))
        return np.stack(out)

def writer(leaf):
    from remax_kb.pack_v2 import KBWriter
    if os.path.exists(KBI): return KBWriter.open(name=NAME, output_dir=IDX, embedder=leaf)
    return KBWriter.create(name=NAME, output_dir=IDX, embedder=leaf, source="~/Projects write-ups", codec="remex", dim=DIM, bits=BITS)

def add_sidecar():
    """Append the rotation (dim x dim, row-major <f4) and centroid table (2^bits <f4) to the .kbi. The writer rebuilds the zip on every commit, so call after each."""
    from remex import Quantizer
    with zipfile.ZipFile(KBI) as z:
        if SIDE_R in z.namelist(): return
        b = json.loads(z.read("manifest.json"))["binarizer"]
    q = Quantizer(d=b["dim"], bits=b["bits"], seed=b["seed"])
    with zipfile.ZipFile(KBI, "a", compression=zipfile.ZIP_STORED) as z:
        z.writestr(SIDE_R, np.ascontiguousarray(q.R, "<f4").tobytes()); z.writestr(SIDE_C, np.ascontiguousarray(q.centroids, "<f4").tobytes())

class Reader:
    """Everything the search server needs, in row order (tombstoned rows stay in place and are masked)."""
    def __init__(self):
        from remax_kb.read_v2 import KB, tokenize_query
        self.kb, self.tokenize = KB.open(KBI), tokenize_query
        self.V = self.kb._remex_doc_matrix(); self.n = len(self.V)
        self.live = ~self.kb._tombstone_mask(); self.rows = np.flatnonzero(self.live)          # BM25 document number -> row
        shards, self.meta = {}, []
        for r in range(self.n):
            sid, flags, _, off, ln, _ = struct.unpack_from("<HBBQIQ", self.kb._chunk_map, r * 24)
            if sid not in shards: shards[sid] = open(f"{KBC}/shard-{sid:04d}.bin", "rb").read()
            head, _, text = shards[sid][off:off + ln].partition(b"\n")
            m = json.loads(head)["meta"]; m["text"] = text.decode().partition("\n")[2]; self.meta.append(m)
    def semantic(self, qv):
        q = qv[:self.V.shape[1]]; s = self.V @ (q / np.linalg.norm(q)); s[~self.live] = -1; return s
    def keyword(self, q):
        s = np.zeros(self.n, np.float32); toks = [t for t in self.tokenize(q) if t in self.kb._bm25.vocab_dict]
        if toks and self.kb._bm25 is not None: s[self.rows] = self.kb._bm25.get_scores(toks)
        return s
